import argparse
from pathlib import Path
import pandas as pd
import torch
import shutil

from proteus.prepare import a3m_tools
from proteus.zeroshot.scorer import ProteusZeroShot
from proteus.zeroshot.metrics import calculate_all_metrics, print_metrics


def main():
    parser = argparse.ArgumentParser(description="Proteus Zero-Shot Fitness Prediction")
    parser.add_argument("--dms-file", type=str, required=True,
                       help="Path to DMS file with mutation data")
    parser.add_argument("--seq-basemodel", type=str, default="esm2_t36_3B_UR50D",
                       help="sequence base encoder (esm2_t36_3B_UR50D or esmc_600m)")
    parser.add_argument("--checkpoint", type=str,
                       help="Proteus model checkpoint path")
    parser.add_argument("--msa-dir", type=str,
                       help="Directory containing MSA files")
    parser.add_argument("--structure-dir", type=str,
                       help="Directory with PDB structure files")
    parser.add_argument("--mt-tokens-dir", type=str,
                       help="Directory with pre-computed MT structure tokens (from Boltz)")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output CSV file path")
    parser.add_argument("--structure-score-dir", type=str, default=None,
                       help="OPTIONAL: directory of precomputed per-variant structure "
                            "scores (CSV with mutant + *_score). When given, it replaces "
                            "the ESM3-token structure modality (used for models whose "
                            "structure modality is ESM-IF1 conditional log-likelihood). "
                            "Omit it to keep the default behaviour unchanged.")

    # Structure perturbation parameters (legacy)
    parser.add_argument("--use-structure-perturbation", action="store_true", default=False,
                       help="Use legacy structure perturbation signal (default: False, deprecated)")
    parser.add_argument("--no-structure-perturbation", action="store_true",
                       help="Disable structure perturbation signal")
    parser.add_argument("--lambda-perturbation", type=float, default=0.1,
                       help="Weight for structure perturbation penalty (default: 0.1)")
    parser.add_argument("--perturbation-window", type=int, default=3,
                       help="Window size around mutation for perturbation (default: 3)")
    parser.add_argument("--perturbation-sigma", type=float, default=1.5,
                       help="Gaussian sigma for distance weighting (default: 1.5)")

    # MT structure-conditioned scoring (NEW)
    parser.add_argument("--use-mt-structure-score", action="store_true", default=True,
                       help="Use MT structure-conditioned scoring (default: True)")
    parser.add_argument("--no-mt-structure-score", action="store_true",
                       help="Disable MT structure-conditioned scoring")
    parser.add_argument("--lambda-mt-structure", type=float, default=0.3,
                       help="Weight for MT structure score contribution (default: 0.3)")
    parser.add_argument("--legacy-a3m", action="store_true",
                       help="Read the alignment exactly as it is on disk. The "
                            "MSA reader assumes a rectangular alignment, so a "
                            "true a3m is otherwise rewritten into that shape "
                            "first; without the rewrite every residue after a "
                            "row's first insertion sits in the wrong column. "
                            "Use this only to reproduce numbers produced "
                            "before the rewrite existed.")

    args = parser.parse_args()

    a3m_tools.set_legacy(args.legacy_a3m)

    # Handle --no-structure-perturbation flag
    use_structure_perturbation = not args.no_structure_perturbation

    # Handle --no-mt-structure-score flag
    use_mt_structure_score = not args.no_mt_structure_score

    # Fixed default parameters
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    msa_samples = 128
    num_seeds = 5
    use_sequence = True
    use_msa = True
    use_structure = True
    offset_idx = 1
    temperature = 1.0
    use_conservation = True
    use_epistasis = True
    binary_threshold = 0.0

    # Auto-detect hhfilter in environment
    path_to_hhfilter = shutil.which('hhfilter')
    filter_msa = path_to_hhfilter is not None
    if filter_msa:
        print(f"hhfilter detected at: {path_to_hhfilter}")
    else:
        print("hhfilter not found in PATH, MSA filtering will be disabled")


    df = pd.read_csv(args.dms_file)
    dataset_name = Path(args.dms_file).stem


    predictor = ProteusZeroShot(
        checkpoint=args.checkpoint,
        use_sequence=use_sequence,
        use_msa=use_msa,
        use_structure=use_structure,
        structure_dir=args.structure_dir,
        device=device,
        temperature=temperature,
        use_conservation=use_conservation,
        seq_basemodel=args.seq_basemodel,
        structure_score_dir=args.structure_score_dir
    )


    scores = predictor.score(
        dms_file=args.dms_file,
        msa_dir=args.msa_dir,
        msa_weights_dir=args.msa_dir,
        structure_dir=args.structure_dir,
        mt_tokens_dir=args.mt_tokens_dir,
        msa_samples=msa_samples,
        offset_idx=offset_idx,
        num_seeds=num_seeds,
        use_epistasis=use_epistasis,
        filter_msa=filter_msa,
        path_to_hhfilter=path_to_hhfilter,
        use_structure_perturbation=use_structure_perturbation,
        lambda_perturbation=args.lambda_perturbation,
        perturbation_window=args.perturbation_window,
        perturbation_sigma=args.perturbation_sigma,
        use_mt_structure_score=use_mt_structure_score,
        lambda_mt_structure=args.lambda_mt_structure
    )

    # Add scores 
    df['Proteus_score'] = scores

    # Evaluate if DMS scores are available
    # Before anything is written: the metrics sidecar below lands in this same
    # directory, and writing it into a directory that does not exist yet would
    # throw away a whole scoring run at the very last step.
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    if 'DMS_score' in df.columns:
        # Remove NaN values for evaluation
        valid_df = df[['Proteus_score', 'DMS_score']].dropna()

        if len(valid_df) > 1:
            y_true = valid_df['DMS_score'].values
            y_pred = valid_df['Proteus_score'].values

            # Calculate all metrics
            metrics = calculate_all_metrics(
                y_true=y_true,
                y_pred=y_pred,
                binary_threshold=binary_threshold
            )

            # Print metrics
            print_metrics(metrics, title=f"Evaluation Results: {dataset_name}")

            # Save metrics to a separate file
            metrics_file = Path(args.output_file).parent / f"{dataset_name}_metrics.csv"
            metrics_df = pd.DataFrame([metrics])
            metrics_df.insert(0, 'dataset', dataset_name)
            metrics_df.insert(1, 'num_samples', len(valid_df))
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Metrics saved to: {metrics_file}")
        else:
            print(f"\nWarning: Only {len(valid_df)} valid data points for evaluation")

    # Save 
    df.to_csv(args.output_file, index=False)
    print(f"\nPredictions saved to: {args.output_file}")
    print(f"Total variants scored: {len(df)}")


if __name__ == "__main__":
    main()
