#!/usr/bin/env python3
"""
MSA processing utilities for Proteus zero-shot fitness prediction.
"""

import random
from typing import List, Optional, Tuple
import numpy as np
from Bio import SeqIO
from proteus.prepare import a3m_tools
from proteus.third_party.proteingym_msa import MSA_processing


def process_msa(filename: str, weight_filename: Optional[str], filter_msa: bool = False,
                path_to_hhfilter: str = None, hhfilter_min_cov: int = 75,
                hhfilter_max_seq_id: int = 100, hhfilter_min_seq_id: int = 0):
    """Process MSA with optional weighting and quality filtering."""
    # MSA_processing assumes a rectangular alignment; a true a3m is not one, and
    # it would be truncated into shape with every residue after an insertion
    # shifted. See proteus.prepare.a3m_tools. `--legacy-a3m` skips this.
    filename, weight_filename = a3m_tools.resolve(filename, weight_filename)

    MSA = MSA_processing(
        MSA_location=filename,
        use_weights=True,
        weights_location=weight_filename
    )

    # Apply HHfilter if requested
    if filter_msa and path_to_hhfilter:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.a3m', delete=False) as tmp:
            # Write MSA to temp file
            for name, seq in MSA.seq_name_to_sequence.items():
                tmp.write(f">{name}\n{seq}\n")
            tmp_path = tmp.name

        # Run hhfilter
        filtered_path = tmp_path + "_filtered"
        cmd = [
            path_to_hhfilter,
            '-i', tmp_path,
            '-o', filtered_path,
            '-cov', str(hhfilter_min_cov),
            '-id', str(hhfilter_max_seq_id),
            '-diff', str(100 - hhfilter_min_seq_id)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            # Reload filtered MSA
            MSA = MSA_processing(
                MSA_location=filtered_path,
                use_weights=True,
                weights_location=weight_filename
            )
        except:
            pass  # Fall back to unfiltered MSA

    return MSA


def sample_msa(filename: str, nseq: int, random_seed: int,
               weight_filename: Optional[str] = None, processed_msa=None) -> List[Tuple[str, str]]:
    """Sample sequences from MSA using sequence-reweighting strategy."""
    random.seed(random_seed)

    # Same rewrite as in process_msa, for the paths that reach the file
    # directly: the SeqIO fallback below and the MSA_processing call.
    filename, weight_filename = a3m_tools.resolve(filename, weight_filename)

    if processed_msa is None:
        if weight_filename is None:
            # Fallback to random sampling
            msa = [
                (record.description, str(record.seq).upper())
                for record in SeqIO.parse(filename, "fasta")
            ]
            nseq = min(len(msa), nseq)
            msa = random.sample(msa, nseq)
        else:
            try:
                MSA = MSA_processing(
                    MSA_location=filename,
                    use_weights=True,
                    weights_location=weight_filename
                )
            except:
                # Fallback to random sampling
                msa = [
                    (record.description, str(record.seq).upper())
                    for record in SeqIO.parse(filename, "fasta")
                ]
                nseq = min(len(msa), nseq)
                msa = random.sample(msa, nseq)
                return msa
    else:
        MSA = processed_msa

    # Keep WT
    msa = [(MSA.focus_seq_name, MSA.raw_seq_name_to_sequence[MSA.focus_seq_name])]

    # Sample non-WT by weights
    non_wt_weights = np.array([
        w for k, w in MSA.seq_name_to_weight.items()
        if k != MSA.focus_seq_name
    ])
    non_wt_sequences = [
        (k, s) for k, s in MSA.seq_name_to_sequence.items()
        if k != MSA.focus_seq_name
    ]

    if len(non_wt_sequences) > 0:
        non_wt_weights = non_wt_weights / non_wt_weights.sum()
        msa.extend(random.choices(non_wt_sequences, weights=non_wt_weights, k=nseq-1))

    # Ensure uppercase and consistent lengths
    msa = [(desc, seq.upper()) for desc, seq in msa]

    # Filter to keep only sequences with same length as WT
    wt_length = len(msa[0][1])
    msa_filtered = [item for item in msa if len(item[1]) == wt_length]

    if len(msa_filtered) < 2:
        return None

    return msa_filtered
