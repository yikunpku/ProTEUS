#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Assemble the CreiLOV demo inputs from a CombinGym checkout.

CombinGym (MIT, https://www.combingym.org/) ships, for each protein, a cleaned
DMS table, a wild-type sequence, a structure and an MSA. Nothing is
redistributed here; point this at your own copy.

    git clone <combingym>  # or download the data release
    python examples/prepare_creilov.py --combingym /path/to/CombinGym --out examples/data

Produces two tables, and the split matters:

    <out>/libraries/CreiLOV.csv        the wild type and the 20 measured singles
    <out>/libraries/CreiLOV_full.csv   every measured variant, all depths
    <out>/proteins/CreiLOV/{fasta,structure,msa}/...   wild-type inputs

`CreiLOV.csv` is what the config points `library_csv` at, so it is what the
pipeline sees -- and `library_for(key, 1)` returns it, which means every
featurize step iterates it. It therefore has to hold the round-1 library and
nothing deeper: pointed at the whole 165,428-row combinatorial table instead,
`featurize --depth 1` builds a mutant structure and four feature arrays for
every one of them (~12 GB of PDBs and ~250 GB of `msa_emb_128`) to train a
surrogate on 20 labels.

`CreiLOV_full.csv` is the ground truth, and only `restrict_to_library.py` and
`score_picks.py` read it -- both by way of an explicit `--library`, so the
loop itself never has access to the answers for anything it has not been
told about.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

# CombinGym's CreiLOV files and the column its fitness lives in.
CLEAN_XLSX = "DMS/Clean/CreiLOV_clean.xlsx"
FASTA = "WT Sequence/CreiLOV.fasta"
PDB = "Structure/CreiLOV.pdb"
# The MSA comes at several coverage thresholds; b0.7 is the one used here.
A2M = "MSA/CreiLOV_b0.7.a2m"
FITNESS_COL = "mean_log"
# CreiLOV genotypes carry a trailing stop character.
STRIP = "*"

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
WT_TAG = "wild_type"


def read_fasta(path: Path) -> str:
    seq = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(">"):
                seq.append(line.strip())
    out = "".join(seq)
    if not out:
        raise ValueError(f"{path}: no sequence found")
    return out


def diff_to_mutant(wt: str, mutated: str) -> str:
    """Canonical variant name from a full mutated sequence."""
    if len(wt) != len(mutated):
        raise ValueError(f"length {len(mutated)} != wild-type {len(wt)}")
    subs = [f"{a}{i + 1}{b}" for i, (a, b) in enumerate(zip(wt, mutated)) if a != b]
    bad = [s for s in subs if s[0] not in AMINO_ACIDS or s[-1] not in AMINO_ACIDS]
    if bad:
        raise ValueError(f"non-standard residue in {bad[:3]}")
    return WT_TAG if not subs else ":".join(subs)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combingym", required=True, type=Path,
                    help="a CombinGym checkout (must contain Data/)")
    ap.add_argument("--out", required=True, type=Path, help="output data root")
    args = ap.parse_args()

    data = args.combingym / "Data"
    if not data.is_dir():
        data = args.combingym
    missing = [p for p in (CLEAN_XLSX, FASTA, PDB, A2M) if not (data / p).exists()]
    if missing:
        raise SystemExit(
            f"not found under {data}: {missing}\n"
            f"Pass --combingym pointing at a CombinGym checkout containing Data/.")

    prot = args.out / "proteins" / "CreiLOV"
    for sub in ("fasta", "structure", "msa"):
        (prot / sub).mkdir(parents=True, exist_ok=True)
    (args.out / "libraries").mkdir(parents=True, exist_ok=True)

    shutil.copy(data / FASTA, prot / "fasta" / "CreiLOV.fasta")
    shutil.copy(data / PDB, prot / "structure" / "CreiLOV.pdb")
    # An .a2m is an aligned FASTA; the MSA reader accepts it under either name.
    shutil.copy(data / A2M, prot / "msa" / "CreiLOV.a3m")
    wt = read_fasta(prot / "fasta" / "CreiLOV.fasta")
    print(f"wild type: {len(wt)} residues")

    df = pd.read_excel(data / CLEAN_XLSX)
    df["genotype"] = df["genotype"].astype(str).str.rstrip(STRIP)
    before = len(df)
    df = df.dropna(subset=[FITNESS_COL]).reset_index(drop=True)
    if len(df) != before:
        print(f"dropped {before - len(df)} rows with no {FITNESS_COL}")

    df["mutated_sequence"] = df["genotype"]
    df["mutant"] = [diff_to_mutant(wt, s) for s in df["mutated_sequence"]]

    # Min-max to [0, 1]. Fitness is used only for ranking and for the surrogate,
    # so the scale is arbitrary -- but it must be recorded, because the base
    # channel's "better than wild type" threshold is read on this scale.
    f = df[FITNESS_COL].astype(float)
    df["DMS_score"] = (f - f.min()) / (f.max() - f.min())
    df["dataset"] = "CreiLOV"

    is_wt = df["mutant"] == WT_TAG
    if not is_wt.any():
        raise SystemExit(
            "the cleaned table has no wild-type row; every downstream step needs "
            "one, so add it before continuing")
    print(f"wild-type DMS_score after rescaling: "
          f"{float(df.loc[is_wt, 'DMS_score'].iloc[0]):.4f}")

    cols = ["dataset", "mutant", "mutated_sequence", "DMS_score"]
    # Wild type first: zero-shot scoring reads it as the reference row.
    ordered = pd.concat([df.loc[is_wt, cols], df.loc[~is_wt, cols]],
                        ignore_index=True)

    full_csv = args.out / "libraries" / "CreiLOV_full.csv"
    ordered.to_csv(full_csv, index=False)
    print(f"{full_csv}: {len(ordered):,} measured variants (ground truth)")

    # The round-1 library: the wild type and the singles, nothing deeper. See
    # this script's docstring for why the depth matters here.
    depth = ordered["mutant"].map(
        lambda m: 0 if m == WT_TAG else m.count(":") + 1)
    out_csv = args.out / "libraries" / "CreiLOV.csv"
    round1 = ordered[depth <= 1]
    round1.to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(round1)} rows (wild type + "
          f"{len(round1) - 1} singles) -- this is `library_csv`")

    depths = ordered[ordered["mutant"] != WT_TAG]["mutant"].str.count(":").add(1)
    print(f"  {full_csv.name} depth histogram: "
          f"{dict(sorted(depths.value_counts().items()))}")
    print("\nnow run:  proteus-doctor --config examples/creilov.yaml")


if __name__ == "__main__":
    main()
