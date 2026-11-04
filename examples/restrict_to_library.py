#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Keep only the candidates a benchmark library has measurements for.

This is for benchmarking, not for a real campaign. `proteus-candidates`
proposes every +1 substitution on every background that beat the wild type,
which for a 119-residue protein is tens of thousands of variants -- correct
for the wet lab, where the point is to search widely, but useless for
measuring the loop against ground truth, because almost none of them were
ever assayed.

A combinatorial benchmark library (CombinGym, say) assays combinations of a
fixed, small set of substitutions. Intersecting the proposal with it leaves a
pool where every candidate has a measured fitness, so the picks can be scored.
It also makes a demo run cheap: a few hundred variants to featurize instead of
tens of thousands.

The scores stay hidden -- only `mutant` and `mutated_sequence` are kept, the
same columns a real round would have. Nothing downstream can see the answers.

    python examples/restrict_to_library.py --config examples/creilov.yaml \\
        --protein CreiLOV --depth 2
"""

import argparse

import pandas as pd

from proteus.al.variants import canonical, depth as variant_depth, is_wild_type
from proteus.config import SEQUENCE_COL, VARIANT_COL, ProteusConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--library", default=None,
                    help="measured library (default: the protein's library_csv)")
    ap.add_argument("--output", default=None,
                    help="default: overwrite the candidates file for this depth")
    args = ap.parse_args()

    cfg = ProteusConfig.from_yaml(args.config)
    spec = cfg.spec(args.protein)
    cand_path = cfg.candidates_csv(args.protein, args.depth)
    if not cand_path.exists():
        raise SystemExit(
            f"no candidate library at {cand_path}.\n"
            f"Run `proteus-candidates --config ... --protein {args.protein} "
            f"--depth {args.depth}` first.")
    cand = pd.read_csv(cand_path)
    lib = pd.read_csv(args.library or spec.library_csv)

    measured = {canonical(m.split(":")) for m in lib[VARIANT_COL]
                if not is_wild_type(m) and variant_depth(m) == args.depth}
    if not measured:
        raise SystemExit(f"the library has no measured depth-{args.depth} variants")

    keep = cand[VARIANT_COL].map(
        lambda m: is_wild_type(m) or canonical(m.split(":")) in measured)
    out = cand[keep][[c for c in (
        "dataset", VARIANT_COL, SEQUENCE_COL) if c in cand.columns]]

    dest = args.output or cand_path
    out.to_csv(dest, index=False)

    n_cand = int((~out[VARIANT_COL].map(is_wild_type)).sum())
    print(f"wrote {dest}")
    print(f"  {len(cand) - 1} proposed -> {n_cand} measured in the library")
    print(f"  (the library has {len(measured)} depth-{args.depth} variants; the "
          f"{len(measured) - n_cand} not kept were never proposed, because no "
          f"background that beat the wild type leads to them)")


if __name__ == "__main__":
    main()
