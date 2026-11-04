#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Turn a benchmark library's ground truth into round-1 labels.

On a real campaign, labels come from the wet lab. On a benchmark they can be
taken from the library's own `DMS_score`, which lets you drive the whole loop
in silico and then check the picks against measurements the loop never saw.

Only variants up to `--depth` are revealed; everything deeper stays hidden, so
the surrogate is in the same position it would be in after one real round.
"""

import argparse
from pathlib import Path

import pandas as pd

from proteus.al.variants import WT_TAG, depth as variant_depth, is_wild_type
from proteus.config import VARIANT_COL, ProteusConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--depth", type=int, default=1,
                    help="reveal variants up to this depth (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N revealed variants, to simulate a "
                         "realistically small first batch")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = ProteusConfig.from_yaml(args.config)
    spec = cfg.spec(args.protein)
    lib = pd.read_csv(spec.library_csv)
    for col in (VARIANT_COL, "DMS_score"):
        if col not in lib.columns:
            raise SystemExit(f"{spec.library_csv}: no {col!r} column")

    lib = lib.dropna(subset=["DMS_score"])
    lib["depth"] = lib[VARIANT_COL].map(lambda v: 0 if is_wild_type(v) else variant_depth(v))
    revealed = lib[(lib["depth"] > 0) & (lib["depth"] <= args.depth)]
    if revealed.empty:
        raise SystemExit(f"no variants of depth 1..{args.depth} in the library")
    if args.limit:
        revealed = revealed.head(args.limit)

    wt = lib[lib["depth"] == 0]
    if wt.empty:
        raise SystemExit("library has no wild-type row")

    # Fitness is WT-normalized downstream: the "better than wild type"
    # threshold is 1.0, so divide through by the wild type's own score rather
    # than passing the raw benchmark scale along.
    wt_score = float(wt.iloc[0]["DMS_score"])
    if wt_score <= 0:
        raise SystemExit(
            f"wild-type DMS_score is {wt_score}; cannot normalize by it. Rescale "
            f"the library so the wild type is positive.")

    rows = [{"variant": WT_TAG, "fitness": 1.0, "round": 0}]
    for _, r in revealed.iterrows():
        rows.append({"variant": r[VARIANT_COL],
                     "fitness": float(r["DMS_score"]) / wt_score,
                     "round": 0})
    out_df = pd.DataFrame(rows)

    out = Path(args.output) if args.output else cfg.labels_csv(args.protein, args.depth)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)

    # No `- 1` for the wild type: its fitness is exactly 1.0 by construction
    # (it is the normalizer, and proteus-audit asserts it), so `> 1.0` already
    # excludes it. The subtraction undercounted by one, and the demo's expected
    # output was written down from this line.
    better = int((out_df["fitness"] > 1.0).sum())
    print(f"wrote {out}")
    print(f"  {len(out_df)} labels (1 wild type + {len(out_df) - 1} measured)")
    print(f"  wild-type raw DMS_score {wt_score:.4f} -> fitness 1.0")
    print(f"  {better} revealed variants beat the wild type; those become the "
          f"backgrounds for depth {args.depth + 1}")


if __name__ == "__main__":
    main()
