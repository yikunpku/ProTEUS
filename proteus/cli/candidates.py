# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Build the candidate library for one round."""

import argparse
from pathlib import Path

import pandas as pd

from proteus.al.candidates import build_library, select_backgrounds
from proteus.al.variants import depth as variant_depth
from proteus.config import SEQUENCE_COL, VARIANT_COL, WT_TAG, ProteusConfig


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="protein registry YAML")
    p.add_argument("--protein", required=True)
    p.add_argument("--depth", type=int, required=True,
                   help="depth of the library to build; backgrounds are taken "
                        "from measured depth-(D-1) variants")
    p.add_argument("--labels", default=None,
                   help="labels CSV (default: the previous round's labels)")
    p.add_argument("--threshold", type=float, default=1.0,
                   help="a background must beat this fitness (default: "
                        "%(default)s, i.e. at least wild-type)")
    p.add_argument("--output", default=None,
                   help="output CSV (default: the configured candidates path)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing candidate library. Without this "
                        "an existing file is an error, because from depth 2 on "
                        "that file is also the round's library and anything "
                        "already scored or featurized against it would no "
                        "longer match.")
    return p

def refuse_to_clobber(paths, what: str) -> None:
    """Stop before replacing outputs a round may already have been built on.

    `library_for(key, depth)` IS the candidates file from depth 2 on, so a
    second run would replace the round's library while the zero-shot scores
    and features already on disk still describe the old one.
    """
    existing = [p for p in paths if Path(p).exists()]
    if not existing:
        return
    listed = "\n".join(f"    {p}" for p in existing)
    raise SystemExit(
        f"{what} already on disk:\n{listed}\n"
        f"Re-running would replace them. Anything already scored, featurized "
        f"or ordered against them would then describe a file that no longer "
        f"exists. Pass --overwrite if that is what you want, or --output to "
        f"write somewhere else.")



def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = ProteusConfig.from_yaml(args.config)
    out = Path(args.output) if args.output else cfg.candidates_csv(args.protein, args.depth)
    if not args.overwrite:
        refuse_to_clobber([out], "candidate library")
    spec = cfg.spec(args.protein)

    labels_path = Path(args.labels or cfg.labels_csv(args.protein, args.depth - 1))
    if not labels_path.exists():
        raise SystemExit(
            f"no labels at {labels_path}.\n"
            f"A depth-{args.depth} candidate library is built from what was "
            f"measured at depth {args.depth - 1}, which is why the path says "
            f"round{args.depth - 1}. Write that round's labels first, or pass "
            f"--labels to point somewhere else.")
    labels = pd.read_csv(labels_path)
    print(f"labels: {labels_path} ({len(labels)} rows)")
    counts = labels[VARIANT_COL if VARIANT_COL in labels.columns else "variant"]
    dist = counts.map(lambda v: 0 if v == WT_TAG else variant_depth(v)).value_counts()
    print(f"  measured by depth: {dict(sorted(dist.items()))}")

    wt_frame = pd.read_csv(spec.library_csv, usecols=[VARIANT_COL, SEQUENCE_COL])
    wt_rows = wt_frame[wt_frame[VARIANT_COL] == WT_TAG]
    if wt_rows.empty:
        raise ValueError(
            f"{spec.library_csv} has no {WT_TAG!r} row; the wild-type sequence "
            f"is needed to build mutated sequences")
    wt_sequence = wt_rows.iloc[0][SEQUENCE_COL]
    print(f"  wild type: {len(wt_sequence)} residues")

    backgrounds = select_backgrounds(labels, args.depth - 1, args.threshold)
    print(f"backgrounds (depth {args.depth - 1}, fitness > {args.threshold}): "
          f"{len(backgrounds)}")
    for _, row in backgrounds.head(10).iterrows():
        print(f"    {row['variant']:24} {row['fitness']:.3f}")

    library = build_library(wt_sequence, labels, args.depth,
                            threshold=args.threshold, dataset=args.protein)

    out.parent.mkdir(parents=True, exist_ok=True)
    library.to_csv(out, index=False)
    print(f"\nwrote {out}")
    print(f"  {len(library):,} rows (1 wild type + {len(library) - 1:,} "
          f"depth-{args.depth} candidates)")


if __name__ == "__main__":
    main()
