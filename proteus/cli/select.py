# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Pick the next batch: fuse three ranked channels, vote across a grid."""

import argparse
from pathlib import Path

import pandas as pd

from proteus.al.select import (DEFAULT_BATCH, DEFAULT_GRID, build_pool,
                               consensus, fuse, load_zeroshot_map, score_pool)
from proteus.al.surrogate import EmbeddingStore
from proteus.al.variants import is_wild_type
from proteus.config import VARIANT_COL, ProteusConfig


def parse_grid(text):
    """`"0,0.6;0.2,0.2"` -> [(0.0, 0.6), (0.2, 0.2)]."""
    out = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        a_zs, a_base = (float(x) for x in item.split(","))
        out.append((a_zs, a_base))
    if not out:
        raise argparse.ArgumentTypeError(f"no settings parsed from {text!r}")
    return out


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="protein registry YAML")
    p.add_argument("--protein", required=True)
    p.add_argument("--depth", type=int, required=True,
                   help="depth of the round being selected for")
    p.add_argument("--labels", default=None,
                   help="labels CSV (default: the previous round's labels)")
    p.add_argument("--embeddings", nargs="*", default=None,
                   help="embedding .npz files (default: depths 1..D of this protein)")
    p.add_argument("--zeroshot", nargs="*", default=None,
                   help="zero-shot CSVs (default: depths 1..D of this protein). "
                        "Later files win on collision.")
    p.add_argument("--score-col", default="Proteus_score",
                   help="zero-shot score column (default: %(default)s)")
    p.add_argument("-n", "--batch", type=int, default=DEFAULT_BATCH,
                   help="picks per setting and in the consensus (default: %(default)s)")
    p.add_argument("--grid", type=parse_grid, default=None,
                   help='"a_zs,a_base;..." (default: the five published settings)')
    p.add_argument("--base", choices=["max", "sum"], default="max",
                   help="how the base channel aggregates measured sub-variants. "
                        "max is the default and the right choice from depth 3 on; "
                        "sum tracks true fitness slightly better at depth 2 but "
                        "inflates with depth (default: %(default)s)")
    p.add_argument("--bad-threshold", type=float, default=1.0,
                   help="drop candidates containing a single that measured below "
                        "this (default: %(default)s; pass a negative number to disable)")
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--candidates-from", default=None,
                   help="the candidate list every pick must have an embedding "
                        "for (default: the configured candidates path). Point "
                        "it at a subset when ranking a subset is the "
                        "intention.")
    p.add_argument("--overwrite", action="store_true",
                   help="replace existing picks. Without this an "
                        "existing picks file is an error: a batch that "
                        "has already passed proteus-audit should not be "
                        "replaced in place, because the audit would "
                        "pass again on the new one.")
    p.add_argument("--write-settings", action="store_true",
                   help="also write each grid setting's own top-N")
    p.add_argument("--write-pool", action="store_true",
                   help="also write the whole ranked pool, every candidate "
                        "with all three channels and each setting's fused "
                        "score. Needed to grade a round against ground truth: "
                        "the picks file alone cannot tell you what the pool "
                        "the model actually ranked looked like.")
    return p



def refuse_to_clobber(paths, what: str) -> None:
    """Stop before replacing outputs a round may already have been built on.

    A batch that has passed `proteus-audit` must not be replaced in place:
    the audit would pass again on the new file, because its consensus group
    only checks that the file is self-consistent.
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
    grid = args.grid or list(DEFAULT_GRID)

    out = Path(args.output) if args.output else cfg.picks_csv(args.protein, args.depth)
    if not args.overwrite:
        planned = [out]
        if args.write_pool:
            planned.append(out.with_name(f"{out.stem}_pool.csv"))
        if args.write_settings:
            planned += [out.with_name(
                f"{out.stem}_zs{int(round(a * 100)):03d}_base{int(round(b * 100)):03d}.csv")
                for a, b in grid]
        refuse_to_clobber(planned, "picks")

    labels_path = Path(args.labels or cfg.labels_csv(args.protein, args.depth - 1))
    if not labels_path.exists():
        raise SystemExit(
            f"no labels at {labels_path}.\n"
            f"The surrogate trains on what was measured at depth "
            f"{args.depth - 1}, which is why the path says "
            f"round{args.depth - 1}. Write that round's labels first, or pass "
            f"--labels to point somewhere else.")
    labels = pd.read_csv(labels_path)
    print(f"labels    : {labels_path} ({len(labels)} measured)")

    emb_paths = args.embeddings or [
        cfg.embeddings_npz(args.protein, d) for d in range(1, args.depth + 1)
    ]
    stores = []
    for path in emb_paths:
        if Path(path).exists():
            store = EmbeddingStore.from_npz(path)
            print(f"embeddings: {path} ({len(store):,} variants, dim={store.dim})")
            stores.append(store)
        else:
            print(f"embeddings: {path} -- absent, skipped")
    if not stores:
        raise FileNotFoundError(f"none of these embedding files exist: {emb_paths}")
    embeddings = EmbeddingStore.concat(stores)
    print(f"            merged {len(embeddings):,} variants")

    # The pool is built by walking the embedding store, so a candidate that is
    # in the library but has no embedding is never seen -- not counted, not
    # warned about. A featurize run killed half way therefore produced a
    # perfectly ordinary-looking batch chosen from half the library. A measured
    # variant in the same position already raises; this makes the two
    # symmetric.
    cand_path = (Path(args.candidates_from) if args.candidates_from
                 else cfg.candidates_csv(args.protein, args.depth))
    if cand_path.exists():
        wanted = [v for v in pd.read_csv(cand_path)[VARIANT_COL]
                  if not is_wild_type(v)]
        # `v in embeddings` hits EmbeddingStore's dict. Rebuilding a set inside
        # the comprehension made this O(n*m): 20k candidates against 20k
        # embeddings took 19 s, 40k took 58 s, for a check that should be
        # instant.
        absent = [v for v in wanted if v not in embeddings]
        if absent:
            # Deliberately not "pass --embeddings instead": this check reads the
            # candidates CSV, which is the other side of the comparison, so a
            # narrower --embeddings makes the message repeat rather than go
            # away. When the narrower pool is the intention, say so with
            # --candidates-from.
            raise SystemExit(
                f"{len(absent)} of {len(wanted)} candidates in "
                f"{cand_path.name} have no embedding (e.g. {absent[:5]}). "
                f"Selecting now would rank only the ones that happen to be "
                f"featurized. Finish featurize for this depth, or point "
                f"--candidates-from at the subset you mean to rank.")

    zs_paths = [str(p) for p in (args.zeroshot or [
        cfg.zeroshot_csv(args.protein, d) for d in range(1, args.depth + 1)
    ]) if Path(p).exists()]
    if not zs_paths:
        raise FileNotFoundError("no zero-shot CSV found; run `proteus-score` first")
    zs_map = load_zeroshot_map(zs_paths, score_col=args.score_col)
    print(f"zero-shot : {len(zs_map):,} scores from {len(zs_paths)} file(s)")

    bad = None if args.bad_threshold < 0 else args.bad_threshold
    names, feats = build_pool(embeddings, labels, max_depth=args.depth,
                              bad_threshold=bad)
    print(f"pool      : {len(names):,} unseen candidates of depth 1..{args.depth}")

    channels = score_pool(names, feats, labels, embeddings, zs_map,
                          n_estimators=args.trees, seed=args.seed,
                          base_agg=args.base)
    n_missing = int(channels["zs_missing"].sum())
    if n_missing:
        print(f"            {n_missing:,} candidates have no zero-shot score "
              f"(ranked last on that channel)")

    out.parent.mkdir(parents=True, exist_ok=True)

    if args.write_pool:
        pool = channels.copy()
        finals = []
        for a_zs, a_base in grid:
            tag = f"zs{int(round(a_zs * 100)):03d}_base{int(round(a_base * 100)):03d}"
            pool[f"final_{tag}"] = fuse(channels, a_zs, a_base)
            finals.append(f"final_{tag}")
        # Deliberately NOT `mean_final_score`: the picks file has a column of
        # that name meaning the mean over the settings that picked the variant,
        # which for a variant with few votes is higher than the mean over all
        # of them. Two files, one name, two quantities is how someone compares
        # them and concludes the tool is inconsistent.
        pool["mean_final_all_settings"] = pool[finals].mean(axis=1)
        pool = pool.sort_values("mean_final_all_settings", ascending=False)
        pool_path = out.with_name(f"{out.stem}_pool.csv")
        pool.to_csv(pool_path, index=False)
        print(f"            wrote {pool_path}  ({len(pool):,} candidates)")

    if args.write_settings:
        for a_zs, a_base in grid:
            tag = f"zs{int(round(a_zs * 100)):03d}_base{int(round(a_base * 100)):03d}"
            top = (channels.assign(final_score=fuse(channels, a_zs, a_base))
                           .sort_values("final_score", ascending=False)
                           .head(args.batch))
            path = out.with_name(f"{out.stem}_{tag}.csv")
            top.to_csv(path, index=False)
            print(f"            wrote {path}")

    picks = consensus(channels, grid=grid, n=args.batch)
    picks.head(args.batch).to_csv(out, index=False)
    print(f"\nwrote {out}  (top {args.batch} by consensus over {len(grid)} settings)")
    print(picks.head(10)[["variant", "votes", "best_rank", "mut_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
