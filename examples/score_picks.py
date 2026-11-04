#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Score a round's picks against measurements the loop never saw.

Only possible on a benchmark. The point of a round is to choose which variants
to make, and on a real protein you find out how good the choice was by making
them. Here the whole library was already assayed, so the answer is on disk and
the selection can be graded directly.

Four numbers, which answer different questions:

    hit rate            of the picks, how many beat the wild type. What the
                        wet lab cares about: a round that returns 30 variants
                        of which 25 are improvements was a good round.
    fraction of truth   how much of the pool's true top-n the picks found.
                        1.0 means a perfect choice; a random choice of n from
                        a pool of N scores about n/N.
    best found          the best measured fitness among the picks, against the
                        best in the pool. A round can score poorly on overlap
                        and still find the single best variant.
    Spearman            rank correlation between each score column and measured
                        fitness, over the picks. Note the restriction: the
                        picks file only holds the top n, so this says how well
                        the model orders the variants it already chose, not how
                        well it orders the whole pool. Expect it to look weak
                        even for a good round -- the picks are a truncated,
                        selected sample, which is exactly the regime where rank
                        correlation is least informative.

    python examples/score_picks.py --config examples/creilov.yaml \\
        --protein CreiLOV --depth 2 --library .../CreiLOV_full.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from proteus.al.variants import canonical, is_wild_type
from proteus.config import VARIANT_COL, ProteusConfig


def key(variant: str) -> str:
    return variant if is_wild_type(variant) else canonical(variant.split(":"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--library", default=None,
                    help="measured library (default: the protein's library_csv)")
    ap.add_argument("--picks", default=None,
                    help="picks CSV (default: the configured path for this depth)")
    ap.add_argument("--pool", default=None,
                    help="the ranked pool written by `proteus-select "
                         "--write-pool` (default: next to the picks file). "
                         "Without it the baselines are computed over the whole "
                         "candidate library instead, which is the wrong "
                         "denominator whenever --bad-threshold dropped any "
                         "candidate.")
    args = ap.parse_args()

    cfg = ProteusConfig.from_yaml(args.config)
    lib = pd.read_csv(args.library or cfg.spec(args.protein).library_csv)
    picks_path = Path(args.picks or cfg.picks_csv(args.protein, args.depth))
    picks = pd.read_csv(picks_path)

    # What the model actually ranked. `select` drops candidates built on a
    # single that measured badly, so the candidate library is a superset of the
    # pool -- scoring against it would understate the random baseline and
    # flatter the round.
    pool_path = Path(args.pool) if args.pool else picks_path.with_name(
        f"{picks_path.stem}_pool.csv")
    ranked = None
    if pool_path.exists():
        ranked = pd.read_csv(pool_path)
        pool = ranked.rename(columns={"variant": VARIANT_COL})
        print(f"pool from  {pool_path}")
    else:
        pool = pd.read_csv(cfg.candidates_csv(args.protein, args.depth))
        print(f"no ranked pool at {pool_path}; falling back to the full "
              f"candidate library. Re-run proteus-select with --write-pool "
              f"for correct baselines.")

    wt = lib[lib[VARIANT_COL].map(is_wild_type)]
    if wt.empty:
        raise SystemExit("the library has no wild-type row to normalize by")
    wt_score = float(wt.iloc[0]["DMS_score"])

    truth = {key(m): s / wt_score for m, s in zip(lib[VARIANT_COL], lib["DMS_score"])
             if not is_wild_type(m)}

    pool_keys = [key(m) for m in pool[VARIANT_COL] if not is_wild_type(m)]
    measured_pool = {k: truth[k] for k in pool_keys if k in truth}
    if not measured_pool:
        raise SystemExit(
            "no candidate in the pool has a measurement; did you run "
            "restrict_to_library.py, and is --library the full table?")

    # The picks table is one of the pipeline's own, so its column is `variant`;
    # `VARIANT_COL` ("mutant") is the library/zero-shot convention used above.
    pick_keys = [key(m) for m in picks["variant"]]
    scored = [(k, measured_pool[k]) for k in pick_keys if k in measured_pool]
    n = len(pick_keys)

    print(f"picks: {n}  ({len(scored)} with a measurement)")
    print(f"pool:  {len(measured_pool)} measured of {len(pool_keys)} candidates")
    if not scored:
        raise SystemExit("none of the picks are measured; nothing to score")

    fits = [f for _, f in scored]
    better = sum(1 for f in fits if f > 1.0)
    top_n = {k for k, _ in sorted(measured_pool.items(), key=lambda kv: -kv[1])[:n]}
    overlap = len({k for k, _ in scored} & top_n)
    best_pool = max(measured_pool.values())

    print()
    print(f"hit rate          {better}/{len(scored)} = {better / len(scored):.3f} "
          f"beat the wild type (pool baseline "
          f"{sum(1 for f in measured_pool.values() if f > 1.0) / len(measured_pool):.3f})")
    print(f"fraction of truth {overlap}/{n} = {overlap / n:.3f} of the true top-{n} "
          f"(random baseline {n / len(measured_pool):.3f})")
    print(f"best found        {max(fits):.4f}  (best in pool {best_pool:.4f})")
    print(f"mean picked       {sum(fits) / len(fits):.4f}  (pool mean "
          f"{sum(measured_pool.values()) / len(measured_pool):.4f})")

    # Rank quality. Over the whole pool when `select --write-pool` gave us
    # every candidate's score; otherwise only over the picks, which is a
    # truncated selected sample and reads much weaker than the model deserves.
    table, scope = (ranked, "pool") if ranked is not None else (picks, "picks")
    names = [key(m) for m in table["variant"]]
    print()
    # Named explicitly as well as by pattern: the pool's aggregate column is
    # `mean_final_all_settings` (the picks file's `mean_final_score` means the
    # mean over the settings that picked the variant, which is a different
    # quantity), and it matches neither suffix below.
    score_cols = [c for c in table.columns
                  if c.endswith("_score") or c == "votes" or c.startswith("final_")
                  or c == "mean_final_all_settings"]
    for col in score_cols:
        rows = [(table[col].iloc[i], measured_pool[k])
                for i, k in enumerate(names) if k in measured_pool]
        if len(rows) > 2:
            rho = spearmanr([r[0] for r in rows], [r[1] for r in rows]).statistic
            print(f"spearman({col:<18}) over the {scope} (n={len(rows)}): {rho:+.4f}")


if __name__ == "__main__":
    main()
