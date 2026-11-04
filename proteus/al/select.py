# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Batch selection: three ranked channels, fused, then voted across a grid.

Every unseen candidate gets three signals, each mapped to a normalized rank
over the unseen pool:

  rank_rf    supervised -- random forest on measured variants' embeddings
  rank_zs    unsupervised -- the zero-shot score
  rank_base  the best fitness already measured among the candidate's own
             sub-variants, which anchors picks on proven backgrounds and
             compensates for mean-pooling diluting a strong sub-variant

    final = a_zs * rank_zs + a_rf * rank_rf + a_base * rank_base,
            a_rf = 1 - a_zs - a_base

A single setting of (a_zs, a_base) is an arbitrary choice, so the batch that
gets ordered is the consensus: run the grid, take each setting's top-N, and
count how many of them each variant appears in.

This replaces the per-round `select_round{R}*.py` scripts, which differed only
in the depth they hardcoded.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from proteus.al.surrogate import EmbeddingStore, RandomForestSurrogate
from proteus.al.variants import (depth, is_wild_type, measured_sub_variants,
                                 split_variant)

# The grid whose consensus produced the ordered batches. Five Strategy-D
# settings as (a_zs, a_base); a_rf is the remainder.
DEFAULT_GRID: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.6), (0.0, 0.2), (0.2, 0.2), (0.0, 0.4), (0.2, 0.4),
)
DEFAULT_BATCH = 30


def load_zeroshot_map(paths: Iterable[str],
                      score_col: str = "Proteus_score") -> Dict[str, float]:
    """Merge zero-shot CSVs into mutant -> score. Later files win collisions."""
    merged: Dict[str, float] = {}
    for path in paths:
        df = pd.read_csv(path)
        if "mutant" not in df.columns:
            raise KeyError(f"{path}: no 'mutant' column")
        col = score_col if score_col in df.columns else None
        if col is None:
            candidates = [c for c in df.columns if c.endswith("_score")
                          and c != "DMS_score"]
            if len(candidates) != 1:
                raise KeyError(
                    f"{path}: cannot find the zero-shot score column; "
                    f"looked for {score_col!r}, saw {list(df.columns)}")
            col = candidates[0]
        for mutant, score in zip(df["mutant"], df[col]):
            if pd.notna(score):
                merged[str(mutant)] = float(score)
    return merged


def zeroshot_for(variant: str, zs_map: Dict[str, float]) -> Optional[float]:
    """Direct zero-shot score, else the sum over constituent singles.

    Deep variants are often absent from a zero-shot run that only covered the
    new depth, so fall back to summing their singles. Returns None when even
    that is impossible; the caller ranks those last rather than guessing.
    """
    if variant in zs_map:
        return zs_map[variant]
    subs = split_variant(variant)
    if not subs:
        return None
    values = [zs_map[s] for s in subs if s in zs_map]
    if len(values) != len(subs):
        return None
    return float(sum(values))


def _normalized_rank(values: np.ndarray,
                     groups: Optional[Sequence[int]] = None) -> np.ndarray:
    """Rank to (0, 1], within `groups` when given.

    Ranking the whole pool at once is wrong as soon as the pool holds more than
    one depth, because two of the three channels are not comparable across
    depths:

    - `zs_score` is additive over substitutions, so a deeper variant scores
      lower whatever its quality. Measured on one round: depth-1 candidates
      averaged -5.20 and depth-2 candidates -9.88.
    - `base_fitness` is the best measured sub-variant, and a candidate whose
      sub-variants have not been measured is floored. Every depth-1 candidate
      is in that position by construction -- a single has no proper
      sub-variant -- so they all land at the bottom.

    Fused at the published `(a_zs=0, a_base=0.6)` setting those two put a
    shallow candidate out of reach of any surrogate score: 0.4 * 1.0 + 0.6 *
    (bottom) loses to a thoroughly mediocre deep one. Ranking inside each depth
    makes the channels say "good for its depth", which is the only thing they
    can say, and leaves the cross-depth choice to the weights.

    Grouped ranks use the midpoint convention, `(rank - 0.5) / size`, rather
    than `rank / size`. That matters because group sizes differ a lot: a deep
    round's pool is mostly its newest depth, with a handful of unmeasured
    shallower candidates left over. Under `rank / size` the best member of a
    group of one scores exactly 1.0 however bad it is, and the best of a group
    of three scores 1.0 too -- so a stale leftover depth outranks the entire
    newest depth on two of the three channels. The midpoint convention puts a
    lone member at 0.5, which is what a group of one actually says: nothing.

    On a single group this is `rank / size` shifted by the constant `0.5 / size`
    for every candidate alike, so the fused score moves by a constant and the
    ordering, the per-setting top-N and the consensus are unchanged -- which is
    what the published rounds were, every one of them a single-depth pool.
    Ungrouped ranking is left as `rank / size` so `rank_rf` is untouched.

    The multi-depth path is the one no published round exercised, since
    every one of them had a single-depth pool. `build_pool` spans depths
    1..R by design, so a deep round with unmeasured shallow leftovers
    takes it -- treat its numbers as new, not as reproduced.
    """
    if len(values) == 0:
        return values
    if groups is None:
        return rankdata(values) / len(values)
    groups = np.asarray(groups)
    out = np.empty(len(values), dtype=np.float64)
    for g in np.unique(groups):
        sel = groups == g
        size = int(sel.sum())
        out[sel] = (rankdata(np.asarray(values)[sel]) - 0.5) / size
    return out


def build_pool(embeddings: EmbeddingStore,
               labels: pd.DataFrame,
               max_depth: int,
               bad_threshold: Optional[float] = 1.0) -> Tuple[List[str], np.ndarray]:
    """Unseen candidates of depth 1..max_depth, with their features.

    Drops the wild type, anything already measured, and -- when
    `bad_threshold` is set -- anything containing a substitution that measured
    worse than the threshold on its own. The pool deliberately spans all
    depths, so a shallower candidate can still win if the surrogate prefers it.
    """
    measured = set(labels["variant"])
    bad: set = set()
    if bad_threshold is not None:
        bad = {
            v for v, f in zip(labels["variant"], labels["fitness"])
            if not is_wild_type(v) and depth(v) == 1 and f < bad_threshold
        }

    names: List[str] = []
    rows: List[int] = []
    for i, v in enumerate(embeddings.variants):
        if is_wild_type(v) or v in measured:
            continue
        d = depth(v)
        if d < 1 or d > max_depth:
            continue
        if bad and any(s in bad for s in split_variant(v)):
            continue
        names.append(v)
        rows.append(i)
    return names, embeddings.features[rows]


def score_pool(pool_names: Sequence[str],
               pool_features: np.ndarray,
               labels: pd.DataFrame,
               embeddings: EmbeddingStore,
               zs_map: Dict[str, float],
               n_estimators: int = 100,
               seed: int = 42,
               base_agg: str = "max") -> pd.DataFrame:
    """Fit the surrogate on everything measured and build the three channels."""
    if not len(pool_names):
        raise ValueError("candidate pool is empty")

    # Training rows are SORTED BY NAME, and the wild type is included when it is
    # labelled (fitness 1.0 by convention).
    #
    # The sort is load-bearing. RandomForestRegressor draws its bootstrap
    # samples by row *position*: a fixed random_state pins the sequence of row
    # indices, not which variant each index refers to. So the same data in a
    # different order is a different forest -- measured here at up to 0.2 in
    # predicted fitness, moving 23 of 30 picks. Embedding-store order is the
    # library CSV's order, which is stable only until something rewrites that
    # CSV (re-running `proteus-candidates` at the same depth does) or until
    # `--embeddings` is given in a different order.
    # Sorting removes the row ORDER as an input. It does not remove the row
    # CONTENTS: a variant present in more than one embedding file -- the wild
    # type is in every one of them, and it is a labelled training row -- is
    # taken from whichever file was listed first. EmbeddingStore.concat warns
    # when those copies differ and says which one won; it does not refuse,
    # because a small difference is expected (the wild type's MSA stream is
    # sampled with a seed and its placeholder structure is rebuilt per round).
    labelled = set(labels["variant"])
    train_names = sorted(v for v in embeddings.variants if v in labelled)
    missing = sorted(labelled - set(train_names))
    if missing:
        raise ValueError(
            f"{len(missing)} measured variants have no embedding "
            f"(e.g. {missing[:5]}). Featurize them before selecting, otherwise "
            f"the surrogate silently trains on a subset of the labels.")
    _, train_features, _ = embeddings.take(train_names)

    fitness = labels.set_index("variant")["fitness"]
    y_train = fitness.loc[train_names].to_numpy(dtype=np.float64)

    surrogate = RandomForestSurrogate(n_estimators=n_estimators, random_state=seed)
    surrogate.fit(train_features, y_train)
    rf_pred = surrogate.predict(pool_features)

    zs_raw = [zeroshot_for(v, zs_map) for v in pool_names]
    n_missing_zs = sum(1 for s in zs_raw if s is None)
    if n_missing_zs == len(zs_raw):
        raise ValueError(
            "no candidate has a zero-shot score; check that the zero-shot CSV "
            "covers this library or its constituent singles")
    floor = min(s for s in zs_raw if s is not None) - 1.0
    zs = np.array([floor if s is None else s for s in zs_raw], dtype=np.float64)

    measured_map = {
        v: f for v, f in zip(labels["variant"], labels["fitness"])
        if not is_wild_type(v)
    }
    # `max` caps the base channel at the best thing already measured, so it
    # cannot grow with depth. `sum` tracks true fitness slightly better for
    # depth-2 candidates but double-counts overlapping sub-variants and inflates
    # as the ladder deepens, which measured worse from depth 3 on.
    if base_agg not in ("max", "sum"):
        raise ValueError(f"base_agg must be 'max' or 'sum', got {base_agg!r}")

    def base_score(variant: str) -> float:
        values = [measured_map[s]
                  for s in measured_sub_variants(variant, measured_map)]
        if not values:
            return -np.inf
        return float(max(values)) if base_agg == "max" else float(sum(values))

    base_raw = np.array([base_score(v) for v in pool_names], dtype=np.float64)
    finite = np.isfinite(base_raw)
    if finite.any():
        base_raw = np.where(finite, base_raw, base_raw[finite].min() - 1.0)
    else:
        base_raw = np.zeros_like(base_raw)

    depths = [depth(v) for v in pool_names]
    # rf is comparable across depths -- the surrogate predicts fitness on the
    # same scale for every variant -- so it is ranked over the whole pool. zs
    # and base are not; see _normalized_rank.
    return pd.DataFrame({
        "variant": list(pool_names),
        "mut_count": depths,
        # Recorded so proteus-audit can check base_fitness against the same
        # aggregation that produced it. Hardcoding `max` there failed every
        # legitimate `--base sum` round.
        "base_agg": base_agg,
        "rf_score": rf_pred,
        "zs_score": zs,
        "zs_missing": [s is None for s in zs_raw],
        "base_fitness": base_raw,
        "rank_rf": _normalized_rank(rf_pred),
        "rank_zs": _normalized_rank(zs, depths),
        "rank_base": _normalized_rank(base_raw, depths),
    })


def fuse(channels: pd.DataFrame, a_zs: float, a_base: float) -> pd.Series:
    a_rf = 1.0 - a_zs - a_base
    if a_rf < -1e-9:
        raise ValueError(
            f"a_zs + a_base must be <= 1, got {a_zs} + {a_base}")
    a_rf = max(0.0, a_rf)
    return (a_zs * channels["rank_zs"]
            + a_rf * channels["rank_rf"]
            + a_base * channels["rank_base"])


def consensus(channels: pd.DataFrame,
              grid: Sequence[Tuple[float, float]] = DEFAULT_GRID,
              n: int = DEFAULT_BATCH) -> pd.DataFrame:
    """Vote each grid setting's top-`n` and rank by agreement.

    Sort order follows the implementation that produced the ordered batches:
    votes descending, then mean final score descending, then best rank
    ascending. (The written guide omits the middle key.)
    """
    if not grid:
        raise ValueError("empty grid")

    per_setting: Dict[str, pd.DataFrame] = {}
    for a_zs, a_base in grid:
        label = f"zs{int(round(a_zs * 100)):03d}_base{int(round(a_base * 100)):03d}"
        scored = channels.assign(final_score=fuse(channels, a_zs, a_base))
        top = (scored.sort_values("final_score", ascending=False)
                     .head(n)
                     .reset_index(drop=True))
        top[f"rank_{label}"] = top.index + 1
        per_setting[label] = top

    union = sorted(set().union(*(set(t["variant"]) for t in per_setting.values())))
    out = pd.DataFrame({"variant": union})
    out["votes"] = [
        sum(1 for t in per_setting.values() if v in set(t["variant"]))
        for v in union
    ]
    for label, top in per_setting.items():
        out = out.merge(top[["variant", f"rank_{label}"]], on="variant", how="left")

    rank_cols = [c for c in out.columns if c.startswith("rank_zs")]
    out["best_rank"] = out[rank_cols].min(axis=1)

    score_by_variant: Dict[str, List[float]] = {v: [] for v in union}
    for top in per_setting.values():
        for v, s in zip(top["variant"], top["final_score"]):
            score_by_variant[v].append(float(s))
    out["mean_final_score"] = [float(np.mean(score_by_variant[v])) for v in union]

    carry = ["variant", "mut_count", "rf_score", "zs_score", "base_fitness"]
    if "base_agg" in channels.columns:
        carry.append("base_agg")       # so proteus-audit knows which one to check
    out = out.merge(channels[carry], on="variant", how="left")
    return out.sort_values(["votes", "mean_final_score", "best_rank"],
                          ascending=[False, False, True]).reset_index(drop=True)
