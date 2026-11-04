# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Candidate library for the next round.

Round R proposes variants one substitution deeper than round R-1: every
measured depth-R variant that beat the threshold becomes a *background*, and
every single substitution is added on top of it at the positions it has not
already used. Canonical naming dedupes backgrounds that meet in the middle,
and anything already measured is dropped.

This replaces the per-round `generate_round{R}_candidates*.py` scripts, which
differed only in the depth they hardcoded.
"""

from typing import Dict, List, Optional, Sequence

import pandas as pd

from proteus.al.variants import (WT_TAG, apply_to, canonical, depth,
                                 is_wild_type, plus_one_neighbours,
                                 split_variant)


def select_backgrounds(labels: pd.DataFrame, background_depth: int,
                       threshold: float = 1.0) -> pd.DataFrame:
    """Measured variants at `background_depth` scoring above `threshold`.

    Fitness is WT-normalized, so the default 1.0 means "at least as good as
    wild type": the ladder is only climbed from variants that did not hurt.
    """
    for col in ("variant", "fitness"):
        if col not in labels.columns:
            raise KeyError(f"labels need a {col!r} column; found {list(labels.columns)}")
    df = labels[~labels["variant"].map(is_wild_type)].copy()
    df["depth"] = df["variant"].map(depth)
    at_depth = df[df["depth"] == background_depth]
    good = at_depth[at_depth["fitness"] > threshold]
    return good.sort_values("fitness", ascending=False)


def build_library(wt_sequence: str,
                  labels: pd.DataFrame,
                  target_depth: int,
                  threshold: float = 1.0,
                  dataset: str = "",
                  exclude: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Enumerate the depth-`target_depth` candidate library.

    Returns a frame with `dataset, mutant, mutated_sequence, DMS_score`, the
    wild type first (so downstream zero-shot scoring has its reference row) and
    `DMS_score` empty because nothing here is measured yet.
    """
    if target_depth < 1:
        raise ValueError(f"target_depth must be >= 1, got {target_depth}")

    backgrounds = select_backgrounds(labels, target_depth - 1, threshold)
    if backgrounds.empty:
        raise ValueError(
            f"no measured depth-{target_depth - 1} variant has fitness > "
            f"{threshold}; there is nothing to build round {target_depth} on. "
            f"Lower --threshold or measure more variants.")

    # Canonical form on both sides. Comparing the raw strings let a label
    # written `T7S:A2C` (not sorted by position) through: the generator emits
    # the same variant as `A2C:T7S`, sees no match, and re-proposes something
    # already measured. The audit's overlap check compared raw strings too, so
    # nothing downstream caught it either.
    measured = {canonical(split_variant(v)) if not is_wild_type(v) else WT_TAG
                for v in labels["variant"]} | {WT_TAG}
    if exclude:
        measured |= set(exclude)

    rows: List[Dict[str, object]] = [{
        "dataset": dataset,
        "mutant": WT_TAG,
        "mutated_sequence": wt_sequence,
        "DMS_score": None,
    }]
    seen = set()
    for background in backgrounds["variant"]:
        for cand in plus_one_neighbours(wt_sequence, background):
            if cand in seen or cand in measured:
                continue
            seen.add(cand)
            rows.append({
                "dataset": dataset,
                "mutant": cand,
                "mutated_sequence": apply_to(wt_sequence, cand),
                "DMS_score": None,
            })

    df = pd.DataFrame(rows)

    # Invariants worth asserting by construction rather than trusting:
    # unique names, uniform length, and exactly the depth we asked for.
    if not df["mutant"].is_unique:
        raise AssertionError("duplicate variant names in the candidate library")
    lengths = df["mutated_sequence"].str.len().unique()
    if len(lengths) != 1:
        raise AssertionError(f"ragged sequences in the library: lengths {lengths}")
    depths = {depth(m) for m in df["mutant"] if not is_wild_type(m)}
    if depths != {target_depth}:
        raise AssertionError(
            f"library should be all depth {target_depth}, found depths {sorted(depths)}")
    return df
