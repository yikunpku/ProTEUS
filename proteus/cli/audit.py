# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Pre-order audit: check a round end to end before anything is synthesized.

Every check here exists because its absence once produced a wrong batch. The
point is that a pipeline run can finish without error and still be invalid --
coverage gaps, a mismatched wild type, a stale candidate library -- so the
state on disk is checked against itself rather than trusted.

Exit code is non-zero if any check fails.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from proteus.al.select import load_zeroshot_map, zeroshot_for
from proteus.al.surrogate import EmbeddingStore
from proteus.al.variants import (apply_to, canonical, depth, is_wild_type,
                                 measured_sub_variants, split_variant)
from proteus.config import (MAX_LEN, SEQUENCE_COL, VARIANT_COL, WT_TAG,
                            ProteusConfig)


class Audit:
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = ""):
        self.results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    def skip(self, name: str, why: str):
        print(f"  [skip] {name} -- {why}")

    @property
    def failed(self) -> List[str]:
        return [n for n, ok, _ in self.results if not ok]


def audit_round(cfg: ProteusConfig, key: str, dep: int,
                labels_path: Optional[Path] = None) -> Audit:
    a = Audit()
    spec = cfg.spec(key)

    labels_path = Path(labels_path or cfg.labels_csv(key, dep - 1))
    if not labels_path.exists():
        raise SystemExit(
            f"no labels at {labels_path}.\n"
            f"A depth-{dep} round is audited against what was measured at "
            f"depth {dep - 1}, which is why the path says round{dep - 1}. "
            f"Either that round's labels have not been written yet, or "
            f"depth {dep} is not the round you meant -- pass --labels to point "
            f"somewhere else.")
    labels = pd.read_csv(labels_path)
    col = "variant" if "variant" in labels.columns else VARIANT_COL
    if col not in labels.columns:
        raise SystemExit(
            f"{labels_path}: no {'variant'!r} or {VARIANT_COL!r} column "
            f"(found {list(labels.columns)}). A labels file is "
            f"`variant,fitness`, fitness normalized so the wild type is "
            f"exactly 1.0.")
    labels = labels.rename(columns={col: "variant"})
    measured = set(labels["variant"])

    print(f"\n1. labels ({labels_path})")
    a.check("labels have variant and fitness columns",
            {"variant", "fitness"} <= set(labels.columns),
            f"columns={list(labels.columns)}")
    a.check("variant names are unique", labels["variant"].is_unique,
            f"{len(labels)} rows, {labels['variant'].nunique()} unique")
    wt_rows = labels[labels["variant"].map(is_wild_type)]
    a.check("wild type is present and labelled", len(wt_rows) == 1,
            f"{len(wt_rows)} wild-type rows")
    if len(wt_rows) == 1:
        a.check("wild-type fitness is 1.0 (WT-normalized convention)",
                abs(float(wt_rows.iloc[0]["fitness"]) - 1.0) < 1e-9,
                f"got {float(wt_rows.iloc[0]['fitness'])}")
    noncanon, unparseable = [], []
    for v in labels["variant"]:
        if is_wild_type(v):
            continue
        try:
            if canonical(split_variant(v)) != v:
                noncanon.append(v)
        except Exception as exc:        # noqa: BLE001 - reported as a check
            # A label like `C2X` raises out of canonical(). Reporting it as a
            # failed check is the whole point of this command; letting it
            # traceback meant the one thing that would have named the bad label
            # never ran.
            unparseable.append(f"{v} ({exc})")
    a.check("all variant names parse as substitutions", not unparseable,
            f"{len(unparseable)} do not, e.g. {unparseable[:3]}")
    a.check("all variant names are canonical (sorted by position)", not noncanon,
            f"{len(noncanon)} not canonical, e.g. {noncanon[:3]}")
    dist = labels["variant"].map(lambda v: 0 if is_wild_type(v) else depth(v))
    print(f"       measured by depth: {dict(sorted(dist.value_counts().items()))}")
    a.check(f"something measured at depth {dep - 1} to build on",
            int((dist == dep - 1).sum()) > 0,
            f"{int((dist == dep - 1).sum())} variants at depth {dep - 1}")

    print("\n2. candidate library")
    cand_path = cfg.candidates_csv(key, dep)
    if not cand_path.exists():
        a.skip("candidate checks", f"no library at {cand_path}")
        cand = None
    else:
        cand = pd.read_csv(cand_path)
        wt_lib = pd.read_csv(spec.library_csv, usecols=[VARIANT_COL, SEQUENCE_COL])
        wt_seq = wt_lib[wt_lib[VARIANT_COL] == WT_TAG].iloc[0][SEQUENCE_COL]
        body = cand[~cand[VARIANT_COL].map(is_wild_type)]
        a.check("names unique", cand[VARIANT_COL].is_unique,
                f"{len(cand)} rows, {cand[VARIANT_COL].nunique()} unique")
        depths = set(body[VARIANT_COL].map(depth))
        a.check(f"every candidate is depth {dep}", depths == {dep}, f"depths={sorted(depths)}")
        a.check("sequence length is uniform",
                cand[SEQUENCE_COL].str.len().nunique() == 1,
                f"lengths={sorted(cand[SEQUENCE_COL].str.len().unique())[:4]}")
        # Canonical on both sides, for the same reason proteus-candidates is:
        # a label whose substitutions are not sorted by position would not
        # match the generator's output as a raw string.
        def _canon(v):
            try:
                return canonical(split_variant(v))
            except Exception:           # noqa: BLE001 - reported above
                return None
        measured_canon = {c for c in (_canon(v) for v in measured
                                      if not is_wild_type(v)) if c}
        overlap = measured_canon & {c for c in (_canon(v) for v in body[VARIANT_COL])
                                    if c}
        a.check("no candidate was already measured", not overlap,
                f"{len(overlap)} overlap, e.g. {sorted(overlap)[:3]}")
        # Reconstruct a sample of sequences from the wild type: this catches an
        # off-by-one in the position offset, which nothing else here would.
        # Every candidate, not a sample of 50. `apply_to` validates the WT
        # residue each substitution names, so this is the one check that
        # catches a position convention off by N -- the failure that makes
        # PyMOL mutate the wrong residue, puts the mask's 1 at the wrong index,
        # and leaves all four streams the right shape and dtype. A sample is
        # the wrong tool for a defect that may touch one position.
        #
        # Vectorized over the column rather than `iterrows`, which took 1.2 s
        # for 33,527 candidates; this is about five times quicker.
        def _mismatch(variant, sequence):
            try:
                return None if apply_to(wt_seq, variant) == sequence else variant
            except ValueError as exc:
                return f"{variant} ({exc})"
        bad = [m for m in map(_mismatch, body[VARIANT_COL], body[SEQUENCE_COL])
               if m is not None]
        a.check("every sequence reconstructs from the wild type", not bad,
                f"{len(bad)} of {len(body)} mismatch, e.g. {bad[:2]}")
        backbones = set(labels["variant"][dist == dep - 1])
        has_backbone = body[VARIANT_COL].map(
            lambda v: bool(set(split_variant(v)) & {s for b in backbones
                                                    for s in split_variant(b)}))
        # .all(), not .any(): every candidate is generated from a measured
        # background, so one that is not points at a labels file that does not
        # match the library it was built from. .any() would pass on 1 of 33,527
        # while the detail line below reported the shortfall.
        #
        # Not applicable at depth 1, where the background is the wild type.
        # It has no substitutions, so no single can share one with it.
        if dep <= 1:
            a.skip("candidates build on measured backgrounds",
                   "depth-1 candidates build on the wild type, which has no "
                   "substitution to share")
        else:
            a.check("candidates build on measured backgrounds",
                    bool(has_backbone.all()),
                    f"{int(has_backbone.sum()):,} of {len(body):,} share a substitution "
                    f"with a measured depth-{dep - 1} variant")

    print("\n3. embedding coverage")
    emb_paths = [cfg.embeddings_npz(key, d) for d in range(1, dep + 1)]
    stores = [EmbeddingStore.from_npz(p) for p in emb_paths if p.exists()]
    if not stores:
        a.skip("embedding checks", f"none of {[str(p) for p in emb_paths]} exist")
        embeddings = None
    else:
        embeddings = EmbeddingStore.concat(stores)
        print(f"       merged {len(embeddings):,} embeddings from {len(stores)} file(s)")
        missing_measured = sorted(measured - set(embeddings.variants))
        a.check("every measured variant has an embedding", not missing_measured,
                f"{len(missing_measured)} missing, e.g. {missing_measured[:3]}")
        if cand is not None:
            body_names = set(cand[VARIANT_COL]) - {WT_TAG}
            missing_cand = body_names - set(embeddings.variants)
            a.check("every candidate has an embedding", not missing_cand,
                    f"{len(missing_cand):,} of {len(body_names):,} missing")

    print("\n4. zero-shot coverage")
    zs_paths = [str(cfg.zeroshot_csv(key, d)) for d in range(1, dep + 1)
                if cfg.zeroshot_csv(key, d).exists()]
    if not zs_paths:
        a.skip("zero-shot checks", "no zero-shot CSV found")
        zs_map = None
    else:
        zs_map = load_zeroshot_map(zs_paths)
        print(f"       {len(zs_map):,} scores from {len(zs_paths)} file(s)")
        if cand is not None:
            body_names = [v for v in cand[VARIANT_COL] if not is_wild_type(v)]
            direct = sum(1 for v in body_names if v in zs_map)
            viafall = sum(1 for v in body_names
                          if v not in zs_map and zeroshot_for(v, zs_map) is not None)
            covered = direct + viafall
            a.check("candidates have a zero-shot score (direct or sum-of-singles)",
                    covered == len(body_names),
                    f"{covered:,}/{len(body_names):,} covered "
                    f"({direct:,} direct, {viafall:,} by fallback)")

    print("\n5. base channel")
    if not measured or len(labels) < 2:
        a.skip("base channel spot-check", "not enough labels")
    else:
        measured_map = {v: f for v, f in zip(labels["variant"], labels["fitness"])
                        if not is_wild_type(v)}
        # Read base_fitness off the picks file and compare it with the max this
        # candidate's measured sub-variants actually reach. The old version
        # asked whether max(subs) was attained by some element of subs, which is
        # true of every non-empty set -- it never opened the picks file at all,
        # so setting every base_fitness to -99 still passed.
        picks_path = cfg.picks_csv(key, dep)
        if not picks_path.exists():
            a.skip("base_fitness equals max over measured sub-variants",
                   f"no picks at {picks_path}")
        else:
            picks_base = pd.read_csv(picks_path)
            if "base_fitness" not in picks_base.columns:
                a.check("picks carry a base_fitness column", False,
                        f"columns={list(picks_base.columns)}")
            else:
                # Which aggregation produced the column. `select` records it;
                # older picks files do not, and for those either answer is
                # accepted rather than failing a round over a missing column.
                agg = (str(picks_base["base_agg"].iloc[0])
                       if "base_agg" in picks_base.columns and len(picks_base)
                       else None)
                bad, compared = [], 0
                for v, got in zip(picks_base["variant"], picks_base["base_fitness"]):
                    subs = measured_sub_variants(v, measured_map)
                    values = [measured_map[s] for s in subs]
                    if not values:
                        continue
                    compared += 1
                    allowed = {"max": [max(values)], "sum": [sum(values)]}.get(
                        agg, [max(values), sum(values)])
                    if not any(np.isclose(float(got), e, atol=1e-9) for e in allowed):
                        bad.append(f"{v}: file {float(got):.6g} vs "
                                   f"{'/'.join(f'{e:.6g}' for e in allowed)}")
                label = (f"base_fitness equals {agg} over measured sub-variants"
                         if agg else
                         "base_fitness equals max or sum over measured sub-variants")
                if not compared:
                    # Every pick is a single, which has no proper sub-variant --
                    # there is nothing for this check to compare. Saying so
                    # beats reporting "checked 30 picks, 0 inconsistent", which
                    # is how the earlier vacuous version looked when it passed.
                    a.skip(label, f"none of the {len(picks_base)} picks has a "
                                  f"measured sub-variant (depth-1 round)")
                else:
                    a.check(label, not bad,
                            f"{compared} of {len(picks_base)} picks have a "
                            f"measured sub-variant, {len(bad)} inconsistent"
                            + (f", e.g. {bad[:2]}" if bad else ""))

    print("\n6. structure and crop consistency")
    a.check("wild-type structure exists", spec.wt_pdb.exists(), str(spec.wt_pdb))
    a.check("wild-type MSA exists", spec.msa_a3m.exists(), str(spec.msa_a3m))
    try:
        wt_lib = pd.read_csv(spec.library_csv, usecols=[VARIANT_COL, SEQUENCE_COL])
        wt_seq = wt_lib[wt_lib[VARIANT_COL] == WT_TAG].iloc[0][SEQUENCE_COL]
        start, end = cfg.crop_window(key)
        if spec.crop == "per-variant":
            # Each variant gets its own window, so there is no single window to
            # measure anything against. `crop_window(key)` with no variant is
            # the WILD TYPE's window, which this policy starts at the
            # N-terminus by design -- asserting it equals the whole sequence
            # failed every short protein under this policy, for nothing.
            a.check("per-variant crop: wild type starts at the N-terminus",
                    start == 0, f"window=({start}, {end})")
            if cand is not None:
                outside = []
                for v in cand[VARIANT_COL]:
                    if is_wild_type(v):
                        continue
                    vs, ve = cfg.crop_window(key, v, len(wt_seq))
                    if any(not (vs <= p < ve)
                           for p in [int(x[1:-1]) - 1 for x in split_variant(v)]):
                        outside.append(v)
                a.check("every candidate fits inside its own crop window",
                        not outside,
                        f"{len(outside):,} of {len(cand):,} do not, "
                        f"e.g. {outside[:2]}")
        elif len(wt_seq) <= MAX_LEN:
            a.check(f"short protein ({len(wt_seq)} aa) uses the whole sequence",
                    (start, end) == (0, len(wt_seq)), f"window=({start}, {end})")
        else:
            a.check(f"long protein ({len(wt_seq)} aa) window is {MAX_LEN} wide",
                    end - start == MAX_LEN, f"window=({start}, {end})")
            if cand is not None:
                outside = [v for v in cand[VARIANT_COL] if not is_wild_type(v)
                           and any(not (start <= p < end)
                                   for p in [int(s[1:-1]) - 1 for s in split_variant(v)])]
                a.check("every candidate mutation falls inside the crop window",
                        not outside,
                        f"{len(outside):,} candidates have a mutation outside "
                        f"({start}, {end}), e.g. {outside[:2]}")
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        a.check("crop window computable", False, f"{type(exc).__name__}: {exc}")

    print("\n7. consensus arithmetic")
    picks_path = cfg.picks_csv(key, dep)
    if not picks_path.exists():
        a.skip("consensus checks", f"no picks at {picks_path}")
    else:
        picks = pd.read_csv(picks_path)
        rank_cols = [c for c in picks.columns if c.startswith("rank_zs")]
        a.check("picks carry per-setting ranks", bool(rank_cols),
                f"{len(rank_cols)} rank columns")
        if rank_cols:
            recomputed = picks[rank_cols].notna().sum(axis=1)
            a.check("votes equal the number of settings that picked the variant",
                    bool((recomputed == picks["votes"]).all()),
                    f"{int((recomputed != picks['votes']).sum())} rows disagree")
            best = picks[rank_cols].min(axis=1)
            a.check("best_rank is the minimum per-setting rank",
                    bool(np.allclose(best.fillna(-1), picks["best_rank"].fillna(-1))),
                    "")
        order = picks.sort_values(["votes", "mean_final_score", "best_rank"],
                                  ascending=[False, False, True])
        a.check("picks are sorted by (votes, mean score, best rank)",
                list(order["variant"]) == list(picks["variant"]), "")
    return a


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="protein registry YAML")
    p.add_argument("--protein", required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--labels", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = ProteusConfig.from_yaml(args.config)
    print(f"auditing {args.protein} round {args.depth}")
    audit = audit_round(cfg, args.protein, args.depth, args.labels)
    print("\n" + "=" * 60)
    if audit.failed:
        print(f"{len(audit.failed)} CHECK(S) FAILED: {audit.failed}")
        sys.exit(1)
    print(f"all {len(audit.results)} checks passed")


if __name__ == "__main__":
    main()
