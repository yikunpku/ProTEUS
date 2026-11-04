# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Mutation-string handling.

A variant is written as colon-separated substitutions, each `<wt><pos><mut>`
with `pos` 1-indexed against the wild-type sequence, e.g. `T7S:A29H`. The
canonical form sorts substitutions by position, so the same variant always has
the same name and set operations behave. The wild type is the literal
`wild_type`.
"""

from typing import Dict, Iterable, List, Tuple

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
WT_TAG = "wild_type"
WT_ALIASES = frozenset({WT_TAG, "WT", "wt", "wildtype"})


def is_wild_type(variant: str) -> bool:
    return variant in WT_ALIASES


def parse_substitution(sub: str) -> Tuple[str, int, str]:
    """`'T7S'` -> `('T', 7, 'S')`."""
    if len(sub) < 3:
        raise ValueError(f"not a substitution: {sub!r}")
    wt, mut = sub[0], sub[-1]
    try:
        pos = int(sub[1:-1])
    except ValueError as exc:
        raise ValueError(f"not a substitution: {sub!r}") from exc
    if wt not in AMINO_ACIDS or mut not in AMINO_ACIDS:
        raise ValueError(f"non-standard amino acid in {sub!r}")
    if pos < 1:
        raise ValueError(f"position must be 1-indexed, got {pos} in {sub!r}")
    return wt, pos, mut


def split_variant(variant: str) -> List[str]:
    """Substitutions of a variant; empty for the wild type."""
    if is_wild_type(variant):
        return []
    return variant.split(":")


def canonical(subs: Iterable[str]) -> str:
    """Canonical variant name: substitutions sorted by position."""
    subs = list(subs)
    if not subs:
        return WT_TAG
    ordered = sorted(subs, key=lambda s: parse_substitution(s)[1])
    positions = [parse_substitution(s)[1] for s in ordered]
    if len(set(positions)) != len(positions):
        raise ValueError(f"two substitutions at the same position: {ordered}")
    return ":".join(ordered)


def depth(variant: str) -> int:
    """Number of substitutions. Wild type is depth 0."""
    return len(split_variant(variant))


def apply_to(wt_sequence: str, variant: str) -> str:
    """Build the mutated sequence, checking every stated WT residue.

    A mismatch means the variant name and the wild-type sequence disagree,
    which is always a data problem -- usually an offset -- so it raises rather
    than silently producing a sequence nobody intended.
    """
    if is_wild_type(variant):
        return wt_sequence
    seq = list(wt_sequence)
    for sub in split_variant(variant):
        wt, pos, mut = parse_substitution(sub)
        if pos > len(seq):
            raise ValueError(
                f"{sub}: position {pos} is past the end of a "
                f"{len(seq)}-residue wild type")
        if seq[pos - 1] != wt:
            raise ValueError(
                f"{sub}: wild type has {seq[pos - 1]!r} at position {pos}, "
                f"not {wt!r} -- check the position offset")
        seq[pos - 1] = mut
    return "".join(seq)


def plus_one_neighbours(wt_sequence: str, background: str) -> List[str]:
    """Every variant one substitution deeper than `background`.

    Positions already substituted in the background are skipped, so the result
    is always exactly one substitution deeper.
    """
    taken = {parse_substitution(s)[1] for s in split_variant(background)}
    base_subs = split_variant(background)
    out: List[str] = []
    for pos in range(1, len(wt_sequence) + 1):
        if pos in taken:
            continue
        wt_aa = wt_sequence[pos - 1]
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            out.append(canonical(base_subs + [f"{wt_aa}{pos}{mut_aa}"]))
    return out


def constituent_singles(variant: str) -> List[str]:
    """The single substitutions a variant is made of."""
    return split_variant(variant)


def measured_sub_variants(variant: str, measured: Dict[str, float]) -> List[str]:
    """Measured variants whose substitutions are a subset of this variant's.

    Used by the `base` channel: it answers "what is the best thing we have
    already measured that this candidate builds on".
    """
    subs = set(split_variant(variant))
    if not subs:
        return []
    out = []
    for name in measured:
        if is_wild_type(name):
            continue
        other = set(split_variant(name))
        if other and other <= subs:
            out.append(name)
    return out
