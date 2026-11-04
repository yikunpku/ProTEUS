#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement in
# LICENSE at the repository root. Not for commercial use or redistribution.
# The rest of this subtree is upstream Boltz under MIT; see
# proteusfold/LICENSE.
"""Put an alignment into the dialect this predictor's a3m parser accepts.

`boltz/data/parse/a3m.py` reads lowercase as an insertion and `-` as a match
gap, and raises `KeyError` on `.`. It also needs every row to cover the same
match columns as the query: the featurizer indexes a row at
`res_start + res_idx` into an array padded with -1, so a row with fewer match
columns than the query reads that padding and the run dies on
`assert torch.all(msa >= 0)` in `process_msa_features`, with nothing in the
message pointing at the alignment.

So every row is projected onto the columns where the query holds a letter,
uppercased, with `.` becoming `-`. Which columns those are depends on the
file's geometry, and the two cases must not be treated alike:

a2m -- rectangular, the query itself carries lowercase column marks
    Columns are shared, so this is a positional selection: keep index `i` of
    every row wherever `query[i]` is a letter.

a3m -- ragged, each row's insertions inline
    The query has no insertions of its own, so a row's match columns are its
    non-lowercase characters: walk the row and drop the lowercase.

    python tools/clean_a3m.py in.a3m out.a3m [--expect-sequence wt.fasta]

Do not shortcut this by deleting every `.` and uppercasing only the query.
That is right for a true a3m and silently wrong for an a2m whose query carries
lowercase marks: it promotes those query columns to match columns while
deleting the `.` standing in the same columns of every hit. Measured on one
119-residue alignment, the query came out with 119 match columns and 51,240 of
51,241 hits with 106, which is the failure above.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

Record = Tuple[str, str]


def read_records(path: Path) -> List[Record]:
    """Headers kept with their `>`, wrapped sequence lines joined."""
    records: List[Record] = []
    header = None
    chunks: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # ColabFold and MMseqs2 open an a3m with `#<len>\t<cardinality>`.
                # boltz/data/parse/a3m.py skips those lines, so this has to as
                # well -- rejecting them turned away the most common source of
                # a3m files with an error about headers that said nothing about
                # the real reason.
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line, []
            else:
                if header is None:
                    raise SystemExit(f"{path}: sequence data before any header")
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise SystemExit(f"{path}: no records")
    return records


def looks_like_a2m(records: List[Record]) -> bool:
    """Decide the geometry from two signals, because neither alone is enough.

    a2m is rectangular with shared insertion columns; a3m is per-row with
    insertions inline. Keying only on "the query carries lowercase marks" sends
    an all-uppercase-query a2m -- common enough -- down the a3m branch, which
    deletes nothing and then reports a perfectly good file as neither shape.
    Keying only on "every row is the same length" is worse: a ragged a3m whose
    rows happen to match the query's length, because inline insertions make up
    for missing match columns, would be read positionally and its insertion
    residues promoted into match columns -- a silently wrong alignment in
    place of a loud length failure, which is exactly what this script is
    for.

    So: not rectangular means a3m. Rectangular with a marked query means a2m,
    the marks being what defines the insertion columns. Rectangular with an
    unmarked query is the ambiguous case, and it is decided by a property a2m
    cannot violate -- insertion columns are shared, so no row may carry a
    lowercase character in a column where the query has a letter.
    """
    query = records[0][1]
    if len({len(seq) for _, seq in records}) != 1:
        return False
    if any(c.islower() for c in query):
        return True
    return not any(seq[i].islower()
                   for _, seq in records
                   for i in range(len(seq)) if query[i].isalpha())


def clean(records: List[Record]) -> List[Record]:
    query = records[0][1]
    out: List[Record] = []
    if any(c.islower() for c in query) and len({len(s) for _, s in records}) != 1:
        raise SystemExit(
            f"the query carries lowercase column marks, which means shared "
            f"insertion columns, but the rows are "
            f"{sorted({len(s) for _, s in records})[:5]} characters wide. An "
            f"a2m is rectangular; this file is neither shape.")
    if looks_like_a2m(records):                  # a2m: shared columns
        keep = [i for i, c in enumerate(query) if c.isalpha()]
        for header, seq in records:
            out.append((header, "".join(
                "-" if seq[i] == "." else seq[i].upper() for i in keep)))
    else:                                        # a3m: insertions inline
        for header, seq in records:
            out.append((header, "".join(
                "-" if c == "." else c.upper()
                for c in seq if not c.islower())))
    return out


def read_fasta(path: Path) -> str:
    return "".join(l.strip() for l in open(path)
                   if not l.startswith(">")).upper()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--expect-sequence", type=Path, default=None,
                    help="a FASTA whose sequence the cleaned query must equal")
    args = ap.parse_args()

    records = read_records(args.src)
    cleaned = clean(records)
    widths = {len(s) for _, s in cleaned}
    if len(widths) != 1:
        raise SystemExit(
            f"{args.src}: rows are still {sorted(widths)[:5]} columns wide "
            f"after cleaning; this alignment is neither a plain a3m nor an a2m")

    query = cleaned[0][1]
    if args.expect_sequence:
        wt = read_fasta(args.expect_sequence)
        if query != wt:
            raise SystemExit(
                f"{args.src}: the cleaned query is {len(query)} residues and "
                f"{args.expect_sequence} is {len(wt)}"
                + ("; same length, different residues"
                   if len(query) == len(wt) else "")
                + ". The predictor would model a different sequence.")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w") as fh:
        for header, seq in cleaned:
            fh.write(f"{header}\n{seq}\n")
    print(f"{args.src.name}: {len(cleaned)} rows x {len(query)} columns "
          f"-> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
