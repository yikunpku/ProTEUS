# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Putting an alignment into the shape the MSA reader actually assumes.

`proteus.third_party.proteingym_msa.MSA_processing` comes from ProteinGym/EVE,
where alignments arrive as **a2m**: every row the same width as the query, an
insertion marked by a lowercase letter or a `.` in a column that every row
shares. Its preprocessing step relies on that (`proteingym_msa.py`, in
`preprocess_msa`):

    sequence.upper()                                     # insertion marks gone
    non_gap_wt_cols = [aa != '-' for aa in query]        # length = len(query)
    ''.join(aa for aa, keep in zip(row, non_gap_wt_cols) if keep)

`zip` stops at the shorter of the two, so for a query with no gaps this is
exactly "keep the first len(query) characters of each row". Given **a3m**,
where a row is the query's columns *plus that row's own insertions inline* and
rows therefore have different widths, the insertion characters have already
been uppercased into ordinary residues and they occupy alignment columns. Every
residue after a row's first insertion is then shifted, and because truncation
makes the matrix rectangular again, nothing raises.

That is not hypothetical. On the AlphaFold-Server alignments this project used,
88.3% of rows carry at least one insertion (median 7, max 63), and 51.9% of all
row positions land out of register.

So the default here is to put an a3m into a2m's shape first. The cheap way to
do it is to drop each row's insertions outright:

    strip_insertions  row -> row without its lowercase characters

which is character-for-character what the a2m route produces. Padding every
row with `.` at the union of insertion columns would give the query a `.` at
each of those columns; `preprocess_msa` turns `.` into `-`, so `non_gap_wt_cols`
is False there and the whole column is dropped from every row. Same output, one
pass, no need to compute the union.

Two guards come with it, because the reader is also strict in ways that are
easy to violate by accident:

* the focus sequence is whichever record sits at **physical line 0**, so a
  leading comment or blank line silently makes the first *hit* the query;
* records are keyed by their entire header line into a `defaultdict(str)` with
  `+=`, so two records sharing a header are silently concatenated. ColabFold's
  `use_env=True` output does exactly that -- the UniRef and environmental
  blocks each open with `>101`.

`--legacy-a3m` turns all of this off and feeds the file through untouched, for
reproducing numbers that were produced before any of it was noticed.
"""

from pathlib import Path
from typing import List, Optional, Tuple

Record = Tuple[str, str]

# Suffix for the rewritten copy. It sits beside the original, like the
# sequence-weight cache does, and carries its own weights file: the weights are
# a function of the alignment, so reusing the original's would pair new columns
# with old weights.
REGISTER_SUFFIX = ".register.a3m"

_LEGACY = False


def set_legacy(value: bool) -> None:
    """Feed alignments through untouched (the `--legacy-a3m` behaviour).

    A module-level switch rather than a parameter threaded through every
    caller: the zero-shot scorer is kept byte-for-byte as it was written, so
    there is nowhere to add an argument without editing it.
    """
    global _LEGACY
    _LEGACY = bool(value)


def legacy() -> bool:
    return _LEGACY


# --------------------------------------------------------------------------
# reading and writing
# --------------------------------------------------------------------------

def read_records(path) -> List[Record]:
    """`[(header, sequence)]`, headers keeping their `>`, wrapped lines joined.

    Wrapping matters downstream as well as here: boltz reads an a3m by taking
    every second line, so a record split over two lines shifts the parity and
    garbles everything after it.
    """
    records: List[Record] = []
    header: Optional[str] = None
    chunks: List[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                if header is None:
                    raise ValueError(
                        f"{path}: blank line before the first header. The MSA "
                        f"reader takes the query from physical line 0 "
                        f"(`if i == 0: focus_seq_name = name`), so a leading "
                        f"blank line makes the query a record named '' whose "
                        f"sequence is empty -- the alignment then has zero "
                        f"focus columns and every row is dropped as all-gap. "
                        f"Delete the blank line.")
                continue
            if line.startswith("#"):
                # ColabFold and MMseqs2 open an a3m with `#<len>\t<cardinality>`.
                # Before this, such a line after the first header was appended
                # to the preceding sequence -- the query came back as
                # 'ACDEF#119\t1' and the width guard then blamed the wrong row.
                if header is None:
                    raise ValueError(
                        f"{path}: comment line before the first header. The MSA "
                        f"reader takes the query from physical line 0, so this "
                        f"makes the query a record that is not a sequence. "
                        f"Delete it.")
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header, chunks = line, []
            else:
                if header is None:
                    raise ValueError(
                        f"{path}: sequence data before the first header. The MSA "
                        f"reader takes whatever record is at line 0 as the query, "
                        f"so this file would silently use a hit as the wild type.")
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def write_records(path, records: List[Record]) -> None:
    with open(path, "w") as fh:
        for header, seq in records:
            fh.write(f"{header}\n{seq}\n")


# --------------------------------------------------------------------------
# the transforms
# --------------------------------------------------------------------------

def query_is_a2m(records: List[Record]) -> bool:
    """True when the query itself carries insertion marks.

    In a2m as ProteinGym writes it, a non-focus column is lowercase in the
    query too, and the alignment is already rectangular. Stripping lowercase
    there would delete real columns, so this is the case to leave alone.
    """
    query = records[0][1]
    return any(c.islower() for c in query)


def strip_insertions(records: List[Record]) -> Tuple[List[Record], int]:
    """Drop every row's insertions. Returns the records and how many changed.

    No-op when the query is itself marked up (see `query_is_a2m`) -- that file
    is already in the shape the reader assumes.
    """
    if query_is_a2m(records):
        return records, 0
    out: List[Record] = []
    changed = 0
    for header, seq in records:
        clean = "".join(c for c in seq if not c.islower())
        if clean != seq:
            changed += 1
        out.append((header, clean))
    return out, changed


def dedupe_headers(records: List[Record]) -> Tuple[List[Record], int]:
    """Make every header unique, keeping the first occurrence as written."""
    seen: dict = {}
    out: List[Record] = []
    renamed = 0
    for header, seq in records:
        if header in seen:
            seen[header] += 1
            header = f"{header}/dup{seen[header]}"
            renamed += 1
        else:
            seen[header] = 0
        out.append((header, seq))
    return out, renamed


def normalize(records: List[Record]) -> Tuple[List[Record], dict]:
    """Everything the reader needs, in one pass, with a report of what moved."""
    records, renamed = dedupe_headers(records)
    records, stripped = strip_insertions(records)
    query_len = len(records[0][1])
    ragged = sum(1 for _, s in records if len(s) != query_len)
    return records, {"renamed_headers": renamed, "rows_stripped": stripped,
                     "query_len": query_len, "rows": len(records),
                     "ragged_rows_after": ragged}


def for_boltz(records: List[Record]) -> List[Record]:
    """A copy the structure predictor can parse.

    Its a3m parser reads lowercase as an insertion and `-` as a match gap, and
    raises `KeyError` on `.`. More importantly, every row must cover the same
    match columns as the query: the featurizer indexes each row at
    `res_start + res_idx` into an array padded with -1, so a row with fewer
    match columns than the query reads that padding and the run dies on
    `assert torch.all(msa >= 0)` with nothing pointing at the alignment.

    So every row is projected onto the columns where the query holds a letter,
    uppercased, with `.` becoming `-`. The result is rectangular, has no
    lowercase and no dots, and every row covers exactly the wild type's
    positions.

    Which columns those are depends on the file's geometry, and the two cases
    cannot be treated alike:

    a2m (rectangular, the query carries the column marks)
        Columns are shared, so this is a positional selection: keep index `i`
        of every row wherever `query[i]` is a letter.

    a3m (ragged, insertions inline in each row)
        The query has no insertions of its own, so a row's match columns are
        its non-lowercase characters; walk the row and drop the lowercase.

    Getting this wrong is not loud. The recipe this replaces -- delete every
    `.`, uppercase the query only, which is what `clean_a3m_for_boltz.py` in
    the structure-prediction repository does -- is correct for a true a3m and
    silently wrong for an a2m whose query carries lowercase marks: it promotes
    those query columns to match columns while deleting the `.` that stands in
    the same columns of every hit. Measured on the bundled demo alignment, the
    query came out with 119 match columns and 51,240 of 51,241 hits with 106.

    This is for the predictor only. The alignment the feature code reads is the
    one `resolve` produces, and the two are not interchangeable.
    """
    query = records[0][1]
    out: List[Record] = []

    if query_is_a2m(records):
        keep = [i for i, c in enumerate(query) if c.isalpha()]
        width = len(query)
        for header, seq in records:
            if len(seq) != width:
                raise ValueError(
                    f"{header}: this alignment's query carries lowercase column "
                    f"marks, which means a2m with shared columns, but the row is "
                    f"{len(seq)} characters against the query's {width}. Mixed "
                    f"geometry cannot be projected; fix the alignment.")
            out.append((header, "".join(
                "-" if seq[i] == "." else seq[i].upper() for i in keep)))
        return out

    width = len(query)
    for header, seq in records:
        row = "".join("-" if c == "." else c.upper()
                      for c in seq if not c.islower())
        if len(row) != width:
            raise ValueError(
                f"{header}: {len(row)} match columns against the query's "
                f"{width}. The predictor's featurizer indexes a row at "
                f"res_start + res_idx into an array padded with -1, so a short "
                f"row reads that padding and the run dies on "
                f"`assert torch.all(msa >= 0)` with nothing in the message "
                f"pointing at the alignment.")
        out.append((header, row))
    return out


def describe(path) -> dict:
    """Read-only statistics, for `proteus-doctor` and `proteus-audit`."""
    records = read_records(path)
    query = records[0][1]
    widths = {len(s) for _, s in records}
    headers = [h for h, _ in records]
    return {
        "path": str(path),
        "rows": len(records),
        "query_len": len(query),
        "query_has_lowercase": any(c.islower() for c in query),
        "query_has_gap": "-" in query or "." in query,
        "distinct_widths": len(widths),
        "rows_with_insertions": sum(
            1 for _, s in records[1:] if any(c.islower() for c in s)),
        "duplicate_headers": len(headers) - len(set(headers)),
        "looks_like_a2m": query_is_a2m(records),
    }


# --------------------------------------------------------------------------
# what the readers call
# --------------------------------------------------------------------------

def register_path(a3m: Path) -> Path:
    name = a3m.name
    for suffix in (".a3m", ".a2m", ".fasta", ".afa"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return a3m.with_name(name + REGISTER_SUFFIX)


def resolve(a3m, weights=None) -> Tuple[str, Optional[str]]:
    """The alignment to actually read, and the weights file that goes with it.

    Returns the inputs unchanged under `--legacy-a3m`, and also when the file
    already needs nothing done to it -- so a well-formed a2m, which is what the
    bundled demo ships, keeps its existing weight cache and its existing
    outputs. Otherwise a rewritten copy is cached beside the original and its
    own weights path is returned, because weights computed against different
    columns do not transfer.
    """
    a3m = Path(a3m)
    if _LEGACY:
        return str(a3m), (str(weights) if weights else None)

    target = register_path(a3m)
    if target.exists() and target.stat().st_mtime >= a3m.stat().st_mtime:
        return str(target), str(target.with_suffix(".npy"))

    records = read_records(a3m)
    fixed, report = normalize(list(records))

    # Before the no-op shortcut, not after it. An a2m whose query carries
    # lowercase column marks makes strip_insertions a no-op, so a ragged file
    # of that shape would otherwise come back unchanged and unreported, with
    # this very report holding the row widths that show it is broken.
    if report["ragged_rows_after"]:
        raise ValueError(
            f"{a3m}: {report['ragged_rows_after']} of {report['rows']} rows are "
            f"still not {report['query_len']} columns wide after removing "
            f"insertions, so this is neither a plain a3m nor an a2m and the MSA "
            f"reader would truncate rows to fit. Fix the alignment, or pass "
            f"--legacy-a3m to read it as-is.")

    if fixed == records:
        return str(a3m), (str(weights) if weights else None)

    write_records(target, fixed)
    print(f"[a3m] {a3m.name}: stripped insertions from "
          f"{report['rows_stripped']}/{report['rows']} rows"
          + (f", renamed {report['renamed_headers']} duplicate headers"
             if report["renamed_headers"] else "")
          + f" -> {target.name}")
    return str(target), str(target.with_suffix(".npy"))
