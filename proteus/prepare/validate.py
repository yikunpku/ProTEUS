# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Do the alignment, the structure and the sequence agree?

Three files have to describe the same protein in the same coordinates, and
nothing downstream checks that they do:

* a variant is named by a **1-based position in the sequence**, and
  `featurize.structure_prep` looks that number up in the structure's `resSeq`
  field. If the structure is numbered from anything other than 1 -- an
  experimental structure usually is -- PyMOL mutates a different residue, or
  none, and the run continues.
* `featurize.structure_tokens` asserts only that the residue *count* matches
  the sequence length. Equal counts with different residues pass that check.
* the MSA's query row defines the columns every per-position MSA feature is
  read from, and `MSA_processing` takes whichever record sits at physical line
  0 as that query.

So the failure mode is not a crash, it is a batch of variants where the
structure channel describes mutations at the wrong positions. These checks are
what `proteus-prepare` runs before it writes anything into the registry, and
what `proteus-doctor` re-runs for every configured protein.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from proteus.prepare import a3m_tools

# Three-letter to one-letter, for reading a sequence back out of a PDB.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # What a predictor or a refinement program may leave behind.
    "MSE": "M", "SEC": "U", "PYL": "O",
}


class Issues:
    """Errors that must stop the run, and warnings that must not."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.notes: List[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, other: "Issues") -> None:
        self.errors += other.errors
        self.warnings += other.warnings
        self.notes += other.notes

    def report(self, prefix: str = "  ") -> None:
        for m in self.notes:
            print(f"{prefix}[ ok ] {m}")
        for m in self.warnings:
            print(f"{prefix}[warn] {m}")
        for m in self.errors:
            print(f"{prefix}[FAIL] {m}")

    def raise_if_failed(self, what: str) -> None:
        if self.errors:
            raise ValueError(
                f"{what}: " + "; ".join(self.errors)
                + ". Nothing was written.")


# --------------------------------------------------------------------------
# reading a structure the way the pipeline reads it
# --------------------------------------------------------------------------

def read_structure_residues(pdb_path) -> Tuple[List[Tuple[str, int, str, str]], List[str]]:
    """`[(chain, resSeq, insertion_code, resName)]` in file order, plus chains.

    `ATOM` only, first occurrence of each (chain, number, insertion) wins --
    the same reading `featurize.structure_prep.residue_chain_map` does, so what
    this sees is what the pipeline will see.
    """
    seen = set()
    residues: List[Tuple[str, int, str, str]] = []
    chain_order: List[str] = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            try:
                res_seq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26].strip()
            key = (chain, res_seq, icode)
            if key in seen:
                continue
            seen.add(key)
            residues.append((chain, res_seq, icode, line[17:20].strip()))
            if chain not in chain_order:
                chain_order.append(chain)
    return residues, chain_order


def check_structure(pdb_path, wt_sequence: str) -> Issues:
    """The structure must be addressable by the library's variant positions."""
    issues = Issues()
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        issues.error(f"structure {pdb_path} does not exist")
        return issues

    residues, chain_order = read_structure_residues(pdb_path)
    if not residues:
        issues.error(f"{pdb_path.name} has no ATOM records")
        return issues

    # Two places in the codebase pick a chain, by different rules. They agree
    # for a single-chain file, which is the only case that is unambiguous.
    first_in_file = chain_order[0]
    first_alphabetical = sorted(chain_order)[0]
    if len(chain_order) > 1:
        issues.warn(
            f"{pdb_path.name} has {len(chain_order)} chains {chain_order}. "
            f"featurize.structure_prep mutates the chain that appears first in "
            f"the file ({first_in_file!r}); zeroshot.scorer reads the "
            f"alphabetically first one ({first_alphabetical!r})"
            + (" -- and those are different chains here, so the two paths "
               "would not describe the same molecule"
               if first_in_file != first_alphabetical
               else ". They are the same chain here"))

    chain = [r for r in residues if r[0] == first_in_file]
    icodes = [r for r in chain if r[2]]
    if icodes:
        issues.warn(
            f"{len(icodes)} residues carry insertion codes (e.g. "
            f"{icodes[0][1]}{icodes[0][2]}); structure_prep keys on the number "
            f"alone, so these collapse onto their un-suffixed neighbours")

    numbers = [r[1] for r in chain]
    expected = list(range(1, len(wt_sequence) + 1))

    if len(chain) != len(wt_sequence):
        issues.error(
            f"{pdb_path.name} chain {first_in_file!r} has {len(chain)} "
            f"residues but the wild-type sequence has {len(wt_sequence)}. "
            f"featurize.structure_tokens requires these to be equal")
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        extra = sorted(set(numbers) - set(expected))
        detail = []
        if missing:
            detail.append(f"{len(missing)} sequence positions absent "
                          f"(first: {missing[:5]})")
        if extra:
            detail.append(f"{len(extra)} residue numbers outside 1..{len(wt_sequence)} "
                          f"(first: {extra[:5]})")
        if not detail:
            detail.append(f"numbering runs {numbers[0]}..{numbers[-1]} but not in order")
        issues.error(
            f"{pdb_path.name} chain {first_in_file!r} is not numbered 1.."
            f"{len(wt_sequence)}: " + ", ".join(detail)
            + ". A variant names a sequence position and structure_prep looks "
              "that number up directly, so renumber the structure to match the "
              "library sequence")

    # Identity, not just length. Equal counts with different residues is a
    # silent failure: every mutant structure would be built from the wrong
    # starting residue.
    unknown = sorted({r[3] for r in chain if r[3] not in THREE_TO_ONE})
    if unknown:
        issues.warn(f"non-standard residues present: {unknown[:6]}")
    struct_seq = "".join(THREE_TO_ONE.get(r[3], "X") for r in chain)
    if len(struct_seq) == len(wt_sequence) and struct_seq != wt_sequence.upper():
        diffs = [(i + 1, a, b) for i, (a, b)
                 in enumerate(zip(wt_sequence.upper(), struct_seq)) if a != b]
        issues.error(
            f"{pdb_path.name} has the right number of residues but "
            f"{len(diffs)} of them differ from the wild-type sequence "
            f"(first: " + ", ".join(f"pos {p} seq={a} structure={b}"
                                    for p, a, b in diffs[:3]) + ")")

    # One summary line, and only when every check above held -- a note that
    # repeats a property a failure has just denied is worse than no note.
    if issues.ok:
        issues.note(
            f"structure: {len(chain)} residues, chain {first_in_file!r}, "
            f"numbered 1..{len(chain)}, sequence matches the wild type")
    return issues


# --------------------------------------------------------------------------
# the alignment
# --------------------------------------------------------------------------

def check_alignment(a3m_path, wt_sequence: Optional[str] = None) -> Issues:
    """The query row must be the wild type, and the rows must line up."""
    issues = Issues()
    a3m_path = Path(a3m_path)
    if not a3m_path.exists():
        issues.error(f"alignment {a3m_path} does not exist")
        return issues

    try:
        info = a3m_tools.describe(a3m_path)
    except ValueError as exc:
        issues.error(str(exc))
        return issues

    if info["duplicate_headers"]:
        issues.error(
            f"{a3m_path.name} has {info['duplicate_headers']} duplicate header "
            f"line(s). MSA_processing keys records on the whole header and "
            f"appends to a defaultdict, so duplicates are concatenated into one "
            f"over-long row instead of being kept apart")

    if info["rows"] < 2:
        issues.error(f"{a3m_path.name} has {info['rows']} record(s); "
                     f"weighted sampling needs at least two")

    if info["looks_like_a2m"]:
        issues.note(
            f"alignment: {info['rows']} rows, {info['distinct_widths']} distinct "
            f"width(s), a2m style (the query carries the column marks), read as-is")
    elif info["distinct_widths"] > 1 or info["rows_with_insertions"]:
        if a3m_tools.legacy():
            issues.warn(
                f"{a3m_path.name} is a true a3m ({info['rows_with_insertions']} "
                f"of {info['rows'] - 1} rows carry insertions) and --legacy-a3m "
                f"is set, so it is read unchanged -- every residue after a "
                f"row's first insertion will be one or more columns out of "
                f"register. This reproduces older numbers; it is not correct")
        else:
            issues.note(
                f"alignment: {info['rows']} rows, a3m with insertions in "
                f"{info['rows_with_insertions']} of them; will be rewritten to "
                f"{info['query_len']} columns before it is read")
    else:
        issues.note(f"alignment: {info['rows']} rows x {info['query_len']} "
                    f"columns, already rectangular")

    if info["query_has_gap"] and not info["looks_like_a2m"]:
        issues.warn(
            f"{a3m_path.name}: the query row contains gap characters; those "
            f"columns are dropped from every row before the features are built")

    if wt_sequence is not None:
        records = a3m_tools.read_records(a3m_path)
        query = records[0][1]
        bare = "".join(c for c in query if c not in ".-").upper()
        if bare != wt_sequence.upper():
            issues.error(
                f"{a3m_path.name}: the query row is not the wild-type sequence "
                f"({len(bare)} residues after removing gaps vs "
                f"{len(wt_sequence)})"
                + ("; same length, different residues"
                   if len(bare) == len(wt_sequence) else "")
                + ". Every per-position MSA feature is indexed from this row")
    return issues


# --------------------------------------------------------------------------
# all three together
# --------------------------------------------------------------------------

def check_protein(wt_sequence: str, a3m_path, pdb_path,
                  label: str = "", verbose: bool = True) -> Issues:
    """Alignment, structure and sequence, checked against each other."""
    issues = Issues()
    head = f"{label}: " if label else ""
    if verbose:
        print(f"{head}validating inputs ({len(wt_sequence)} residues)")
    issues.extend(check_alignment(a3m_path, wt_sequence))
    issues.extend(check_structure(pdb_path, wt_sequence))
    if verbose:
        issues.report()
    return issues


def check_registry(cfg, keys: Optional[List[str]] = None,
                   verbose: bool = True) -> Dict[str, Issues]:
    """Run the checks for every protein in a loaded config.

    The wild-type sequence comes from the library CSV's `wild_type` row, which
    is the same row the zero-shot scorer reads as its reference, so this checks
    what the pipeline will actually use rather than a separate FASTA.
    """
    import pandas as pd

    from proteus.config import SEQUENCE_COL, VARIANT_COL, WT_TAG

    out: Dict[str, Issues] = {}
    for key in (keys or sorted(cfg.proteins)):
        spec = cfg.spec(key)
        issues = Issues()
        library = spec.library_csv
        if not Path(library).exists():
            issues.error(f"library {library} does not exist")
            if verbose:
                issues.report()
            out[key] = issues
            continue
        df = pd.read_csv(library, usecols=lambda c: c in (VARIANT_COL, SEQUENCE_COL))
        wt_rows = df[df[VARIANT_COL] == WT_TAG]
        if wt_rows.empty:
            issues.error(
                f"{library} has no {WT_TAG!r} row; the wild-type sequence is "
                f"read from it")
            if verbose:
                issues.report()
            out[key] = issues
            continue
        wt = str(wt_rows[SEQUENCE_COL].iloc[0]).upper()
        out[key] = check_protein(wt, spec.msa_a3m, spec.wt_pdb, label=key,
                                 verbose=verbose)
    return out
