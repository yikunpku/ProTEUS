# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Getting a wild-type MSA, from a sequence or from a finished folding job.

Three backends, because the right one depends on what you are trying to match:

`colabfold`  (default)
    The public MMseqs2 API at api.colabfold.com -- the same search the
    structure predictor runs, so the alignment behind a structure and the
    alignment behind the MSA features come from one database build. Needs no
    local database and no GPU. It does send the sequence to a third-party
    server, which is a decision to make deliberately for an unpublished
    target; the other two backends never leave your machine.

`af3-zip`
    An AlphaFold Server job archive. Its `fold_*_unpaired_msa_chains_a.a3m` is
    the alignment this project's published numbers were produced from, so this
    is the backend to use when matching them. Submitting the job is still
    manual; everything after the download is not.

`a3m`
    An alignment you already have. Validated, normalised and copied into
    place, nothing more.

Whatever the source, the file written here is put through
`proteus.prepare.a3m_tools` first: the reader downstream assumes a rectangular
alignment, and none of these three sources produces one by default.
"""

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from proteus.prepare import a3m_tools

DEFAULT_HOST = "https://api.colabfold.com"
BACKENDS = ("colabfold", "af3-zip", "a3m")

# What an AlphaFold Server archive calls the single-chain unpaired alignment.
AF3_MSA_GLOB = "*unpaired_msa*.a3m"


def _finish(records: List[Tuple[str, str]], wt_sequence: str,
            out_a3m: Path, source: str) -> Path:
    """Normalise, check against the wild type, write.

    The query row has to be the wild type exactly. If it is not, every
    downstream column index is wrong -- the mutation mask, the crop window and
    the per-position scores all count from it -- so this is the place to stop.
    """
    records, report = a3m_tools.normalize(records)
    query_header, query_seq = records[0]

    if report["ragged_rows_after"]:
        raise ValueError(
            f"{source}: {report['ragged_rows_after']} of {report['rows']} rows "
            f"are not {report['query_len']} columns wide even after removing "
            f"insertions; this alignment is malformed")

    if query_seq.upper() != wt_sequence.upper():
        raise ValueError(
            f"{source}: the alignment's query row does not match the wild-type "
            f"sequence ({len(query_seq)} vs {len(wt_sequence)} residues"
            + (", same length but different residues"
               if len(query_seq) == len(wt_sequence) else "")
            + "). Every column index downstream is counted from this row, so "
            "this cannot be allowed through.")

    # One canonical name for the query row, matching what the published inputs
    # use, so a file from any backend looks the same to the reader.
    records[0] = (">query", query_seq)
    out_a3m.parent.mkdir(parents=True, exist_ok=True)
    a3m_tools.write_records(out_a3m, records)
    print(f"  MSA: {report['rows']} rows x {report['query_len']} columns "
          f"-> {out_a3m}")
    return out_a3m


def _drop_repeated_query(records: List[Tuple[str, str]]
                         ) -> List[Tuple[str, str]]:
    """Remove the query row that each MMseqs2 database block opens with.

    `run_mmseqs2(use_env=True)` concatenates the UniRef result and the
    environmental result, and each block begins with the query -- all of them
    labelled `>101`. Those repeats are an artefact of the concatenation, not
    hits, and they would otherwise be merged into one record by the reader,
    which keys records on their header line and appends to a `defaultdict(str)`.
    Only exact repeats of the query are dropped; a genuine hit identical to the
    query keeps its own header and therefore survives.

    That rests on an assumption about the server, not something checked here:
    that the repeated rows carry the SAME header as the query. They do today
    (`>101` in every block). If that ever changes, this stops dropping them,
    and `dedupe_headers` then renames the survivor to `>101/dup1` and the
    reader treats it as a real hit -- changing the depth and the sequence
    weights. `proteus-doctor` reports the depth, which is where that would
    show up.
    """
    query_header, query = records[0]
    out = [records[0]]
    dropped = 0
    for header, seq in records[1:]:
        # Both, not just the sequence. A genuine UniRef hit identical to the
        # query is common in a close family, and it carries its own header --
        # dropping it would change the depth and the sequence weights.
        if header == query_header and seq == query:
            dropped += 1
            continue
        out.append((header, seq))
    if dropped:
        print(f"  dropped {dropped} repeated query row(s) from the "
              f"concatenated database blocks")
    return out


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def from_colabfold(wt_sequence: str, out_a3m: Path,
                   host_url: str = DEFAULT_HOST,
                   use_env: bool = True, use_filter: bool = True,
                   keep_raw: Optional[Path] = None) -> Path:
    """Search the public MMseqs2 server and write the alignment."""
    from proteus.third_party.colabfold_msa import run_mmseqs2

    print(f"  searching {host_url} for a {len(wt_sequence)}-residue query "
          f"(use_env={use_env}, use_filter={use_filter})")
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "mmseqs"
        result = run_mmseqs2([wt_sequence], str(prefix), use_env=use_env,
                             use_filter=use_filter, use_pairing=False,
                             host_url=host_url)
        a3m_text = result[0]
        if keep_raw:
            keep_raw.parent.mkdir(parents=True, exist_ok=True)
            keep_raw.write_text(a3m_text)
            print(f"  raw server output kept at {keep_raw}")
        raw = Path(tmp) / "server.a3m"
        raw.write_text(a3m_text)
        records = a3m_tools.read_records(raw)

    records = _drop_repeated_query(records)
    return _finish(records, wt_sequence, out_a3m, host_url)


def from_af3_zip(zip_path, wt_sequence: str, out_a3m: Path) -> Path:
    """Pull the unpaired single-chain alignment out of an AlphaFold Server job."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist()
                 if Path(n).match(AF3_MSA_GLOB) and not n.endswith("/")]
        if not names:
            raise FileNotFoundError(
                f"{zip_path}: no member matching {AF3_MSA_GLOB}. An "
                f"AlphaFold Server archive contains "
                f"fold_<name>_unpaired_msa_chains_a.a3m; this archive holds "
                f"{len(zf.namelist())} members, first few: "
                f"{zf.namelist()[:5]}")
        if len(names) > 1:
            raise ValueError(
                f"{zip_path}: {len(names)} alignments match {AF3_MSA_GLOB} "
                f"({names}). This path handles a single-chain job; extract the "
                f"chain you want and pass it with --msa a3m.")
        with tempfile.TemporaryDirectory() as tmp:
            # Read the one member we chose rather than extracting the archive,
            # so a crafted member name cannot write outside the temp directory.
            staged = Path(tmp) / "af3.a3m"
            with zf.open(names[0]) as src, open(staged, "wb") as dst:
                shutil.copyfileobj(src, dst)
            records = a3m_tools.read_records(staged)
    print(f"  took {names[0]} from {zip_path.name}")
    return _finish(records, wt_sequence, out_a3m, str(zip_path))


def from_a3m(path, wt_sequence: str, out_a3m: Path) -> Path:
    """Validate and normalise an alignment you already have."""
    path = Path(path)
    records = a3m_tools.read_records(path)
    if path.resolve() == out_a3m.resolve():
        raise ValueError(f"{path}: source and destination are the same file")
    return _finish(records, wt_sequence, out_a3m, str(path))


def fetch(backend: str, wt_sequence: str, out_a3m: Path,
          source: Optional[str] = None, host_url: str = DEFAULT_HOST,
          keep_raw: Optional[Path] = None) -> Path:
    """Dispatch to a backend. `source` is the ZIP or a3m path where needed."""
    out_a3m = Path(out_a3m)
    if backend == "colabfold":
        return from_colabfold(wt_sequence, out_a3m, host_url=host_url,
                              keep_raw=keep_raw)
    if backend == "af3-zip":
        if not source:
            raise ValueError("--msa af3-zip needs --msa-source <job.zip>")
        return from_af3_zip(source, wt_sequence, out_a3m)
    if backend == "a3m":
        if not source:
            raise ValueError("--msa a3m needs --msa-source <alignment.a3m>")
        return from_a3m(source, wt_sequence, out_a3m)
    raise ValueError(f"unknown MSA backend {backend!r}; expected one of {BACKENDS}")
