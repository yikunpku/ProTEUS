#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Get a protein's two static inputs: its MSA and its wild-type structure.

Everything else in the pipeline is derived from the library CSV. These two are
not, and obtaining them used to be the one manual step: submit a web job, wait,
download an archive, pick two files out of it, convert one, edit the config.
This does that from a sequence.

    proteus-prepare --config cfg.yaml --protein MY_PROTEIN --fasta wt.fasta

which searches for an MSA, predicts the wild-type structure from it, checks
the three descriptions of the protein against each other, and prints the
registry entry to add.

Matching the published inputs instead of searching fresh:

    proteus-prepare --config cfg.yaml --protein MY_PROTEIN --fasta wt.fasta \
        --msa af3-zip --msa-source fold_job.zip \
        --structure af3-zip --structure-source fold_job.zip

One AlphaFold Server archive carries both, which is how the published proteins
were set up by hand.

Nothing is written outside `proteins/<key>/`, and the config file is left alone
unless you pass --register.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from proteus.config import (SEQUENCE_COL, VARIANT_COL, WT_TAG, ProteusConfig,
                            default_library_path, default_msa_path,
                            default_pdb_path)
from proteus.prepare import a3m_tools, msa as msa_backend, structure as structure_backend
from proteus.prepare import validate

REGISTRY_TEMPLATE = """  {key}:
    library_csv: {library}
    wt_pdb:      {pdb}
    msa_a3m:     {a3m}
    crop: {crop}
"""


def config_root(config_path: Path) -> Path:
    """`root` from the config, resolved the same way ProteusConfig resolves it."""
    import yaml

    raw = yaml.safe_load(config_path.read_text()) or {}
    root = Path(raw.get("root", config_path.parent))
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def read_fasta(path) -> str:
    """The sequence from a single-record FASTA."""
    seq: List[str] = []
    headers = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                headers += 1
                continue
            seq.append(line)
    if headers > 1:
        raise SystemExit(
            f"{path} holds {headers} records; this takes one wild-type "
            f"sequence. Split the file, or pass --sequence.")
    out = "".join(seq).upper()
    if not out:
        raise SystemExit(f"{path}: no sequence found")
    return out


def sequence_from_library(library: Path) -> str:
    """The `wild_type` row of the library, which is what the pipeline reads."""
    import pandas as pd

    if not library.exists():
        raise SystemExit(
            f"{library} does not exist, so the wild-type sequence cannot be "
            f"read from it. Pass --fasta or --sequence.")
    df = pd.read_csv(library, usecols=lambda c: c in (VARIANT_COL, SEQUENCE_COL))
    rows = df[df[VARIANT_COL] == WT_TAG]
    if rows.empty:
        raise SystemExit(
            f"{library} has no {WT_TAG!r} row. Every step reads the wild type "
            f"from it, so add one, or pass --fasta.")
    return str(rows[SEQUENCE_COL].iloc[0]).upper()


def register(config_path: Path, block: str) -> None:
    """Insert a protein block into the config, leaving the rest byte-for-byte.

    A text insertion rather than a load-and-dump: the config is hand-written
    and usually commented, and round-tripping it through a YAML dumper would
    drop the comments and reorder the keys.
    """
    text = config_path.read_text()
    if not text.endswith("\n"):
        text += "\n"
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.rstrip() == "proteins:":
            lines.insert(i + 1, block)
            config_path.write_text("".join(lines))
            print(f"  registered in {config_path}")
            return
    raise SystemExit(
        f"{config_path} has no top-level `proteins:` key to insert under. Add "
        f"the block below by hand.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Proteus YAML config")
    ap.add_argument("--protein", required=True,
                    help="registry key for this protein; also the file stem "
                         "under proteins/<key>/")

    src = ap.add_argument_group("wild-type sequence (one of these)")
    src.add_argument("--fasta", help="single-record FASTA")
    src.add_argument("--sequence", help="the sequence itself")

    m = ap.add_argument_group("MSA")
    m.add_argument("--msa", default="colabfold", choices=msa_backend.BACKENDS,
                   help="colabfold: search the public MMseqs2 server (default; "
                        "sends the sequence to a third party). af3-zip: take it "
                        "from an AlphaFold Server archive. a3m: use a file you "
                        "have.")
    m.add_argument("--msa-source",
                   help="the archive or alignment, for --msa af3-zip / a3m")
    m.add_argument("--msa-host", default=msa_backend.DEFAULT_HOST,
                   help="MMseqs2 server (default: %(default)s)")
    m.add_argument("--legacy-a3m", action="store_true",
                   help="do not put the alignment into the rectangular shape "
                        "the MSA reader assumes. Reproduces numbers produced "
                        "before that rewrite existed; see the README.")

    s = ap.add_argument_group("structure")
    s.add_argument("--structure", default="boltz",
                   choices=structure_backend.BACKENDS,
                   help="boltz: predict it (default). af3-zip: take the "
                        "top-ranked model from an archive. pdb: a file you "
                        "have. rcsb: download by PDB ID.")
    s.add_argument("--structure-source",
                   help="the archive, file or PDB ID, for the backends that "
                        "need one")
    s.add_argument("--boltz-exe", default=None,
                   help="the `boltz` launcher of the structure predictor's own "
                        "environment (or set PROTEUS_BOLTZ). It has to be a "
                        "separate environment: its torch and scipy conflict "
                        "with this one's.")
    s.add_argument("--boltz-python", default=None,
                   help="a python interpreter that has boltz installed, "
                        "instead of --boltz-exe")
    s.add_argument("--boltz-cache", default=None,
                   help="the predictor's weight cache. Point this at a "
                        "populated directory: a fresh cache downloads about "
                        "6.2 GB (boltz2_conf.ckpt 2.29, boltz2_aff.ckpt 2.06 "
                        "-- affinity weights this never uses -- and mols.tar "
                        "1.86), and mols.tar is then extracted beside itself, "
                        "so budget about 8 GB.")
    s.add_argument("--accelerator", default="gpu", choices=("gpu", "cpu"))
    s.add_argument("--devices", type=int, default=1)
    s.add_argument("--msa-server-for-structure", action="store_true",
                   help="let the predictor run its own MSA search instead of "
                        "giving it the alignment from above. Two searches, two "
                        "alignments -- only useful for comparing against a run "
                        "done that way.")

    ap.add_argument("--work-dir", default=None,
                    help="keep the predictor's inputs and raw outputs here "
                         "instead of a temporary directory")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo steps whose output is already on disk")
    ap.add_argument("--register", action="store_true",
                    help="insert the protein into the config file instead of "
                         "only printing the block")
    args = ap.parse_args(argv)

    a3m_tools.set_legacy(args.legacy_a3m)

    config_path = Path(args.config).resolve()
    key = args.protein

    # A config with an empty `proteins:` is exactly the state this command is
    # for -- the first protein has nowhere to be registered yet -- and
    # ProteusConfig rejects that, rightly, because every other command needs at
    # least one. So the root is read directly and the registry lookup is
    # attempted separately.
    root = config_root(config_path)
    spec = None
    try:
        cfg = ProteusConfig.from_yaml(config_path)
        root = cfg.root
        if key in cfg.proteins:
            spec = cfg.spec(key)
    except ValueError as exc:
        if "no proteins defined" not in str(exc):
            raise
        print(f"{config_path.name} has no proteins yet; {key} will be the first")

    known = spec is not None
    if known:
        a3m_path, pdb_path, library = spec.msa_a3m, spec.wt_pdb, spec.library_csv
        crop = spec.crop
    else:
        a3m_path = default_msa_path(root, key)
        pdb_path = default_pdb_path(root, key)
        library = default_library_path(root, key)
        crop = "dataset"

    if args.sequence:
        wt = args.sequence.strip().upper()
    elif args.fasta:
        wt = read_fasta(args.fasta)
    else:
        wt = sequence_from_library(Path(library))
    print(f"{key}: wild type is {len(wt)} residues"
          f"{' (from the library)' if not (args.sequence or args.fasta) else ''}")

    # Which files THIS run produced, so that a validation failure can set them
    # aside without touching inputs that were already there.
    msa_written = pdb_written = False

    # --- MSA ---------------------------------------------------------------
    if a3m_path.exists() and not args.overwrite:
        print(f"  MSA already present, keeping it: {a3m_path}")
    else:
        msa_backend.fetch(args.msa, wt, a3m_path, source=args.msa_source,
                          host_url=args.msa_host)
        msa_written = True

    # --- structure ---------------------------------------------------------
    if pdb_path.exists() and not args.overwrite:
        print(f"  structure already present, keeping it: {pdb_path}")
    else:
        kwargs = {}
        if args.structure == "boltz":
            kwargs = dict(
                boltz_exe=args.boltz_exe, boltz_python=args.boltz_python,
                cache=args.boltz_cache, accelerator=args.accelerator,
                devices=args.devices,
                use_msa_server=args.msa_server_for_structure,
                work_dir=Path(args.work_dir) if args.work_dir else None)
        structure_backend.fetch(
            args.structure, wt, pdb_path,
            source=args.structure_source,
            msa_a3m=None if args.msa_server_for_structure else a3m_path,
            **kwargs)
        pdb_written = True

    # --- do the three descriptions agree? ----------------------------------
    print()
    issues = validate.check_protein(wt, a3m_path, pdb_path, label=key)
    if not issues.ok:
        # Move what this run wrote out of the way. The resume checks above are
        # `path.exists()`, so leaving a rejected MSA or structure at its final
        # name made the next run -- even one asking for a different backend --
        # print "already present, keeping it" and skip it.
        moved = []
        for path, produced in ((a3m_path, msa_written), (pdb_path, pdb_written)):
            if produced and path.exists():
                rejected = path.with_name(path.name + ".rejected")
                path.replace(rejected)
                moved.append(rejected)
        print("\nThe inputs do not agree, so this protein is not ready to "
              "register. Each failure above says which step reads the thing "
              "that is wrong.")
        if moved:
            print("What this run wrote has been set aside so a re-run does not "
                  "mistake it for finished work:")
            for path in moved:
                print(f"    {path}")
        return 1

    # --- registry ----------------------------------------------------------
    block = REGISTRY_TEMPLATE.format(
        key=key,
        library=_relative(library, root),
        pdb=_relative(pdb_path, root),
        a3m=_relative(a3m_path, root),
        crop=crop)
    print()
    if known:
        print(f"{key} is already registered in {config_path}; nothing to add.")
    elif args.register:
        register(config_path, block)
    else:
        print(f"Add this to {config_path} under `proteins:` "
              f"(or re-run with --register):\n")
        print(block)

    if not Path(library).exists():
        print(f"Still needed: the depth-1 library at {library} "
              f"(columns: {VARIANT_COL}, {SEQUENCE_COL}, and a fitness column), "
              f"with a {WT_TAG!r} row.")
    return 0


def _relative(path, root) -> str:
    """Config paths are written relative to `root` where possible."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
