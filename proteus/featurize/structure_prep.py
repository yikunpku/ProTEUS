# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Mutated structures for a library, via the PyMOL mutagenesis wizard.

Two stages, because the second one is the expensive one and is worth resuming:

    pml   library CSV + wild-type PDB -> one `.pml` script per variant
    pdb   run those scripts through PyMOL -> one mutated PDB per variant

Both stages are keyed by `proteus.config.variant_stem`, the stem shared with
every other modality, and both skip work whose output is already on disk.

    python -m proteus.featurize.structure_prep --config cfg.yaml \
        --protein KEY --depth 2 --stage both

The `pdb` stage needs the `pymol` executable (conda-forge package
`pymol-open-source`); nothing else here does.
"""

import argparse
import shutil
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from proteus.al import variants as variant_utils
from proteus.config import VARIANT_COL, WT_TAG, ProteusConfig, variant_stem

AA_THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

PYMOL_MISSING = (
    "pymol was not found on PATH. Install the open-source build "
    "(`conda install -c conda-forge pymol-open-source`) or point "
    "--pymol-binary at an existing one."
)


def residue_chain_map(pdb_path: Path) -> Dict[int, str]:
    """`{residue number: chain id}` from the ATOM records, first occurrence wins.

    A variant string names a residue number but no chain, so the chain has to
    come from the structure. When two chains share a residue number -- a
    homodimer, or a partner chain modelled alongside the target -- the chain
    whose atoms come first in the file wins. That is the rule the released
    structure features were generated with, so it is load-bearing rather than
    arbitrary: picking the other chain would mutate a different copy.
    """
    chain_of: Dict[int, str] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            try:
                res_seq = int(line[22:26])
            except ValueError:
                continue
            chain_of.setdefault(res_seq, line[21])
    return dict(sorted(chain_of.items()))


def render_pml(variant: str, chain_of: Dict[int, str], wt_pdb: Path,
               out_pdb: Path) -> str:
    """The mutagenesis script for one variant.

    Raises if a substitution is unparseable or names a residue the structure
    does not contain: PyMOL would accept the empty selection, save an
    unmutated copy, and that copy would then travel through the pipeline as if
    it were a mutant.
    """
    lines = [
        f"# MUTATION: {variant}",
        f"load {wt_pdb}, orig",
        "wizard mutagenesis",
        "refresh_wizard",
    ]
    for sub in variant_utils.split_variant(variant):
        _, pos, mut_aa = variant_utils.parse_substitution(sub)
        if pos not in chain_of:
            raise ValueError(
                f"{sub}: residue {pos} is not in {wt_pdb.name} -- "
                f"check the position offset")
        lines += [
            f"select res{pos}, chain {chain_of[pos]} and resi {pos}",
            f'cmd.get_wizard().do_select("res{pos}")',
            f'cmd.get_wizard().set_mode("{AA_THREE_LETTER[mut_aa]}")',
            "cmd.get_wizard().apply()",
        ]
    lines += [f"save {out_pdb}", "disable wizard"]
    return "\n".join(lines)


def write_pml_scripts(cfg: ProteusConfig, key: str, depth: int,
                      overwrite: bool = False,
                      allow_failures: int = 0) -> int:
    """Write one `.pml` per variant of `key`'s depth-`depth` library.

    Returns the number of scripts written. Existing scripts are left alone
    unless `overwrite`, so a re-run only fills gaps -- note that each script
    hard-codes its output path, so a config whose `root` moved needs
    `overwrite` to repoint them.
    """
    spec = cfg.spec(key)
    library = cfg.library_for(key, depth)
    if not library.exists():
        raise FileNotFoundError(f"{key}: library CSV not found: {library}")
    if not spec.wt_pdb.exists():
        raise FileNotFoundError(f"{key}: wild-type PDB not found: {spec.wt_pdb}")

    pml_dir = cfg.pmls_dir(key, depth)
    pdb_dir = cfg.mutant_pdbs_dir(key, depth)
    pml_dir.mkdir(parents=True, exist_ok=True)
    pdb_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(library)
    if VARIANT_COL not in df.columns:
        raise ValueError(
            f"{library}: no {VARIANT_COL!r} column; found {list(df.columns)}")

    chain_of = residue_chain_map(spec.wt_pdb)

    # The wild type has nothing to mutate; the pdb stage copies its structure
    # in under the same stem so downstream stages see it like any variant.
    library_variants = [v for v in df[VARIANT_COL]
                        if not variant_utils.is_wild_type(v)]

    written = 0
    skipped = 0
    failures: List[Tuple[str, str]] = []
    for variant in tqdm(library_variants, desc=f"{key} pml", leave=False):
        stem = variant_stem(key, variant)
        script = pml_dir / f"{stem}.pml"
        if script.exists() and not overwrite:
            skipped += 1
            continue
        try:
            text = render_pml(variant, chain_of, spec.wt_pdb,
                              pdb_dir / f"{stem}_mutated.pdb")
            script.write_text(text)
        except (ValueError, KeyError, OSError) as exc:
            failures.append((str(variant), str(exc)))
            continue
        written += 1

    print(f"[{key}] pml: {written} written, {skipped} already present, "
          f"{len(failures)} failed, {len(library_variants)} variants "
          f"-> {pml_dir}")
    _report_failures(failures, allow_failures, f"{key}: .pml generation")
    return written


def resolve_pymol(pymol_binary: Optional[str] = None) -> str:
    """Absolute path to the PyMOL executable."""
    candidate = pymol_binary or shutil.which("pymol")
    if not candidate:
        raise FileNotFoundError(PYMOL_MISSING)
    # Accepts either a bare name on PATH or a path to the binary.
    resolved = shutil.which(candidate)
    if not resolved:
        raise FileNotFoundError(
            f"{candidate!r} is not an executable PyMOL. {PYMOL_MISSING}")
    return resolved


def _run_one(task):
    binary, pml, out_pdb, log = task
    with open(log, "w") as log_fh:
        rc = subprocess.run([binary, "-cq", str(pml)],
                            stdout=log_fh, stderr=log_fh).returncode
    # PyMOL exits 0 even when the script failed before its `save`, so the
    # product on disk is the only trustworthy success signal.
    return pml.name, rc, out_pdb.exists()


def run_pymol(cfg: ProteusConfig, key: str, depth: int,
              workers: Optional[int] = None,
              pymol_binary: Optional[str] = None,
              allow_failures: int = 0) -> int:
    """Run every pending `.pml` of `key` at `depth` through PyMOL.

    Returns the number of structures produced by this call. Variants whose
    mutated PDB already exists are not re-run.
    """
    binary = resolve_pymol(pymol_binary)
    workers = workers or max(1, cpu_count() // 2)

    spec = cfg.spec(key)
    pml_dir = cfg.pmls_dir(key, depth)
    pdb_dir = cfg.mutant_pdbs_dir(key, depth)
    log_dir = cfg.workspace(depth) / "pymol_logs" / key
    if not pml_dir.exists():
        raise FileNotFoundError(
            f"{key}: no .pml directory at {pml_dir}; run --stage pml first")

    # The mirror of the guard in featurize.structure_predict. Both write into
    # pdb_dir -- PyMOL as `<stem>_mutated.pdb`, the predictor as `<stem>.pdb` --
    # and structure_tokens globs `*.pdb`, so with both present it reads
    # whichever sorts first ('.' < '_', so predictions win) and `--total-tasks`
    # hands two shards the same stem set, racing on one temporary file.
    predicted = ([f for f in pdb_dir.glob("*.pdb")
                  if not f.stem.endswith("_mutated")]
                 if pdb_dir.is_dir() else [])
    if predicted:
        raise SystemExit(
            f"{pdb_dir} already holds {len(predicted)} structure(s) from "
            f"featurize.structure_predict (named `<stem>.pdb`). Mutating the "
            f"wild type into the same directory would leave two files per "
            f"variant, and structure_tokens would read whichever sorts first. "
            f"Pick one source: delete those, or keep using structure_predict.")

    pml_files = sorted(pml_dir.glob("*.pml"))
    if not pml_files:
        raise FileNotFoundError(
            f"{key}: {pml_dir} holds no .pml scripts; run --stage pml first")
    pdb_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for pml in pml_files:
        out_pdb = pdb_dir / f"{pml.stem}_mutated.pdb"
        if out_pdb.exists():
            continue
        tasks.append((binary, pml, out_pdb, log_dir / f"{pml.stem}.log"))

    print(f"[{key}] pdb: {len(tasks)}/{len(pml_files)} to build "
          f"(pymol={binary}, workers={workers})")

    failures: List[Tuple[str, str]] = []
    produced = 0
    if tasks:
        with Pool(processes=workers) as pool:
            for name, rc, exists in tqdm(pool.imap_unordered(_run_one, tasks),
                                         total=len(tasks), desc=f"{key} pdb",
                                         leave=False):
                if rc != 0:
                    failures.append((name, f"pymol exit status {rc}"))
                elif not exists:
                    failures.append((name, "pymol exited 0 but saved no PDB"))
                else:
                    produced += 1

    _copy_wild_type(key, spec.wt_pdb, pdb_dir)

    print(f"[{key}] pdb: {produced} built, "
          f"{len(pml_files) - len(tasks)} already present, "
          f"{len(failures)} failed -> {pdb_dir}")
    if failures:
        print(f"[{key}] pdb: logs under {log_dir}")
    _report_failures(failures, allow_failures, f"{key}: PyMOL mutagenesis")
    return produced


def _copy_wild_type(key: str, wt_pdb: Path, pdb_dir: Path) -> None:
    """Place the wild-type structure under its variant stem.

    It has no `.pml` of its own, and downstream stages address every structure
    by stem, so without this copy the wild type would simply be missing.
    """
    if not wt_pdb.exists():
        raise FileNotFoundError(f"{key}: wild-type PDB not found: {wt_pdb}")
    dst = pdb_dir / f"{variant_stem(key, WT_TAG)}_mutated.pdb"
    if not dst.exists():
        shutil.copy(wt_pdb, dst)
        print(f"[{key}] pdb: wild type copied to {dst.name}")


def _report_failures(failures: List[Tuple[str, str]], allow_failures: int,
                     what: str) -> None:
    if not failures:
        return
    for name, reason in failures[:10]:
        print(f"  FAILED {name}: {reason}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")
    if len(failures) > allow_failures:
        raise RuntimeError(
            f"{what}: {len(failures)} failure(s), "
            f"--allow-failures is {allow_failures}")
    print(f"  tolerating {len(failures)} failure(s) "
          f"(--allow-failures {allow_failures})")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="protein registry YAML")
    p.add_argument("--protein", required=True, help="protein key in the config")
    p.add_argument("--depth", type=int, required=True,
                   help="round depth; selects the library and the output dirs")
    p.add_argument("--stage", choices=("pml", "pdb", "both"), default="both",
                   help="pml: write scripts; pdb: run PyMOL (default: %(default)s)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel PyMOL processes (default: half the CPUs)")
    p.add_argument("--pymol-binary", default=None,
                   help="PyMOL executable (default: pymol on PATH)")
    p.add_argument("--overwrite", action="store_true",
                   help="rewrite .pml scripts that already exist")
    p.add_argument("--allow-failures", type=int, default=0,
                   help="failed variants to tolerate before raising "
                        "(default: %(default)s)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = ProteusConfig.from_yaml(args.config)
    if args.stage in ("pml", "both"):
        write_pml_scripts(cfg, args.protein, args.depth,
                          overwrite=args.overwrite,
                          allow_failures=args.allow_failures)
    if args.stage in ("pdb", "both"):
        run_pymol(cfg, args.protein, args.depth, workers=args.workers,
                  pymol_binary=args.pymol_binary,
                  allow_failures=args.allow_failures)


if __name__ == "__main__":
    main()
