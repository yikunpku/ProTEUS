# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Getting a wild-type structure: predict one, or import one you have.

The pipeline only ever needs the **wild-type** structure. Mutant structures
are built from it with PyMOL's mutagenesis wizard
(`proteus.featurize.structure_prep`), which is what the released features were
produced with, so predicting each variant separately is not on this path. That
variant-by-variant mode exists as an ablation and is a separate entry point.

Backends:

`boltz`  (default)
    Boltz-2 with the Gauss-Seidel coordinate projection (`--project_pbd`),
    which is the predictor and the setting this project's structures were
    produced with. Run as a subprocess in **its own environment**, not
    imported: the two dependency sets conflict outright -- measured, the
    predictor's environment carries torch 2.12/scipy 1.13/pandas 2.3 against
    this one's torch 2.7/scipy 1.15/pandas 2.2, and moving either side is how
    you stop reproducing the published selections. Point `--boltz-exe` or
    `$PROTEUS_BOLTZ` at the other environment's `boltz`.

`af3-zip`
    The `fold_*_model_0.cif` out of an AlphaFold Server job, converted to PDB.
    Pairs with `--msa af3-zip`: one archive supplies both inputs, which is how
    the published proteins were set up by hand.

`pdb` / `rcsb`
    A file you have, or an experimental structure by PDB ID. Validated, not
    modified. `rcsb` is how one of the published targets was set up -- an
    experimental structure beats a prediction when one exists.

Everything here writes PDB, never mmCIF, because that is all the downstream
code reads: `featurize/structure_tokens.py` globs `*.pdb` and
`featurize/structure_prep.py` parses fixed-column `ATOM` records directly.
"""

import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

BACKENDS = ("boltz", "af3-zip", "pdb", "rcsb")

# What an AlphaFold Server archive calls the top-ranked model.
AF3_MODEL_GLOB = "*model_0.cif"
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# The predictor settings these structures were produced with, copied from the
# project's own run configuration (`config/<target>/boltz-2-gs-10step-*.yaml`)
# rather than from the predictor's defaults -- they differ. Notably the
# defaults are 200 sampling steps and step_scale 1.5.
BOLTZ_SETTINGS = {
    "model": "boltz2",
    "sampling_steps": 10,
    "diffusion_samples": 5,
    "seed": 101,
    "step_scale": 1.0,
    "noise_scale": 1.0,
    "gamma_0": 0.0,
    "project_pbd": True,     # the Gauss-Seidel projection
    "use_potentials": False,
}


def find_boltz(exe: Optional[str] = None,
               python: Optional[str] = None) -> List[str]:
    """The command prefix that runs the predictor, or a message saying how to.

    Resolution order: what was passed, then `$PROTEUS_BOLTZ`, then whatever is
    on PATH. A python interpreter is accepted instead of the launcher because
    a conda environment is often referred to by its interpreter.
    """
    if python:
        # Checked for the same reason `exe` is: an interpreter path with a typo
        # should say so here, not at the subprocess boundary. Note `python`
        # wins over `exe` when both are given.
        if not (Path(python).exists() or shutil.which(str(python))):
            raise SystemExit(
                f"--boltz-python {python} does not exist and is not on PATH.")
        return [str(python), "-c",
                "from boltz.main import cli; cli()"]
    # An explicit --boltz-exe that does not resolve is an error, not a reason
    # to try the next candidate: silently running a different environment is
    # exactly how you stop reproducing the published selections.
    if exe and not (Path(exe).exists() or shutil.which(exe)):
        raise SystemExit(
            f"--boltz-exe {exe} does not exist and is not on PATH. Fix the "
            f"path rather than letting $PROTEUS_BOLTZ or PATH decide which "
            f"environment runs the prediction.")
    for candidate in (exe, os.environ.get("PROTEUS_BOLTZ"),
                      shutil.which("boltz")):
        if candidate and (Path(candidate).exists() or shutil.which(candidate)):
            return [str(candidate)]
    raise SystemExit(
        "the structure predictor was not found. It lives in its own "
        "environment on purpose -- its torch and scipy versions conflict with "
        "this one's -- so tell Proteus where that environment is:\n"
        "    proteus-prepare ... --boltz-exe /path/to/env/bin/boltz\n"
        "or export PROTEUS_BOLTZ=/path/to/env/bin/boltz\n"
        "See the structure-prediction repository for how to install it, or "
        "use --structure af3-zip / --structure pdb to import a structure "
        "instead of predicting one.")


def _boltz_safe_alignment(msa_a3m, out_a3m: Path, wt_sequence: str) -> Path:
    """Rewrite the alignment into the dialect the predictor's parser accepts.

    See `a3m_tools.for_boltz` for what differs and why. The query check here is
    the one that matters: if the cleaned query is not the wild type, the
    predictor would fold a different sequence than the one being engineered.
    """
    from proteus.prepare import a3m_tools

    records = a3m_tools.for_boltz(a3m_tools.read_records(msa_a3m))
    query = records[0][1]
    if query != wt_sequence.upper():
        raise ValueError(
            f"{msa_a3m}: after cleaning for the predictor the query row is "
            f"{len(query)} residues and the wild type is {len(wt_sequence)}"
            + ("; same length, different residues" if len(query) == len(wt_sequence)
               else "")
            + ". The predictor would model a different sequence, so stopping "
              "here.")
    out_a3m.parent.mkdir(parents=True, exist_ok=True)
    a3m_tools.write_records(out_a3m, records)
    return out_a3m


def write_boltz_yaml(sequence: str, out_yaml: Path,
                     msa_a3m: Optional[Path] = None,
                     chain_id: str = "A") -> Path:
    """The predictor's single-chain input.

    `msa: empty` is the predictor's own spelling for single-sequence mode. We
    write the alignment path when we have one, so the structure and the MSA
    features come from the same alignment instead of two separate searches.
    """
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    lines = ["sequences:", "- protein:", "    id:", f"    - {chain_id}",
             f"    sequence: {sequence}"]
    if msa_a3m is not None:
        lines.append(f"    msa: {Path(msa_a3m).resolve()}")
    out_yaml.write_text("\n".join(lines) + "\n")
    return out_yaml


def from_boltz(wt_sequence: str, out_pdb: Path,
               msa_a3m: Optional[Path] = None,
               use_msa_server: bool = False,
               boltz_exe: Optional[str] = None,
               boltz_python: Optional[str] = None,
               cache: Optional[str] = None,
               accelerator: str = "gpu",
               devices: int = 1,
               work_dir: Optional[Path] = None,
               settings: Optional[dict] = None,
               extra_args: Optional[List[str]] = None) -> Path:
    """Predict the wild-type structure and place the top-ranked model."""
    cfg = dict(BOLTZ_SETTINGS)
    if settings:
        cfg.update(settings)
    if msa_a3m is None and not use_msa_server:
        raise ValueError(
            "predicting a structure needs an alignment: pass msa_a3m (what "
            "`proteus-prepare` does, so the structure and the MSA features "
            "share one alignment) or use_msa_server=True to let the predictor "
            "run its own search.")

    cmd = find_boltz(boltz_exe, boltz_python)
    keep = work_dir is not None
    work = Path(work_dir) if keep else Path(tempfile.mkdtemp(prefix="proteus-boltz-"))
    work.mkdir(parents=True, exist_ok=True)

    stem = out_pdb.stem
    if msa_a3m is not None:
        msa_a3m = _boltz_safe_alignment(msa_a3m, work / f"{stem}.boltz.a3m",
                                        wt_sequence)
    yaml_path = write_boltz_yaml(wt_sequence, work / f"{stem}.yaml", msa_a3m)
    out_root = work / "out"

    cmd += ["predict", str(yaml_path),
            "--out_dir", str(out_root),
            "--output_format", "pdb",
            "--model", str(cfg["model"]),
            "--seed", str(cfg["seed"]),
            "--diffusion_samples", str(cfg["diffusion_samples"]),
            "--sampling_steps", str(cfg["sampling_steps"]),
            "--step_scale", str(cfg["step_scale"]),
            "--noise_scale", str(cfg["noise_scale"]),
            "--gamma_0", str(cfg["gamma_0"]),
            "--accelerator", accelerator,
            "--devices", str(devices),
            "--override"]
    if cfg["project_pbd"]:
        cmd.append("--project_pbd")
    if cfg["use_potentials"]:
        cmd.append("--use_potentials")
    if use_msa_server:
        cmd.append("--use_msa_server")
    if cache:
        cmd += ["--cache", str(cache)]
    if extra_args:
        cmd += list(extra_args)

    print(f"  predicting the wild-type structure "
          f"({len(wt_sequence)} residues, {cfg['sampling_steps']} sampling "
          f"steps, {cfg['diffusion_samples']} samples, seed {cfg['seed']}, "
          f"project_pbd={cfg['project_pbd']})")
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(work))
    if proc.returncode != 0:
        raise RuntimeError(
            f"the structure predictor exited {proc.returncode}. Its own output "
            f"is above; the inputs it was given are under {work}"
            + ("" if keep else ", which is a temporary directory kept on "
                               "purpose so you can look at it -- delete it "
                               "when you are done, or pass --work-dir to put "
                               "it somewhere you choose"))

    # predictions/<stem>/<stem>_model_0.pdb, where _model_0 is the sample with
    # the highest confidence (the writer ranks them before naming).
    pred_dir = out_root / f"boltz_results_{stem}" / "predictions" / stem
    best = pred_dir / f"{stem}_model_0.pdb"
    if not best.exists():
        found = sorted(p.name for p in pred_dir.glob("*")) if pred_dir.is_dir() else []
        raise FileNotFoundError(
            f"the predictor reported success but {best} is not there. "
            f"{pred_dir} holds: {found or 'nothing'}")

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out_pdb)
    confidence = pred_dir / f"confidence_{stem}_model_0.json"
    if confidence.exists():
        shutil.copy(confidence, out_pdb.with_name(f"{out_pdb.stem}_confidence.json"))
        try:
            import json
            c = json.loads(confidence.read_text())
            print(f"  confidence: score={c.get('confidence_score'):.4f} "
                  f"plddt={c.get('complex_plddt'):.4f} ptm={c.get('ptm'):.4f}")
        except Exception:
            pass
    print(f"  structure -> {out_pdb}")
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return out_pdb


# --------------------------------------------------------------------------
# importing a structure
# --------------------------------------------------------------------------

def cif_to_pdb(cif_path, pdb_path) -> Path:
    """Convert mmCIF to PDB, keeping only the amino-acid chain(s).

    Waters, ions and ligands are dropped: `structure_tokens` requires the
    residue count to equal the library's sequence length, and a HETATM water
    would break that. Hydrogens go too, since neither consumer reads them and
    PyMOL re-derives them.
    """
    from biotite.structure import filter_amino_acids
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure

    cif = CIFFile.read(str(cif_path))
    atoms = get_structure(cif, model=1)
    atoms = atoms[filter_amino_acids(atoms)]
    atoms = atoms[atoms.element != "H"]
    out = PDBFile()
    out.set_structure(atoms)
    Path(pdb_path).parent.mkdir(parents=True, exist_ok=True)
    out.write(str(pdb_path))
    return Path(pdb_path)


def from_af3_zip(zip_path, out_pdb: Path) -> Path:
    """Take the top-ranked model out of an AlphaFold Server job archive."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist()
                 if Path(n).match(AF3_MODEL_GLOB) and not n.endswith("/")]
        if not names:
            raise FileNotFoundError(
                f"{zip_path}: no member matching {AF3_MODEL_GLOB}. An "
                f"AlphaFold Server archive contains fold_<name>_model_0.cif; "
                f"this one holds {len(zf.namelist())} members, first few: "
                f"{zf.namelist()[:5]}")
        name = sorted(names)[0]
        with tempfile.TemporaryDirectory() as tmp:
            # Read the chosen member rather than extracting the archive, so a
            # crafted member name cannot write outside the temp directory.
            staged = Path(tmp) / "model.cif"
            with zf.open(name) as src, open(staged, "wb") as dst:
                shutil.copyfileobj(src, dst)
            cif_to_pdb(staged, out_pdb)
    print(f"  took {name} from {zip_path.name} -> {out_pdb}")
    return out_pdb


def from_pdb(path, out_pdb: Path) -> Path:
    """Place a structure you already have. Converted if it is mmCIF."""
    path = Path(path)
    if path.resolve() == out_pdb.resolve():
        print(f"  structure already in place: {out_pdb}")
        return out_pdb
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".cif", ".mmcif"):
        cif_to_pdb(path, out_pdb)
    else:
        shutil.copy(path, out_pdb)
    print(f"  structure <- {path} -> {out_pdb}")
    return out_pdb


def from_rcsb(pdb_id: str, out_pdb: Path) -> Path:
    """Download an experimental structure by PDB ID."""
    pdb_id = pdb_id.strip().upper()
    if not pdb_id.isalnum() or len(pdb_id) != 4:
        raise ValueError(f"{pdb_id!r} is not a 4-character PDB ID")
    url = RCSB_URL.format(pdb_id=pdb_id)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        body = resp.read()
    out_pdb.write_bytes(body)
    print(f"  structure -> {out_pdb} ({len(body)} bytes)")
    print("  note: an experimental structure usually has gaps, extra chains "
          "and its own residue numbering. The validation step will say whether "
          "it lines up with your library's sequence positions.")
    return out_pdb


def fetch(backend: str, wt_sequence: str, out_pdb: Path,
          source: Optional[str] = None,
          msa_a3m: Optional[Path] = None, **kwargs) -> Path:
    """Dispatch to a backend. `source` is the ZIP, file or PDB ID where needed."""
    out_pdb = Path(out_pdb)
    if backend == "boltz":
        return from_boltz(wt_sequence, out_pdb, msa_a3m=msa_a3m, **kwargs)
    if backend == "af3-zip":
        if not source:
            raise ValueError("--structure af3-zip needs --structure-source <job.zip>")
        return from_af3_zip(source, out_pdb)
    if backend == "pdb":
        if not source:
            raise ValueError("--structure pdb needs --structure-source <file.pdb>")
        return from_pdb(source, out_pdb)
    if backend == "rcsb":
        if not source:
            raise ValueError("--structure rcsb needs --structure-source <PDB ID>")
        return from_rcsb(source, out_pdb)
    raise ValueError(
        f"unknown structure backend {backend!r}; expected one of {BACKENDS}")
