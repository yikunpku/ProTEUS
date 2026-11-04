#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Predict a structure for every variant, instead of mutating the wild type.

This **replaces** `featurize.structure_prep` in the chain, and it is opt-in:

    structure_prep       wild-type PDB --PyMOL--> one mutated PDB per variant
    structure_predict    one folded structure per variant, from its own sequence

The default is `structure_prep`, for two reasons that are worth stating plainly
rather than leaving implicit. It is what the released features were produced
with, so changing it changes the structure tokens and the published numbers
stop being reproducible. And it is three orders of magnitude cheaper: measured
on a 119-residue protein, one prediction takes about 70 s against well under a
second for a PyMOL mutagenesis, so a 173-variant library is minutes versus
hours, and a deeper round with thousands of candidates is not comparable at
all.

This path exists because it is the arm the structure-encoder ablation was run
with, and an ablation nobody can re-run is not much of an ablation. In that
comparison it did not beat mutating the wild type.

Each variant is folded from the wild-type alignment with the query row replaced
by the variant's own sequence -- the recipe the original top-30 runs used --
so the evolutionary context is held fixed and only the query changes.

**A prediction depends on how many inputs were folded with it.** Measured: the
same wild type, same seed, same settings, came out at confidence 0.8963 folded
alone and 0.9022 folded in a batch of three, and the two structures sit 2.76 Å
CA-RMSD apart after superposition -- a different conformation, not a different
frame. Folded alone twice it is bit-identical. So the batch is part of the
input: reproducing a set of these structures needs the same sharding it was
produced with, which for the original top-30 runs was one shard of thirty. The
sidecar records the batch size and the shard for that reason.

    python -m proteus.featurize.structure_predict --config cfg.yaml \
        --protein MY_PROTEIN --depth 2 --boltz-exe /path/to/env/bin/boltz

Output goes where `structure_prep` would have put it, under the same stems, so
`featurize.structure_tokens` consumes it with no change.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from proteus.config import (SEQUENCE_COL, VARIANT_COL, WT_TAG, ProteusConfig,
                            variant_stem)
from proteus.prepare import a3m_tools
from proteus.prepare.structure import (BOLTZ_SETTINGS, find_boltz,
                                       write_boltz_yaml)


def variant_rows(cfg: ProteusConfig, key: str, depth: int
                 ) -> List[Tuple[str, str]]:
    """`[(variant, mutated_sequence)]` for this depth, wild type included.

    The wild type is folded too: the structure channel needs it, and
    `structure_prep` copies it in for the same reason.
    """
    library = cfg.library_for(key, depth)
    df = pd.read_csv(library)
    for col in (VARIANT_COL, SEQUENCE_COL):
        if col not in df.columns:
            raise ValueError(f"{library} has no {col!r} column")
    out = [(str(r[VARIANT_COL]), str(r[SEQUENCE_COL]).upper())
           for _, r in df.iterrows()]
    if not any(v == WT_TAG for v, _ in out):
        raise ValueError(
            f"{library} has no {WT_TAG!r} row; the structure channel needs the "
            f"wild type as well as the variants")
    return out


def write_inputs(records: List[Tuple[str, str]], wt_sequence: str,
                 work: Path, jobs: List[Tuple[str, str]]) -> Dict[str, str]:
    """One YAML and one alignment per variant. Returns job id -> variant.

    The alignment and the input live in separate directories on purpose: the
    predictor, given a directory, rejects any file that is not a `.yaml` or a
    `.fasta`, so an `.a3m` sitting beside the inputs stops the whole batch.
    """
    inputs = work / "inputs"
    msas = work / "msa"
    inputs.mkdir(parents=True, exist_ok=True)
    msas.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str] = {}
    for i, (variant, sequence) in enumerate(jobs):
        if len(sequence) != len(wt_sequence):
            raise ValueError(
                f"{variant}: sequence is {len(sequence)} residues and the wild "
                f"type is {len(wt_sequence)}. The alignment's query row is "
                f"replaced in place, so the lengths have to match.")
        rows = [(">query", sequence)] + records[1:]
        a3m = msas / f"{i}.a3m"
        a3m_tools.write_records(a3m, rows)
        write_boltz_yaml(sequence, inputs / f"{i}.yaml", msa_a3m=a3m)
        mapping[str(i)] = variant
    return inputs, mapping


def collect(out_root: Path, inputs_name: str, mapping: Dict[str, str],
            key: str, pdb_dir: Path) -> Tuple[int, List[str], Dict[str, dict]]:
    """Move the top-ranked model of each job to where structure_tokens looks."""
    pdb_dir.mkdir(parents=True, exist_ok=True)
    pred_root = out_root / f"boltz_results_{inputs_name}" / "predictions"
    written, missing = 0, []
    confidences: Dict[str, dict] = {}
    for job, variant in mapping.items():
        src = pred_root / job / f"{job}_model_0.pdb"
        if not src.exists():
            missing.append(variant)
            continue
        shutil.copy(src, pdb_dir / f"{variant_stem(key, variant)}.pdb")
        conf = pred_root / job / f"confidence_{job}_model_0.json"
        if conf.exists():
            try:
                confidences[variant] = json.loads(conf.read_text())
            except Exception:  # noqa: BLE001
                pass
        written += 1
    return written, missing, confidences


def run(cfg: ProteusConfig, key: str, depth: int,
        boltz_exe: Optional[str] = None, boltz_python: Optional[str] = None,
        cache: Optional[str] = None, accelerator: str = "gpu",
        devices: int = 1, task_id: int = 0, total_tasks: int = 1,
        limit: Optional[int] = None, work_dir: Optional[Path] = None,
        settings: Optional[dict] = None) -> dict:
    spec = cfg.spec(key)
    pdb_dir = cfg.mutant_pdbs_dir(key, depth)
    records = a3m_tools.for_boltz(a3m_tools.read_records(spec.msa_a3m))
    wt_alignment_query = records[0][1]

    rows = variant_rows(cfg, key, depth)
    wt_sequence = next(s for v, s in rows if v == WT_TAG)
    if wt_alignment_query != wt_sequence:
        raise ValueError(
            f"{key}: the alignment's query row ({len(wt_alignment_query)} "
            f"residues) is not the library's wild type ({len(wt_sequence)}). "
            f"Run proteus-prepare, which checks this before registering.")

    # `structure_tokens` keys on the stem and accepts both `<stem>.pdb` and
    # `<stem>_mutated.pdb`, so a directory holding both has two files per
    # variant and which one is read comes down to sort order. Refuse rather
    # than quietly pick: the two are different structures, and the published
    # features came from the mutated one.
    pymol_made = sorted(pdb_dir.glob("*_mutated.pdb")) if pdb_dir.is_dir() else []
    if pymol_made:
        raise SystemExit(
            f"{pdb_dir} already holds {len(pymol_made)} structures built by "
            f"featurize.structure_prep (`*_mutated.pdb`). Folding each variant "
            f"into the same directory would leave two files per variant and "
            f"featurize.structure_tokens would read whichever sorts first. "
            f"Choose one path: delete those files to fold instead, or keep "
            f"using featurize.structure_prep and do not run this step.")

    todo = [(v, s) for v, s in rows
            if not (pdb_dir / f"{variant_stem(key, v)}.pdb").exists()]
    skipped = len(rows) - len(todo)
    todo = todo[task_id::total_tasks]
    if limit is not None:
        todo = todo[:limit]

    print(f"{key} depth {depth} shard {task_id}/{total_tasks}: "
          f"{len(todo)} to fold, {skipped} already present")
    if not todo:
        return {"written": 0, "skipped": skipped, "missing": []}

    cfg_settings = dict(BOLTZ_SETTINGS)
    if settings:
        cfg_settings.update(settings)
    print(f"  one prediction took about 70 s for a 119-residue protein at "
          f"these settings, so expect roughly "
          f"{len(todo) * 70 / 3600:.1f} h of GPU time for {len(todo)} variants")

    cmd = find_boltz(boltz_exe, boltz_python)
    keep = work_dir is not None
    work = Path(work_dir) if keep else Path(
        tempfile.mkdtemp(prefix=f"proteus-fold-{key}-"))
    work.mkdir(parents=True, exist_ok=True)

    inputs, mapping = write_inputs(records, wt_sequence, work, todo)
    out_root = work / "out"

    cmd += ["predict", str(inputs),
            "--out_dir", str(out_root),
            "--output_format", "pdb",
            "--model", str(cfg_settings["model"]),
            "--seed", str(cfg_settings["seed"]),
            "--diffusion_samples", str(cfg_settings["diffusion_samples"]),
            "--sampling_steps", str(cfg_settings["sampling_steps"]),
            "--step_scale", str(cfg_settings["step_scale"]),
            "--noise_scale", str(cfg_settings["noise_scale"]),
            "--gamma_0", str(cfg_settings["gamma_0"]),
            "--accelerator", accelerator,
            "--devices", str(devices),
            "--override"]
    if cfg_settings["project_pbd"]:
        cmd.append("--project_pbd")
    if cfg_settings["use_potentials"]:
        cmd.append("--use_potentials")
    if cache:
        cmd += ["--cache", str(cache)]

    print(f"  $ {' '.join(cmd[:4])} ... ({len(todo)} inputs)")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(work))
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"the structure predictor exited {proc.returncode} after "
            f"{elapsed:.0f}s; its inputs are under {work}"
            + ("" if keep else ", a temporary directory kept on purpose so you "
                               "can look at it -- delete it when you are done, "
                               "or pass --work-dir to choose where it goes"))

    written, missing, confidences = collect(out_root, inputs.name, mapping,
                                            key, pdb_dir)
    print(f"  {written} structures -> {pdb_dir}  ({elapsed:.0f}s, "
          f"{elapsed / max(1, written):.1f}s each)")
    if missing:
        print(f"  [fail] no model produced for {len(missing)}: {missing[:5]}")

    sidecar = pdb_dir.parent / f"{key}_structure_predict.json"
    payload = {
        "protein": key, "depth": depth,
        "settings": cfg_settings,
        "alignment": str(spec.msa_a3m),
        "query_row": "replaced per variant",
        # Load-bearing, not bookkeeping: a prediction is not reproducible
        # without the batch it was folded in. See the module docstring.
        "batch": {"inputs_folded": len(todo),
                  "task_id": task_id, "total_tasks": total_tasks,
                  "variants": [v for v, _ in todo]},
        "written": written, "missing": missing,
        "seconds": round(elapsed, 1),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "confidence": {v: {k: c.get(k) for k in
                           ("confidence_score", "complex_plddt", "ptm")}
                       for v, c in confidences.items()},
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"  sidecar: {sidecar}")
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return {"written": written, "skipped": skipped, "missing": missing}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--protein", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--boltz-exe", default=None)
    ap.add_argument("--boltz-python", default=None)
    ap.add_argument("--boltz-cache", default=None)
    ap.add_argument("--accelerator", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--total-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="fold only this many variants (testing)")
    ap.add_argument("--work-dir", default=None,
                    help="keep the predictor's inputs and raw outputs here")
    args = ap.parse_args(argv)

    cfg = ProteusConfig.from_yaml(args.config)
    counts = run(cfg, args.protein, args.depth,
                 boltz_exe=args.boltz_exe, boltz_python=args.boltz_python,
                 cache=args.boltz_cache, accelerator=args.accelerator,
                 devices=args.devices, task_id=args.task_id,
                 total_tasks=args.total_tasks, limit=args.limit,
                 work_dir=Path(args.work_dir) if args.work_dir else None)
    return 1 if counts["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
