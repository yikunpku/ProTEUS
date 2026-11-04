#!/usr/bin/env python
# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement in
# LICENSE at the repository root. Not for commercial use or redistribution.
# The rest of this subtree is upstream Boltz under MIT; see
# proteusfold/LICENSE.
"""Run a prediction over a directory of inputs, sharded across GPUs.

Replaces the per-dataset shell scripts and the generated per-shard config
files that differed only in a dataset name, an input path and a shard range.

    # one input, one GPU
    python predict.py --input inputs/1.yaml --out out/

    # a directory, four GPUs, model loaded once per GPU
    python predict.py --input inputs/ --out out/ --gpus 0,1,2,3

    # one shard of a larger sweep, for a second machine
    python predict.py --input inputs/ --out out/ --shard 1/4

The Gauss-Seidel projection is on by default, which is what this fork adds and
what its results were produced with; `--no-project-pbd` gives plain Boltz-2.
The sampling defaults are this fork's, not upstream's -- 10 steps and
step_scale 1.0 against upstream's 200 and 1.5.

Inputs are the `boltz` schema: one YAML (or FASTA) per structure. See
docs/prediction.md. An MSA is required for every protein chain: give each
chain an `msa:` path, or pass --use-msa-server to have one searched.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

INPUT_SUFFIXES = (".yaml", ".yml", ".fasta", ".fa", ".fas")

# This fork's settings, which differ from the upstream defaults.
DEFAULTS = {
    "sampling_steps": 10,
    "diffusion_samples": 5,
    "seed": 101,
    "step_scale": 1.0,
    "noise_scale": 1.0,
    "gamma_0": 0.0,
}


def find_inputs(path: Path) -> List[Path]:
    if not path.exists():
        raise SystemExit(f"--input {path} does not exist")
    if path.is_file():
        return [path]
    files = sorted(p for p in path.iterdir()
                   if p.suffix.lower() in INPUT_SUFFIXES)
    if not files:
        raise SystemExit(
            f"{path} holds no {'/'.join(INPUT_SUFFIXES)} inputs. Note that the "
            f"predictor rejects a directory containing anything else, so keep "
            f"alignments somewhere other than beside the inputs.")
    return files


def stage_shard(files: List[Path], work: Path) -> Path:
    """Symlink one shard's inputs into their own directory.

    The predictor takes a directory and folds everything in it in one process,
    loading the model once. Sharding therefore means handing each worker its
    own directory rather than its own index range.
    """
    # Refuse before touching anything. A staging directory only ever holds
    # symlinks this function made, so anything else in it means the path is not
    # ours -- `--work-dir` pointing at the inputs, or at a scratch directory
    # with real work in it. Either has to be caught here, before the pruning
    # loop below deletes what it finds.
    wanted = {f.name: f.resolve() for f in files}
    if work.exists():
        if not work.is_dir():
            raise SystemExit(f"--work-dir {work} is not a directory")
        intruders = sorted(p.name for p in work.iterdir() if not p.is_symlink())
        if intruders:
            raise SystemExit(
                f"{work} holds {len(intruders)} entr{'y' if len(intruders) == 1 else 'ies'} "
                f"that this script did not put there "
                f"({', '.join(intruders[:4])}{', ...' if len(intruders) > 4 else ''}).\n"
                f"Staging directories hold nothing but symlinks, so refusing to "
                f"prune this one. Point --work-dir at a new or empty directory; "
                f"in particular it must not be the input directory, whose files "
                f"would be the first thing deleted.")
    if work.resolve() in {f.resolve().parent for f in files} if work.exists() else False:
        raise SystemExit(f"--work-dir {work} is the input directory itself")

    work.mkdir(parents=True, exist_ok=True)
    # Prune links this run does not want. Removing only the names about to be
    # created left a previous run's inputs in place, and the predictor folds
    # everything in the directory -- so the batch silently grew, and with it
    # every structure in it changed (see Reproducibility notes).
    for stale in work.iterdir():
        if wanted.get(stale.name) == stale.resolve():
            continue
        stale.unlink()
    for name, target in wanted.items():
        link = work / name
        if link.is_symlink():
            continue
        link.symlink_to(target)
    return work


def boltz_cmd(data: Path, out: Path, args) -> List[str]:
    exe = args.boltz or shutil.which("boltz")
    if exe:
        head = [exe]
    else:
        # Fall back to this interpreter. The console script is not always on
        # PATH -- an unactivated virtualenv, or a tree on PYTHONPATH -- but if
        # `boltz` imports here it can be driven directly.
        try:
            import boltz  # noqa: F401
        except ImportError:
            raise SystemExit(
                "`boltz` is neither on PATH nor importable. Install this "
                "package (`pip install -e .`) or pass --boltz "
                "/path/to/boltz.") from None
        head = [sys.executable, "-c", "from boltz.main import cli; cli()"]
    cmd = head + ["predict", str(data),
           "--out_dir", str(out),
           "--output_format", args.output_format,
           "--model", args.model,
           "--seed", str(args.seed),
           "--diffusion_samples", str(args.diffusion_samples),
           "--sampling_steps", str(args.sampling_steps),
           "--step_scale", str(args.step_scale),
           "--noise_scale", str(args.noise_scale),
           "--gamma_0", str(args.gamma_0),
           "--accelerator", args.accelerator,
           "--devices", "1"]
    if args.project_pbd:
        cmd.append("--project_pbd")
    if args.use_potentials:
        cmd.append("--use_potentials")
    if args.use_msa_server:
        cmd.append("--use_msa_server")
    if args.cache:
        cmd += ["--cache", str(args.cache)]
    if args.override:
        cmd.append("--override")
    if args.extra:
        cmd += args.extra
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="an input file, or a directory of them")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gpus", default=None,
                    help="comma-separated device ids; one worker each, each "
                         "loading the model once (default: one worker on "
                         "whatever CUDA_VISIBLE_DEVICES already says)")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="fold only shard I of N (0-based). For splitting a "
                         "sweep across machines; --gpus shards within one.")
    ap.add_argument("--work-dir", default=None, type=Path,
                    help="where to stage per-worker input directories "
                         "(default: <out>/.shards)")

    m = ap.add_argument_group("prediction settings (this fork's defaults)")
    m.add_argument("--model", default="boltz2", choices=("boltz1", "boltz2"))
    m.add_argument("--sampling-steps", type=int, default=DEFAULTS["sampling_steps"])
    m.add_argument("--diffusion-samples", type=int,
                   default=DEFAULTS["diffusion_samples"])
    m.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    m.add_argument("--step-scale", type=float, default=DEFAULTS["step_scale"])
    m.add_argument("--noise-scale", type=float, default=DEFAULTS["noise_scale"])
    m.add_argument("--gamma-0", type=float, default=DEFAULTS["gamma_0"],
                   dest="gamma_0")
    m.add_argument("--project-pbd", dest="project_pbd", action="store_true",
                   default=True,
                   help="Gauss-Seidel coordinate projection (default: on)")
    m.add_argument("--no-project-pbd", dest="project_pbd", action="store_false",
                   help="plain Boltz-2, without this fork's projection")
    m.add_argument("--use-potentials", action="store_true", default=False)
    m.add_argument("--use-msa-server", action="store_true", default=False,
                   help="search an MSA instead of requiring one per chain")
    m.add_argument("--output-format", default="pdb", choices=("pdb", "mmcif"))
    m.add_argument("--accelerator", default="gpu", choices=("gpu", "cpu"))
    m.add_argument("--cache", default=os.environ.get("BOLTZ_CACHE"),
                   help="weight cache (default: $BOLTZ_CACHE, else ~/.boltz). "
                        "A cold cache downloads about 3 GB.")
    m.add_argument("--no-override", dest="override", action="store_false",
                   default=True,
                   help="keep existing results instead of recomputing them")
    m.add_argument("--boltz", default=None, help="path to the boltz launcher")
    m.add_argument("extra", nargs="*",
                   help="further flags passed straight to `boltz predict`. "
                        "Put them after a bare `--`, otherwise argparse claims "
                        "them: `predict.py --input in/ --out out/ -- "
                        "--recycling_steps 5`")
    args = ap.parse_args(argv)

    files = find_inputs(args.input)
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard wants I/N, got {args.shard!r}") from None
        if not 0 <= i < n:
            raise SystemExit(f"--shard {args.shard}: need 0 <= I < N")
        files = files[i::n]
        print(f"shard {i}/{n}: {len(files)} of the inputs")
    if not files:
        print("nothing to do")
        return 0

    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else [None]
    work_root = args.work_dir or (args.out / ".shards")
    args.out.mkdir(parents=True, exist_ok=True)

    # Folding results depend on what else was in the batch, so say what the
    # batches were. Reproducing a set of structures needs the same split.
    print(f"{len(files)} input(s), {len(gpus)} worker(s), "
          f"project_pbd={args.project_pbd}, {args.sampling_steps} steps, "
          f"{args.diffusion_samples} samples, seed {args.seed}")

    procs = []
    t0 = time.time()
    for w, gpu in enumerate(gpus):
        mine = files[w::len(gpus)]
        if not mine:
            continue
        tag = f"w{w}" if not args.shard else f"s{args.shard.replace('/', '-')}w{w}"
        data = (stage_shard(mine, work_root / tag)
                if len(files) > 1 or args.input.is_dir() else mine[0])
        out = args.out if len(gpus) == 1 else args.out / tag
        cmd = boltz_cmd(data, out, args)
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        print(f"  [w{w}] {len(mine)} input(s)"
              + (f" on GPU {gpu}" if gpu is not None else "")
              + f" -> {out}")
        procs.append((w, gpu, len(mine),
                      subprocess.Popen(cmd, env=env)))

    failed = []
    for w, gpu, n, proc in procs:
        rc = proc.wait()
        if rc == 0:
            print(f"  [w{w}] done ({n} inputs)")
        else:
            print(f"  [w{w}] FAILED with exit {rc}")
            failed.append(w)

    print(f"{time.time() - t0:.0f}s total")
    if failed:
        print(f"workers that failed: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
