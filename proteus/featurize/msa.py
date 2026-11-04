# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""The `msa_emb_128` modality: ESM-MSA-1b layer-12 embeddings, one per variant.

For each protein we sample `--rows` sequences (128 by default) from its
wild-type MSA once, with a fixed seed, crop every row to the protein's crop
window, tokenize the stack and run frozen ESM-MSA-1b. The query row's layer-12
representation is saved as `(L, 768)` float32 under
`rounds/round<D>/workspace/pre_tokens/msa_emb_128/<stem>.npy`.

Two row conventions exist upstream and they are NOT interchangeable:

`wt-msa` (default, what the released checkpoint was trained on)
    The sampled wild-type MSA is used unchanged for every variant of a
    protein. Row 0 is the MSA's focus (wild-type) sequence. The per-variant
    signal in this feature therefore comes only from the crop window, so under
    the `dataset` crop policy -- or whenever the protein is shorter than
    `max_len` -- every variant of a protein gets a byte-identical array. That
    is expected, not a bug: the trunk reads the per-variant signal from the
    sequence and mutation-mask streams, and this modality supplies the
    evolutionary context they sit in.

`mutant-row0` (opt-in, `--sampler mutant-row0`)
    Row 0 is replaced by the variant's own mutated sequence, which makes the
    feature vary per variant. It is off-distribution for the released weights:
    the checkpoint never saw a mutated residue in row 0, so scores computed
    from these features are not comparable to published numbers. Only use it
    when retraining from scratch.

Whichever is used is recorded in a JSON sidecar beside the output directory,
together with seed, row count, crop windows, model and layer, so a feature
directory is self-describing.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from proteus.config import (MAX_LEN, SEQUENCE_COL, VARIANT_COL, ProteinSpec,
                            ProteusConfig, variant_stem)
from proteus.prepare import a3m_tools
from proteus.zeroshot.msa_sampling import process_msa, sample_msa

MODALITY = "msa_emb_128"
MODEL_NAME = "esm_msa1b_t12_100M_UR50S"
REPR_LAYER = 12
EMB_DIM = 768
DEFAULT_ROWS = 128
DEFAULT_SEED = 42

SAMPLER_WT_MSA = "wt-msa"
SAMPLER_MUTANT_ROW0 = "mutant-row0"
SAMPLERS = (SAMPLER_WT_MSA, SAMPLER_MUTANT_ROW0)

DTYPES = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def load_esm_msa(device: str = "cpu", dtype: str = "float32",
                 checkpoint: Optional[str] = None):
    """Load frozen ESM-MSA-1b and its batch converter.

    Deliberately NOT `torch.hub.load('facebookresearch/esm', ...)`, which is
    what the upstream script did: hubconf.py does `from esm.pretrained import
    ...`, and in any environment that also has EvolutionaryScale's `esm` (v3)
    installed that name resolves to the unrelated v3 package and the load
    fails or, worse, returns a different model. Importing `fair_esm` by its
    distribution name leaves no room for that confusion.

    `checkpoint` loads a specific `.pt` instead of whatever the torch hub
    cache holds. Worth using: the hub cache is keyed by file name only, so a
    stale or hand-placed `esm_msa1b_t12_100M_UR50S.pt` is picked up silently.
    We measured one such cache whose embeddings were orthogonal (mean cosine
    0.017) to those of the official release -- a different model entirely,
    with no error anywhere. The file's size is recorded in the sidecar so the
    identity of the weights travels with the features.
    """
    import fair_esm

    if checkpoint:
        # Trusted local file chosen by the operator; fairseq checkpoints carry
        # an argparse.Namespace, which torch>=2.6 refuses under weights_only.
        import argparse as _argparse

        torch.serialization.add_safe_globals([_argparse.Namespace])
        original = torch.load
        torch.load = lambda *a, **k: original(*a, **{**k, "weights_only": False})
        try:
            model, alphabet = fair_esm.pretrained.load_model_and_alphabet_local(
                checkpoint)
        finally:
            torch.load = original
    else:
        model, alphabet = fair_esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    model = model.to(DTYPES[dtype]).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    batch_converter = alphabet.get_batch_converter(truncation_seq_length=MAX_LEN)
    return model, alphabet, batch_converter


# --------------------------------------------------------------------------
# MSA rows
# --------------------------------------------------------------------------

def _weights_identity(checkpoint: Optional[str]) -> Optional[dict]:
    """The file the MSA weights were loaded from, with its size and target.

    `fair_esm` resolves an unqualified model name through the torch hub cache,
    which is a flat directory keyed on filename: anything called
    `esm_msa1b_t12_100M_UR50S.pt` that lands there is what gets loaded,
    including a symlink into another project's data. Recording the resolved
    path and size is the cheapest way to make two runs comparable.
    """
    if checkpoint:
        path = Path(checkpoint)
    else:
        try:
            path = Path(torch.hub.get_dir()) / "checkpoints" / f"{MODEL_NAME}.pt"
        except Exception:  # noqa: BLE001
            return None
    if not path.exists():
        return {"path": str(path), "present": False}
    return {"path": str(path), "present": True,
            "realpath": str(path.resolve()),
            "bytes": path.stat().st_size}


def alignment_inputs(spec: ProteinSpec) -> Tuple[str, Path]:
    """The alignment file actually read, and the weight file paired with it.

    `process_msa` rewrites a true a3m into the rectangular shape its reader
    assumes (see `proteus.prepare.a3m_tools`), and the weights are a function
    of the columns, so the rewritten copy gets its own cache. Resolving here
    too, rather than letting `process_msa` do it silently, keeps the existence
    check below and the sidecar pointing at the files that were used.
    """
    aln, weights = a3m_tools.resolve(spec.msa_a3m, spec.msa_weights)
    return aln, Path(weights if weights else Path(aln).with_suffix(".npy"))


def ensure_weights(spec: ProteinSpec, num_cpus: int = 8) -> Path:
    """Return the sequence-weight file, computing and persisting it if absent.

    Weighted sampling is part of the feature definition, so an unweighted
    fallback is not acceptable: it would make a first run (weights missing)
    differently distributed from a second (weights on disk). We compute once,
    save to `spec.msa_weights`, and fail if that did not happen.
    """
    aln, weights = alignment_inputs(spec)
    if weights.exists():
        return weights

    print(f"  sequence weights missing, computing once -> {weights}")
    weights.parent.mkdir(parents=True, exist_ok=True)
    process_msa(aln, str(weights), filter_msa=False,
                path_to_hhfilter=None)
    if not weights.exists():
        raise RuntimeError(
            f"{spec.key}: sequence weights were not written to {weights}; "
            f"refusing to continue with unweighted sampling")
    return weights


def sample_rows(spec: ProteinSpec, rows: int, seed: int,
                num_cpus: int = 8) -> List[Tuple[str, str]]:
    """Sample `rows` weighted sequences from the wild-type MSA.

    Row 0 is the MSA focus sequence; the rest are drawn with
    `random.choices` weighted by the EVE sequence weights under `seed`, so the
    stack is identical for every variant of the protein.
    """
    weights = ensure_weights(spec, num_cpus=num_cpus)
    aln, _ = alignment_inputs(spec)
    msa = process_msa(aln, str(weights), filter_msa=False,
                      path_to_hhfilter=None)
    sampled = sample_msa(aln, rows, seed,
                         weight_filename=str(weights), processed_msa=msa)
    if not sampled:
        raise RuntimeError(f"{spec.key}: MSA sampling returned no usable rows")
    if len(sampled) != rows:
        print(f"  warning: MSA yielded {len(sampled)} of {rows} requested rows")
    return [(name, str(seq).upper()) for name, seq in sampled]


def crop_rows(rows: Sequence[Tuple[str, str]],
              window: Tuple[int, int]) -> List[Tuple[str, str]]:
    start, end = window
    return [(name, seq[start:end]) for name, seq in rows]


def build_stack(rows: Sequence[Tuple[str, str]], mutated_sequence: str,
                window: Tuple[int, int], sampler: str) -> List[Tuple[str, str]]:
    """The MSA as it is handed to the model, already cropped."""
    cropped = crop_rows(rows, window)
    if sampler == SAMPLER_WT_MSA:
        return cropped
    if sampler == SAMPLER_MUTANT_ROW0:
        start, end = window
        return [("query", mutated_sequence[start:end])] + cropped[1:]
    raise ValueError(f"unknown sampler {sampler!r}; expected one of {SAMPLERS}")


# --------------------------------------------------------------------------
# tokenization + forward
# --------------------------------------------------------------------------

def tokenize(stack: Sequence[Tuple[str, str]], batch_converter, alphabet,
             pad_to: Optional[int]) -> torch.Tensor:
    """`(1, depth, tokens)` int64 on the CPU, optionally padded in length.

    Padding to a fixed token count keeps every array in the modality the same
    shape, which the tar shards and the trunk's fixed-size streams rely on.
    """
    _, _, tokens = batch_converter([list(stack)])
    if pad_to is None:
        return tokens
    cur = tokens.shape[2]
    if cur < pad_to:
        pad = torch.full((tokens.shape[0], tokens.shape[1], pad_to - cur),
                         fill_value=alphabet.padding_idx, dtype=tokens.dtype)
        tokens = torch.cat([tokens, pad], dim=2)
    elif cur > pad_to:
        tokens = tokens[:, :, :pad_to]
    return tokens


def embed(model, tokens: torch.Tensor, device: str) -> np.ndarray:
    """Layer-`REPR_LAYER` representation of the query row, BOS dropped."""
    with torch.no_grad():
        out = model(tokens.to(device), repr_layers=[REPR_LAYER],
                    return_contacts=False)
    emb = out["representations"][REPR_LAYER][:, 0, 1:, :]
    return emb.float().cpu().numpy()


def _check_and_save(path: Path, array: np.ndarray, expect_len: int) -> None:
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.dtype != np.float32:
        raise AssertionError(f"{path.name}: dtype {array.dtype}, expected float32")
    if array.shape != (expect_len, EMB_DIM):
        raise AssertionError(
            f"{path.name}: shape {array.shape}, expected {(expect_len, EMB_DIM)}")
    if not np.isfinite(array).all():
        raise AssertionError(f"{path.name}: non-finite values in embedding")
    # Write through a temporary name so an interrupted run never leaves a
    # half-written array that the idempotency check would then skip. The file
    # object matters: np.save(path) would append a second ".npy".
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    tmp.replace(path)


# --------------------------------------------------------------------------
# sidecar
# --------------------------------------------------------------------------

GLOBAL_KEYS = ("modality", "model", "layer", "embedding_dim", "sampler",
               "rows", "seed", "pad_to_tokens", "max_len", "dtype",
               "checkpoint", "checkpoint_bytes")


def write_sidecar(out_dir: Path, meta: Dict) -> Path:
    """Record how the directory was produced, merging per-protein entries.

    Raises if an existing sidecar disagrees on anything global: mixing two
    sampling conventions or two seeds in one feature directory silently
    poisons every downstream comparison.
    """
    path = out_dir.parent / f"{out_dir.name}.json"
    if path.exists():
        with open(path) as fh:
            old = json.load(fh)
        clashes = {k: (old.get(k), meta[k]) for k in GLOBAL_KEYS
                   if k in old and old[k] != meta[k]}
        if clashes:
            raise RuntimeError(
                f"{path} was written with different settings {clashes}; "
                f"write to a fresh directory instead of mixing conventions")
        proteins = dict(old.get("proteins") or {})
        proteins.update(meta.get("proteins") or {})
        meta = {**old, **meta, "proteins": proteins}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return path


# --------------------------------------------------------------------------
# library function
# --------------------------------------------------------------------------

def featurize_protein(cfg: ProteusConfig, key: str, depth: int,
                      sampler: str = SAMPLER_WT_MSA,
                      rows: int = DEFAULT_ROWS,
                      seed: int = DEFAULT_SEED,
                      max_len: int = MAX_LEN,
                      pad: bool = True,
                      device: str = "cpu",
                      dtype: str = "float32",
                      batch_size: int = 1,
                      task_id: int = 0,
                      total_tasks: int = 1,
                      limit: Optional[int] = None,
                      out_dir: Optional[Path] = None,
                      model_bundle=None,
                      checkpoint: Optional[str] = None,
                      num_cpus: int = 8) -> Dict[str, int]:
    """Write one `.npy` per variant of `key`; returns written/skipped/failed."""
    if sampler not in SAMPLERS:
        raise ValueError(f"unknown sampler {sampler!r}; expected one of {SAMPLERS}")
    if not (0 <= task_id < total_tasks):
        raise ValueError(f"invalid task_id {task_id} for total_tasks {total_tasks}")

    spec = cfg.spec(key)
    library = cfg.library_for(key, depth)
    for path in (library, spec.msa_a3m):
        if not Path(path).exists():
            raise FileNotFoundError(f"{key}: missing input {path}")

    out_dir = Path(out_dir) if out_dir is not None else cfg.pre_token_dir(MODALITY, depth)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(library)
    for col in (VARIANT_COL, SEQUENCE_COL):
        if col not in df.columns:
            raise ValueError(f"{library}: missing column {col!r}")
    if total_tasks > 1:
        df = df.iloc[task_id::total_tasks].reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit].reset_index(drop=True)

    print(f"{key}[{task_id}/{total_tasks}]: {len(df)} variants, "
          f"sampler={sampler}, rows={rows}, seed={seed}")

    if model_bundle is None:
        model_bundle = load_esm_msa(device=device, dtype=dtype,
                                    checkpoint=checkpoint)
    model, alphabet, batch_converter = model_bundle

    msa_rows = sample_rows(spec, rows, seed, num_cpus=num_cpus)
    seq_len = len(df[SEQUENCE_COL].iloc[0])
    msa_len = len(msa_rows[0][1])
    if msa_len != seq_len:
        raise ValueError(
            f"{key}: MSA rows are {msa_len} columns wide but the library's "
            f"sequences are {seq_len} residues; crop windows are wild-type "
            f"coordinates, so the two must agree")

    pad_to = max_len + 1 if pad else None
    windows: Dict[str, Tuple[int, int]] = {}
    written = skipped = failed = 0
    pending: List[Tuple[torch.Tensor, Path, int, Tuple[int, int]]] = []

    # Under the default sampler, `build_stack` ignores the variant's sequence
    # entirely -- the model input is the wild-type MSA cropped to the window --
    # so every variant sharing a window gets the same array. The docstring at
    # the top of this module says so; computing it once per window instead of
    # once per variant makes a 173-variant library one forward pass rather than
    # 173 identical ones, which on CPU is the difference between a minute and
    # three hours. The bytes written are the same either way, and the condition
    # is checked rather than assumed: `mutant-row0` does depend on the variant,
    # and `crop: per-variant` gives different variants different windows.
    reuse = sampler == SAMPLER_WT_MSA
    by_window: Dict[Tuple[int, int], Tuple[np.ndarray, int]] = {}

    def flush() -> int:
        if not pending:
            return 0
        tokens = torch.cat([t for t, _, _, _ in pending], dim=0)
        arrays = embed(model, tokens, device)
        for i, (_, path, expect_len, window) in enumerate(pending):
            _check_and_save(path, arrays[i], expect_len)
            if reuse:
                by_window.setdefault(window, (arrays[i], expect_len))
        n = len(pending)
        pending.clear()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return n

    for _, row in df.iterrows():
        variant = str(row[VARIANT_COL])
        mutated_sequence = str(row[SEQUENCE_COL])
        path = out_dir / f"{variant_stem(key, variant)}.npy"
        try:
            # Recorded even for variants we skip, so a resumed run still
            # describes the windows its directory was built with.
            window = cfg.crop_window(key, variant=variant,
                                     seq_len=len(mutated_sequence),
                                     max_len=max_len)
            windows[variant] = window
            if path.exists():
                skipped += 1
                continue
            done = by_window.get(window) if reuse else None
            if done is not None:
                _check_and_save(path, done[0], done[1])
                written += 1
                continue
            stack = build_stack(msa_rows, mutated_sequence, window, sampler)
            tokens = tokenize(stack, batch_converter, alphabet, pad_to)
            expect_len = tokens.shape[2] - 1
            # Everything in one forward pass must share a token length.
            if pending and pending[-1][0].shape[2] != tokens.shape[2]:
                written += flush()
            pending.append((tokens, path, expect_len, window))
            if len(pending) >= batch_size:
                written += flush()
        except Exception as exc:  # noqa: BLE001 -- counted and re-raised below
            failed += 1
            print(f"  [fail] {variant}: {type(exc).__name__}: {exc}")
    written += flush()

    distinct = sorted(set(windows.values()))
    sidecar = write_sidecar(out_dir, {
        "modality": MODALITY,
        "model": MODEL_NAME,
        "layer": REPR_LAYER,
        "embedding_dim": EMB_DIM,
        "sampler": sampler,
        "rows": rows,
        "seed": seed,
        "pad_to_tokens": pad_to,
        "max_len": max_len,
        "dtype": dtype,
        "checkpoint": str(checkpoint) if checkpoint else "torch hub cache",
        # Which file the weights actually came from, resolved rather than
        # described. The hub cache is keyed on filename alone and is commonly a
        # symlink into another project, so "torch hub cache" on its own does not
        # identify a model: two runs of this step produced embeddings differing
        # by 15.06 in absolute value because $TORCH_HOME pointed at a different
        # 1.39 GB file one time and a 1.38 GB one the other.
        "weights_resolved": _weights_identity(checkpoint),
        "checkpoint_bytes": (Path(checkpoint).stat().st_size
                             if checkpoint and Path(checkpoint).exists() else None),
        # On CPU this step is bit-reproducible; a CUDA run of the same input
        # differs by about 6e-06 from it, which is accumulation order, not a
        # different model. Recorded so the two are never silently compared.
        "device": str(device),
        "off_distribution_for_released_weights": sampler != SAMPLER_WT_MSA,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "proteins": {key: {
            "library": str(library),
            "msa_a3m": str(spec.msa_a3m),
            # What was read, which differs from msa_a3m when a true a3m had to
            # be put into the reader's rectangular shape.
            "msa_alignment_read": alignment_inputs(spec)[0],
            "legacy_a3m": a3m_tools.legacy(),
            "msa_weights": str(alignment_inputs(spec)[1]),
            "crop_policy": spec.crop,
            "crop_windows": [list(w) for w in distinct],
            "depth": depth,
        }},
    })

    print(f"  [{key}] {written} written, {skipped} skipped, {failed} failed "
          f"-> {out_dir}")
    print(f"  sidecar: {sidecar}")
    return {"written": written, "skipped": skipped, "failed": failed}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Proteus YAML config")
    ap.add_argument("--protein", action="append", dest="proteins",
                    help="protein key; repeat for several, omit for all")
    ap.add_argument("--depth", type=int, required=True,
                    help="round depth; picks the library and the output round")
    ap.add_argument("--sampler", default=SAMPLER_WT_MSA, choices=SAMPLERS,
                    help=f"MSA row convention (default {SAMPLER_WT_MSA}; "
                         f"{SAMPLER_MUTANT_ROW0} is off-distribution for the "
                         f"released weights)")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help="MSA rows per stack")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="sampling seed, recorded in the sidecar")
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--no-pad", action="store_true",
                    help="keep the natural cropped length instead of padding "
                         "every stack to --max-len tokens")
    ap.add_argument("--checkpoint", default=None,
                    help="path to esm_msa1b_t12_100M_UR50S.pt; omit to use "
                         "the torch hub cache, but see load_esm_msa -- a "
                         "stale cache file is loaded without a warning")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--total-tasks", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many variants (testing)")
    ap.add_argument("--out-dir", default=None,
                    help="override the output directory (testing)")
    ap.add_argument("--num-cpus", type=int, default=8,
                    help="cores for the one-off sequence-weight computation")
    ap.add_argument("--allow-failures", type=int, default=0,
                    help="tolerated per-variant failures before raising")
    ap.add_argument("--legacy-a3m", action="store_true",
                    help="read the alignment exactly as it is on disk. The "
                         "MSA reader assumes a rectangular alignment, so a "
                         "true a3m is otherwise rewritten first; without that "
                         "rewrite every residue after a row's first insertion "
                         "sits in the wrong column. Use this only to reproduce "
                         "numbers produced before the rewrite existed.")
    args = ap.parse_args(argv)

    a3m_tools.set_legacy(args.legacy_a3m)
    cfg = ProteusConfig.from_yaml(args.config)
    keys = args.proteins or sorted(cfg.proteins)
    bundle = load_esm_msa(device=args.device, dtype=args.dtype,
                          checkpoint=args.checkpoint)

    totals = {"written": 0, "skipped": 0, "failed": 0}
    for key in keys:
        counts = featurize_protein(
            cfg, key, args.depth,
            sampler=args.sampler, rows=args.rows, seed=args.seed,
            max_len=args.max_len, pad=not args.no_pad,
            device=args.device, dtype=args.dtype,
            batch_size=args.batch_size,
            task_id=args.task_id, total_tasks=args.total_tasks,
            limit=args.limit,
            out_dir=Path(args.out_dir) if args.out_dir else None,
            model_bundle=bundle, checkpoint=args.checkpoint,
            num_cpus=args.num_cpus)
        for k in totals:
            totals[k] += counts[k]

    print(f"total: {totals['written']} written, {totals['skipped']} skipped, "
          f"{totals['failed']} failed")
    if totals["failed"] > args.allow_failures:
        raise SystemExit(
            f"{totals['failed']} variants failed, which is more than the "
            f"{args.allow_failures} allowed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
