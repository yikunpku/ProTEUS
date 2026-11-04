# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Run the trunk encoder over pre-tokenized variants and save embeddings.

This is the step that actually uses the released weights: the four input
modality streams go through their frozen adapters, get fused, and pass through
the 12-layer encoder. The pooled output is what the downstream surrogate learns on.

Input enumeration, which layer to read, how to pool and which structure
modality to name are all flags.

`--legacy-masking` feeds the encoder an all-False padding mask, so `<pad>`
positions are attended to as if they were residues and mean pooling averages
over them. It is there to reproduce embeddings made that way; it changes the
encoder pass as well as the pooling, so it is not a pooling flag.
"""

import argparse
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from proteus.config import (INPUT_MODALITIES as CONFIG_INPUT_MODALITIES,
                            MODALITIES, VARIANT_COL,
                            ProteusConfig, variant_stem)

# ESM2 alphabet pad index. The sequence stream is ESM2 token ids, so this is
# how padding is identified; there is no separate length array in the tars.
PAD_IDX = 1

# The same four streams as proteus.config.INPUT_MODALITIES, kept in the order
# the original pipeline built its encoder dict in. Asserted rather than
# imported so the ordering stays visible, and so adding a modality in one
# place and not the other fails at import instead of silently dropping it.
INPUT_MODALITIES = ("sequence", "msa_emb_128", "structure", "mutation_mask")
assert set(INPUT_MODALITIES) == set(CONFIG_INPUT_MODALITIES), (
    f"modality sets have drifted: {INPUT_MODALITIES} vs {CONFIG_INPUT_MODALITIES}")

DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


def check_device_dtype(device: str, dtype: torch.dtype) -> None:
    """Reject the one combination that always fails, before any slow work.

    The weights get cast to `dtype`, but the autocast in `encode_batch` is a
    CUDA autocast, so on CPU there is nothing to reconcile a float32 input
    (`msa_emb_128` is stored as float32) with a bfloat16 weight: the first
    matmul raises `mat1 and mat2 must have the same dtype`. Say so here rather
    than after loading the tars and an 8 GiB checkpoint.
    """
    if not str(device).startswith("cuda") and dtype != torch.float32:
        raise ValueError(
            f"device={device!r} with dtype={dtype} cannot work: reduced "
            f"precision here relies on CUDA autocast. Pass --dtype float32 "
            f"for CPU runs (slower, but the embeddings are the reference "
            f"values), or point --device at a GPU.")


def build_model(checkpoint: str, device: str = "cuda:0",
                dtype: torch.dtype = torch.bfloat16):
    """Rebuild the architecture and load a trunk-only checkpoint.

    The architecture is read from the checkpoint's own `args` where possible,
    so a checkpoint carries enough to reconstruct the model it belongs to.
    """
    from proteus.data.modality_info import MODALITY_INFO
    from proteus.utils import create_model
    import proteus.models.fm  # noqa: F401  registers the model factories

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    if hasattr(args, "__dict__"):
        args = vars(args)

    model_name = args.get("model", "fm_base_12e_12d_swiglu_nobias")
    seq_basemodel = args.get("seq_basemodel", "esm2_t36_3B_UR50D")
    seq_emb_dim = int(args.get("seq_emb_dim", 2560))
    num_register_tokens = int(args.get("num_register_tokens", 1))
    all_domains = list(args.get("all_domains", MODALITIES))

    modality_info = {d: MODALITY_INFO[d] for d in all_domains if d in MODALITY_INFO}
    encoder_embeddings = {}
    for mod in INPUT_MODALITIES:
        info = modality_info.get(mod)
        if info is None or info.get("encoder_embedding") is None:
            continue
        if info["type"] == "protein_sequence":
            encoder_embeddings[mod] = info["encoder_embedding"](
                orig_emb_dim=seq_emb_dim, base_model=seq_basemodel)
        else:
            encoder_embeddings[mod] = info["encoder_embedding"]()

    check_device_dtype(device, dtype)
    model = create_model(
        model_name,
        encoder_embeddings=encoder_embeddings,
        decoder_embeddings={},
        modality_info=modality_info,
        num_register_tokens=num_register_tokens,
    )
    model = model.to(device).to(dtype).eval()

    state = ckpt.get("model", ckpt)
    msg = model.load_state_dict(state, strict=False)
    if msg.unexpected_keys:
        raise RuntimeError(
            f"{checkpoint}: {len(msg.unexpected_keys)} keys do not belong to "
            f"this architecture, e.g. {msg.unexpected_keys[:3]}")
    # `*.pretrained_model.*` are the only keys allowed to be absent, for a
    # checkpoint whose frozen encoders were dropped (tools/strip_checkpoint.py).
    # Anything else missing means the checkpoint does not match the
    # architecture.
    unexplained = [k for k in msg.missing_keys if "pretrained_model" not in k]
    if unexplained:
        raise RuntimeError(
            f"{checkpoint}: {len(unexplained)} trunk tensors are missing, e.g. "
            f"{unexplained[:5]}")
    print(f"loaded {len(state)} tensors from the checkpoint")
    if msg.missing_keys:
        print(f"  warning: {len(msg.missing_keys)} encoder tensors were NOT in "
              f"the checkpoint and come from upstream instead; this is a "
              f"different model from the released one")
    return model


def load_tar_members(tar_dir: Path, keep: Optional[set] = None) -> Dict[str, np.ndarray]:
    """Load every `.npy` member from every shard in a modality directory."""
    shards = sorted(tar_dir.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"no .tar shards in {tar_dir}")
    out: Dict[str, np.ndarray] = {}
    for shard in shards:
        with tarfile.open(shard) as tf:
            for member in tf.getmembers():
                if not member.name.endswith(".npy"):
                    continue
                stem = Path(member.name).stem
                if keep is not None and stem not in keep:
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    raise IOError(f"{shard}:{member.name} could not be read")
                out[stem] = np.load(BytesIO(fh.read()), allow_pickle=False)
    return out


def prepare_batch(stems: Sequence[str], data: Dict[str, Dict[str, np.ndarray]],
                  device: str, legacy_masking: bool = False) -> Tuple[dict, List[str]]:
    """Assemble one batch. Raises on any variant missing a modality.

    Upstream skipped such variants silently, so a partially featurized library
    produced a quietly smaller embedding set than the caller asked for.
    """
    missing: Dict[str, List[str]] = {}
    for mod in INPUT_MODALITIES:
        gone = [s for s in stems if s not in data[mod]]
        if gone:
            missing[mod] = gone
    if missing:
        detail = "; ".join(f"{m}: {len(v)} (e.g. {v[0]})" for m, v in missing.items())
        raise KeyError(f"variants missing modality data -- {detail}")

    batch = {}
    seq = torch.from_numpy(np.stack([data["sequence"][s] for s in stems])).to(device)
    mut = torch.from_numpy(np.stack([data["mutation_mask"][s] for s in stems])).to(device)

    if legacy_masking:
        seq_input_mask = torch.zeros_like(seq, dtype=torch.bool)
    else:
        seq_input_mask = (seq == PAD_IDX)

    batch["sequence"] = {
        "tensor": seq,
        "mutation_mask": mut,
        "input_mask": seq_input_mask,
        "target_mask": torch.zeros_like(seq, dtype=torch.bool),
        "decoder_attention_mask": torch.zeros_like(seq, dtype=torch.int),
    }
    for mod in ("msa_emb_128", "structure", "mutation_mask"):
        t = torch.from_numpy(np.stack([data[mod][s] for s in stems])).to(device)
        shape = t.shape if t.ndim == 2 else t.shape[:2]
        batch[mod] = {
            "tensor": t,
            "input_mask": torch.zeros(shape, dtype=torch.bool, device=device),
            "target_mask": torch.zeros(shape, dtype=torch.bool, device=device),
            "decoder_attention_mask": torch.zeros(shape, dtype=torch.int, device=device),
        }
    return batch, list(stems)


@torch.no_grad()
def encode_batch(model, batch: dict, num_encoder_tokens: int = 512,
                 reduction: str = "mean", dtype: torch.dtype = torch.bfloat16,
                 legacy_masking: bool = False) -> torch.Tensor:
    """Encoder output, pooled. Mirrors `FourM.forward` up to the trunk output."""
    use_amp = dtype != torch.float32 and str(
        next(model.parameters()).device).startswith("cuda")
    with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
        encoder_mod_dict = {
            mod: model.encoder_embeddings[mod](d)
            for mod, d in batch.items() if mod in model.encoder_embeddings
        }
        tokens, emb, mask, _ = model.forward_mask_encoder(
            encoder_mod_dict, num_encoder_tokens)
        x = tokens + emb
        x = model.forward_encoder(
            x, encoder_mask=mask,
            mut_mask=encoder_mod_dict["sequence"]["mutation_mask"])

    out = x.detach().cpu().float()
    if reduction == "none":
        return out
    if reduction == "bos":
        # The register token; this is also what the regression head reads.
        return out[:, 0, :]
    if reduction != "mean":
        raise ValueError(f"unknown reduction {reduction!r}")

    if legacy_masking:
        return out.mean(dim=1)
    valid = (~mask.detach().cpu().squeeze(1)).float().unsqueeze(-1)
    return (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def extract(cfg: ProteusConfig, key: str, depth: int, checkpoint: str,
            device: str = "cuda:0", dtype: str = "bfloat16",
            reduction: str = "mean", batch_size: int = 8,
            legacy_masking: bool = False,
            task_id: int = 0, total_tasks: int = 1,
            overwrite: bool = False) -> Optional[Path]:
    """Embed every variant of one library, in CSV order."""
    torch_dtype = DTYPES[dtype]
    check_device_dtype(device, torch_dtype)
    library = cfg.library_for(key, depth)
    df = pd.read_csv(library, usecols=[VARIANT_COL])
    variants = list(df[VARIANT_COL])
    stems = [variant_stem(key, v) for v in variants]

    if total_tasks > 1:
        stems = stems[task_id::total_tasks]
        variants = variants[task_id::total_tasks]
        out_path = (cfg.round_dir(depth) / "embeddings" / "shards" / key /
                    f"shard_{task_id:02d}_of_{total_tasks:02d}.npz")
    else:
        out_path = cfg.embeddings_npz(key, depth)
    if out_path.exists() and not overwrite:
        # Existence is not completion. A run killed inside the write, or an
        # out-of-space one, leaves a short or zero-byte .npz, and accepting it
        # here makes the next reader fail with `EOFError: No data left in
        # file`, nowhere near the cause. Open it and say so instead.
        try:
            with np.load(out_path, allow_pickle=False) as probe:
                n = int(probe["embeddings"].shape[0])
                if n != len(probe["variants"]):
                    raise ValueError(
                        f"{n} embeddings for {len(probe['variants'])} variants")
                # Coverage too, not just internal consistency: a valid npz of
                # 2 rows is still the wrong answer for a library of 3.
                if n != len(stems):
                    raise ValueError(
                        f"{n} embeddings for {len(stems)} variants in "
                        f"{cfg.library_for(key, depth).name}")
        except Exception as exc:
            raise SystemExit(
                f"{out_path} exists but is not a complete embedding file "
                f"({type(exc).__name__}: {exc}). Delete it, or pass "
                f"--overwrite to rebuild it.") from None
        print(f"[skip] {out_path} exists ({n} variants)")
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keep = set(stems)
    data = {}
    for mod in INPUT_MODALITIES:
        data[mod] = load_tar_members(cfg.tar_dir(key, mod, depth), keep=keep)
        print(f"  {mod:14} {len(data[mod]):,} members")

    model = build_model(checkpoint, device=device, dtype=torch_dtype)

    feats: List[np.ndarray] = []
    names: List[str] = []
    for i in tqdm(range(0, len(stems), batch_size), desc=f"encoding {key}"):
        chunk = stems[i:i + batch_size]
        chunk_variants = variants[i:i + batch_size]
        batch, used = prepare_batch(chunk, data, device, legacy_masking)
        emb = encode_batch(model, batch, reduction=reduction, dtype=torch_dtype,
                           legacy_masking=legacy_masking)
        feats.append(emb.numpy())
        names.extend(chunk_variants)

    features = np.concatenate(feats, axis=0)
    if features.shape[0] != len(names):
        raise AssertionError(
            f"{features.shape[0]} embeddings for {len(names)} variants")
    # Write and rename, like every other writer in this package. Writing
    # straight to the final name meant an interrupted or out-of-space run left
    # a short or zero-byte .npz that the `out_path.exists()` resume check above
    # accepts as finished, and the next reader fails with `EOFError: No data
    # left in file` nowhere near the cause.
    # Through a file handle, not a path: np.savez_compressed appends `.npz` to
    # a name that does not already end in it, so a `.partial` path would come
    # back as `.partial.npz` and the rename below would miss it.
    tmp = out_path.with_name(out_path.name + ".partial")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, embeddings=features,
                            variants=np.array(names), dataset=key,
                            reduction=reduction,
                            masking="legacy" if legacy_masking else "fixed")
    tmp.replace(out_path)
    print(f"wrote {out_path}  embeddings={features.shape}")
    return out_path


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="protein registry YAML")
    p.add_argument("--protein", required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    p.add_argument("--reduction", default="mean", choices=["mean", "bos", "none"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--legacy-masking", action="store_true",
                   help="treat <pad> as residues and mean-pool over them, as "
                        "before the padding fix. Only for reproducing "
                        "embeddings produced by the old code.")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--total-tasks", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = ProteusConfig.from_yaml(args.config)
    extract(cfg, args.protein, args.depth, args.checkpoint,
            device=args.device, dtype=args.dtype, reduction=args.reduction,
            batch_size=args.batch_size, legacy_masking=args.legacy_masking,
            task_id=args.task_id, total_tasks=args.total_tasks,
            overwrite=args.overwrite)


if __name__ == "__main__":
    main()
