#!/usr/bin/env python

import argparse
from pathlib import Path

import torch

FROZEN_PREFIXES = (
    "encoder_embeddings.sequence.pretrained_model.",
    "encoder_embeddings.structure.pretrained_model.",
)

STRUCTURE_EMBED_KEY = (
    "encoder_embeddings.structure.pretrained_model.encoder.structure_tokens_embed.weight"
)

# Everything needed to rebuild the architecture, and nothing else. Paths, output
# dirs, run names, distributed settings and optimizer schedule are all dropped.
ARGS_WHITELIST = (
    "model",
    "in_domains",
    "out_domains",
    "all_domains",
    "num_input_tokens",
    "num_target_tokens",
    "num_register_tokens",
    "seq_basemodel",
    "seq_emb_dim",
    "dtype",
    "loss_type",
)


def sanitize_args(args_obj):
    """Keep the architecture whitelist; refuse anything that smells like a path."""
    src = vars(args_obj) if hasattr(args_obj, "__dict__") else dict(args_obj)
    out = {}
    for key in ARGS_WHITELIST:
        if key not in src:
            continue
        value = src[key]
        if isinstance(value, str) and ("/" in value or "\\" in value):
            raise ValueError(
                f"whitelisted arg {key!r} looks like a path ({value!r}); "
                "refusing to ship it"
            )
        out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="input training checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="output trunk-only checkpoint (.pth)")
    parser.add_argument("--drop-frozen-encoders", action="store_true",
                        help="ALSO drop encoder_embeddings.*.pretrained_model.*. "
                             "Off by default: dropping them changes the model.")
    args = parser.parse_args()

    src = Path(args.checkpoint)
    dst = Path(args.output)
    if not src.is_file():
        raise SystemExit(f"no such checkpoint: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"reading {src}  ({src.stat().st_size / 2**30:.2f} GiB)")
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    state = ckpt["model"]

    kept, dropped = {}, {"keys": 0, "params": 0}
    for key, tensor in state.items():
        frozen = key.startswith(FROZEN_PREFIXES)
        if frozen and args.drop_frozen_encoders:
            dropped["keys"] += 1
            dropped["params"] += tensor.numel()
            continue
        kept[key] = tensor.clone()

    if not args.drop_frozen_encoders and STRUCTURE_EMBED_KEY not in kept:
        raise SystemExit(f"expected to keep {STRUCTURE_EMBED_KEY} but it is absent")

    out = {"model": kept, "args": sanitize_args(ckpt["args"])}
    if "epoch" in ckpt:
        out["epoch"] = ckpt["epoch"]

    torch.save(out, dst)

    kept_params = sum(t.numel() for t in kept.values())
    print(f"\nkept    {len(kept):5d} keys  {kept_params / 1e6:10.3f} M params")
    print(f"dropped {dropped['keys']:5d} keys  {dropped['params'] / 1e6:10.3f} M params")
    print(f"args kept: {sorted(out['args'])}")
    print(f"\nwrote {dst}  ({dst.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
