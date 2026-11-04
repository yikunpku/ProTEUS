# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Sequence modality: ESM2 token ids for every variant in a library.

One `(514,) int64` array per variant -- BOS, then the `MAX_LEN`-residue crop
window of its mutated sequence, then EOS -- written to
`cfg.pre_token_dir("sequence", depth)`.

    python -m proteus.featurize.sequence --config cfg.yaml --protein KEY --depth 2
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from proteus.config import (MAX_LEN, SEQUENCE_COL, VARIANT_COL, ProteusConfig,
                            variant_stem)
from proteus.featurize._common import (Result, load_library, save_array,
                                       shard)

# The checkpoint the published tokens were produced with. It only names an
# alphabet: see `batch_converter`.
DEFAULT_ESM2_MODEL = "esm2_t33_650M_UR50D"

# Literal text of the ESM pad token. Crops shorter than MAX_LEN are padded with
# it as *text*, because the alphabet's tokenizer splits special tokens out of a
# sequence string before mapping residues, so each occurrence becomes one pad
# id rather than five residue ids.
PAD_TEXT = "<pad>"

TOKEN_LEN = MAX_LEN + 2  # BOS + MAX_LEN + EOS


def batch_converter(model_name: str = DEFAULT_ESM2_MODEL):
    """An ESM2 batch converter, loaded through `fair_esm.pretrained`.

    Not through `torch.hub.load('facebookresearch/esm', ...)`: the hub entry
    point's `hubconf.py` does `from esm.pretrained import ...`, which resolves
    to the unrelated EvolutionaryScale `esm` (v3) package whenever that is also
    installed, and the load fails. `fair_esm.pretrained` is the same code
    without the name collision.

    Which ESM2 size this is does not change the output. Every ESM2 checkpoint
    shares one alphabet -- `fair_esm` builds them all with
    `Alphabet.from_architecture("ESM-1b")` -- so tokenizing with the 650M model
    for a 3B trunk is not a mismatch, it is the same vocabulary. The default
    just records which checkpoint the released tokens came from.
    """
    import fair_esm.pretrained

    loader = getattr(fair_esm.pretrained, model_name, None)
    if loader is None:
        raise ValueError(f"fair_esm.pretrained has no model {model_name!r}")
    _, alphabet = loader()
    return alphabet.get_batch_converter()


def tokenize(sequence: str, crop: Tuple[int, int], converter) -> np.ndarray:
    """`(514,) int64` tokens for one mutated sequence and its crop window."""
    start, end = crop
    cropped = sequence[start:end]
    if len(cropped) > MAX_LEN:
        raise ValueError(
            f"crop window {crop} yielded {len(cropped)} residues, "
            f"more than MAX_LEN={MAX_LEN}")
    cropped += PAD_TEXT * (MAX_LEN - len(cropped))
    _, _, tokens = converter([("variant", cropped)])
    return tokens[0].cpu().numpy()


def write_sequence_tokens(cfg: ProteusConfig, key: str, depth: int,
                          converter,
                          task_id: int = 0, total_tasks: int = 1,
                          allow_failures: int = 0,
                          progress: bool = True) -> Result:
    """Tokenize the `(key, depth)` library, skipping variants already on disk."""
    df = shard(load_library(cfg, key, depth), task_id, total_tasks)
    out_dir = cfg.pre_token_dir("sequence", depth)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = Result(key=key, total=len(df))
    rows = tqdm(df.itertuples(index=False), total=len(df),
                desc=f"sequence {key} d{depth} [{task_id}/{total_tasks}]",
                disable=not progress, leave=False)
    for row in rows:
        variant = getattr(row, VARIANT_COL)
        out_path = out_dir / f"{variant_stem(key, variant)}.npy"
        if out_path.exists():
            result.skipped += 1
            continue
        try:
            sequence = getattr(row, SEQUENCE_COL)
            crop = cfg.crop_window(key, variant, len(sequence))
            save_array(out_path, tokenize(sequence, crop, converter),
                       np.int64, (TOKEN_LEN,))
        except Exception as exc:  # noqa: BLE001 -- recorded, then raised in finish()
            result.fail(variant, exc)
            continue
        result.written += 1

    return result.finish("sequence", out_dir, allow_failures)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="protein registry YAML")
    ap.add_argument("--protein", nargs="+", default=None,
                    help="protein keys; default every protein in the config")
    ap.add_argument("--depth", type=int, required=True,
                    help="round depth; 1 is the saturation library")
    ap.add_argument("--esm-model", default=DEFAULT_ESM2_MODEL,
                    help="fair_esm model whose alphabet is used "
                         f"(default {DEFAULT_ESM2_MODEL}; all ESM2 sizes share it)")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--total-tasks", type=int, default=1)
    ap.add_argument("--allow-failures", type=int, default=0,
                    help="tolerate this many unprocessable variants per protein")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    cfg = ProteusConfig.from_yaml(args.config)
    keys = args.protein or sorted(cfg.proteins)
    for key in keys:
        cfg.spec(key)  # fail before loading a 650M checkpoint for a typo

    print(f"loading the ESM2 alphabet from fair_esm ({args.esm_model})")
    converter = batch_converter(args.esm_model)

    for key in keys:
        write_sequence_tokens(cfg, key, args.depth, converter,
                              task_id=args.task_id,
                              total_tasks=args.total_tasks,
                              allow_failures=args.allow_failures,
                              progress=not args.no_progress)


if __name__ == "__main__":
    main()
