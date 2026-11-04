# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Mutation-mask modality: which positions of the crop window were mutated.

One `(512,) uint8` array per variant, 1 at every mutated position, written to
`cfg.pre_token_dir("mutation_mask", depth)`. Positions are relative to the same
crop window the sequence tokens use, so mask index `i` lines up with sequence
token `i + 1` (the token array carries a leading BOS). The wild type's mask is
all zeros.

    python -m proteus.featurize.mask --config cfg.yaml --protein KEY --depth 2
"""

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from proteus.al.variants import parse_substitution, split_variant
from proteus.config import (MAX_LEN, SEQUENCE_COL, VARIANT_COL, ProteusConfig,
                            variant_stem)
from proteus.featurize._common import (Result, load_library, save_array,
                                       shard)


def mutated_positions(variant: str) -> List[int]:
    """1-indexed positions a variant names; empty for the wild type.

    `parse_substitution` raises on anything that is not `<wt><pos><mut>` over
    the standard 20 amino acids. The old step silently dropped parts it could
    not read, which turns a corrupt library into an all-zero mask instead of an
    error, so unparseable names are now a failure the caller reports.
    """
    return [parse_substitution(sub)[1] for sub in split_variant(variant)]


def mutation_mask(variant: str, crop: Tuple[int, int],
                  mask_len: int = MAX_LEN) -> Tuple[np.ndarray, int]:
    """`(512,) uint8` mask plus the number of positions outside the window.

    Under the dataset-wide crop policy a protein longer than `mask_len` shares
    one window across its whole library, so a variant can name positions the
    window does not cover. Those cannot be represented in a `mask_len` mask;
    they are counted and returned rather than quietly dropped, because a large
    count means the crop policy is wrong for that protein.
    """
    start, end = crop
    mask = np.zeros(mask_len, dtype=np.uint8)
    # The window, not the array. `index < mask_len` let a position past the end
    # of the protein through: on a 119-residue protein cropped to (0, 119),
    # `T300S` lands at index 299, which is inside the 512-slot array but 180
    # residues past the sequence -- a 1 in the padding region, where the
    # sequence stream has <pad> and the structure stream has its pad token.
    width = min(mask_len, end - start)
    outside = 0
    for pos in mutated_positions(variant):
        index = pos - 1 - start
        if 0 <= index < width:
            mask[index] = 1
        else:
            outside += 1
    return mask, outside


def write_mutation_masks(cfg: ProteusConfig, key: str, depth: int,
                         task_id: int = 0, total_tasks: int = 1,
                         allow_failures: int = 0,
                         progress: bool = True) -> Result:
    """Build the `(key, depth)` masks, skipping variants already on disk."""
    df = shard(load_library(cfg, key, depth), task_id, total_tasks)
    out_dir = cfg.pre_token_dir("mutation_mask", depth)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = Result(key=key, total=len(df))
    outside_positions = 0
    empty_masks = 0
    rows = tqdm(df.itertuples(index=False), total=len(df),
                desc=f"mutation_mask {key} d{depth} [{task_id}/{total_tasks}]",
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
            mask, outside = mutation_mask(variant, crop)
            if outside:
                outside_positions += outside
                if not mask.any():
                    # Every mutated position is outside the window, so this
                    # variant's mask is the wild type's -- and so is its
                    # cropped sequence, since `sequence[start:end]` drops the
                    # same residues. Four streams byte-identical to the wild
                    # type, carrying a different label. A line of output is not
                    # enough for that, so it is a failure.
                    #
                    # Note `--allow-failures` does not get you past it in any
                    # useful way: a tolerated variant has no mutation_mask
                    # .npy, and `tars` then refuses the whole round because
                    # that variant is featurized for some streams and not
                    # others. The way through is to fix the crop policy or drop
                    # the variant from the library.
                    empty_masks += 1
                    raise ValueError(
                        f"all {outside} mutated position(s) fall outside the "
                        f"crop window {crop}, so the mask is all zeros and this "
                        f"variant is indistinguishable from the wild type in "
                        f"every stream. The crop policy is wrong for this "
                        f"protein, or this variant does not belong in this "
                        f"library.")
            save_array(out_path, mask, np.uint8, (MAX_LEN,))
        except Exception as exc:  # noqa: BLE001 -- recorded, then raised in finish()
            result.fail(variant, exc)
            continue
        result.written += 1

    if outside_positions:
        print(f"[mutation_mask] {key}: {outside_positions} mutated position(s) "
              f"fall outside the crop window and are not representable; "
              f"{empty_masks} variant(s) would have had an all-zero mask and "
              f"were recorded as failures")
    return result.finish("mutation_mask", out_dir, allow_failures)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="protein registry YAML")
    ap.add_argument("--protein", nargs="+", default=None,
                    help="protein keys; default every protein in the config")
    ap.add_argument("--depth", type=int, required=True,
                    help="round depth; 1 is the saturation library")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--total-tasks", type=int, default=1)
    ap.add_argument("--allow-failures", type=int, default=0,
                    help="tolerate this many unprocessable variants per protein")
    ap.add_argument("--no-progress", action="store_true")
    args = ap.parse_args(argv)

    cfg = ProteusConfig.from_yaml(args.config)
    for key in args.protein or sorted(cfg.proteins):
        write_mutation_masks(cfg, key, args.depth,
                             task_id=args.task_id,
                             total_tasks=args.total_tasks,
                             allow_failures=args.allow_failures,
                             progress=not args.no_progress)


if __name__ == "__main__":
    main()
