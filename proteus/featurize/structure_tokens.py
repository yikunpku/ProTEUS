# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Structure modality: one mutated PDB -> one `(512,) int64` token array.

Every mutated structure is passed through the ESM3 VQ-VAE structure encoder,
cropped to the same `MAX_LEN` window the other modalities use
(`ProteusConfig.crop_window`), right-padded with the ESM3 structure pad token
and written to `pre_tokens/structure/<variant_stem>.npy`.

The window matters more than it looks: the crop is what keeps the structure
stream positionally aligned with the sequence, mask and MSA streams for the
same variant, so it is read from the config rather than recomputed here.

Fan out over GPUs with `--task-id i --total-tasks T`; shards partition the PDB
list, so they never write the same file and can run concurrently.

The encoder weights are gated. See the ImportError message below.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from proteus.config import (MAX_LEN, SEQUENCE_COL, VARIANT_COL, ProteusConfig,
                            variant_stem)

MISSING_ESM = (
    "the ESM3 structure encoder is required to tokenize structures.\n"
    "  1. pip install 'proteus[structure]'\n"
    "  2. accept the EvolutionaryScale licence for esm3-sm-open-v1 at\n"
    "     https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1\n"
    "  3. export HF_TOKEN=<your Hugging Face token>\n"
    "Pre-tokenized structure inputs need none of this."
)


def _load_encoder(device: str):
    """The gated bits, imported late so the rest of the package stays usable.

    `esm.login()` is deliberately not called: it writes an interactive token to
    the user's home directory. `HF_TOKEN` is enough for the download.
    """
    import torch                    # a hard dependency; not part of the extra

    try:
        from esm.pretrained import ESM3_structure_encoder_v0
        from esm.utils.constants import esm3 as esm3_constants
        from esm.utils.structure.protein_chain import ProteinChain
    except ImportError as exc:
        raise ImportError(f"{MISSING_ESM}\n(import failed: {exc})") from exc

    print(f"loading the ESM3 structure encoder on {device}")
    encoder = ESM3_structure_encoder_v0(device)
    encoder.eval()
    return torch, encoder, esm3_constants, ProteinChain


def library_stems(cfg: ProteusConfig, key: str, depth: int) -> Dict[str, Tuple[str, str]]:
    """`variant_stem -> (variant, sequence)` for the round's library.

    Doubles as a collision check: two variants hashing to one stem would make
    the modality streams ambiguous, so it is an error rather than a warning.
    """
    library = cfg.library_for(key, depth)
    df = pd.read_csv(library, usecols=[VARIANT_COL, SEQUENCE_COL])
    out: Dict[str, Tuple[str, str]] = {}
    for variant, sequence in zip(df[VARIANT_COL], df[SEQUENCE_COL]):
        stem = variant_stem(key, variant)
        previous = out.get(stem)
        if previous is not None and previous[0] != variant:
            raise ValueError(
                f"{library}: variants {previous[0]!r} and {variant!r} share the "
                f"stem {stem!r}; the md5 prefix collided and the per-variant "
                f"files would overwrite each other")
        out[stem] = (str(variant), str(sequence))
    return out


def pdb_stem(path: Path) -> str:
    """`<stem>_mutated.pdb` and `<stem>.pdb` both key on `<stem>`."""
    stem = path.stem
    return stem[: -len("_mutated")] if stem.endswith("_mutated") else stem


def encode_one(pdb_path: Path, torch, encoder, ProteinChain) -> np.ndarray:
    """Raw structure tokens for a PDB, one per residue, no special tokens."""
    chain = ProteinChain.from_pdb(str(pdb_path))
    coords, _plddt, residue_index = chain.to_structure_encoder_inputs()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        _, tokens = encoder.encode(coords.to(device),
                                   residue_index=residue_index.to(device))
    return tokens.squeeze(0).cpu().numpy()


def _strip_special(tokens: np.ndarray, constants) -> np.ndarray:
    """Drop the BOS/EOS pair if this ESM release brackets the encoding.

    `encode()` returned bare per-residue tokens when the published checkpoint
    was built. A release that starts bracketing them would shift every token by
    one position against the crop window -- silent corruption -- so the two
    cases are told apart explicitly instead of assumed.
    """
    if (tokens.size >= 2
            and tokens[0] == constants.STRUCTURE_BOS_TOKEN
            and tokens[-1] == constants.STRUCTURE_EOS_TOKEN):
        return tokens[1:-1]
    return tokens


def tokenize_structures(cfg: ProteusConfig,
                        key: str,
                        depth: int,
                        device: str = "cuda:0",
                        task_id: int = 0,
                        total_tasks: int = 1,
                        allow_failures: int = 0,
                        limit: Optional[int] = None) -> Dict[str, int]:
    """Tokenize this shard's mutated PDBs. Returns per-outcome counts.

    Already-written variants are skipped, so a killed run resumes where it
    stopped. Anything that cannot be tokenized is counted and reported, and a
    non-zero failure count above `allow_failures` raises: a half-populated
    structure stream would otherwise be packed into tars and quietly break the
    join with the other modalities.
    """
    if not 0 <= task_id < total_tasks:
        raise ValueError(f"task_id {task_id} out of range for total_tasks {total_tasks}")

    cfg.spec(key)                      # fail now if the protein is not configured
    pdb_dir = cfg.mutant_pdbs_dir(key, depth)
    out_dir = cfg.pre_token_dir("structure", depth)
    if not pdb_dir.is_dir():
        raise FileNotFoundError(
            f"no mutated structures for {key!r} at depth {depth}: {pdb_dir} does "
            f"not exist (build them before tokenizing)")

    all_pdbs = sorted(pdb_dir.glob("*.pdb"))
    if not all_pdbs:
        raise FileNotFoundError(f"no .pdb files in {pdb_dir}")

    stems = library_stems(cfg, key, depth)
    shard = all_pdbs[task_id::total_tasks]
    if limit is not None:
        shard = shard[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{key} depth {depth} shard {task_id}/{total_tasks}: "
          f"{len(shard)} of {len(all_pdbs)} structures -> {out_dir}")

    torch, encoder, constants, ProteinChain = _load_encoder(device)
    pad = constants.STRUCTURE_PAD_TOKEN

    counts = {"written": 0, "skipped": 0, "failed": 0}
    failures: List[Tuple[str, str]] = []
    for i, pdb_path in enumerate(shard, 1):
        stem = pdb_stem(pdb_path)
        save_path = out_dir / f"{stem}.npy"
        if save_path.exists():
            counts["skipped"] += 1
            continue
        try:
            if stem not in stems:
                raise KeyError(
                    f"{pdb_path.name} has no row in {cfg.library_for(key, depth)}")
            variant, sequence = stems[stem]

            tokens = _strip_special(encode_one(pdb_path, torch, encoder, ProteinChain),
                                    constants)
            if len(tokens) != len(sequence):
                # The crop window is in sequence coordinates. Applying it to a
                # structure with a different residue count would crop the wrong
                # region, which no downstream check would catch.
                raise ValueError(
                    f"{len(tokens)} residues in the structure but "
                    f"{len(sequence)} in the library sequence")

            start, end = cfg.crop_window(key, variant=variant,
                                         seq_len=len(sequence), max_len=MAX_LEN)
            cropped = tokens[start:end]
            if cropped.size == 0:
                raise ValueError(f"crop window [{start}, {end}) is empty")
            if len(cropped) < MAX_LEN:
                cropped = np.pad(cropped, (0, MAX_LEN - len(cropped)),
                                 constant_values=pad)
            cropped = np.ascontiguousarray(cropped, dtype=np.int64)
            assert cropped.shape == (MAX_LEN,), cropped.shape
            assert cropped.dtype == np.int64, cropped.dtype

            # Written aside and renamed: a killed process must not leave a
            # truncated .npy that the resume check above accepts as done.
            tmp_path = save_path.with_name(save_path.name + ".partial")
            with open(tmp_path, "wb") as fh:   # a handle, so np.save cannot
                np.save(fh, cropped)           # re-append .npy to the name
            os.replace(tmp_path, save_path)
            counts["written"] += 1
        except Exception as exc:                        # noqa: BLE001 -- reported
            counts["failed"] += 1
            failures.append((pdb_path.name, f"{type(exc).__name__}: {exc}"))
        if i % 500 == 0:
            print(f"  {i}/{len(shard)} {counts}", flush=True)

    print(f"{key} depth {depth} shard {task_id}/{total_tasks}: "
          f"{counts['written']} written, {counts['skipped']} already present, "
          f"{counts['failed']} failed")
    for name, reason in failures[:20]:
        print(f"  FAILED {name}: {reason}")
    if len(failures) > 20:
        print(f"  ... and {len(failures) - 20} more failures")
    if counts["failed"] > allow_failures:
        raise RuntimeError(
            f"{counts['failed']} structures failed to tokenize "
            f"(allowed {allow_failures}); the structure stream is incomplete. "
            f"Fix the inputs or raise --allow-failures deliberately.")
    return counts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="protein registry YAML")
    p.add_argument("--protein", required=True, help="protein key in the config")
    p.add_argument("--depth", type=int, required=True,
                   help="round depth (substitutions per variant)")
    p.add_argument("--device", default="cuda:0", help="torch device, e.g. cpu")
    p.add_argument("--task-id", type=int, default=0,
                   help="this shard's index, for multi-GPU fan-out")
    p.add_argument("--total-tasks", type=int, default=1, help="number of shards")
    p.add_argument("--allow-failures", type=int, default=0,
                   help="tolerate up to N untokenizable structures")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N structures of this shard (smoke tests)")
    args = p.parse_args(argv)

    cfg = ProteusConfig.from_yaml(args.config)
    try:
        tokenize_structures(cfg, args.protein, args.depth, device=args.device,
                            task_id=args.task_id, total_tasks=args.total_tasks,
                            allow_failures=args.allow_failures, limit=args.limit)
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
