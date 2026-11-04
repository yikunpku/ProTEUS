# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Pack per-variant `.npy` features into WebDataset shards, one set per modality.

The loader reads the modalities as parallel streams and zips them positionally:
member *k* of `sequence/chunk_00003.tar` is assumed to describe the same variant
as member *k* of `structure/chunk_00003.tar`. Nothing downstream re-checks that,
so a single missing or reordered member silently mislabels every variant after
it. This module therefore derives one member list for the whole round from
`variant_stem`, uses it for every modality, and refuses to write anything it
cannot write identically everywhere.

Fan out with `--task-id i --total-tasks T`; shards split the chunk indices, so
chunk numbering is global and independent of how many workers ran.
"""

import argparse
import os
import sys
import tarfile
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from proteus.config import (INPUT_MODALITIES, MODALITIES, VARIANT_COL,
                            ProteusConfig, variant_stem)

DEFAULT_CHUNK_SIZE = 10000


def member_names(cfg: ProteusConfig, key: str, depth: int) -> List[str]:
    """The round's member names, sorted -- the join key for every modality.

    Sorted rather than CSV order so the shards are reproducible from the
    library alone, however the CSV rows happen to be arranged.
    """
    library = cfg.library_for(key, depth)
    df = pd.read_csv(library, usecols=[VARIANT_COL])
    stems: Dict[str, str] = {}
    for variant in df[VARIANT_COL]:
        stem = variant_stem(key, variant)
        previous = stems.get(stem)
        if previous is not None and previous != variant:
            raise ValueError(
                f"{library}: variants {previous!r} and {variant!r} share the "
                f"stem {stem!r}; the md5 prefix collided")
        stems[stem] = variant
    return sorted(f"{stem}.npy" for stem in stems)


def _resolve_members(cfg: ProteusConfig, key: str, depth: int,
                     modalities: Sequence[str],
                     allow_failures: int) -> Tuple[List[str], Dict[str, Path]]:
    """Member list restricted to variants every modality can supply.

    Two kinds of gap, deliberately treated differently:

    * missing from *some* modalities -- always fatal. One modality being behind
      the others is exactly the asymmetry that corrupts the join, and no
      threshold makes it safe.
    * missing from *all* modalities -- the variant simply has not been
      featurized. Counted as a failure, so it raises unless `allow_failures`
      covers it, and then dropped from every modality at once.
    """
    dirs = {mod: cfg.pre_token_dir(mod, depth) for mod in modalities}
    for mod, directory in dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(
                f"no {mod!r} features at depth {depth}: {directory} does not exist")

    names = member_names(cfg, key, depth)
    present = {mod: set(os.listdir(directory)) for mod, directory in dirs.items()}

    # Completeness is judged over EVERY input stream, not just the ones being
    # packed now. Judging only `--modality sequence` would make `all(...)` and
    # `any(...)` range over one element, so the "featurized for some modalities
    # but not others" check could not fire -- and packing one stream on its own
    # then produces a shard whose k-th member is a different variant from the
    # k-th member of its neighbours, which nothing downstream re-checks.
    judged = tuple(dict.fromkeys(tuple(INPUT_MODALITIES) + tuple(modalities)))
    for mod in judged:
        if mod in present:
            continue
        directory = cfg.pre_token_dir(mod, depth)
        present[mod] = set(os.listdir(directory)) if directory.is_dir() else set()

    judged_counts = {mod: len([n for n in names if n in present[mod]])
                     for mod in judged}
    print(f"{key} depth {depth}: {len(names)} variants in "
          f"{cfg.library_for(key, depth).name}; packing {list(modalities)}; "
          f"present per stream (completeness is judged over all of these): "
          f"{judged_counts}")

    complete = [n for n in names if all(n in present[mod] for mod in judged)]
    complete_set = set(complete)
    partial = [n for n in names
               if n not in complete_set and any(n in present[mod] for mod in judged)]
    if partial:
        detail = []
        for name in partial[:10]:
            missing_in = [mod for mod in judged if name not in present[mod]]
            detail.append(f"    {name} missing from {missing_in}")
        raise RuntimeError(
            f"{len(partial)} variant(s) are featurized for some modalities but "
            f"not others, so the streams cannot be joined:\n" + "\n".join(detail)
            + (f"\n    ... and {len(partial) - 10} more" if len(partial) > 10 else "")
            + "\nFinish the missing modalities before packing.")

    unfeaturized = len(names) - len(complete)
    if unfeaturized:
        print(f"  {unfeaturized} variant(s) have no features in any modality")
        if unfeaturized > allow_failures:
            raise RuntimeError(
                f"{unfeaturized} of {len(names)} variants are not featurized "
                f"(allowed {allow_failures}). Run the missing featurization "
                f"steps, or pass --allow-failures {unfeaturized} to pack the "
                f"{len(complete)} complete variants only.")
    if not complete:
        raise RuntimeError(f"nothing to pack for {key!r} at depth {depth}")
    return complete, dirs


def _validate(paths: Sequence[Path], modality: str,
              expected: Optional[Tuple[Tuple[int, ...], np.dtype]] = None
              ) -> Tuple[Tuple[int, ...], np.dtype]:
    """Every array of a modality must agree on dtype and shape.

    Header-only reads via `mmap_mode`, so this costs a stat per file rather
    than the whole feature.
    """
    for path in paths:
        array = np.load(path, mmap_mode="r")
        if expected is None:
            expected = (tuple(array.shape), array.dtype)
            continue
        if (tuple(array.shape), array.dtype) != expected:
            raise ValueError(
                f"{modality}: {path.name} is {array.dtype}{tuple(array.shape)}, "
                f"expected {expected[1]}{expected[0]}")
    return expected


def pack_tars(cfg: ProteusConfig,
              key: str,
              depth: int,
              modalities: Sequence[str] = INPUT_MODALITIES,
              chunk_size: int = DEFAULT_CHUNK_SIZE,
              task_id: int = 0,
              total_tasks: int = 1,
              allow_failures: int = 0,
              validate: bool = True) -> Dict[str, int]:
    """Write `<tar_dir>/chunk_NNNNN.tar` for each modality. Returns counts.

    Existing shards are re-opened and their member lists checked instead of
    being trusted, so a shard left truncated by a killed run is rebuilt rather
    than silently reused.
    """
    if not 0 <= task_id < total_tasks:
        raise ValueError(f"task_id {task_id} out of range for total_tasks {total_tasks}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    unknown = [mod for mod in modalities if mod not in MODALITIES]
    if unknown:
        raise ValueError(f"unknown modalities {unknown}; expected from {MODALITIES}")
    if not modalities:
        raise ValueError("no modalities requested")

    cfg.spec(key)
    members, dirs = _resolve_members(cfg, key, depth, modalities, allow_failures)

    n_chunks = ceil(len(members) / chunk_size)
    mine = list(range(n_chunks))[task_id::total_tasks]
    print(f"  packing {len(members)} members into {n_chunks} chunk(s) of "
          f"{chunk_size}; shard {task_id}/{total_tasks} owns {len(mine)}")

    out_dirs = {mod: cfg.tar_dir(key, mod, depth) for mod in modalities}
    for directory in out_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    # A shorter member list means fewer chunks, so shards above the new count
    # are left behind unless pruned. `featurize.embed` survives that --
    # it globs `*.tar` but then keys members by stem, so a duplicate just
    # overwrites itself. The consumer that does not survive it is a positional
    # WebDataset loader, which is what these shards exist for: member k of one
    # modality is assumed to be member k of every other. Only the owning shard
    # prunes, so this is safe under --task-id.
    if task_id == 0:
        # Over every stream, not just the ones being packed. Pruning only the
        # requested modalities left the others with orphans AND -- because a
        # shorter member list moves every chunk boundary -- with chunk k holding
        # a different set of variants from chunk k of the stream just repacked.
        # That is the misalignment this prune exists to prevent, displaced onto
        # the streams nobody asked for.
        prune_dirs = {mod: cfg.tar_dir(key, mod, depth)
                      for mod in set(modalities) | set(INPUT_MODALITIES)}
        for mod, directory in prune_dirs.items():
            if not directory.is_dir():
                continue
            for stale in sorted(directory.glob("chunk_*.tar")):
                try:
                    index = int(stale.stem.split("_")[-1])
                except ValueError:
                    continue
                if index >= n_chunks:
                    stale.unlink()
                    print(f"  removed orphan {mod}/{stale.name} "
                          f"(only {n_chunks} chunk(s) now)")

    # A single-stream repack is only safe while the member list is unchanged.
    # Pruning orphans above fixes the shard COUNT; it cannot fix the contents
    # of the shards that survive, because a shorter member list moves every
    # chunk boundary. So check the streams this run is not packing against the
    # chunks it is about to write, and refuse rather than leave chunk k of one
    # stream describing different variants from chunk k of another.
    not_packed = [m for m in dict.fromkeys(tuple(INPUT_MODALITIES) + tuple(modalities))
                  if m not in set(modalities)]
    if not_packed:
        stale = []
        for mod in not_packed:
            directory = cfg.tar_dir(key, mod, depth)
            if not directory.is_dir():
                continue
            for index in range(n_chunks):
                chunk = members[index * chunk_size:(index + 1) * chunk_size]
                tar_path = directory / f"chunk_{index:05d}.tar"
                if tar_path.exists() and _shard_names(tar_path) != list(chunk):
                    stale.append(f"{mod}/{tar_path.name}")
        if stale:
            raise RuntimeError(
                f"{len(stale)} shard(s) of the streams this run is not packing "
                f"no longer match the member list "
                f"({', '.join(stale[:4])}{', ...' if len(stale) > 4 else ''}).\n"
                f"The member list changed, which moves every chunk boundary, so "
                f"repacking one stream would leave chunk k of it describing "
                f"different variants from chunk k of the others -- and nothing "
                f"downstream re-checks that. Repack every stream: drop "
                f"--modality, or delete {cfg.tar_dir(key, modalities[0], depth).parent} "
                f"and start over.")

    seen: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
    counts = {"written": 0, "reused": 0, "members": 0}
    for index in mine:
        chunk = members[index * chunk_size:(index + 1) * chunk_size]

        # The assertion the whole format rests on: identical names, identical
        # order, in every modality of this chunk. Checked against the files on
        # disk, before a byte is written.
        plans = {mod: [dirs[mod] / name for name in chunk] for mod in modalities}
        for mod, paths in plans.items():
            got = [p.name for p in paths]
            if got != chunk:
                raise AssertionError(
                    f"chunk {index} member list for {mod!r} does not match the "
                    f"join key (first difference at "
                    f"{next(i for i, (a, b) in enumerate(zip(got, chunk)) if a != b)})")
            absent = [p.name for p in paths if not p.exists()]
            if absent:
                raise RuntimeError(
                    f"chunk {index}: {len(absent)} member(s) vanished from "
                    f"{dirs[mod]} since the scan, e.g. {absent[:3]}")

        for mod in modalities:
            if validate:
                seen[mod] = _validate(plans[mod], mod, seen.get(mod))
            tar_path = out_dirs[mod] / f"chunk_{index:05d}.tar"
            if _shard_is_good(tar_path, chunk):
                counts["reused"] += 1
                continue
            tmp_path = tar_path.with_suffix(".tar.partial")
            with tarfile.open(tmp_path, "w") as tar:
                for path in plans[mod]:
                    tar.add(path, arcname=path.name)
            written = _shard_names(tmp_path)
            if written != chunk:
                raise AssertionError(
                    f"{tmp_path} came back with {len(written)} members, "
                    f"expected {len(chunk)} in join order")
            os.replace(tmp_path, tar_path)
            counts["written"] += 1
        counts["members"] += len(chunk)

    for mod in modalities:
        if mod in seen:
            shape, dtype = seen[mod]
            print(f"  {mod}: {dtype}{shape}")
    print(f"{key} depth {depth} shard {task_id}/{total_tasks}: "
          f"{counts['written']} shard(s) written, {counts['reused']} already "
          f"complete, {counts['members']} member(s) per modality")
    return counts


def _shard_names(tar_path: Path) -> Optional[List[str]]:
    """Member names of a shard, or None if it cannot be read."""
    try:
        with tarfile.open(tar_path) as tar:
            return tar.getnames()
    except (tarfile.TarError, OSError):
        return None


def _shard_is_good(tar_path: Path, expected: Sequence[str]) -> bool:
    if not tar_path.exists():
        return False
    names = _shard_names(tar_path)
    if names == list(expected):
        return True
    if names is None:
        reason = "unreadable"
    elif len(names) != len(expected):
        reason = f"holds {len(names)} members, expected {len(expected)}"
    else:
        first = next(i for i, (a, b) in enumerate(zip(names, expected)) if a != b)
        reason = (f"holds {len(names)} members but they are not the expected "
                  f"ones in the expected order (position {first}: "
                  f"{names[first]} vs {expected[first]})")
    print(f"  rebuilding {tar_path}: {reason}")
    return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="protein registry YAML")
    p.add_argument("--protein", required=True, help="protein key in the config")
    p.add_argument("--depth", type=int, required=True,
                   help="round depth (substitutions per variant)")
    p.add_argument("--modality", action="append", dest="modalities",
                   choices=list(MODALITIES),
                   help=f"pack only this modality; repeatable. Default: the "
                        f"four encoder inputs, {', '.join(INPUT_MODALITIES)}. "
                        f"scores_normalized is the training target and has no "
                        f"featurizer here, so ask for it only if you brought "
                        f"your own.")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help="members per shard")
    p.add_argument("--task-id", type=int, default=0,
                   help="this shard's index, for parallel fan-out")
    p.add_argument("--total-tasks", type=int, default=1, help="number of workers")
    p.add_argument("--allow-failures", type=int, default=0,
                   help="tolerate up to N library variants with no features at all")
    p.add_argument("--no-validate", action="store_true",
                   help="skip the per-array dtype/shape check")
    args = p.parse_args(argv)

    cfg = ProteusConfig.from_yaml(args.config)
    modalities = args.modalities or list(INPUT_MODALITIES)
    try:
        pack_tars(cfg, args.protein, args.depth, modalities=modalities,
                  chunk_size=args.chunk_size, task_id=args.task_id,
                  total_tasks=args.total_tasks,
                  allow_failures=args.allow_failures,
                  validate=not args.no_validate)
    except (RuntimeError, ValueError, AssertionError, FileNotFoundError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
