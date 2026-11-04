# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Pieces every per-variant featurizer needs.

Loading and sharding a library, writing an array only after checking it, and
accounting for the variants that could not be processed. Each modality writes
`<protein>_<n>mut_<md5[:16]>.npy` into `cfg.pre_token_dir(<modality>, depth)`;
the stems come from `proteus.config.variant_stem` because the modalities must
agree on them or the shards built from them will not join.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from proteus.config import SEQUENCE_COL, VARIANT_COL, ProteusConfig


def load_library(cfg: ProteusConfig, key: str, depth: int) -> pd.DataFrame:
    """The library for `(key, depth)`, checked for the columns featurizers read."""
    path = cfg.library_for(key, depth)
    if not path.exists():
        raise FileNotFoundError(
            f"{key} depth {depth}: library CSV not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in (VARIANT_COL, SEQUENCE_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: library is missing column(s) {missing}")
    return df


def shard(df: pd.DataFrame, task_id: int, total_tasks: int) -> pd.DataFrame:
    """Stride slice of `df` for one worker out of `total_tasks`."""
    if total_tasks < 1 or not 0 <= task_id < total_tasks:
        raise ValueError(
            f"invalid --task-id {task_id} for --total-tasks {total_tasks}")
    return df if total_tasks == 1 else df.iloc[task_id::total_tasks]


def save_array(path: Path, array: np.ndarray, dtype, shape: Tuple[int, ...]) -> None:
    """Write `array` only if it is exactly the shape and dtype downstream expects.

    The write goes to a temporary file and is renamed into place. Featurizers
    skip variants whose output already exists, so a half-written file from an
    interrupted run would otherwise be mistaken for a finished one forever.
    """
    if array.dtype != dtype or array.shape != shape:
        raise AssertionError(
            f"{path.name}: expected {shape} {np.dtype(dtype)}, "
            f"got {array.shape} {array.dtype}")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    tmp.replace(path)


@dataclass
class Result:
    """What one (protein, depth, modality) pass did, including what it could not do."""

    key: str
    total: int = 0
    written: int = 0
    skipped: int = 0
    failures: List[Tuple[str, str]] = field(default_factory=list)

    def fail(self, variant: str, reason: BaseException) -> None:
        self.failures.append((str(variant), f"{type(reason).__name__}: {reason}"))

    def finish(self, label: str, out_dir: Path, allow_failures: int = 0) -> "Result":
        """Print the tally and raise unless the failures were allowed for."""
        print(f"[{label}] {self.key}: {self.written} written, "
              f"{self.skipped} already present, {self.total} in library "
              f"-> {out_dir}")
        if self.failures:
            print(f"[{label}] {self.key}: {len(self.failures)} variant(s) failed")
            for variant, reason in self.failures[:20]:
                print(f"    {variant}: {reason}")
            if len(self.failures) > 20:
                print(f"    ... and {len(self.failures) - 20} more")
            if len(self.failures) > allow_failures:
                raise RuntimeError(
                    f"{label}: {self.key}: {len(self.failures)} variant(s) could "
                    f"not be featurized, allowance is {allow_failures}; pass "
                    f"--allow-failures to proceed anyway")
        return self
