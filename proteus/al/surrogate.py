# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Embedding store and the supervised surrogate used for batch selection.

Both are deliberately small. The surrogate is a random forest over mean-pooled
Proteus embeddings -- the selection logic in `proteus.al.select` is where the
method lives, not here.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

DEFAULT_TREES = 100
DEFAULT_SEED = 42


class EmbeddingStore:
    """Variant name -> feature vector, loaded from one or more `.npz` files.

    Reads the format written by `proteus.cli.featurize` (and by the pipeline's
    step 10): arrays `embeddings` of shape (N, D) and `variants` of shape (N,).
    """

    def __init__(self, variants: Sequence[str], features: np.ndarray, source: str = ""):
        if len(variants) != features.shape[0]:
            raise ValueError(
                f"{source or 'embeddings'}: {len(variants)} variant names but "
                f"{features.shape[0]} feature rows")
        self.variants: List[str] = [str(v) for v in variants]
        self.features: np.ndarray = np.asarray(features, dtype=np.float32)
        self.source = source
        self._index: Dict[str, int] = {v: i for i, v in enumerate(self.variants)}
        if len(self._index) != len(self.variants):
            dupes = len(self.variants) - len(self._index)
            raise ValueError(f"{source or 'embeddings'}: {dupes} duplicate variant names")

    @property
    def dim(self) -> int:
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return len(self.variants)

    def __contains__(self, variant: str) -> bool:
        return variant in self._index

    @classmethod
    def from_npz(cls, path) -> "EmbeddingStore":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"embeddings not found: {path}")
        data = np.load(path, allow_pickle=False)
        for key in ("embeddings", "variants"):
            if key not in data:
                raise KeyError(
                    f"{path}: missing array {key!r}; found {sorted(data.files)}. "
                    f"Expected the output of `proteus featurize`.")
        return cls(data["variants"], data["embeddings"], source=str(path))

    @classmethod
    def concat(cls, stores: Iterable["EmbeddingStore"]) -> "EmbeddingStore":
        """Merge stores from several depths. Dimensions must agree exactly."""
        stores = [s for s in stores if len(s)]
        if not stores:
            raise ValueError("nothing to concatenate")
        dims = {s.dim for s in stores}
        if len(dims) != 1:
            detail = ", ".join(f"{s.source}={s.dim}" for s in stores)
            raise ValueError(f"feature dimension mismatch across inputs: {detail}")

        variants: List[str] = []
        first: Dict[str, Tuple[int, int]] = {}      # variant -> (store, row)
        keep_rows: List[np.ndarray] = []
        clashes: List[str] = []
        for si, store in enumerate(stores):
            mask = np.zeros(len(store.variants), dtype=bool)
            for ri, v in enumerate(store.variants):
                if v not in first:
                    first[v] = (si, ri)
                    mask[ri] = True
                    continue
                # Shared variants are kept from the FIRST store, so if the two
                # copies differ the result depends on the order the files were
                # given -- and the wild type is in every depth's file, and it is
                # a labelled training row. Different depths featurized at
                # different --device/--dtype/--reduction give different numbers
                # for it, and sorting the training rows does not help: the rows
                # themselves differ. Say so rather than letting argument order
                # pick the batch.
                sj, rj = first[v]
                if not np.allclose(stores[sj].features[rj], store.features[ri],
                                   rtol=0, atol=1e-6):
                    delta = float(np.abs(stores[sj].features[rj].astype(np.float64)
                                         - store.features[ri].astype(np.float64)).max())
                    clashes.append(f"{v} ({stores[sj].source} vs {store.source}, "
                                   f"max|diff| {delta:.3g})")
            variants.extend([v for v, m in zip(store.variants, mask) if m])
            keep_rows.append(store.features[mask])
        if clashes:
            # A warning, not an error. These differences are expected to be
            # small but not zero even in a correct run: the wild type is in
            # every depth's library, and its msa_emb_128 stream is sampled with
            # a seed while its placeholder structure is rebuilt per round, so
            # the trunk sees slightly different inputs. Measured on the bundled
            # demo: max|diff| 0.05, cosine 0.99997. Refusing would reject every
            # legitimate multi-depth run; saying nothing would leave the batch
            # quietly dependent on the order the files were listed in.
            print(f"[embeddings] {len(clashes)} variant(s) appear in more than "
                  f"one file with different values; the FIRST file listed wins:")
            for line in clashes[:3]:
                print(f"    {line}")
            if len(clashes) > 3:
                print(f"    ... and {len(clashes) - 3} more")
            print("    A large difference here means the depths were "
                  "featurized with different --device/--dtype/--reduction, "
                  "and the batch depends on argument order.")
        return cls(variants, np.concatenate(keep_rows, axis=0),
                   source=" + ".join(s.source for s in stores))

    def take(self, wanted: Sequence[str]) -> Tuple[List[str], np.ndarray, List[str]]:
        """Return (found names, their features, missing names) preserving order."""
        found, rows, missing = [], [], []
        for v in wanted:
            i = self._index.get(v)
            if i is None:
                missing.append(v)
            else:
                found.append(v)
                rows.append(i)
        feats = self.features[rows] if rows else np.empty((0, self.dim), dtype=np.float32)
        return found, feats, missing


class RandomForestSurrogate:
    """Random forest regressor over variant embeddings.

    Thin on purpose: 100 trees, fixed seed, no feature engineering.
    """

    def __init__(self, n_estimators: int = DEFAULT_TREES,
                 random_state: int = DEFAULT_SEED,
                 max_depth: Optional[int] = None,
                 n_jobs: Optional[int] = None):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs or -1,
        )

    def fit(self, features: np.ndarray, fitness: np.ndarray) -> "RandomForestSurrogate":
        features = np.asarray(features, dtype=np.float32)
        fitness = np.asarray(fitness, dtype=np.float64)
        if features.shape[0] != fitness.shape[0]:
            raise ValueError(
                f"{features.shape[0]} feature rows vs {fitness.shape[0]} labels")
        if features.shape[0] < 5:
            raise ValueError(
                f"only {features.shape[0]} labelled variants; need at least 5 to "
                f"fit a surrogate")
        self.model.fit(features, fitness)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(features, dtype=np.float32)).astype(np.float32)
