# Copyright (c) 2026 Foundry Biosciences Inc. All Rights Reserved.
#
# Use of this file is governed by the Internal Research End User Agreement
# in LICENSE at the root of this repository. Not for commercial use or
# redistribution.
"""Protein registry and the on-disk layout, parameterized by round depth.

Everything is driven by one YAML file, so adding a protein means adding an
entry, and running one round deeper means passing a bigger `--depth`. There is
no depth ceiling.

Layout under `root`:

    proteins/<key>/structure/<key>.pdb      wild-type structure
    proteins/<key>/msa/<key>.a3m            wild-type MSA
    proteins/<key>/msa/<key>.npy            sequence weights (auto-cached)
    libraries/<key>.csv                     depth-1 (saturation) library
    rounds/round<D>/
        labels/<key>.csv                    all measurements so far
        candidates/<key>.csv                depth-D candidate library
        zeroshot/<key>.csv                  zero-shot scores
        embeddings/<key>.npz                trunk embeddings
        picks/<key>_consensus.csv           the batch to order
        workspace/pmls/<key>/               PyMOL scripts
        workspace/mutant_pdbs/<key>/        mutated structures
        workspace/pre_tokens/<modality>/    per-variant .npy
        workspace/tars/<key>/<modality>/    WebDataset shards
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

MAX_LEN = 512
VARIANT_COL = "mutant"
SEQUENCE_COL = "mutated_sequence"
WT_TAG = "wild_type"
# The four streams the encoder reads. Each has a featurizer in
# proteus.featurize, and the trunk's in_domains are exactly these.
INPUT_MODALITIES = ("sequence", "structure", "msa_emb_128", "mutation_mask")

# `scores_normalized` is the trunk's output domain -- the quantile-transformed
# fitness label. It only exists as a feature when you are training, which this
# repository does not do, so it is named here but has no featurizer and is not
# packed by default.
MODALITIES = INPUT_MODALITIES + ("scores_normalized",)


def default_msa_path(root, key: str) -> Path:
    """Where `proteus-prepare` puts a protein's alignment, per the layout above."""
    return Path(root) / "proteins" / key / "msa" / f"{key}.a3m"


def default_pdb_path(root, key: str) -> Path:
    """Where `proteus-prepare` puts a protein's wild-type structure."""
    return Path(root) / "proteins" / key / "structure" / f"{key}.pdb"


def default_library_path(root, key: str) -> Path:
    """Where the depth-1 library for a protein is expected."""
    return Path(root) / "libraries" / f"{key}.csv"


@dataclass
class ProteinSpec:
    """One protein's static inputs."""

    key: str
    library_csv: Path
    wt_pdb: Path
    msa_a3m: Path
    msa_weights: Optional[Path] = None
    crop: str = "dataset"          # "dataset" or "per-variant"

    def __post_init__(self):
        if self.crop not in ("dataset", "per-variant"):
            raise ValueError(
                f"{self.key}: crop must be 'dataset' or 'per-variant', "
                f"got {self.crop!r}")
        if self.msa_weights is None:
            self.msa_weights = self.msa_a3m.with_suffix(".npy")


@dataclass
class ProteusConfig:
    root: Path
    proteins: Dict[str, ProteinSpec] = field(default_factory=dict)
    _crop_cache: Dict[Tuple[str, int], Tuple[int, int]] = field(
        default_factory=dict, repr=False)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_yaml(cls, path) -> "ProteusConfig":
        path = Path(path).resolve()
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}

        # Relative paths resolve against `root`, which itself defaults to the
        # directory holding the config -- so a config and its data move together.
        root = Path(raw.get("root", path.parent))
        if not root.is_absolute():
            root = (path.parent / root).resolve()

        def resolve(value) -> Path:
            p = Path(value)
            return p if p.is_absolute() else root / p

        entries = raw.get("proteins") or {}
        if not entries:
            raise ValueError(f"{path}: no proteins defined")

        proteins = {}
        for key, spec in entries.items():
            missing = [f for f in ("library_csv", "wt_pdb", "msa_a3m")
                       if f not in spec]
            if missing:
                raise ValueError(f"{path}: protein {key!r} is missing {missing}")
            proteins[key] = ProteinSpec(
                key=key,
                library_csv=resolve(spec["library_csv"]),
                wt_pdb=resolve(spec["wt_pdb"]),
                msa_a3m=resolve(spec["msa_a3m"]),
                msa_weights=resolve(spec["msa_weights"]) if spec.get("msa_weights") else None,
                crop=spec.get("crop", "dataset"),
            )
        return cls(root=root, proteins=proteins)

    def spec(self, key: str) -> ProteinSpec:
        if key not in self.proteins:
            raise KeyError(
                f"unknown protein {key!r}; configured: {sorted(self.proteins)}")
        return self.proteins[key]

    # ---- layout -----------------------------------------------------------

    def round_dir(self, depth: int) -> Path:
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth}")
        return self.root / "rounds" / f"round{depth}"

    def labels_csv(self, key: str, depth: int) -> Path:
        return self.round_dir(depth) / "labels" / f"{key}.csv"

    def candidates_csv(self, key: str, depth: int) -> Path:
        return self.round_dir(depth) / "candidates" / f"{key}.csv"

    def zeroshot_csv(self, key: str, depth: int) -> Path:
        return self.round_dir(depth) / "zeroshot" / f"{key}.csv"

    def embeddings_npz(self, key: str, depth: int) -> Path:
        return self.round_dir(depth) / "embeddings" / f"{key}.npz"

    def picks_csv(self, key: str, depth: int) -> Path:
        return self.round_dir(depth) / "picks" / f"{key}_consensus.csv"

    def workspace(self, depth: int) -> Path:
        return self.round_dir(depth) / "workspace"

    def pmls_dir(self, key: str, depth: int) -> Path:
        return self.workspace(depth) / "pmls" / key

    def mutant_pdbs_dir(self, key: str, depth: int) -> Path:
        return self.workspace(depth) / "mutant_pdbs" / key

    def pre_token_dir(self, modality: str, depth: int) -> Path:
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}")
        return self.workspace(depth) / "pre_tokens" / modality

    def tar_dir(self, key: str, modality: str, depth: int) -> Path:
        if modality not in MODALITIES:
            raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}")
        return self.workspace(depth) / "tars" / key / modality

    def library_for(self, key: str, depth: int) -> Path:
        """Depth 1 is the saturation library; deeper rounds use candidates."""
        return self.spec(key).library_csv if depth <= 1 else self.candidates_csv(key, depth)

    # ---- crop window ------------------------------------------------------

    def crop_window(self, key: str, variant: Optional[str] = None,
                    seq_len: Optional[int] = None,
                    max_len: int = MAX_LEN) -> Tuple[int, int]:
        """The `max_len`-residue window of the wild type that gets tokenized.

        Two policies, chosen per protein:

        `dataset` (default) centres ONE window on the mutation range of the
        whole library, so every variant and the wild type share it and their
        embeddings stay positionally comparable. This is what the released
        checkpoint was trained with.

        `per-variant` centres the window on each variant's own positions.
        Necessary when mutations span more than `max_len`, but variants then
        have different crop starts, so anything comparing embeddings across
        variants by absolute position is no longer valid.
        """
        spec = self.spec(key)
        if spec.crop == "dataset":
            return self._dataset_window(key, max_len)

        # Under the per-variant policy an unspecified variant means the wild
        # type, not "fall back to the dataset window": falling back would start
        # the window at the dataset's mut_min and put the wild type out of frame
        # with every variant mutated near the N-terminus.
        if not variant or variant == WT_TAG:
            # Start at the N-terminus so the wild type stays in frame with
            # variants mutated near the start; the dataset window would begin
            # at mut_min and drop the leading residues.
            return (0, max_len)

        positions = _mutation_positions(variant)
        if not positions:
            return self._dataset_window(key, max_len)
        if seq_len is None:
            seq_len = self._wt_length(key)
        return _centre_window(positions, seq_len, max_len)

    def _dataset_window(self, key: str, max_len: int) -> Tuple[int, int]:
        cache_key = (key, max_len)
        if cache_key in self._crop_cache:
            return self._crop_cache[cache_key]

        import pandas as pd
        df = pd.read_csv(self.spec(key).library_csv)
        seq_len = len(df[SEQUENCE_COL].iloc[0])

        if seq_len <= max_len:
            out = (0, seq_len)
        else:
            positions: List[int] = []
            for variant in df[VARIANT_COL]:
                if variant == WT_TAG:
                    continue
                positions.extend(_mutation_positions(variant))
            out = _centre_window(positions, seq_len, max_len) if positions else (0, max_len)

        self._crop_cache[cache_key] = out
        return out

    def _wt_length(self, key: str) -> int:
        import pandas as pd
        df = pd.read_csv(self.spec(key).library_csv, usecols=[SEQUENCE_COL], nrows=1)
        return len(df[SEQUENCE_COL].iloc[0])


def variant_stem(protein: str, variant: str) -> str:
    """Per-variant file stem, shared by every modality.

    `<protein>_<n>mut_<md5(variant)[:16]>` is the contract that keys the five
    modality streams together: all of them must emit the same stem for the same
    variant or the tar shards will not join. Note the wild type lands on
    `_1mut_` because `'wild_type'.count(':') + 1 == 1`; that is historical and
    deliberate -- a lone `_1mut_` entry for a protein is the wild-type
    placeholder.

    The hash is truncated to 16 hex characters. An earlier revision of the
    pipeline used 8 and observed collisions, so do not shorten it.
    """
    import hashlib
    mut_count = str(variant).count(":") + 1
    digest = hashlib.md5(str(variant).encode()).hexdigest()[:16]
    return f"{protein}_{mut_count}mut_{digest}"


def _mutation_positions(variant: str) -> List[int]:
    """0-indexed positions named by a variant string; unparseable parts skipped."""
    out: List[int] = []
    for part in str(variant).split(":"):
        try:
            out.append(int(part[1:-1]) - 1)
        except ValueError:
            continue
    return out


def _centre_window(positions: Iterable[int], seq_len: int,
                   max_len: int) -> Tuple[int, int]:
    positions = list(positions)
    mut_min, mut_max = min(positions), max(positions)
    region = mut_max - mut_min + 1
    if region >= max_len:
        # Mutations span more than the window; starting at mut_min keeps the
        # earliest ones and loses the tail. Nothing better is available.
        return (mut_min, mut_min + max_len)
    start = max(0, mut_min - (max_len - region) // 2)
    end = start + max_len
    if end > seq_len:
        end = seq_len
        start = max(0, end - max_len)
    return (start, end)
