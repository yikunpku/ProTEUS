# Environments

Two of them, deliberately. Proteus and the structure predictor cannot share
one: measured conflicts are torch 2.12 vs 2.7, scipy 1.13 vs 1.15, pandas 2.3
vs 2.2, biopython 1.84 vs 1.85. The Proteus pins are what reproduce the
published selections, so the predictor is reached as a subprocess instead.

## Proteus

Three paths, in order of preference. The deciding factor is two binaries that
are not installable from PyPI: `pymol` (builds mutant structures in
`structure_prep`) and `hhfilter`.

**`hhfilter` is not optional in the sense of not mattering.** `proteus-score`
auto-detects it with `shutil.which` and filters the alignment only when it is
found, so the same command gives different zero-shot scores on an environment
that has it and one that does not. The uv path has no `hhfilter`; Docker and
conda do. Any claim that the three environments agree has to be read as "the
two that carry the same binaries agree". Do not compare a uv score against a
Docker score.

| | Python | `pymol` | `hhfilter` | pinned by |
|---|---|---|---|---|
| **uv** | yes | no | no | `uv.lock`, with hashes |
| **Docker** | yes | yes | yes | image + `uv.lock` |
| conda | yes | yes | yes | version ranges only |

```bash
uv sync --frozen
uv run python tools/install_fair_esm.py   # required
uv run proteus-doctor
```

`tools/install_fair_esm.py` is not optional: it installs the unpatched
`fair-esm` under a name that does not collide with the EvolutionaryScale `esm`
SDK. Skipping it is one of the failure modes in `pitfalls.md`.

Use uv if you bring your own structures or are only scoring and selecting.
Use Docker (≈11 GB) to run the loop end to end; keep `--user "$(id -u):$(id -g)"`
and `-e HOME=/tmp` or every file written into a bind mount ends up owned by
root, and mount the ESM caches or each start re-downloads 6.6 GB.

`uv.lock` pins `torch 2.7.1+cu126` from PyTorch's own index. For another CUDA
or for CPU, change `[[tool.uv.index]]` and re-run `uv lock`, then re-check
your embeddings, not just that it imports.

## The structure predictor (ProteusFold)

`proteusfold/` in the same checkout, but its own distribution and its own
virtualenv, because its torch and scipy pins conflict with the pipeline's, so they
cannot share one.

```bash
cd proteusfold
python -m venv .venv && . .venv/bin/activate && pip install -e .
export PROTEUS_BOLTZ="$PWD/.venv/bin/boltz"
cd ..
```

- `warp-lang` is a hard dependency, not an extra: `diffusion.py` imports it at
  module level.
- A CUDA GPU is required for the projection kernels.
- First run downloads about **6.2 GB** into `$BOLTZ_CACHE` (or `~/.boltz`),
  measured on disk: `boltz2_conf.ckpt` 2.29 GB, `mols.tar` 1.86 GB,
  `boltz2_aff.ckpt` 2.06 GB, the last fetched even though nothing here asks for
  affinity. `mols.tar` is then extracted beside itself, so budget ~8 GB. The
  **CCD** download is keyed on `mols.tar` alone, so keep the tar; an extracted
  `mols/` does not prevent re-fetching it. Each checkpoint is keyed on its own
  file, and each is written straight to its final name with no size check, so an
  ENOSPC-truncated checkpoint looks complete to the next run.
- It installs under the import name `boltz`, same as upstream, so do not put it
  beside an upstream install.

## Caches and weights

| what | where | note |
|---|---|---|
| ESM2-3B, ESM-MSA-1b | `$TORCH_HOME/hub/checkpoints` | flat namespace keyed on filename, see `pitfalls.md` |
| ESM3 (gated) | HuggingFace cache | needed by the zero-shot structure channel |
| Proteus checkpoint | wherever you put it | **not in the repository**; `--checkpoint` |
| predictor weights | `$BOLTZ_CACHE` | public Boltz-2 weights; this fork trains none |

Set `TORCH_HOME` explicitly if you keep more than one hub cache. The default
`~/.cache/torch` is where a stray symlink does the most damage.

## GPU

| step | needs GPU | rough cost |
|---|---|---|
| `proteus-score` | yes | the expensive one: ESM-MSA-1b does one forward pass per alignment column per sampled alignment, measured 5 × 120 for a 119-residue query (prepend_bos, no append_eos), ~23 s per sample on an H100 |
| `structure_predict` | yes | ~40 s per variant in a batch, 119-residue protein, H100 |
| `structure_tokens`, `msa`, `embed` | yes, and all three take `--device` | defaults disagree: `msa` is `cpu`, the other two `cuda:0`. `msa` on CPU is ~55 s per variant without the window memoization; `embed --device cpu` also needs `--dtype float32` |
| `sequence`, `mask` | **no** | pure tokenization and bit-twiddling; no model forward, no `--device` |
| `structure_prep` | no | PyMOL, `--workers` processes |
| `candidates`, `tars`, `select`, `audit` | no | seconds to minutes |

Shard with `--task-id` / `--total-tasks`. Check free memory before starting:
these steps load multi-GB encoders and will not fit beside an active training
job on the same device.
