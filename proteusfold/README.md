# ProteusFold

A fork of [Boltz](https://github.com/jwohlwend/boltz) that adds a differentiable
**Gauss-Seidel / position-based-dynamics projection** to the diffusion sampler,
and the batch driver used to fold variant libraries with it.

The projection pushes sampled coordinates back onto geometric constraints
(bond lengths, angles, stereochemistry, planarity) during sampling rather than
afterwards. It is implemented as NVIDIA Warp CUDA kernels and switched on with
`--project_pbd`.

This is the structure predictor behind the structures in
[ProTEUS](../README.md), one directory up. It needs its **own environment**, because
its torch and scipy pins conflict with the pipeline's, which reaches it as a
subprocess through `PROTEUS_BOLTZ`.

> Everything outside **What this fork changes** is upstream Boltz, unmodified.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
```

**Use a fresh environment.** Both the import name and the distribution name
are `boltz` at version `2.2.1`, unchanged from upstream on purpose, because renaming
would make `boltz.__version__` raise. So installing here over an upstream boltz
replaces its metadata, and `pip show boltz` reports upstream's version.

`warp-lang` is a hard dependency, not an extra: `diffusion.py` imports it at
module level. A CUDA GPU is required for the projection kernels.

### Weights and the chemical component dictionary

On first run the predictor downloads into `$BOLTZ_CACHE`, or `~/.boltz` if
unset: **6.2 GB**, and `mols.tar` is extracted beside itself, so budget 8 GB:

| file                 | size    | note                                                                           |
| -------------------- | ------- | ------------------------------------------------------------------------------ |
| `boltz2_conf.ckpt` | 2.29 GB | the public Boltz-2 weights; this fork trains no new ones for the `boltz2` path |
| `boltz2_aff.ckpt`  | 2.06 GB | affinity head, downloaded even when nothing asks for affinity                  |
| `mols.tar`         | 1.86 GB | chemical component dictionary, extracted to `mols/`                            |

The download is keyed on `mols.tar` alone, so an extracted `mols/` does **not**
prevent re-downloading the archive, so keep the tar. `--cache` pointed at a
populated directory skips all of it.

## Predict

Inputs are the upstream schema, one YAML or FASTA per structure; see
[docs/prediction.md](docs/prediction.md). A minimal single-chain protein with
its own alignment:

```yaml
sequences:
- protein:
    id:
    - A
    sequence: MAGLRHTFVVADATLPDCPLVYASEGFYAMTGY...
    msa: /abs/path/to/wild_type.a3m
```

```bash
# one input -- examples/ ships a few
python predict.py --input examples/prot.yaml --out out/

# a directory, four GPUs, one model load per GPU
python predict.py --input my_inputs/ --out out/ --gpus 0,1,2,3

# one shard of a sweep, for a second machine
python predict.py --input my_inputs/ --out out/ --shard 1/4
```

Results are named after what the predictor was handed, which for a directory
input is the staging directory, not a file:

| invocation                              | results under                                                  |
| --------------------------------------- | -------------------------------------------------------------- |
| `--input one.yaml --out out/`         | `out/boltz_results_one/predictions/<id>/`                    |
| `--input dir/ --out out/`             | `out/boltz_results_w0/predictions/<id>/`                     |
| `--input dir/ --out out/ --gpus 0,1`  | `out/w0/boltz_results_w0/…`, `out/w1/boltz_results_w1/…` |
| `--input dir/ --out out/ --shard 1/4` | `out/boltz_results_s1-4w0/…`                                |

In each, the model is `<id>_model_0.pdb`, ranked first by confidence, with
`confidence_<id>_model_0.json` beside it.

`predict.py` replaces eleven `inference_*.sh` scripts and 437 generated
per-shard config files that differed only in a dataset name and an index range.

### The defaults are this fork's, not upstream's

|                         | here | upstream    |
| ----------------------- | ---- | ----------- |
| `--sampling-steps`    | 10   | 200         |
| `--step-scale`        | 1.0  | 1.5         |
| `--diffusion-samples` | 5    | 1           |
| `--project-pbd`       | on   | not present |
| `--output-format`     | pdb  | mmcif       |

`--no-project-pbd` gives plain Boltz-2 at **these** settings, the comparison
for what the projection does, not a comparison against published Boltz-2
numbers.

**On `--model boltz1` the flag did nothing until recently.** `Boltz1` never
read its `physically_informed` attribute, both `project_pbd` call sites were
unconditional, and the `project_pbd=` from `main.py` was filtered out by
Lightning. Fixed, but it means **any Boltz-1 result from before, including
anything labelled a projection-free baseline, had the projection on.** The
training config `structure_boltz-1.yaml` had the same problem and now says
`physically_informed: false`.

### Alignments

The a3m parser reads lowercase as an insertion, `-` as a match gap, raises
`KeyError` on `.`, and needs every row to cover the query's match columns: a
short row reads into a `-1` pad and dies on an assertion that never mentions
the alignment. `tools/clean_a3m.py` converts either geometry and checks the
result:

```bash
python tools/clean_a3m.py raw.a2m clean.a3m --expect-sequence wild_type.fasta
```

Use it on anything that is not already a plain uppercase-query a3m.

## What this fork changes

New:

```
boltz/model/models/gauss_seidel_fast_diff.py                      the kernels
boltz/model/models/gauss_seidel_fast_diff_implicit_warp_cg.py     + CG solver
boltz/model/models/gauss_seidel_fast_diff_implicit_warp_cg_all.py + batched
boltz/main_large_dataset.py            config-driven driver kept for reference
```

Modified upstream files, all of it reachable from `--project_pbd`:

```
boltz/model/modules/diffusion.py       project_pbd, called from the sampler
boltz/model/modules/diffusionv2.py     the same for the Boltz-2 sampler
boltz/model/models/boltz1.py           physically_informed / project_pbd, gated
boltz/model/models/boltz2.py           project_pbd plumbed to the output dict
boltz/main.py                          the --project_pbd flag
boltz/model/potentials/potentials.py   three `pdb.set_trace()` calls -> errors
boltz/data/feature/symmetry.py         one `pdb.set_trace()` call -> raise
```

The last two are not about the projection. Upstream's four `pdb.set_trace()`
calls on the `--use_potentials` path block on stdin in a subprocess, so a
failure there looks like a worker that never returns and a GPU that stays busy.
They now raise instead.

Also added: `predict.py`, and under `tools/`, `clean_a3m.py` plus four
dataset-inspection scripts. See [tests/README.md](tests/README.md) for what
`pytest` actually covers.

`extern/PhysProtein/physcialsim_metrics.py` extends upstream's
`scripts/eval/physcialsim_metrics.py` with chirality checks and `--model_dir`.
Use the `extern/` one: the upstream copy still hardcodes five absolute paths on
the upstream authors' cluster.

`main_large_dataset.py` predates `predict.py` and is kept because the published
runs went through it. It needs integer filename stems (`1.yaml`, `2.yaml`, …)
and has no `output_format` key, so it always writes mmCIF.

## Reproducibility notes

**A prediction depends on what else was in the batch.** On a 119-residue
protein, same sequence, same seed, same settings: confidence 0.8963 folded
alone, 0.9022 in a batch of three, and the two structures 2.76 Å CA-RMSD apart
after superposition. Folded alone twice it is bit-identical. So to reproduce a
set of structures, shard it the way it was sharded originally.

**`--cache` is part of the configuration too.** It selects which checkpoint
file is loaded, by directory rather than by content.

## Training

`scripts/train/` is upstream's training code with this fork's additions:
`configs/structure_boltz-gs.yaml` trains the projection-enabled model, and
`train.py` gained deterministic seeding and keeps every epoch's checkpoint.

The configs reference a processed RCSB dataset that is not distributed here;
[docs/training.md](docs/training.md) says how to build it. Set `wandb.entity`,
or delete the `wandb` block to log locally.

## Citation

Cite upstream Boltz for the model:

```bibtex
@article{passaro2025boltz2,
  title   = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  author  = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and others},
  journal = {bioRxiv},
  year    = {2025}
}

@article{wohlwend2024boltz1,
  title   = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  author  = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and others},
  journal = {bioRxiv},
  year    = {2024}
}
```

## License

This subtree keeps upstream's MIT, see [LICENSE](LICENSE). Developed upon
[Boltz](https://github.com/jwohlwend/boltz) by Jeremy Wohlwend, Gabriele Corso,
Saro Passaro and contributors, whose copyright notice `LICENSE` retains, and
[ColabFold](https://github.com/sokrypton/ColabFold), the MSA client under
`boltz/data/msa/`, whose public server a query is sent to. The files this fork
changed are listed above; the rest is upstream. The Boltz-2 weights are
downloaded at run time, not shipped here.

Two files were written for this release and are not upstream: `predict.py` and
`tools/clean_a3m.py`. Each carries its own notice and is governed by
[../LICENSE](../LICENSE), the pipeline's Internal Research End User Agreement,
not by MIT.
