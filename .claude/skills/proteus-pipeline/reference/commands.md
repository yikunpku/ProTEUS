# Command reference

Only the flags that change results are listed. `--help` on any of them is
authoritative; this file exists to say which defaults matter and why.

## proteus-doctor

```bash
proteus-doctor [--config cfg.yaml]
```

Non-zero exit on a blocking problem. `--config` adds the three-way consistency
check per registered protein.

**Do not treat the printed count as fixed.** Checks that end in a warning are
not counted in the denominator, so the "N/M" line moves with the machine,
19 to 28 depending on what warns. Measured here: `24/25 ... 3 warning(s)` on
one machine and `22/23 ... 5 warning(s)` on another. The exit code is the
pass condition, not the count. A missing `PROTEUS_BOLTZ` and a patched
`fair_esm` are warnings: neither blocks, neither is counted.

## proteus-prepare

```bash
proteus-prepare --config cfg.yaml --protein KEY [--fasta f | --sequence S] [--register]
```

| flag | default | note |
|---|---|---|
| `--msa` | `colabfold` | **sends the sequence to a public third-party server.** `af3-zip` takes it from an AlphaFold Server archive, `a3m` uses your own file. Both need `--msa-source`. |
| `--structure` | `boltz` | predicts it, which needs `proteusfold/` installed in its own environment. `af3-zip` / `pdb` / `rcsb` import one instead and each needs `--structure-source`. |
| `--boltz-python` | none | takes precedence over `--boltz-exe`. An explicit `--boltz-exe` that does not resolve is a hard error, not a fallback. |
| `--msa-server-for-structure` | off | lets the predictor run its own search: two alignments instead of one, so it changes results. |

Neither `--fasta` nor `--sequence` is required and they are not mutually
exclusive: with neither, the wild type is read from the library CSV's
`wild_type` row. `--register` on an already-registered key is a silent no-op.

`--msa a3m` validates your file rather than trusting it, and it **rejects a
leading `#` comment line**, which is exactly the shape raw ColabFold/MMseqs2
output has. Run it through `proteus/prepare/a3m_tools.py` first.

When the three-way check fails, whatever that run wrote is renamed
`<file>.rejected` and the command exits 1. Two consequences worth knowing: a
second failure overwrites the first `.rejected`, and the predictor's
`<key>_confidence.json` is not renamed with it, so a stale confidence file can
sit beside a structure that was set aside.
| `--legacy-a3m` | off | keeps the pre-rewrite alignment shape. Only for reproducing old numbers. |
| `--register` | off | without it, the registry block is printed, not written. |
| `--boltz-exe` | `$PROTEUS_BOLTZ` | the predictor's launcher, in its own environment. |
| `--boltz-cache` | unset | forwarded only when given; otherwise the predictor uses `$BOLTZ_CACHE`, else `~/.boltz`. A cold cache downloads about **6.2 GB** (measured: `boltz2_conf.ckpt` 2.29, `mols.tar` 1.86, `boltz2_aff.ckpt` 2.06). `mols.tar` is then extracted beside itself, so budget ~8 GB. |
| `--overwrite` | off | redo steps whose output exists. Otherwise idempotent. |

Leaves the config alone unless `--register`. On success the only outputs it
keeps are under `proteins/<key>/`, but **on failure it deliberately keeps the
predictor's working directory**, a `/tmp/proteus-boltz-*` outside
`proteins/`, because that is where the inputs and the predictor's own log are.
Nothing cleans those up; pass `--work-dir` to put them somewhere you will
look. It is not a cheap call either way: on the default `--structure boltz` it
populates the predictor's multi-GB weight cache.

## proteus-score

```bash
proteus-score --dms-file LIB.csv --output-file OUT.csv \
              --msa-dir proteins/KEY/msa --structure-dir proteins/KEY/structure \
              [--checkpoint CKPT]
```

There is no switch to drop a modality, but **leaving out `--msa-dir` or
`--structure-dir` drops one anyway, silently**: a modality that produced no
logits never enters `active_modalities`, and the ensemble weights are
renormalized over what is left. The output column is still called
`Proteus_score`. Treat both directories as required, and the same goes for the
structure directory actually containing `<stem>.pdb`.

**Its cost is per protein, not per variant.** The masked marginals are computed
once over the sequence and the alignment, and each variant is then a lookup, so
scoring 20 variants and scoring 165,000 cost about the same. Measured: the whole
165,428-row CreiLOV library in 361 s on one H100. Do not shard this by variant
count, and do not restrict the input to save time. It will not.

That holds for the invocation above. `--mt-tokens-dir` changes it: with that
directory given, the structure-consistency path runs one ESM3 forward pass per
variant per mutated position, and the cost becomes per variant. The published
pipeline leaves it unset. Do not add it without saying what it costs.

- `--checkpoint` is **optional, and a path that does not exist is not an
  error**: it falls back to equal ensemble weights, silently. So does a path
  that exists but fails to load: that prints `Error loading checkpoint` and
  carries on with equal weights. Check the path, and check that line. Measured
  difference on 197 CreiLOV variants: Spearman 0.7083 vs 0.7077, identical
  top-30. Reproducible either way, but check the path is real.
- `--seq-basemodel` has no `choices` and no validation. A value whose
  lowercase form starts with `esmc` selects ESMC (`esmc_600m`, `esmc_6b`);
  **anything else is ignored entirely** and ESM2-3B is loaded, so
  `--seq-basemodel esm2_t33_650M_UR50D` silently scores with the 3B model.
- A metrics sidecar is written **only when the input carries a `DMS_score`
  column**, which a candidate library does not, so this applies to benchmark
  inputs. It lands next to `--output-file` but is named after `--dms-file`'s
  stem, so two runs on the same input overwrite each other's metrics even with
  different `--output-file`. Give each arm its own directory.

## proteus-candidates

```bash
proteus-candidates --config cfg.yaml --protein KEY --depth D
```

Backgrounds are measured depth-(D−1) variants with fitness **strictly above**
`--threshold` (default 1.0), crossed with every substitution at an unused
position, canonically deduplicated, already-measured variants removed. Strictly
above matters: a background that measured exactly at wild type is excluded.

No GPU, seconds. It refuses to overwrite an existing candidates file and names
`--overwrite` / `--output` in the message. Think before passing `--overwrite`:
for depth ≥ 2 that path is also `library_for(key, depth)`, so replacing it
discards whatever replaced it (`restrict_to_library.py`'s output in the demo)
and leaves existing zero-shot scores and features describing a library that is
no longer on disk. `--labels` overrides where the measurements are read from;
without it a missing labels file is a bare pandas traceback.

## featurize

All take `--config --protein --depth`. Ordering constraints are in SKILL.md.

| step | GPU | `--device` default | flag worth knowing |
|---|---|---|---|
| `structure_prep` | no | n/a | `--workers` (PyMOL processes, default half the CPUs); `--stage pml\|pdb\|both`; `--overwrite`, `--allow-failures`, `--pymol-binary` |
| `structure_predict` | **yes** | n/a (`--accelerator gpu\|cpu`) | replaces `structure_prep`; `--boltz-exe`, `--limit` for smoke tests |
| `sequence` | **no** | n/a | no model forward, but it still loads ESM2-**650M** just for its alphabet, so a cold or offline machine dies here on a ~2.5 GB download that `proteus-doctor` does not check for. `--esm-model` names that checkpoint (same 33-token vocab as 3B) |
| `mask` | no | n/a | `--task-id`/`--total-tasks`, `--allow-failures`, `--no-progress` |
| `structure_tokens` | yes | `cuda:0` | `--limit` for smoke tests |
| `msa` | yes | **`cpu`** | `--rows`, `--seed` (recorded in the sidecar), `--sampler`, `--legacy-a3m`, `--num-cpus` for the one-off sequence-weight computation |
| `tars` | no | n/a | `--modality`, but see below: it is not a free "redo one stream" |
| `embed` | yes | `cuda:0` | `--checkpoint` **required**; `--reduction mean\|bos\|none`; `--legacy-masking`; `--overwrite` |

`tars` refuses more than the table suggests. Completeness is judged over all
four input streams regardless of `--modality`, so a variant featurized for some
streams and not others stops the pack with a message naming the gap. And once
the member list has changed, a single-stream repack is refused outright. The
fix is to drop `--modality` and repack everything.

`embed` skips an output that already exists, and **exits** if that file exists
but is not a complete embedding file (an interrupted run leaves one): delete it
or pass `--overwrite`. A `msa` re-run at the same depth with different
`--rows`/`--seed`/`--sampler` is refused by the sidecar, and that sidecar is
shared by every protein at that depth.

A variant whose mutations all fall outside the crop window gets an all-zero
mask, identical to the wild type, and is recorded as a failure.
`--allow-failures` lets the step pass, but the tolerated variant has no `.npy`,
so `tars` then refuses the whole round.

**The defaults do not agree with each other.** `msa` defaults to `cpu` while
`structure_tokens` and `embed` default to `cuda:0`, so running the six steps
with no `--device`, which is how the loop in SKILL.md reads, computes the MSA
stream on CPU and the rest on GPU. CPU and GPU are not bit-identical
(max|diff| 5.7e-06 measured on the MSA embeddings), so pass `--device`
explicitly to all three and keep it the same for a whole campaign.

`embed --device cpu` additionally needs `--dtype float32`: `--dtype` defaults to
`bfloat16` and the autocast is CUDA-only, so the combination is rejected up
front rather than failing inside the first matmul.

`--task-id` / `--total-tasks` shard `sequence`, `mask`, `structure_tokens`,
`msa`, `tars`, `embed` and `structure_predict`, but **not** `structure_prep`,
which has neither flag and parallelizes with `--workers` instead. Passing them
to it is an argparse error.

## proteus-select

```bash
proteus-select --config cfg.yaml --protein KEY --depth D [--write-pool]
```

RandomForest (`--trees` 100, `--seed` 42) over the measured variants, then
three ranked channels (`rank_rf` / `rank_zs` / `rank_base`) fused across five
`(a_zs, a_base)` settings and voted.

- `--base max` is the default and right from depth 3 on. `sum` tracks true
  fitness slightly better at depth 2 but inflates with depth.
- `--bad-threshold` (default 1.0) drops candidates containing a single that
  measured badly. This makes the candidate library a **superset** of what was
  actually ranked, which is why grading needs `--write-pool`.
- `-n` defaults to 30 picks.
- Defaults for `--embeddings` / `--zeroshot` are depths 1..D of this protein.
- `rank_rf` ranks over the whole pool; `rank_zs` and `rank_base` rank **within
  each depth** (midpoint convention), because neither is comparable across
  depths. The chosen aggregation is recorded as `base_agg` in the picks file.
- It refuses to overwrite the picks, the `_pool.csv` and any
  `--write-settings` file; `--overwrite` forces it.
- It reads the candidates CSV and **refuses if any candidate has no
  embedding**, naming them, rather than quietly ranking the subset that happens
  to be featurized. That is what catches a featurize shard that died.
  `--candidates-from` is the escape hatch for deliberately ranking a subset.

## proteus-audit

```bash
proteus-audit --config cfg.yaml --protein KEY --depth D
```

7 groups: labels, candidate construction, embedding coverage, zero-shot
coverage, the base channel, structure/crop consistency, consensus arithmetic.
The count depends on the protein and the crop policy: 24 for a protein under
the encoder window, 25 under `crop: per-variant`. Do not treat a fixed
number as the pass condition; the exit code is. Some checks report `[skip]`
when they do not apply, which is not a failure. Free to run; run it every
round.

## ProteusFold (the structure predictor, `proteusfold/`)

```bash
cd proteusfold && python predict.py --input inputs/ --out out/ --gpus 0,1,2,3
```

Defaults are this fork's, not upstream Boltz's: 10 sampling steps (vs 200),
step_scale 1.0 (vs 1.5), 5 diffusion samples, seed 101, `--project-pbd` on,
`--output-format pdb`. `--no-project-pbd` gives plain Boltz-2 **at this fork's
settings** (10 steps, step_scale 1.0, seed 101), not upstream's, so it is the
comparison for what the projection does, not a comparison against published
Boltz-2 numbers.

`proteusfold/tools/clean_a3m.py` (that subtree's `tools/`, not the top-level
one) converts an alignment into the dialect its a3m parser accepts; use it on
anything that is not already a plain uppercase-query a3m. Proteus does the
equivalent itself in `proteus/prepare/a3m_tools.py`, so this is only needed
when driving the predictor directly.
