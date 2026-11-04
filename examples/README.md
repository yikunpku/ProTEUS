# Demo

One closed-loop round on a public benchmark protein, with ground truth, so
you can confirm your install reproduces sensible numbers before pointing it at
a protein you care about.

CreiLOV is a 119-residue LOV-domain fluorescent protein. It stays under the
512-position encoder input, so it never exercises the crop window, the part
of the pipeline easiest to get wrong, and CombinGym ships measured fitness
for its whole combinatorial library, which lets the loop run in silico and
then be scored against measurements it never saw.

## Inputs

`prepare_creilov.py` assembles the inputs from a CombinGym checkout:

| File | Source |
|---|---|
| `libraries/CreiLOV.csv` | the round-1 library: the wild type and the 20 measured singles, as `dataset, mutant, mutated_sequence, DMS_score`. The config points `library_csv` here, and `library_for(key, 1)` returns it, so this is what every featurize step iterates, and it must not hold the deeper variants. |
| `libraries/CreiLOV_full.csv` | all 165,428 measured variants, every depth. Ground truth, read only by `restrict_to_library.py` and `score_picks.py` through an explicit `--library`, so the loop never sees the answers for anything it was not told. |
| `proteins/CreiLOV/structure/CreiLOV.pdb` | wild-type structure |
| `proteins/CreiLOV/msa/CreiLOV.a3m` | wild-type MSA |

CombinGym is MIT-licensed; see <https://www.combingym.org/>. Nothing is
redistributed here. The MSA is the input that moves the numbers most, so a
different alignment gives different scores. That is expected, not a broken
install.

The library is combinatorial rather than saturating: 1 wild type, 20 single
mutants, then combinations of those same 20 substitutions up to depth 15. So
"round 1" here means those 20 measured singles.

## Run

**Install the package first**, with `uv sync --frozen` or `pip install -e .` in an
activated environment. Three of the four scripts here import `proteus`, and
unlike `python -m proteus.<module>` (which finds the package through the current
directory) a script run as `python examples/x.py` puts `examples/` on the import
path, not the repository root. Without the install those three stop at
`ModuleNotFoundError: No module named 'proteus'` while the `python -m` steps in
the same sequence keep working, which makes the cause look stranger than it is.

```bash
python examples/prepare_creilov.py --combingym /path/to/CombinGym --out examples/data
CFG=examples/creilov.yaml
CKPT=weights/proteus_release.pth

# Treat the measured depth-1 variants as the first round's wet-lab readout.
# Fitness is normalized by the wild type's own score, so 1.0 means "as good
# as wild type" exactly as it would in a real campaign.
python examples/make_round1_labels.py --config $CFG --protein CreiLOV --depth 1

# Propose depth-2 variants: every measured single that beat wild type, times
# every substitution at a position it has not used.
proteus-candidates --config $CFG --protein CreiLOV --depth 2

# Benchmark-only: keep just the proposals CombinGym actually measured, so the
# picks can be scored. In a real campaign you skip this and order from the
# full proposal. See the script's docstring -- it drops the scores, so nothing
# downstream can see the answers.
python examples/restrict_to_library.py --config $CFG --protein CreiLOV --depth 2 \
    --library examples/data/libraries/CreiLOV_full.csv

# Zero-shot both pools. This is the expensive step; see Cost below.
for D in 1 2; do
  proteus-score --dms-file $(python -c "from proteus.config import ProteusConfig as C; print(C.from_yaml('$CFG').library_for('CreiLOV', $D))") \
                --checkpoint $CKPT \
                --msa-dir examples/data/proteins/CreiLOV/msa \
                --structure-dir examples/data/proteins/CreiLOV/structure \
                --output-file $(python -c "from proteus.config import ProteusConfig as C; print(C.from_yaml('$CFG').zeroshot_csv('CreiLOV', $D))")
done

# Featurize both depths. The surrogate trains on the measured variants'
# embeddings, so depth 1 is not optional.
# Only structure_tokens, msa and embed take --device, and their defaults
# disagree (msa: cpu, the other two: cuda:0). The numbers below were produced
# on CPU in float32, so this pins both.
for D in 1 2; do
  for STEP in structure_prep sequence mask; do
    python -m proteus.featurize.$STEP --config $CFG --protein CreiLOV --depth $D
  done
  for STEP in structure_tokens msa; do
    python -m proteus.featurize.$STEP --config $CFG --protein CreiLOV --depth $D --device cpu
  done
  python -m proteus.featurize.tars  --config $CFG --protein CreiLOV --depth $D
  python -m proteus.featurize.embed --config $CFG --protein CreiLOV --depth $D \
         --checkpoint $CKPT --device cpu --dtype float32
done

# Rank, fuse, vote. --write-pool also dumps every candidate's three channels
# and fused scores, which score_picks needs below: `select` drops candidates
# built on a single that measured badly, so the candidate library is a superset
# of what the model actually ranked, and grading against the superset
# understates the random baseline.
proteus-select --config $CFG --protein CreiLOV --depth 2 --write-pool

# The pre-order checks: seven groups, 24 of them for this protein.
proteus-audit --config $CFG --protein CreiLOV --depth 2

# Finally, the thing a real campaign cannot do: score the picks against
# measurements the loop never saw.
python examples/score_picks.py --config $CFG --protein CreiLOV --depth 2 \
    --library examples/data/libraries/CreiLOV_full.csv
```

## Your own protein

`prepare_creilov.py` exists because CombinGym ships this protein's structure
and MSA already. For a protein of your own, those two files are what
`proteus-prepare` is for:

```bash
proteus-prepare --config cfg.yaml --protein MY_PROTEIN --fasta wt.fasta --register
```

Then the only thing left to write by hand is the library CSV (`mutant`,
`mutated_sequence`, and a fitness column, with a `wild_type` row) and the
first round's labels. Everything from `proteus-candidates` onwards is the same
commands as above with `CreiLOV` replaced.

Two differences from this demo are worth expecting. The demo's alignment is an
a2m and is read as-is; a freshly searched one is an a3m, whose lowercase
insertions are stripped so that every row lines up with the query before the
MSA reader sees it (`proteus/prepare/a3m_tools.py`; `--legacy-a3m` keeps the
older shape, which does not). And the demo's
structure is CombinGym's; a predicted one is not the same structure. On this
protein a Boltz-2 prediction sits 2.2 Å CA-RMSD from the reference, so the
structure channel, and through it the surrogate, sees different inputs.

## What you should see

Run on CPU against the release checkpoint, this is the whole demo's output.
Use it to tell a working install from a broken one.

| step | expected |
|---|---|
| labels, depth 1 | 21 rows: the wild type plus 20 singles; `wild_type` fitness exactly 1.0; 15 singles beat it (the 15th, `L4N`, by 0.0001; if you count 14, that is why) |
| candidates, depth 2 | 33,527 generated, then 172 after `restrict_to_library.py` (CombinGym's CreiLOV library is combinatorial, not saturating, so only 176 of the doubles were ever measured and 4 of those are unreachable from the measured singles) |
| featurize | depth 1: 21 per modality; depth 2: 173 (172 candidates + the wild type). `sequence int64(514,)`, `structure int64(512,)`, `msa_emb_128 float32(512,768)`, `mutation_mask uint8(512,)` |
| embed | `(21, 768)` and `(173, 768)`; the loader reports 1296 tensors from the checkpoint and no missing trunk tensors |
| select | pool of 100 (the 172 minus candidates built on a single that measured below the wild type), all 5 grid settings agreeing on the top of the batch |
| audit | 24 checks, all pass |
| score_picks | hit rate 30/30, fraction of truth 0.700, best found 1.1206 = the best in the pool |

The selection quality numbers, in full:

```
hit rate          30/30 = 1.000 beat the wild type (pool baseline 0.750)
fraction of truth 21/30 = 0.700 of the true top-30 (random baseline 0.300)
best found        1.1206  (best in pool 1.1206)
mean picked       1.0715  (pool mean 1.0307)

spearman(rf_score)           over the pool (n=100): +0.6672
spearman(zs_score)           over the pool (n=100): +0.5029
spearman(mean_final_all_settings) over the pool (n=100): +0.7508
```

Two things that read as results rather than as a smoke test: the fused ranking
(+0.75) beats either channel alone (+0.67 random forest, +0.50 zero-shot), which
is the whole argument for the three-channel fusion; and the round found the
single best double mutant in its pool, which is the outcome a wet-lab round is
actually judged on.

These numbers are for the run as documented above: **CPU, and
`featurize.embed --dtype float32`**. The dtype is not a detail. Measured on
this demo, embedding with the GPU default `bfloat16` instead moves the
embeddings by up to 0.11 and the selection with them -- fraction of truth
0.700 -> 0.767, `spearman(rf_score)` +0.667 -> +0.659, 2 of 30 picks different.
Same code, same seed, same labels. Device alone, at the same dtype, is a
4-order-of-magnitude smaller effect (max|diff| 5.7e-06). So pick one dtype for
a campaign and record it; and if your numbers differ from the table, check the
dtype before suspecting your install.

What else can legitimately differ: the zero-shot numbers, if your MSA differs
from the bundled `.a3m` (see Inputs) or if `hhfilter` is not on PATH.
`featurize.embed` is deterministic -- re-run on the same tars at the same
device and dtype it reproduces the previous `.npz` bit-for-bit -- and so is
`proteus-select`, which sorts its training rows precisely so that the batch
does not depend on the order rows happen to sit in on disk.

## Cost

Zero-shot scoring dominates: ESM2-3B and ESM-MSA-1b each need one forward pass
per residue (and the MSA model repeats that per sampled alignment), so a
119-residue protein is minutes on a GPU and a couple of hours on CPU. The
featurization steps are cheap by comparison, and `proteus-candidates`,
`proteus-select` and `proteus-audit` are seconds.

Everything in this demo runs on CPU, and the `--device cpu` flags in the
commands above are what make that true rather than optional: `structure_tokens`
and `embed` default to `cuda:0`, so without them a GPU machine silently runs a
different arm, and a CPU-only machine fails.

One flag matters for CPU: `featurize.embed` defaults to `--dtype bfloat16`,
which only works under CUDA autocast, so on CPU add `--dtype float32`:

```bash
python -m proteus.featurize.embed --config $CFG --protein CreiLOV --depth $D \
       --checkpoint $CKPT --device cpu --dtype float32
```

The step refuses the bad combination up front rather than failing inside the
first matmul. float32 is also what the reference embeddings were computed in,
so this is the exact arm, just slower.
