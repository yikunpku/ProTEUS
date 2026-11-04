---
name: proteus-pipeline
description: Run the Proteus protein-fitness pipeline end to end, from searching an MSA and predicting a wild-type structure for a new protein through scoring variants zero-shot, featurizing them across four encoder streams, training the surrogate, and picking the next wet-lab batch. Use when the user wants to run, resume, debug or audit a Proteus round, prepare a protein's inputs, fold variants with ProteusFold/Boltz-GS, or reproduce the CreiLOV demo. Also use when a Proteus command fails and the cause is not obvious from its message.
metadata:
  short-description: Drive the ProTEUS closed-loop pipeline and ProteusFold structure prediction
---

# Proteus pipeline

A round goes: **measured labels → candidates → zero-shot → featurize → select → audit**.
`proteus-prepare` runs once per protein before any of that, to get the two inputs
the pipeline cannot derive from the library CSV: the MSA and the wild-type structure.

## Rule: plan, then stop

Several steps cost a GPU and hours, and some of them are hard to undo (they
overwrite features on disk). So:

1. Read the config, see what is already on disk, and work out which steps are
   actually needed.
2. Print the plan: the commands in order, what each writes, a rough cost, and
   which GPU it will take.
3. **Stop and wait for the user to confirm.** Do not start a GPU step or a
   multi-minute step on your own initiative.

Steps that are free to run unasked: `proteus-doctor`, `proteus-audit`, and
reading any output. Everything else waits.

`proteus-candidates` and `proteus-select` both refuse to overwrite their
outputs. They exit non-zero before doing any work and name `--overwrite` /
`--output`. Do not reach for `--overwrite` to get past that message without
saying what it will destroy: for depth ≥ 2 the candidates file *is* the
round's library, so replacing it after `restrict_to_library.py`, or after
anything has been scored or featurized against it, leaves the features on
disk describing a library that is gone. `proteus-select --candidates-from` is
the supported way to rank a subset without rewriting the library.

If the user's instruction is already specific ("run featurize depth 2"), that is
the confirmation, so do it, but still say what you are about to run.

## First: preflight

```bash
proteus-doctor --config <cfg.yaml>
```

It exits non-zero on a blocking problem and names the fix. Run it before
anything else in a fresh environment, and again whenever a step fails for a
reason that smells environmental. With `--config` it also runs the three-way
consistency check on every protein in the registry: the MSA's query row, the
structure's residues, and the wild-type sequence agree. That sequence comes
from the library CSV's `wild_type` row. The registry has no FASTA field, so a
`proteins/<key>/fasta/` file is not what to inspect when this check fails.

Do not work around what it reports. A blocking failure means the numbers would
be wrong, not that the run would crash. See `reference/pitfalls.md` for the
ones that have actually happened.

## Two environments, not one

Structure prediction lives in its own environment and is reached as a
subprocess. Its torch and scipy conflict with Proteus's, and the pin set here
is what reproduces the published selections, so do not try to merge them.

```bash
export PROTEUS_BOLTZ=/path/to/proteusfold-env/bin/boltz   # or pass --boltz-exe
```

Without it, `proteus-doctor` warns and the `boltz` structure backend is
unavailable; `--structure af3-zip|pdb|rcsb` still work.

## A new protein

```bash
proteus-prepare --config cfg.yaml --protein KEY --fasta wt.fasta --register
```

Searches an MSA (ColabFold's public MMseqs2 server by default, which **sends the
sequence to a third party**; use `--msa af3-zip` or `--msa a3m` for anything
unpublished), predicts the wild-type structure from it, checks all three
descriptions of the protein against each other, and inserts the registry block.

Two CSVs are still written by hand, and they have different schemas:

| file | columns | written by |
|---|---|---|
| the library, at `library_csv` in the registry | `mutant`, `mutated_sequence`, a fitness column, and a `wild_type` row | you |
| the round's labels, at `rounds/round<D>/labels/<KEY>.csv` | `variant`, `fitness`, normalized so the wild type is exactly 1.0 | you; **no `proteus-*` command writes it** |

`proteus-candidates`, `proteus-select` and `proteus-audit` all read the labels
for round D−1, so a protein with only a library CSV gets as far as
`proteus-candidates --depth 2` and then dies on a path it was never told about.
The demo generates its labels with `examples/make_round1_labels.py`, which is a
benchmark helper, not part of the pipeline.

## A round

```bash
CFG=cfg.yaml; KEY=MY_PROTEIN; D=2
proteus-candidates --config $CFG --protein $KEY --depth $D
proteus-score --dms-file <library for depth $D> --checkpoint <ckpt> \
              --msa-dir <proteins/$KEY/msa> --structure-dir <proteins/$KEY/structure> \
              --output-file <zeroshot csv for depth $D>
# featurize: see the dependency graph below
proteus-select --config $CFG --protein $KEY --depth $D --write-pool
proteus-audit  --config $CFG --protein $KEY --depth $D
```

Resolve the paths with the config rather than guessing them:

```bash
python -c "from proteus.config import ProteusConfig as C; c=C.from_yaml('$CFG'); \
print(c.library_for('$KEY',$D), c.zeroshot_csv('$KEY',$D), c.embeddings_npz('$KEY',$D), c.picks_csv('$KEY',$D))"
```

The surrogate trains on the measured variants, so **every depth from 1 to D has
to be featurized and scored**, not just the new one.

Two things to say out loud when reporting a round's picks:

- The candidate pool spans depths 1..D. `rank_rf` ranks over the whole pool,
  but `rank_zs` and `rank_base` rank **within each depth**, because neither is
  comparable across depths. The picks file records which aggregation the base
  channel used in a `base_agg` column, and `proteus-audit` reads that column to
  know what to verify. With the column missing it accepts either `max` or
  `sum`, so an older picks file gets a weaker check.
- No published round exercised a pool with more than one depth in it. If
  unmeasured shallow candidates are still in the pool, the numbers are new
  results, not reproduced ones. Say so rather than claiming a reproduction.

### featurize dependency graph

Only three orderings are forced; the rest of the six steps are independent:

```
structure_prep ──> structure_tokens ─┐
sequence ────────────────────────────┤
mask ────────────────────────────────┼──> tars ──> embed
msa ─────────────────────────────────┘
```

Only `structure_tokens`, `msa` and `embed` accept `--device`, and their
defaults disagree (`msa` is `cpu`, the other two `cuda:0`), so a loop that
passes nothing builds a mixed-device feature set. Passing `--device` to the
other four is an argparse error. Split the loop:

```bash
for STEP in structure_prep sequence mask; do
  python -m proteus.featurize.$STEP --config $CFG --protein $KEY --depth $D
done
for STEP in structure_tokens msa; do
  python -m proteus.featurize.$STEP --config $CFG --protein $KEY --depth $D --device cuda:0
done
python -m proteus.featurize.tars  --config $CFG --protein $KEY --depth $D
python -m proteus.featurize.embed --config $CFG --protein $KEY --depth $D \
       --checkpoint <ckpt> --device cuda:0
```

On CPU, `embed` also needs `--dtype float32`; it refuses the default
`bfloat16` up front rather than failing inside the first matmul.

`structure_predict` is an opt-in replacement for `structure_prep` alone: it
folds each variant instead of mutating the wild type with PyMOL. Pick one: the
two write differently-named files into the same directory and `structure_tokens`
would read whichever sorts first. Each refuses to run if the other's files are
already there, but `structure_prep`'s half of that guard lives in its PyMOL
stage, so `--stage pml` still writes into the same directory.

## When something fails

Read `reference/pitfalls.md` before theorising. Every entry there is a failure
that actually happened, with the measurement that identified it. The common
shapes: a step "succeeds" but produced wrong features; a re-run reuses stale
intermediates; the wrong copy of a third-party checkpoint got loaded.

`proteus-audit` is the backstop: 24 checks for a short protein, 25 under
`crop: per-variant`, non-zero exit on any failure. A round that has not passed
it has not finished, however clean the logs looked.

## Reference

- `reference/commands.md`: every flag that changes results, and the defaults
- `reference/environments.md`: installing both environments, the caches, the GPU story
- `reference/pitfalls.md`: the measured failure modes, and how each is detected
