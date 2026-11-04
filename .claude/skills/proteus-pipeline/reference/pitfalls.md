# Measured failure modes

Every entry is something that actually happened, with the evidence that
identified it. The pattern they share: the command exits 0 and the output looks
reasonable. Check for these before believing a number.

## A step succeeds and the features are wrong

**The wrong copy of a third-party checkpoint.** `torch.hub`'s cache is a flat
namespace keyed on filename, so anything that drops
`esm_msa1b_t12_100M_UR50S.pt` there is what gets loaded. On one machine that
file was another project's unrelated weights: median cosine 0.0008 against the
published model, embeddings absmax 15.27 instead of 11.38. It loaded without
complaint. `proteus-doctor` detects it by shape
(`msa_position_embedding` is `(1, 1024, 1, 768)` in the published model) and
fails. Fix it, do not route around it; set `TORCH_HOME` if you keep several
caches.

**A patched `fair_esm`.** One install had `MSABatchConverter` hardcoded to pad
to 512 columns. Note which path that reaches, because the two MSA consumers do
not share a copy of fair-esm: `featurize.msa` imports the installed one and
gets the patch, while the zero-shot scorer loads its own unpatched copy from
`proteus.third_party.esm_fork` and is unaffected. Measured, its masked-marginal
loop ran 120 columns for a 119-residue query on the patched machine, not 513.
For `featurize.msa` an alignment of ≤512 columns is padded rather than changed,
so the embeddings still match stock fair-esm; past 512 it is truncated and they
do not. `proteus-doctor` probes the behaviour rather than the version number
and names the affected path.

**Insertions read as alignment columns.** The inherited ProteinGym MSA reader
assumes a2m geometry (rectangular, shared insertion columns). Given a true a3m
it truncates every row to the first L characters, with insertion characters
still occupying positions, and because the result is rectangular, nothing
errors. Measured on one 508-residue alignment: 88.3% of rows affected, 51.9%
of residue positions out of register. Fixed by default now; `--legacy-a3m`
restores it. Isolated effect on the MSA channel's Spearman: gb1 0.1065 →
0.1230, gfp 0.6669 → 0.6793. **The magnitude does not follow the insertion
fraction**: gb1 had 6.8% of rows affected and gained more than gfp's 79.4%.

## A re-run reuses stale state

**Stale `.pml` scripts.** `structure_prep` processes the whole `pmls/`
directory by glob. Each round has its own `rounds/round<D>/workspace/pmls/<KEY>`,
so the previous round cannot leak in. The real trigger is re-running
`proteus-candidates --overwrite` for the **same** depth, which replaces the
library while the earlier attempt's `.pml` and `.pdb` files are still sitting
in that round's workspace. Clear `workspace/pmls/<KEY>` and
`mutant_pdbs/<KEY>` before redoing a round at the same depth. Without
`--overwrite` the command refuses instead, which is the point of the guard.

**Two structure sources in one directory.** `structure_prep` writes
`<stem>_mutated.pdb`, `structure_predict` writes `<stem>.pdb`, and
`structure_tokens` globs `*.pdb`, so with both present it reads whichever
sorts first. Both directions are guarded now: each refuses to start if the
other's files are present. The remaining gap is that `structure_prep`'s guard
sits in its PyMOL stage, so `--stage pml` still writes into a directory that
holds predictions.

**A multi-chain wild-type PDB, where two code paths pick different chains.**
`featurize.structure_prep` mutates the chain that appears **first in the
file**; `zeroshot.scorer` reads the **alphabetically first** one. For a
single-chain file those are the same chain and nothing is wrong. For a
multi-chain file they can be different molecules, and the structure stream then
describes one chain while the zero-shot structure logits describe another.

`proteus-prepare` and `proteus-doctor` only **warn** about this. It does not
block, and it does not trigger the `.rejected` path, so it is on you to read
that warning. The detection is in `proteus/prepare/validate.py`, and the
warning names both chains so you can see whether they differ. Fix it by
stripping the PDB to the one chain the library's sequence describes.

**The metrics sidecar.** `proteus-score` names it after `--dms-file`, not
`--output-file`. Two arms on the same input overwrite each other's metrics.
This has already destroyed one comparison's numbers. Separate directories.

## A prediction is not reproducible the way you expect

**Folding depends on batch composition.** Same sequence, same seed, same
settings: confidence 0.8963 folded alone, 0.9022 folded in a batch of three,
and the two structures sit 2.76 Å CA-RMSD apart after superposition: a
different conformation, not a different frame. Folded alone twice it is
bit-identical. To reproduce a set of structures, shard it the way it was
sharded. `structure_predict` records the batch in its sidecar for this reason.

**A predicted structure is not the reference structure.** On CreiLOV a Boltz-2
prediction sits 2.2 Å CA-RMSD from CombinGym's. Downstream: 173/173 structure
tokens differ, a mean 15.8% of positions per variant; trunk embeddings cosine
0.9991/0.9988; 28 of 30 picks survive, `rank_base` ρ=1.0000, `rank_zs` 0.9879,
`rank_rf` 0.8878. Switching structure source changes the batch, not
catastrophically, but it is a different run.

**CPU and GPU are not bit-identical.** max|diff| 5.7e-06 on the MSA
embeddings. Do not mix devices within a campaign.

## Environment

**scikit-learn version changes the picks.** The surrogate is a RandomForest;
sklearn 1.7.2 selects differently from 1.6.1. The lockfile constrains it for
this reason. Three independently built environments selected the **same 30
picks** with `max|diff| = 0` on every numeric column. That is the bar.

**PyMOL has to actually run, not just be on PATH.** A Debian `pymol` wrapper
invoking bare `python3` gets intercepted by an active virtualenv and fails
every time. `proteus-doctor` executes it rather than looking it up.

**The predictor's cache is part of its configuration.** `--boltz-cache`
selects which checkpoint is loaded, by directory rather than by content. A
cold cache downloads about 6.2 GB (measured), including a 2.06 GB affinity
checkpoint nothing here uses, and the CCD download is keyed on `mols.tar`
alone, and an extracted `mols/` does not prevent re-downloading it.

## Known not reproducible

- Re-featurizing **aav** today gives different structure tokens from the
  published numbers: those arrays predate a crop-window fix and used a
  per-variant window with a clamped right edge.
- The published aav zero-shot numbers were produced before the zero-shot path
  was given its own vendored `esm_fork`, under a `fair_esm` carrying the 512
  patch, at L=735 > 512. The path as it stands today is not patch-sensitive
  (that is the point of the vendored copy), so re-running aav now does not
  reproduce those numbers, not because the patch still bites, but because it
  no longer does.
