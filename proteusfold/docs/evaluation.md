# Evaluation

This fork does not distribute evaluation tooling. Upstream's `evaluation.sh`
referenced a vendored `extern/PXMeter/` that was not part of this tree, so it
was left out rather than shipped broken, and the benchmark predictions and
input sets it consumed are not here either, being hundreds of gigabytes of
generated output.

For upstream's benchmarks, predictions and the scripts that produce them, see
the [Boltz repository](https://github.com/jwohlwend/boltz) and the Boltz-1 and
Boltz-2 technical reports.

What this tree does carry, for physical-plausibility scoring of structures you
have predicted yourself:

```bash
python extern/PhysProtein/physcialsim_metrics.py \
    --model_dir out/boltz_results_x/predictions \
    --output_csv metrics.csv --output_txt metrics.txt
```

It needs `ccd.pkl` and `mols/` in `$BOLTZ_CACHE` (or `~/.boltz`), which a
prediction run will already have downloaded. `scripts/eval/physcialsim_metrics.py`
is upstream's earlier version of the same file and still hardcodes four
directories on the upstream authors' cluster plus a cache path; it cannot run
anywhere else. Use the `extern/` one.
