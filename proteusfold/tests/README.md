# tests

```bash
pip install -e ".[test]"
pytest
```

That runs `tests/model/`, which is what is collectible here: two chunked-attention
equivalence tests for `OuterProductMean` and triangle attention. Both pass against
this tree (measured).

The two files beside them are upstream's and are left as they came, but neither
runs under `pytest`, so `testpaths` in `pyproject.toml` keeps them out of the
default run:

- `test_regression.py` does `from boltz.main import MODEL_URL`. That name is not
  in `boltz/main.py` at the commit this fork was taken from, so the module does
  not import. It is written against a different upstream revision.
- `test_kernels.py` is a Triton speed benchmark, not a test: it calls
  `benchmark.run(...)` at module level, so merely importing it starts a GPU
  sweep, and it needs `matplotlib` and a CUDA device. Run it directly
  (`python tests/test_kernels.py`) if you want the numbers.

`profiling.py` and `test_utils.py` are helpers for those two.

The fork's four dataset-inspection scripts take positional arguments and are
command-line tools, not tests, so they sit in `tools/` as `check_dataset.py`,
`check_evaluation.py`, `check_output.py` and `check_supported_data.py`. Each
expects the `dataset/` layout described in its own docstring; that directory is
not distributed with this repository.
