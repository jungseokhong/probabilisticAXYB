# MATLAB/Python parity

`run_parity.m` calls the original MATLAB likelihood, gradient, uncertainty, and
solver functions. The Python driver generates a deterministic shared `.mat`
fixture, invokes this function in MATLAB batch mode, compares every output, and
deletes the temporary files afterward.

From the repository root:

```bash
python -m benchmarks.matlab_parity
```

Use `--keep-files` to copy the exact input and MATLAB output into
`matlab_parity_artifacts/` for manual inspection. Equation-level differences
above `1e-8`, SciPy transform errors above `1e-8`, or other solver errors above
`1e-2` make the command fail.
