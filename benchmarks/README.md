# Backend benchmark

`benchmark_backends.py` creates a deterministic, noiseless `AX=YB` problem and
runs every installed backend from the same perturbed initial estimate. It reports
wall time, iterations or objective evaluations, likelihood, and errors against
the known transformations.

```bash
python -m benchmarks.benchmark_backends \
    --measurements 20 \
    --max-iterations 2000 \
    --tolerance 1e-10 \
    --repeats 3
```

The first run is labeled `cold`; later runs are used for the warm mean. To force
Numba to compile rather than load its disk cache, point `NUMBA_CACHE_DIR` to a
new empty directory before running the command.
