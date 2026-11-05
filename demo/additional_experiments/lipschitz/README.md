# Lipschitz-bound experiment

Implements the plan in `Lipschitz Bound Experiment for RBF Your SDF.md` (repository
root): freeze a fitted partition-of-unity reconstruction, compare certified upper
bounds on its gradient with observed lower bounds, and measure conservative
rejection of cells that cannot meet its zero set. The findings are in
[`REPORT.md`](REPORT.md).

```bash
uv sync --group lipschitz
uv run python -m unittest tests/test_lipschitz_bounds.py          # algebra + regression tests
uv run --group lipschitz demo/additional_experiments/lipschitz/run.py sphere 24
uv run --group lipschitz demo/additional_experiments/lipschitz/run.py bunny 30
uv run --group lipschitz demo/additional_experiments/lipschitz/report.py
```

Results go to `demo/additional_experiments/results/lipschitz/<input>_<grid>[_variant]/`
(`results.json` with the run manifest, `snapshot.npz` with its SHA-256, per-patch
metrics, per-cell certificates, prune label volumes). `report.py` collects every
run into tables and figures under `results/lipschitz/report/`.

| File | Responsibility |
|---|---|
| `adapter.py` | Snapshot of the native fit via `PUInterpolator.export_lipschitz_state()`; independent reference evaluator reproducing the native branch rules |
| `interval.py` | Outward-rounded float64 interval arithmetic (`nextafter` per operation, Higham's a priori bound for reductions) |
| `bounds.py` | Local fit derivatives and bounds B0 (triangle), B1 (idealized moment, diagnostic), B2 (residual-aware moment), B3 (cell Taylor), value-range enclosure |
| `pu_bounds.py` | Support enumeration, certified `W_min`, weight-gradient bounds, patch disagreement, blend bounds, direct interval quotient, coverage statuses |
| `certify.py` | Octree traversal, rejection rules, sign-change validation |
| `inputs.py` | Analytic SDFs (sphere, box, torus, thin shell) and the mesh sampling driver |
| `run.py` | One reproducible run: fit, validate, local/full-field bounds, probes, pruning |
| `report.py` | Tables and figures |

## Conventions

* **Labels.** *Observed lower bound* = sampled/optimized gradient norm of the
  native evaluator (after it was checked against the reference evaluator and
  central differences). *Analytic estimate* = `*_estimate` functions in
  float64. *Certified upper bound* = everything returned by `*_cert` and
  `PUCertifier.certify`, computed with outward rounding over exact stored
  coefficients.
* **Frame.** Everything is computed in the normalized frame of the C++ fit
  (`x' = (x - center)/scale`, `F = scale * F'`). Gradients and Lipschitz
  constants are identical in both frames, so all bounds are dimensionless.
* **Domain.** The sample bounding box (normalized), unless `--expand` is given.
* **Statuses.** `certified` (W_min > 0 on the box, blend formula everywhere),
  `fallback_fixed` (the box misses every support and one patch is provably the
  nearest), `unresolved` (anything else: never given a finite bound).
* **Only the sphere partition** (the Python default) is modelled. The box
  partition raises `NotImplementedError` rather than being silently mis-bounded.
