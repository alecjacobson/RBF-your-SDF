"""Reconstruction accuracy of PU dedup strategies, measured as in the paper (issue #10).

    uv run demo/additional_experiments/dedup_recon.py

Follows demo/SDF_to_surface_3D.py: samples come from generate_test_mesh_data
(grid, or uniform scatter), the fit uses the demo defaults (10 iterations,
clamp, reg=0, PU sphere partition, overlap 0.2, normalize=False), the surface
is extracted with dual contouring at the demo's adaptive resolution over the
sample bounding box, and accuracy is util.mesh_distances against the GT mesh
normalized the same way (Hausdorff, Chamfer, F1 at 1% of the GT diagonal).
Additionally reports the field error |F - SDF| at random points in the sample
box. Only the PU dedup settings vary between rows.
"""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))           # demo/
sys.path.insert(0, str(HERE))
import _common                                  # noqa: E402
import rbfyoursdf                               # noqa: E402
import SDF_to_surface_3D as sdf3d               # noqa: E402
from util import mesh_distances                 # noqa: E402

CONFIGS = {
    'greedy 5e-4 (baseline)': dict(pu_dedup_mode='greedy', pu_dedup_tol=5e-4),
    'greedy 2e-3': dict(pu_dedup_mode='greedy', pu_dedup_tol=2e-3),
    'greedy 5e-3': dict(pu_dedup_mode='greedy', pu_dedup_tol=5e-3),
    'greedy 1e-2': dict(pu_dedup_mode='greedy', pu_dedup_tol=1e-2),
    'adaptive 0.01': dict(pu_dedup_mode='adaptive', pu_dedup_ratio=0.01, pu_dedup_tol=5e-4),
    'adaptive 0.03': dict(pu_dedup_mode='adaptive', pu_dedup_ratio=0.03, pu_dedup_tol=5e-4),
    'adaptive 0.01 + 2e-3 tangent floor': dict(pu_dedup_mode='adaptive', pu_dedup_ratio=0.01, pu_dedup_tol=2e-3),
    'adaptive 0.01 + 5e-3 tangent floor': dict(pu_dedup_mode='adaptive', pu_dedup_ratio=0.01, pu_dedup_tol=5e-3),
    'reg 1e-6 (reference)': dict(pu_dedup_mode='greedy', pu_dedup_tol=5e-4, reg=1e-6),
    'reg 1e-5 (reference)': dict(pu_dedup_mode='greedy', pu_dedup_tol=5e-4, reg=1e-5),
}
INPUTS = [('bunny', 30, False), ('fandisk', 30, False), ('armadillo', 30, False),
          ('horse', 30, False), ('bunny', 30, True)]
OUT = HERE / 'results' / 'dedup_recon'


def true_sdf(mesh, P):
    import igl
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int32)
    d2, _, _ = igl.point_mesh_squared_distance(P, V, F)
    w = igl.winding_number(V, F, P)
    return np.sqrt(d2) * np.where(w > 0.5, -1.0, 1.0)


def run(name, n, scatter, kw):
    import trimesh
    sdf3d.seed = _common.SEED
    gt = _common.gt_mesh(name)
    _, pts, d, _ = sdf3d.generate_test_mesh_data(str(_common.REPO / 'examples' / f'{name}.obj'),
                                                 name, grid_len=n, scatter=scatter)
    t = time.perf_counter()
    args = dict(max_iters=10, clamp=True, reg=0, interp_partition='sphere', interp_overlap=0.2,
                verbose=False, normalize=False)
    args.update(kw)                             # a config may override a demo default (e.g. reg)
    res = rbfyoursdf.main_algorithm(pts, d, **args)
    t_fit = time.perf_counter() - t
    I = res.interpolator
    lo, hi = pts.min(0), pts.max(0)
    r = sdf3d._adaptive_resolution(pts, n)
    V, F = I.extract_surface(lo, hi, r, r, r, use_dual_contouring=True, lipschitz_postfix=False)
    np.random.seed(0)                           # mesh_distances samples both surfaces
    haus, cham, f1 = mesh_distances(trimesh.Trimesh(V, F), gt)
    Q = np.random.default_rng(1).uniform(lo, hi, (20000, 3))
    ferr = np.abs(I.predict(Q) - true_sdf(gt, Q))
    st = I.interpolator.export_lipschitz_state()
    A = [np.abs(p['alpha']).sum() for p in st['patches']]
    return {'hausdorff': haus, 'chamfer': cham, 'f1': f1,
            'field_err_median': float(np.median(ferr)), 'field_err_p99': float(np.quantile(ferr, 0.99)),
            'sample_err_max': float(np.abs(I.predict(pts) - d).max()),
            'constraints': int(sum(len(p['alpha']) for p in st['patches'])),
            'sum_abs_alpha_median': float(np.median(A)), 'faces': int(len(F)), 'fit_s': t_fit}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'results.json'
    res = json.loads(path.read_text()) if path.exists() else {}
    for name, n, scatter in INPUTS:
        key = f'{name}_{n}' + ('_scatter' if scatter else '')
        for cname, kw in CONFIGS.items():
            if cname in res.get(key, {}):
                continue
            r = run(name, n, scatter, kw)
            res.setdefault(key, {})[cname] = r
            path.write_text(json.dumps(res, indent=1))
            print(f"{key:16s} {cname:22s} chamfer={r['chamfer']:.3e} hausdorff={r['hausdorff']:.3e} "
                  f"f1={r['f1']:.4f} field med/p99={r['field_err_median']:.1e}/{r['field_err_p99']:.1e} "
                  f"sample_err={r['sample_err_max']:.1e} A med={r['sum_abs_alpha_median']:.2g} fit={r['fit_s']:.0f}s",
                  flush=True)
    with open(OUT / 'results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['chamfer', 'hausdorff', 'f1', 'field_err_median', 'field_err_p99', 'sample_err_max',
                'constraints', 'sum_abs_alpha_median', 'faces', 'fit_s']
        w.writerow(['input', 'config'] + cols)
        for k, rows in res.items():
            for c, r in rows.items():
                w.writerow([k, c] + [r[x] for x in cols])


if __name__ == '__main__':
    main()
