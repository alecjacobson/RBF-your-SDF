"""Compare PU dedup strategies by hull-test pruning and reconstruction quality (issue #10).

    uv run --group lipschitz demo/additional_experiments/lipschitz/dedup_study.py

For each input and configuration: fit with main_algorithm, then report
* constraints kept, min center separation, per-patch sum |alpha| (median/max);
* max |F - d| at the input samples (exact-interpolation check);
* |F| at true-surface samples and distance from the extracted mesh (128^3
  marching cubes over the sample box) to the true surface, in units of the
  normalized sample box;
* hull-test pruning with the C++ certifier (hull-only mode) on octrees with
  64^3, 128^3 and 256^3 leaves: rejected / unresolved leaf fractions,
  certification time, sparse evaluation time at the retained leaves' corners,
  and the dense evaluation time of all corners (the sparse marching cubes
  comparison).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'lipschitz'

import rbfyoursdf                                                      # noqa: E402
from scipy.spatial import cKDTree                                      # noqa: E402
from lipschitz import pu_bounds as pub                                 # noqa: E402
from lipschitz.adapter import snapshot_from_native                     # noqa: E402
from lipschitz.certify import adaptive_prune, corner_points           # noqa: E402
from lipschitz.inputs import make_input, surface_points, ANALYTIC, REPO   # noqa: E402
from lipschitz.run import RESULTS, jsonable                            # noqa: E402

CONFIGS = {
    'greedy 5e-4 (baseline)': dict(pu_dedup_mode='greedy', pu_dedup_tol=5e-4),
    'greedy 5e-3': dict(pu_dedup_mode='greedy', pu_dedup_tol=5e-3),
    'adaptive 0.01': dict(pu_dedup_mode='adaptive', pu_dedup_ratio=0.01, pu_dedup_tol=5e-4),
    'adaptive 0.03': dict(pu_dedup_mode='adaptive', pu_dedup_ratio=0.03, pu_dedup_tol=5e-4),
    'reg 1e-6 (reference)': dict(pu_dedup_mode='greedy', pu_dedup_tol=5e-4, reg=1e-6),
}
DEPTHS = (4, 5, 6)          # 64^3, 128^3, 256^3 leaves from a 4^3 root
INPUTS = [('bunny', 30, False), ('fandisk', 30, False), ('bunny', 30, True), ('box', 24, False),
          ('sphere', 24, False)]


def truth_distance(name, P):
    """Unsigned distance from points (physical frame) to the true surface."""
    if name in ANALYTIC:
        return np.abs(ANALYTIC[name](P))
    import igl, trimesh
    mesh = trimesh.load(REPO / 'examples' / f'{name}.obj')
    lo, hi = mesh.vertices.min(0), mesh.vertices.max(0)
    V = (np.asarray(mesh.vertices) - (lo + hi) / 2) / np.max(hi - lo)
    d2, _, _ = igl.point_mesh_squared_distance(P, V, np.asarray(mesh.faces, dtype=np.int32))
    return np.sqrt(d2)


def q(x):
    x = np.asarray(x, float)
    return {'median': float(np.median(x)), 'p99': float(np.quantile(x, 0.99)), 'max': float(x.max())}


def one(name, n, scatter, kw):
    pts, vals, _ = make_input(name, n, scatter=scatter)
    t = time.perf_counter()
    res = rbfyoursdf.main_algorithm(pts, vals, max_iters=10, verbose=False, **kw)
    t_fit = time.perf_counter() - t
    I, native = res.interpolator, res.interpolator.interpolator
    snap = snapshot_from_native(I)
    A = np.array([np.abs(f.alpha).sum() for f in snap.fits])
    allc = np.unique(np.concatenate([f.centers for f in snap.fits]), axis=0)
    minsep = float(cKDTree(allc).query(allc, k=2)[0][:, 1].min())
    s = snap.transform_scale
    interp = np.abs(I.predict(pts) - vals) / s
    sp = surface_points(name)
    on_surf = np.abs(I.predict(sp)) / s
    lo, hi = pts.min(0), pts.max(0)
    V, F = I.extract_surface(lo, hi, 128, 128, 128)
    to_truth = truth_distance(name, np.asarray(V)) / s if len(V) else np.array([np.inf])
    # hull-test pruning in the normalized frame
    dlo, dhi = (lo - snap.transform_center) / s, (hi - snap.transform_center) / s
    cert = pub.NativeCertifier(native, hull_only=True)
    hull = {}
    for depth in DEPTHS:
        leaf_n = np.array([4 * 2 ** depth] * 3)
        allpts, _ = corner_points(dlo, dhi, leaf_n)
        t = time.perf_counter(); native.predict(allpts); t_dense = time.perf_counter() - t
        t = time.perf_counter()
        pr = adaptive_prune(cert, dlo, dhi, 4, depth, 'hull')
        t_cert = time.perf_counter() - t
        keep_pts, _ = corner_points(dlo, dhi, leaf_n, pr.leaf_label != 1)
        t = time.perf_counter(); native.predict(keep_pts); t_sparse = time.perf_counter() - t
        hull[int(leaf_n[0])] = {'rejected': float((pr.leaf_label == 1).mean()),
                                'unresolved': float((pr.leaf_label == 2).mean()),
                                'boxes_certified': pr.n_boxes_certified,
                                'certify_s': t_cert, 'sparse_eval_s': t_sparse, 'dense_eval_s': t_dense,
                                'speedup': t_dense / (t_cert + t_sparse)}
    return {'constraints': int(len(allc)), 'patches': snap.num_patches, 'min_center_sep': minsep,
            'sum_abs_alpha_median': float(np.median(A)), 'sum_abs_alpha_max': float(A.max()),
            'interp_err_max': float(interp.max()), 'abs_F_on_true_surface': q(on_surf),
            'mesh_to_truth': q(to_truth), 'hull': hull, 'fit_seconds': t_fit}


def main():
    out_path = RESULTS / 'dedup_study_hull.json'
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name, n, scatter in INPUTS:
        key = f'{name}_{n}' + ('_scatter' if scatter else '')
        for cname, kw in CONFIGS.items():
            if cname in res.get(key, {}):
                continue
            r = one(name, n, scatter, kw)
            res.setdefault(key, {})[cname] = r
            out_path.write_text(json.dumps(jsonable(res), indent=1))
            h = ' '.join(f"{n}^3:rej={v['rejected']:.2f},x{v['speedup']:.2f}" for n, v in r['hull'].items())
            print(f"{key:16s} {cname:22s} n={r['constraints']:6d} A med={r['sum_abs_alpha_median']:.2g} "
                  f"interp={r['interp_err_max']:.1e} surf med/p99={r['abs_F_on_true_surface']['median']:.1e}/"
                  f"{r['abs_F_on_true_surface']['p99']:.1e} mesh med/p99={r['mesh_to_truth']['median']:.1e}/"
                  f"{r['mesh_to_truth']['p99']:.1e} fit={r['fit_seconds']:.0f}s | {h}", flush=True)


if __name__ == '__main__':
    main()
