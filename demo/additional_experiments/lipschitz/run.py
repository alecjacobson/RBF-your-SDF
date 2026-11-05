"""Run the Lipschitz-bound experiment on one input and save machine-readable results.

    uv run --group lipschitz python -m lipschitz.run sphere 24      # from demo/additional_experiments
    uv run --group lipschitz demo/additional_experiments/lipschitz/run.py bunny 30

Stages (all on the same frozen snapshot, normalized frame, sample bounding box):

1. fit + snapshot, 2. reference-vs-native validation and derivative checks,
3. per-patch diagnostics and local bounds, 4. full-field bounds on regular
cell grids with observed lower bounds, 5. transition probes, 6. conservative
cell rejection versus the unpruned query.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'lipschitz'

from lipschitz import interval as iv                                   # noqa: E402
from lipschitz.adapter import (snapshot_from_native, evaluate as ref_eval,  # noqa: E402
                               nearest_support_patch, Snapshot)
from lipschitz.bounds import (fit_constants, b0_cert, b2_cert, b3_cert, b0_estimate,  # noqa: E402
                              b2_estimate, b3_estimate, evaluate as local_eval)
from lipschitz import pu_bounds as pub                                 # noqa: E402
from lipschitz.certify import (regular_cells, adaptive_prune, corner_points,  # noqa: E402
                               sign_change_violations, METHODS)
from lipschitz.inputs import make_input, surface_points, REPO                          # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'lipschitz'


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def quantiles(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {'n': 0}
    return {'n': int(len(x)), 'median': float(np.median(x)), 'p90': float(np.quantile(x, 0.9)),
            'p99': float(np.quantile(x, 0.99)), 'max': float(x.max()), 'min': float(x.min())}


def timed(fn, reps=3, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def manifest(args, desc, snap, t_fit):
    import rbfyoursdf
    def sh(c):
        try:
            return subprocess.check_output(c, cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {'command': ' '.join(sys.argv), 'commit': sh(['git', 'rev-parse', 'HEAD']),
            'dirty': bool(sh(['git', 'status', '--porcelain', '--untracked-files=no'])),
            'input': desc, 'snapshot_sha256': snap.digest(), 'num_patches': snap.num_patches,
            'normalization': {'center': snap.transform_center, 'scale': snap.transform_scale,
                              'note': 'x_norm = (x - center)/scale; F = scale*F_norm; gradients equal'},
            'policy': snap.policy, 'fit_seconds': t_fit, 'python': sys.version.split()[0],
            'numpy': np.__version__, 'platform': platform.platform(),
            'machine': platform.machine(), 'cpu_count': os.cpu_count(),
            'openmp': bool(rbfyoursdf.has_openmp()), 'arithmetic': 'float64 RN; outward-rounded intervals via nextafter',
            'args': vars(args)}


# ── stage 1 ─────────────────────────────────────────────────────────

def fit(name, n, max_iters, fit_kwargs=None):
    import rbfyoursdf
    pts, vals, desc = make_input(name, n)
    t = time.perf_counter()
    res = rbfyoursdf.main_algorithm(pts, vals, max_iters=max_iters, verbose=False, **(fit_kwargs or {}))
    t_fit = time.perf_counter() - t
    desc = dict(desc, fit_kwargs=fit_kwargs or {})
    snap = snapshot_from_native(res.interpolator, meta=desc)
    native = res.interpolator.interpolator           # normalized frame
    lo = (pts.min(0) - snap.transform_center) / snap.transform_scale
    hi = (pts.max(0) - snap.transform_center) / snap.transform_scale
    xs = (pts - snap.transform_center) / snap.transform_scale
    vs = vals / snap.transform_scale
    return snap, native, (lo, hi), (xs, vs), t_fit


# ── stage 2 ─────────────────────────────────────────────────────────

def stratified_points(snap, native, dom, rng, n=1500):
    lo, hi = dom
    strata = {}
    strata['uniform'] = lo + (hi - lo) * rng.random((n, 3))
    lat = lo + (hi - lo) * rng.random((40 * n, 3))
    F, _, info = ref_eval(snap, lat, return_info=True)
    scale = np.abs(F).max()
    strata['near_zero'] = lat[np.argsort(np.abs(F))[:n]]
    strata['overlap_ge4'] = lat[info['n_active'] >= 4][:n]
    W = info['W']
    strata['small_denominator'] = lat[(W > 0)][np.argsort(W[W > 0])[:n]]
    strata['fallback'] = lat[info['branch'] == 1][:n]
    j = rng.integers(0, snap.num_patches, n)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    sgn = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    pb = snap.patch_centers[j] + u * (snap.patch_radii[j] * (1 + sgn * 1e-7))[:, None]
    strata['support_boundary'] = pb[np.all((pb >= lo) & (pb <= hi), axis=1)]
    return strata


def validate(snap, native, dom, rng):
    out = {}
    strata = stratified_points(snap, native, dom, rng)
    for name, X in strata.items():
        if len(X) == 0:
            out[name] = {'n': 0}
            continue
        Fr, Gr, info = ref_eval(snap, X, return_info=True)
        Fn, Gn = native.predict_with_gradients(X)
        Fp = native.predict(X)
        Gp = native.predict_gradients(X)
        gs = np.maximum(np.linalg.norm(Gr, axis=1), 1e-300)
        out[name] = {'n': int(len(X)),
                     'value_absdiff_max': float(np.abs(Fr - Fn).max()),
                     'value_predict_vs_fused_max': float(np.abs(Fp - Fn).max()),
                     'grad_absdiff_max': float(np.linalg.norm(Gr - Gn, axis=1).max()),
                     'grad_reldiff_median': float(np.median(np.linalg.norm(Gr - Gn, axis=1) / gs)),
                     'grad_predict_vs_fused_max': float(np.linalg.norm(Gp - Gn, axis=1).max()),
                     'fallback_fraction': float((info['branch'] == 1).mean()),
                     'nearest_ties': int(info['tie'].sum())}
    # derivative check: central differences of the native value at smooth points
    X = strata['uniform']
    _, _, info = ref_eval(snap, X, return_info=True)
    X = X[(info['branch'] == 0) & (info['W'] > 1e-3)][:400]
    _, Gn = native.predict_with_gradients(X)
    fd = {}
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        G = np.zeros_like(X)
        for k in range(3):
            e = np.zeros(3); e[k] = h
            G[:, k] = (native.predict(X + e) - native.predict(X - e)) / (2 * h)
        fd[str(h)] = float(np.median(np.linalg.norm(G - Gn, axis=1) / np.linalg.norm(Gn, axis=1)))
    out['central_difference_relerr_median'] = fd
    return out, strata


# ── stage 3 ─────────────────────────────────────────────────────────

def patch_diagnostics(snap, cert, samples):
    xs, vs = samples
    rows = []
    for j, (fit, k) in enumerate(zip(snap.fits, cert.consts)):
        d = np.linalg.norm(fit.centers - snap.patch_centers[j], axis=1)
        rows.append({'n_centers': len(fit.centers), 'R': float(snap.patch_radii[j]),
                     'A_sum_abs_alpha': k.A_up, 'max_abs_alpha': float(np.abs(fit.alpha).max()),
                     'center_spread_rms': float(np.sqrt(k.S_est / max(k.A_up, 1e-300))),
                     'affine_grad': k.linear_norm_est, 'm0': max(abs(k.m0[0]), abs(k.m0[1])),
                     'm1': k.m1_norm_up, 'B1_ideal': k.linear_norm_est + 3 * k.S_est,
                     'centers_outside_support': int((d > snap.patch_radii[j]).sum())})
    F, _ = ref_eval(snap, xs)
    interp = np.abs(F - vs)
    keys = rows[0].keys()
    summary = {key: quantiles([r[key] for r in rows]) for key in keys}
    return rows, summary, {'abs_error_at_samples': quantiles(interp)}


def local_bound_study(snap, cert, dom, rng, n_patches=40, grids=(4, 8, 16, 32, 64)):
    """Certified B0/B2/B3 and estimates for single local fits versus observed maxima."""
    lo_d, hi_d = dom
    pick = rng.choice(snap.num_patches, min(n_patches, snap.num_patches), replace=False)
    res = {}
    for g in grids:
        lo, hi, _ = regular_cells(lo_d, hi_d, g)
        x0 = 0.5 * (lo + hi)
        rec = {k: [] for k in ('B0', 'B1', 'B2', 'B3', 'B0_est', 'B2_est', 'B3_est', 'obs', 'rho')}
        for j in pick:
            fit, k = snap.fits[j], cert.consts[j]
            dmin = np.linalg.norm(np.maximum(np.maximum(lo - snap.patch_centers[j],
                                                        snap.patch_centers[j] - hi), 0), axis=1)
            sel = np.flatnonzero(dmin <= snap.patch_radii[j])
            if len(sel) > 60:
                sel = rng.choice(sel, 60, replace=False)
            if len(sel) == 0:
                continue
            l, h = lo[sel], hi[sel]
            t = np.linspace(0, 1, 4)
            T = np.stack(np.meshgrid(t, t, t, indexing='ij'), -1).reshape(-1, 3)
            P = (l[:, None, :] + (h - l)[:, None, :] * T[None]).reshape(-1, 3)
            _, gl = local_eval(fit, P)
            obs = np.linalg.norm(gl, axis=1).reshape(len(sel), -1).max(1)
            rec['B0'] += list(b0_cert(fit, l, h, k)); rec['B2'] += list(b2_cert(fit, l, h, k))
            rec['B3'] += list(b3_cert(fit, l, h, k))
            rec['B1'] += [k.linear_norm_est + 3 * k.S_est] * len(sel)
            rec['B0_est'] += list(b0_estimate(fit, l, h)); rec['B2_est'] += list(b2_estimate(fit, l, h, k))
            rec['B3_est'] += list(b3_estimate(fit, l, h))
            rec['obs'] += list(obs)
            rec['rho'] += list(0.5 * np.linalg.norm(h - l, axis=1))
        rec = {k: np.asarray(v) for k, v in rec.items()}
        obs = np.maximum(rec['obs'], 1e-300)
        res[g] = {'n_boxes': int(len(obs)), 'rho': float(np.median(rec['rho'])),
                  'ratio': {m: quantiles(rec[m] / obs) for m in ('B0', 'B1', 'B2', 'B3')},
                  'ratio_min_cert': quantiles(np.minimum(np.minimum(rec['B0'], rec['B2']), rec['B3']) / obs),
                  'cert_over_estimate': {m: quantiles(rec[m] / np.maximum(rec[m + '_est'], 1e-300))
                                         for m in ('B0', 'B2', 'B3')},
                  'B1_violations': int((rec['B1'] < rec['obs']).sum()),
                  'obs': quantiles(rec['obs'])}
    return res


# ── stage 4 ─────────────────────────────────────────────────────────

def observed_cell_max(native, dom, n, sub=2):
    """Per-cell max |grad F| over a (sub+1)^3 lattice in each closed cell (native evaluator)."""
    lo, hi = dom
    m = n * sub + 1
    axes = [np.linspace(lo[k], hi[k], m) for k in range(3)]
    X = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    _, G = native.predict_with_gradients(X)
    gn = np.linalg.norm(G, axis=1).reshape(m, m, m)
    out = np.zeros((n, n, n))
    for a in range(sub + 1):
        for b in range(sub + 1):
            for c in range(sub + 1):
                out = np.maximum(out, gn[a:a + n * sub:sub, b:b + n * sub:sub, c:c + n * sub:sub])
    return out


def optimized_lower(native, lo, hi, rng, starts=3):
    """Locally maximize |grad F| in a cell from several starts (lower bound only)."""
    from scipy.optimize import minimize
    best = 0.0
    def obj(x):
        _, g = native.predict_with_gradients(x[None])
        return -float(np.dot(g[0], g[0]))
    for _ in range(starts):
        x = lo + (hi - lo) * rng.random(3)
        r = minimize(obj, x, method='L-BFGS-B', bounds=list(zip(lo, hi)),
                     options={'maxiter': 30, 'eps': 1e-7 * float(np.max(hi - lo))})
        best = max(best, np.sqrt(-r.fun), np.sqrt(-obj(x)))
    return best


def grid_study(snap, cert, native, dom, n, rng, subset=None, n_opt=60):
    lo, hi, ijk = regular_cells(*dom, n)
    if subset is not None and subset < len(lo):
        sel = rng.choice(len(lo), subset, replace=False)
        lo, hi, ijk = lo[sel], hi[sel], ijk[sel]
    t = time.perf_counter()
    out = cert.certify(lo, hi)
    t_cert = time.perf_counter() - t
    if subset is None:
        obs = observed_cell_max(native, dom, n)[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    else:
        tt = np.linspace(0, 1, 3)
        T = np.stack(np.meshgrid(tt, tt, tt, indexing='ij'), -1).reshape(-1, 3)
        P = (lo[:, None] + (hi - lo)[:, None] * T[None]).reshape(-1, 3)
        _, G = native.predict_with_gradients(P)
        obs = np.linalg.norm(G, axis=1).reshape(len(lo), -1).max(1)
    st = out['status']
    res = {'cells': int(len(lo)), 'subset': subset, 'certify_seconds': t_cert,
           'cost_bound_terms': int(out['cost_bound_terms']),
           'status_counts': {pub.STATUS_NAMES[s]: int((st == s).sum()) for s in pub.STATUS_NAMES},
           'n_active': quantiles(out['n_active']), 'W_min': quantiles(out['W_min'][st == pub.CERTIFIED]),
           'zero_observed': int((obs == 0).sum())}
    ok = (st != pub.UNRESOLVED) & (obs > 0)
    res['ratio'] = {k: quantiles(out[k][ok] / obs[ok]) for k in
                    ('L_best', 'L_pu', 'L_interval', 'L_simple', 'L_B0', 'L_local_only', 'L_local_B0',
                     'L_local_B2', 'L_local_B3')}
    res['violations'] = {k: int((out[k][ok] < obs[ok]).sum())
                         for k in ('L_best', 'L_pu', 'L_interval', 'L_simple', 'L_B0')}
    res['interval_wins_fraction'] = float((out['L_interval'][ok] < out['L_pu'][ok]).mean()) if ok.any() else None
    res['local_only_below_observed'] = int((out['L_local_only'][ok] < obs[ok]).sum())
    cert_ok = st == pub.CERTIFIED
    res['blend_share_best'] = quantiles(out['blend_best'][cert_ok] /
                                        np.maximum(out['L_best'][cert_ok], 1e-300))
    # optimized lower bounds on a few certified cells
    cand = np.flatnonzero(ok)
    if len(cand):
        pick = rng.choice(cand, min(n_opt, len(cand)), replace=False)
        opt = np.array([optimized_lower(native, lo[i], hi[i], rng) for i in pick])
        best_lower = np.maximum(opt, obs[pick])
        res['optimized'] = {'n': int(len(pick)), 'gain_over_lattice': quantiles(best_lower / obs[pick]),
                            'ratio_L_best': quantiles(out['L_best'][pick] / best_lower),
                            'violations': int((out['L_best'][pick] < best_lower).sum())}
    arrays = {'lo': lo, 'hi': hi, 'obs': obs, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)}}
    return res, arrays


# ── stage 5 ─────────────────────────────────────────────────────────

def transition_probes(snap, native, dom, rng, n=200_000):
    """Locate fallback regions, nearest-patch switches and union-boundary corners in the domain."""
    lo, hi = dom
    X = lo + (hi - lo) * rng.random((n, 3))
    _, _, info = ref_eval(snap, X, return_info=True)
    fb = info['branch'] == 1
    res = {'fallback_volume_fraction': float(fb.mean()), 'samples': n}
    witnesses = []
    # (a) nearest-patch switch inside the fallback region: bisection between two
    # fallback points with different nearest patches, then two-sided probes.
    Y = X[fb]
    if len(Y) >= 2:
        near, _ = nearest_support_patch(snap, Y)
        tree_pairs = 0
        for t in range(min(4000, len(Y))):
            a, b = rng.integers(0, len(Y), 2)
            if near[a] == near[b] or np.linalg.norm(Y[a] - Y[b]) > 0.05:
                continue
            pa, pb = Y[a].copy(), Y[b].copy()
            ja = near[a]
            ok = True
            for _ in range(60):
                mid = 0.5 * (pa + pb)
                Fm, _, im = ref_eval(snap, mid[None], return_info=True)
                if im['branch'][0] != 1:
                    ok = False
                    break
                jm, _ = nearest_support_patch(snap, mid[None])
                if jm[0] == ja:
                    pa = mid
                else:
                    pb = mid
            if not ok:
                continue
            seps, diffs = [], []
            dirv = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-300)
            c = 0.5 * (pa + pb)
            for s in 10.0 ** -np.arange(2, 10):
                F2, _ = ref_eval(snap, np.stack([c - 0.5 * s * dirv, c + 0.5 * s * dirv]))
                seps.append(s); diffs.append(abs(F2[1] - F2[0]))
            witnesses.append({'kind': 'fallback_nearest_switch', 'point': c,
                              'patches': [int(ja), int(nearest_support_patch(snap, pb[None])[0][0])],
                              'separation': seps, 'value_jump': diffs,
                              'limiting_jump_estimate': diffs[-1],
                              'difference_quotient_at_min_sep': diffs[-1] / seps[-1]})
            tree_pairs += 1
            if tree_pairs >= 5:
                break
    # (b) union-boundary corner: a point on two support spheres, inside no other.
    C, R = snap.patch_centers, snap.patch_radii
    from scipy.spatial import cKDTree
    tree = cKDTree(C)
    found = 0
    for (j, k) in tree.query_pairs(2 * R.max(), output_type='ndarray')[rng.permutation(
            len(tree.query_pairs(2 * R.max(), output_type='ndarray')))][:20000]:
        d = np.linalg.norm(C[k] - C[j])
        if not (abs(R[j] - R[k]) < d < R[j] + R[k]):
            continue
        e = (C[k] - C[j]) / d
        a = (d * d + R[j] ** 2 - R[k] ** 2) / (2 * d)
        rc = np.sqrt(max(R[j] ** 2 - a * a, 0))
        u = np.cross(e, rng.normal(size=3)); u /= np.linalg.norm(u)
        v = np.cross(e, u)
        for th in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            p = C[j] + a * e + rc * (np.cos(th) * u + np.sin(th) * v)
            if np.any(p < lo) or np.any(p > hi):
                continue
            others = np.linalg.norm(C - p, axis=1) < R * (1 - 1e-9)
            others[[j, k]] = False
            if others.any():
                continue
            nj = (p - C[j]) / R[j]; nk = (p - C[k]) / R[k]
            u1 = -(nj + 0.1 * nk); u1 /= np.linalg.norm(u1)
            u2 = -(0.1 * nj + nk); u2 /= np.linalg.norm(u2)
            seps, diffs = [], []
            for s in 10.0 ** -np.arange(2, 9):
                P = np.stack([p + s * u1, p + s * u2])
                F2, _, inf2 = ref_eval(snap, P, return_info=True)
                if np.any(inf2['branch'] != 0):
                    continue
                seps.append(float(np.linalg.norm(P[0] - P[1]))); diffs.append(abs(F2[1] - F2[0]))
            fj = local_eval(snap.fits[j], p[None])[0][0]
            fk = local_eval(snap.fits[k], p[None])[0][0]
            if len(diffs) >= 3:
                witnesses.append({'kind': 'union_boundary_corner', 'point': p, 'patches': [int(j), int(k)],
                                  'f_j_minus_f_k': float(fj - fk), 'separation': seps,
                                  'value_jump': diffs, 'limiting_jump_estimate': diffs[-1],
                                  'difference_quotient_at_min_sep': diffs[-1] / seps[-1]})
                found += 1
            break
        if found >= 5:
            break
    res['witnesses'] = witnesses
    return res


# ── stage 6 ─────────────────────────────────────────────────────────

def kernel_terms(snap, pts):
    """Kernel terms the blend evaluator spends on pts: sum over containing patches of their center counts."""
    from scipy.spatial import cKDTree
    if len(pts) == 0:
        return 0
    tree = cKDTree(pts)
    total = 0
    for j, f in enumerate(snap.fits):
        total += len(tree.query_ball_point(snap.patch_centers[j], snap.patch_radii[j])) * len(f.centers)
    return int(total)


def prune_study(snap, cert, native, dom, root_n, depth, rng, reps=3):
    lo, hi = dom
    leaf_n = np.array([root_n * 2 ** depth] * 3)
    allpts, _ = corner_points(lo, hi, leaf_n)
    t_full = timed(lambda: native.predict(allpts), reps=reps)
    ref_vals = native.predict(allpts)
    res = {'leaf_n': leaf_n, 'root_n': root_n, 'depth': depth,
           'baseline': {'field_evaluations': int(len(allpts)), 'seconds': t_full,
                        'kernel_terms': kernel_terms(snap, allpts)}}
    cache = {}
    for method in METHODS:
        pr = adaptive_prune(cert, lo, hi, root_n, depth, method, cache=cache)
        rejected = pr.leaf_label == 1
        retained = ~rejected
        pts, _ = corner_points(lo, hi, leaf_n, retained)
        t_eval = timed(lambda: native.predict(pts), reps=reps) if len(pts) else 0.0
        viol, n_sc = sign_change_violations(ref_vals, leaf_n, rejected)
        # independent probes inside rejected leaves: all values one-signed per leaf
        probe_viol = 0
        cells = np.argwhere(rejected)
        if len(cells):
            sel = cells[rng.choice(len(cells), min(3000, len(cells)), replace=False)]
            h = (hi - lo) / leaf_n
            P = lo + (sel[:, None, :] + rng.random((len(sel), 6, 3))) * h
            Fp, _ = ref_eval(snap, P.reshape(-1, 3))
            Fp = Fp.reshape(len(sel), 6)
            probe_viol = int(((Fp.min(1) <= 0) & (Fp.max(1) >= 0)).sum())
        res[method] = {
            'rejected_leaf_fraction': float(rejected.mean()),
            'unresolved_leaf_fraction': float((pr.leaf_label == 2).mean()),
            'retained_leaves': int(retained.sum()),
            'field_evaluations': int(len(pts)),
            'evaluations_saved_fraction': 1 - len(pts) / len(allpts),
            'boxes_certified': pr.n_boxes_certified,
            'bound_kernel_terms': int(pr.cost_bound_terms),
            'eval_kernel_terms': kernel_terms(snap, pts),
            'certify_seconds': pr.time_certify,
            'eval_seconds': t_eval,
            'total_seconds': pr.time_certify + t_eval,
            'speedup_vs_unpruned': t_full / max(pr.time_certify + t_eval, 1e-12),
            'sign_change_edges': n_sc, 'violations_sign_edges': viol,
            'violations_interior_probes': probe_viol,
            'per_level': pr.per_level}
        np.save(RESULTS_DIR / f'prune_labels_{method}.npy', pr.leaf_label)
    return res


RESULTS_DIR = RESULTS


def main(argv=None):
    global RESULTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('grid', type=int)
    ap.add_argument('--max-iters', type=int, default=10)
    ap.add_argument('--grids', type=int, nargs='*', default=[32, 64])
    ap.add_argument('--subset-grid', type=int, default=128)
    ap.add_argument('--subset-cells', type=int, default=20000)
    ap.add_argument('--prune-root', type=int, default=4)
    ap.add_argument('--prune-depth', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip', nargs='*', default=[])
    ap.add_argument('--reg', type=float, default=None,
                    help='fit variant: RBF regularization (baseline: package default 0)')
    ap.add_argument('--dedup-tol', type=float, default=None,
                    help='fit variant: PU dedup tolerance (baseline: 5e-4)')
    ap.add_argument('--expand', type=float, default=0.0,
                    help='expand the query domain by this fraction per side (extrapolation stress)')
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    fit_kwargs = {}
    if args.reg is not None:
        fit_kwargs['reg'] = args.reg
    if args.dedup_tol is not None:
        fit_kwargs['pu_dedup_tol'] = args.dedup_tol
    tag = (f'{args.input}_{args.grid}' + ''.join(f'_{k}{v:g}' for k, v in fit_kwargs.items())
           + (f'_expand{args.expand}' if args.expand else ''))
    RESULTS_DIR = RESULTS / tag
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def log(*a):
        print(f'[{tag} {time.strftime("%H:%M:%S")}]', *a, flush=True)

    snap, native, dom, samples, t_fit = fit(args.input, args.grid, args.max_iters, fit_kwargs)
    if args.expand:
        ext = (dom[1] - dom[0]) * args.expand
        dom = (dom[0] - ext, dom[1] + ext)
    snap.save(RESULTS_DIR / 'snapshot.npz')
    out = {'manifest': manifest(args, snap.meta, snap, t_fit), 'domain': dom}
    log('fit', f'{t_fit:.1f}s', snap.num_patches, 'patches', snap.digest()[:12])

    def save():
        (RESULTS_DIR / 'results.json').write_text(json.dumps(jsonable(out), indent=1))

    t = time.perf_counter()
    cert = pub.PUCertifier(snap)
    out['preprocess_seconds'] = time.perf_counter() - t
    out['pair_shared_fraction'] = quantiles(cert.pair_shared_fraction)
    log('preprocess', f"{out['preprocess_seconds']:.1f}s")

    if 'validate' not in args.skip:
        out['validation'], _ = validate(snap, native, dom, rng)
        save(); log('validation done')
    rows, summ, interp = patch_diagnostics(snap, cert, samples)
    out['patches'] = summ
    out['interpolation'] = interp
    sp = surface_points(args.input)
    Fs = native.predict((sp - snap.transform_center) / snap.transform_scale) * snap.transform_scale
    out['quality'] = {'abs_F_on_true_surface': quantiles(np.abs(Fs)),
                      'note': '|F| at true-surface samples, physical units (~ distance to the zero set)'}
    (RESULTS_DIR / 'patches.json').write_text(json.dumps(jsonable(rows)))
    if 'local' not in args.skip:
        out['local_bounds'] = local_bound_study(snap, cert, dom, rng)
        save(); log('local bounds done')
    if 'grid' not in args.skip:
        out['grids'] = {}
        for n in args.grids:
            r, arrays = grid_study(snap, cert, native, dom, n, rng)
            out['grids'][str(n)] = r
            np.savez_compressed(RESULTS_DIR / f'grid_{n}.npz', **arrays)
            save(); log('grid', n, f"{r['certify_seconds']:.1f}s", r['status_counts'])
        if args.subset_grid:
            r, arrays = grid_study(snap, cert, native, dom, args.subset_grid, rng, subset=args.subset_cells)
            out['grids'][f'{args.subset_grid}_subset'] = r
            np.savez_compressed(RESULTS_DIR / f'grid_{args.subset_grid}_subset.npz', **arrays)
            save(); log('grid subset done')
    if 'probes' not in args.skip:
        out['transitions'] = transition_probes(snap, native, dom, rng)
        save(); log('probes done', len(out['transitions']['witnesses']), 'witnesses')
    if 'prune' not in args.skip:
        out['prune'] = prune_study(snap, cert, native, dom, args.prune_root, args.prune_depth, rng)
        save(); log('prune done')
    save()
    return out


if __name__ == '__main__':
    main()
