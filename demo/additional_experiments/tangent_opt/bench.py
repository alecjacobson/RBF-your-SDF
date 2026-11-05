"""Benchmark tangent-direction optimizers on a frozen reconstruction (issue #11).

    uv run --group lipschitz demo/additional_experiments/tangent_opt/bench.py bunny 30

Per sample (z, d) the paper (Sec. 4.2) minimizes, over unit directions x,

    f(x) = D(q) / d,   q = z - d x,

whose Euclidean gradient is -grad D(q) and Euclidean Hessian d * Hess D(q).
On S^2: Riemannian gradient P grad f (P = I - x x^T); Riemannian Hessian
P Hess f P - (x . grad f) P; retraction R(x, v) = normalize(x + v).

Every method runs on the same frozen interpolator (fitted once with
main_algorithm), from the same starts, to the same tolerance |P grad f| <= 1e-4
(the default solver's own). All methods advance all active points in lockstep
and make one batched native evaluation per round, like the production solver.
Points retire when converged, so later rounds are cheaper. Counted per method:
rounds (batched calls), evaluated rows by kind, native evaluation time and
Python overhead. The production C++ batched BFGS is timed directly as a
reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rbfyoursdf                                                     # noqa: E402
from lipschitz.inputs import make_input                               # noqa: E402

TOL = 1e-4
RESULTS = Path(__file__).resolve().parents[1] / 'results' / 'tangent_opt'


class Oracle:
    """Batched native evaluations with counting and timing."""

    def __init__(self, native, z, d):
        self.native, self.z, self.d = native, z, d
        self.calls = 0
        self.rows = {'grad': 0, 'hess': 0}
        self.seconds = 0.0

    def __call__(self, idx, x, hess=False):
        q = self.z[idx] - self.d[idx, None] * x
        t = time.perf_counter()
        if hess:
            v, g, H = self.native.predict_hessians(q)
            H = H.reshape(-1, 3, 3)
        else:
            v, g = self.native.predict_with_gradients(q)
            H = None
        self.seconds += time.perf_counter() - t
        self.calls += 1
        self.rows['hess' if hess else 'grad'] += len(idx)
        inv = 1.0 / self.d[idx]
        f = v * inv
        G = -g                                    # Euclidean grad of f
        Hf = None if H is None else H * self.d[idx, None, None]
        return f, G, Hf


def proj(x, v):
    return v - np.einsum('ij,ij->i', x, v)[:, None] * x


def retract(x, v):
    y = x + v
    return y / np.linalg.norm(y, axis=1, keepdims=True)


def tnorm(x, G):
    return np.linalg.norm(proj(x, G), axis=1)


# ── methods ─────────────────────────────────────────────────────────
# Each takes (oracle, x0, max_rounds) and returns (x, converged mask, rounds
# at which each point converged (-1 if not)). The Armijo-based methods use the
# same lockstep trial/accept scheme as the production BFGS: one trial point per
# active point per round, backtracking by halving on rejection.

def _armijo_driver(oracle, x0, max_rounds, direction, cap=1.0, first_cap=0.2, need_hess=False,
                   on_accept=None):
    N = len(x0)
    x = x0.copy()
    xt = x0.copy()
    fk = np.zeros(N); gk = np.zeros((N, 3)); pk = np.zeros((N, 3)); alpha = np.ones(N)
    gTp = np.zeros(N); nbt = np.zeros(N, int)
    Hk = np.zeros((N, 3, 3))
    done = np.full(N, -1)
    act = np.arange(N)
    for r in range(max_rounds):
        if not len(act):
            break
        f, G, H = oracle(act, xt[act], hess=need_hess)
        acc = np.ones(len(act), bool) if r == 0 else f <= fk[act] + 1e-4 * alpha[act] * gTp[act]
        rej = act[~acc]
        nbt[rej] += 1
        alpha[rej] *= 0.5
        give_up = rej[nbt[rej] >= 30]
        a = act[acc]
        if len(a):
            x[a] = xt[a]
            fk[a] = f[acc]
            gk[a] = proj(x[a], G[acc])
            if need_hess:
                Hk[a] = H[acc]
            if on_accept is not None:
                on_accept(a, x, gk, Hk, first=(r == 0))
        gn = np.linalg.norm(gk[a], axis=1) if len(a) else np.zeros(0)
        conv = a[gn <= TOL]
        done[conv] = r
        go = a[gn > TOL]
        if len(go):
            p = direction(go, x, gk, Hk, fk)
            pn = np.linalg.norm(p, axis=1)
            c = first_cap if r == 0 else cap
            alpha[go] = np.minimum(1.0, c / np.maximum(pn, 1e-300))
            pk[go] = p
            gTp[go] = np.einsum('ij,ij->i', p, gk[go])
            nbt[go] = 0
        live = np.setdiff1d(np.concatenate([go, rej]), give_up)
        xt[live] = retract(x[live], alpha[live, None] * pk[live])
        act = np.sort(live)
    return x, done >= 0, done


def gd(oracle, x0, max_rounds):
    return _armijo_driver(oracle, x0, max_rounds, lambda i, x, g, H, f: -g[i])


def bfgs(oracle, x0, max_rounds):
    """Port of the production Riemannian BFGS (projection transport, H0 scaling)."""
    N = len(x0)
    Hinv = np.repeat(np.eye(3)[None], N, 0)
    scaled = np.zeros(N, bool)
    prev_x = x0.copy(); prev_g = np.zeros((N, 3))

    def on_accept(a, x, g, H, first):
        if not first:
            P = np.eye(3)[None] - x[a, :, None] * x[a, None, :]
            s = np.einsum('nij,nj->ni', P, x[a] - prev_x[a])
            y = g[a] - np.einsum('nij,nj->ni', P, prev_g[a])
            sTy = np.einsum('ij,ij->i', s, y)
            Hi = P @ Hinv[a] @ P + x[a, :, None] * x[a, None, :]
            ok = sTy > 1e-14 * np.linalg.norm(s, axis=1) * np.linalg.norm(y, axis=1)
            yy = np.einsum('ij,ij->i', y, y)
            sc = ok & ~scaled[a] & (yy > 0)
            Hi[sc] *= (sTy[sc] / yy[sc])[:, None, None]
            scaled[a[ok]] = True
            rho = np.where(ok, 1.0 / np.where(ok, sTy, 1.0), 0.0)
            V = np.eye(3)[None] - rho[:, None, None] * s[:, :, None] * y[:, None, :]
            Hn = V @ Hi @ V.transpose(0, 2, 1) + rho[:, None, None] * s[:, :, None] * s[:, None, :]
            Hinv[a] = np.where(ok[:, None, None], Hn, Hi)
        prev_x[a] = x[a]; prev_g[a] = g[a]

    def direction(i, x, g, H, f):
        p = -np.einsum('nij,nj->ni', Hinv[i], g[i])
        p = proj(x[i], p)
        bad = np.einsum('ij,ij->i', p, g[i]) >= 0
        p[bad] = -g[i][bad]
        Hinv[i[bad]] = np.eye(3)
        scaled[i[bad]] = False
        return p

    return _armijo_driver(oracle, x0, max_rounds, direction, on_accept=on_accept)


def cg(oracle, x0, max_rounds):
    """Riemannian nonlinear CG (Polak-Ribiere+, projection transport, restart on non-descent)."""
    N = len(x0)
    prev_g = np.zeros((N, 3)); prev_p = np.zeros((N, 3)); has = np.zeros(N, bool)

    def direction(i, x, g, H, f):
        gt = g[i]
        pg = proj(x[i], prev_g[i]); pp = proj(x[i], prev_p[i])
        den = np.einsum('ij,ij->i', prev_g[i], prev_g[i])
        beta = np.where(has[i] & (den > 0), np.einsum('ij,ij->i', gt, gt - pg) / np.where(den > 0, den, 1), 0.0)
        beta = np.maximum(beta, 0.0)
        p = -gt + beta[:, None] * pp
        bad = np.einsum('ij,ij->i', p, gt) >= 0
        p[bad] = -gt[bad]
        prev_g[i] = gt; prev_p[i] = p; has[i] = True
        return p

    return _armijo_driver(oracle, x0, max_rounds, direction)


def newton(oracle, x0, max_rounds):
    """Riemannian Newton with eigenvalue-modified (positive definite) Hessian and Armijo."""
    def direction(i, x, g, H, f):
        xi = x[i]; gi = g[i]
        P = np.eye(3)[None] - xi[:, :, None] * xi[:, None, :]
        G_e = gi                         # tangent gradient (the normal part was projected away)
        # Riemannian Hessian: P Hf P - (x . grad_e f) P. grad_e f's normal part is
        # recovered from the oracle through f's homogeneity: store it in H's caller.
        Hr = P @ H[i] @ P - _xdotg[i][:, None, None] * P
        # tangent basis
        a = np.where(np.abs(xi[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
        e1 = proj(xi, a); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(xi, e1)
        B = np.stack([e1, e2], 2)                          # (n,3,2)
        H2 = B.transpose(0, 2, 1) @ Hr @ B                 # (n,2,2)
        g2 = np.einsum('nij,ni->nj', B, G_e)
        w, V = np.linalg.eigh(H2)
        w = np.maximum(np.abs(w), 1e-3 * np.maximum(np.abs(w).max(1, keepdims=True), 1e-12))
        step2 = -np.einsum('nij,nj->ni', V, np.einsum('nji,nj->ni', V, g2) / w)
        return np.einsum('nij,nj->ni', B, step2)

    N = len(x0)
    _xdotg = np.zeros(N)

    # wrap the oracle to also record x . grad_e f at accepted points
    class O:
        def __init__(s, base): s.base = base
        def __call__(s, idx, x, hess=False):
            f, G, H = s.base(idx, x, hess=True)
            _xdotg_trial[idx] = np.einsum('ij,ij->i', x, G)
            return f, G, H
    _xdotg_trial = np.zeros(N)
    wrapped = O(oracle)

    def on_acc(a, x, g, H, first):
        _xdotg[a] = _xdotg_trial[a]

    return _armijo_driver(wrapped, x0, max_rounds, direction, need_hess=True, on_accept=on_acc)


def fixed_point(oracle, x0, max_rounds):
    """Planar Gauss-Newton: x <- grad D(q)/|grad D(q)| (curvature term dropped), Armijo-safeguarded."""
    def direction(i, x, g, H, f):
        target = -_G[i] / np.maximum(np.linalg.norm(_G[i], axis=1, keepdims=True), 1e-300)
        return proj(x[i], target - x[i]) / np.maximum(np.einsum('ij,ij->i', target, x[i]), 0.2)[:, None]

    N = len(x0)
    _G = np.zeros((N, 3)); _Gt = np.zeros((N, 3))

    class O:
        def __init__(s, base): s.base = base
        def __call__(s, idx, x, hess=False):
            f, G, H = s.base(idx, x)
            _Gt[idx] = G
            return f, G, H

    def on_acc(a, x, g, H, first):
        _G[a] = _Gt[a]

    return _armijo_driver(O(oracle), x0, max_rounds, direction, on_accept=on_acc)


def vector_adam(lr):
    """VectorAdam (Ling et al. 2022) on S^2: rotation-equivariant second moment |g|^2,
    momentum transported by projection, retraction by normalization. No line search."""
    def run(oracle, x0, max_rounds, b1=0.9, b2=0.999, eps=1e-12):
        N = len(x0)
        x = x0.copy(); m = np.zeros((N, 3)); v = np.zeros(N); t = np.zeros(N)
        done = np.full(N, -1)
        act = np.arange(N)
        for r in range(max_rounds):
            if not len(act):
                break
            f, G, _ = oracle(act, x[act])
            g = proj(x[act], G)
            gn = np.linalg.norm(g, axis=1)
            conv = gn <= TOL
            done[act[conv]] = r
            a = act[~conv]; g = g[~conv]
            t[a] += 1
            m[a] = b1 * proj(x[a], m[a]) + (1 - b1) * g
            v[a] = b2 * v[a] + (1 - b2) * np.einsum('ij,ij->i', g, g)
            mh = m[a] / (1 - b1 ** t[a])[:, None]
            vh = v[a] / (1 - b2 ** t[a])
            x[a] = retract(x[a], -lr * mh / (np.sqrt(vh) + eps)[:, None])
            act = a
        return x, done >= 0, done
    return run


METHODS = {'gd (Armijo)': gd, 'bfgs (port)': bfgs, 'cg (PR+)': cg, 'newton (modified)': newton,
           'fixed-point (planar GN)': fixed_point,
           'vectoradam lr=0.05': vector_adam(0.05), 'vectoradam lr=0.2': vector_adam(0.2)}


def fibonacci(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def cold_start(native, z, d, n=64):
    F = fibonacci(n)
    Q = (z[:, None, :] - d[:, None, None] * F[None]).reshape(-1, 3)
    v = native.predict(Q).reshape(len(z), n) / d[:, None]
    return F[np.argmin(v, axis=1)]


def rotate(x, deg, rng):
    a = rng.normal(size=x.shape)
    a = proj(x, a)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    th = np.deg2rad(deg)
    return np.cos(th) * x + np.sin(th) * a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input'); ap.add_argument('grid', type=int)
    ap.add_argument('--rounds', type=int, default=60)
    ap.add_argument('--scatter', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    pts, vals, _ = make_input(a.input, a.grid, scatter=a.scatter)
    res = rbfyoursdf.main_algorithm(pts, vals, max_iters=10, verbose=False)
    native = res.interpolator.interpolator
    z = (pts - res.center) / res.scale
    d = vals / res.scale
    keep = np.abs(d) > 1e-12
    z, d = z[keep], d[keep]
    tag = f'{a.input}_{a.grid}' + ('_scatter' if a.scatter else '')
    # reference solution: Newton to the tolerance from the cold start
    x_cold = cold_start(native, z, d)
    x_ref, ok_ref, _ = newton(Oracle(native, z, d), x_cold, 200)
    f_ref = native.predict(z - d[:, None] * x_ref) / d
    starts = {'cold (Fibonacci 64)': x_cold,
              'warm 5 deg': rotate(x_ref, 5, rng), 'warm 20 deg': rotate(x_ref, 20, rng)}
    out = {'input': tag, 'N': int(len(z)), 'tol': TOL, 'max_rounds': a.rounds,
           'reference_converged': float(ok_ref.mean()), 'results': {}}
    for sname, x0 in starts.items():
        out['results'][sname] = {}
        for mname, fn in METHODS.items():
            orc = Oracle(native, z, d)
            t = time.perf_counter()
            x, conv, rounds = fn(orc, x0, a.rounds)
            wall = time.perf_counter() - t
            f = native.predict(z - d[:, None] * x) / d
            worse = (f > f_ref + 1e-7) & conv
            ang = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', x, x_ref), -1, 1)))
            r = rounds[conv]
            res_ = {'converged': float(conv.mean()),
                    'rounds_median': float(np.median(r)) if len(r) else None,
                    'rounds_p90': float(np.quantile(r, 0.9)) if len(r) else None,
                    'rounds_max': int(r.max()) if len(r) else None,
                    'batched_calls': orc.calls, 'rows_grad': orc.rows['grad'], 'rows_hess': orc.rows['hess'],
                    'native_eval_s': orc.seconds, 'wall_s': wall,
                    'other_minimum_fraction': float(worse.mean()),
                    'angle_to_ref_deg_median': float(np.median(ang))}
            # what the production budget (5 rounds) would achieve
            orc5 = Oracle(native, z, d)
            x5, conv5, _ = fn(orc5, x0, 5)
            res_['converged_within_5'] = float(conv5.mean())
            res_['tangent_grad_after_5_median'] = float(np.median(
                tnorm(x5, -native.predict_with_gradients(z - d[:, None] * x5)[1])))
            out['results'][sname][mname] = res_
            print(f"{tag} {sname:20s} {mname:24s} conv={res_['converged']:.3f} rounds med/p90/max="
                  f"{res_['rounds_median']}/{res_['rounds_p90']}/{res_['rounds_max']} calls={orc.calls} "
                  f"rows g/h={orc.rows['grad']}/{orc.rows['hess']} eval={orc.seconds:.2f}s wall={wall:.2f}s "
                  f"other-min={res_['other_minimum_fraction']:.3f} conv@5={res_['converged_within_5']:.3f}",
                  flush=True)
        # production C++ batched BFGS for reference
        for steps in (5, a.rounds):
            native.optimize_best_gradients(z[:100], d[:100], x0[:100], steps)      # warmup
            t = time.perf_counter()
            xc = native.optimize_best_gradients(z, d, x0, steps)
            wall = time.perf_counter() - t
            gt = tnorm(xc, -native.predict_with_gradients(z - d[:, None] * xc)[1])
            out['results'][sname][f'C++ bfgs, {steps} steps'] = {
                'wall_s': wall, 'converged': float((gt <= TOL).mean()), 'tangent_grad_median': float(np.median(gt))}
            print(f"{tag} {sname:20s} C++ bfgs {steps:3d} steps     conv={(gt <= TOL).mean():.3f} wall={wall:.2f}s",
                  flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f'{tag}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
