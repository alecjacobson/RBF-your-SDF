"""Analytic verification of discontinuity witnesses found by run.py's probes.

Numerical two-sided probes only suggest a jump. This script turns each witness
into a certificate about the exact field (stored coefficients as reals):

* ``fallback_nearest_switch`` at point c between patches a and b. If a small
  box around c misses every support (so W = 0 there and F is the nearest-patch
  fit), the nearest patch is a on one side and b on the other, then the
  one-sided limits at c are f_a(c) and f_b(c). An enclosure of
  ``f_a(c) - f_b(c)`` excluding zero proves a jump of at least its mignitude.

* ``union_boundary_corner`` at p on the boundaries of supports j and k and
  inside no other support. Approaching p along a direction u that enters both
  supports, ``1 - t_j ~ s (-n_j.u)/R_j`` so ``v_j/v_k -> lam(u) =
  ((n_j.u / R_j) / (n_k.u / R_k))^4`` and ``F -> (lam f_j + f_k)/(lam + 1)``.
  Two directions with different lam give limits differing by
  ``|f_j - f_k| |lam1/(lam1+1) - lam2/(lam2+1)|``. The analysis uses the exact
  circle point near the floating-point p; enclosures account for the offset.

Usage: uv run python -m lipschitz.witness <run dir>  (from demo/additional_experiments)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'lipschitz'

from lipschitz import interval as iv                            # noqa: E402
from lipschitz.adapter import Snapshot                          # noqa: E402
from lipschitz.bounds import local_derivs_cert, b0_cert, fit_constants   # noqa: E402
from lipschitz.pu_bounds import _pair_near_far                  # noqa: E402


def _diff_enclosure(snap, a, b, x, slack):
    """Enclose f_a - f_b on the ball of radius ``slack`` around x."""
    fa = local_derivs_cert(snap.fits[a], x[None])['f']
    fb = local_derivs_cert(snap.fits[b], x[None])['f']
    d = iv.sub(fa, fb)
    lo, hi = x - slack, x + slack
    L = b0_cert(snap.fits[a], lo[None], hi[None]) + b0_cert(snap.fits[b], lo[None], hi[None])
    rad = iv.up(float(L[0]) * iv.up(slack * np.sqrt(3.0) * (1 + 1e-12)))
    return iv.down(d[0][0] - rad), iv.up(d[1][0] + rad)


def verify_switch(snap, w, h=1e-7):
    c = np.asarray(w['point'], float)
    a, b = w['patches']
    lo, hi = (c - h)[None], (c + h)[None]
    P = snap.num_patches
    dmin, dmax = _pair_near_far(np.repeat(lo, P, 0), np.repeat(hi, P, 0), snap.patch_centers)
    uncovered = bool(np.all(dmin > snap.patch_radii))
    s = iv.down(dmin - snap.patch_radii)          # support distance lower bounds
    S = iv.up(dmax - snap.patch_radii)
    others = np.ones(P, bool); others[[a, b]] = False
    # a and b are the only candidates for nearest support anywhere in the box
    only_ab = bool(np.max([S[a], S[b]]) < np.min(s[others], initial=np.inf))
    d = _diff_enclosure(snap, a, b, c, h * np.sqrt(3))
    jump = float(iv.mig(d))
    ok = uncovered and only_ab and jump > 0
    return {'certified': ok, 'box_uncovered': uncovered, 'only_two_candidates': only_ab,
            'jump_lower_bound': jump if ok else 0.0, 'f_a_minus_f_b': [float(d[0]), float(d[1])]}


def verify_corner(snap, w):
    p = np.asarray(w['point'], float)
    j, k = w['patches']
    C, R = snap.patch_centers, snap.patch_radii
    # exact circle point is within eps of p (p was constructed from C, R in float64)
    eps = 1e-12 * (1 + np.abs(p).max())
    dist = np.linalg.norm(C - p, axis=1)
    others = np.ones(len(R), bool); others[[j, k]] = False
    margin = float(np.min(dist[others] - R[others])) if others.any() else np.inf
    outside_others = margin > 10 * eps
    on_j = abs(dist[j] - R[j]) < 1e3 * eps
    on_k = abs(dist[k] - R[k]) < 1e3 * eps
    nj = (p - C[j]) / R[j]; nk = (p - C[k]) / R[k]
    u1 = -(nj + 0.1 * nk); u2 = -(0.1 * nj + nk)
    def lam(u):
        # along x = p + s u:  1 - t_j ~ s (-n_j.u) / R_j, and psi(t) ~ 5 (1-t)^4 near t = 1
        a, b = -(nj @ u) / R[j], -(nk @ u) / R[k]
        return (a / b) ** 4, a > 0 and b > 0
    l1, ok1 = lam(u1); l2, ok2 = lam(u2)
    factor = abs(l1 / (l1 + 1) - l2 / (l2 + 1))
    # distance from p to the exact circle: residuals over the sine of the intersection angle
    sin_t = np.linalg.norm(np.cross(nj, nk))
    slack = 2 * (abs(dist[j] - R[j]) + abs(dist[k] - R[k])) / max(sin_t, 1e-300) + eps
    d = _diff_enclosure(snap, j, k, p, slack)
    jump = float(iv.mig(d)) * factor * (1 - 1e-6)     # factor computed in float64; keep a margin
    ok = bool(outside_others and on_j and on_k and ok1 and ok2 and jump > 0)
    return {'certified': ok, 'outside_other_supports_margin': margin, 'lambda': [l1, l2],
            'direction_factor': factor, 'f_j_minus_f_k': [float(d[0]), float(d[1])],
            'limit_gap_lower_bound': jump if ok else 0.0}


def main(run_dir):
    run_dir = Path(run_dir)
    snap = Snapshot.load(run_dir / 'snapshot.npz')
    res = json.loads((run_dir / 'results.json').read_text())
    out = []
    for w in res.get('transitions', {}).get('witnesses', []):
        v = verify_switch(snap, w) if w['kind'] == 'fallback_nearest_switch' else verify_corner(snap, w)
        out.append({'kind': w['kind'], 'point': w['point'], 'patches': w['patches'], **v})
    (run_dir / 'witnesses_verified.json').write_text(json.dumps(out, indent=1, default=float))
    n = sum(o['certified'] for o in out)
    print(f'{run_dir.name}: {n}/{len(out)} witnesses certified')
    return out


if __name__ == '__main__':
    for d in sys.argv[1:]:
        main(d)
