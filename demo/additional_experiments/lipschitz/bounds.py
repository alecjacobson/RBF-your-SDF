"""Gradient bounds for a single affine-augmented cubic RBF fit.

A fit is ``f(x) = c + p.x + sum_i alpha_i |x - q_i|^3`` with the stored
float64 numbers treated as exact reals. Coefficients may be intervals (used for
difference fits ``f_j - f_k``, whose merged coefficients are not representable).

Two families of functions:

* ``*_estimate``: real-arithmetic formulas evaluated in ordinary float64
  ("analytic estimate").
* ``*_cert``: the same formulas with every operation outward rounded
  (``interval.py``), giving certified upper bounds for the exact field.

Bounds (see the design document, section "Bounds for an individual RBF"):

* B0 triangle: ``|p| + 3 sum |alpha_i| R_i(B)^2``
* B1 idealized moment: ``|p| + 3 sum |alpha_i| |q_i - c|^2`` (assumes exact
  zero moments; diagnostic only, never certified)
* B2 residual-aware moment: B1 + ``3 |m0| Z^2 + 6 |m1| Z``
* B3 cell Taylor: ``|grad f(x0)| + rho |H(x0)|_2 + 3 A rho^2``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import interval as iv

# Peak elements of a (boxes x centers) temporary in the certified paths.
CHUNK_ELEMS = 400_000


@dataclass
class Fit:
    """An affine-augmented cubic RBF fit with (possibly interval) coefficients."""

    centers: np.ndarray            # (n, 3) exact float64
    alpha_lo: np.ndarray           # (n,)
    alpha_hi: np.ndarray           # (n,)
    linear_lo: np.ndarray          # (3,)
    linear_hi: np.ndarray          # (3,)
    const_lo: float
    const_hi: float

    @staticmethod
    def exact(centers, alpha, linear, constant):
        """Build a fit whose coefficients are exact float64 numbers."""
        centers = np.ascontiguousarray(centers, dtype=np.float64).reshape(-1, 3)
        alpha = np.asarray(alpha, dtype=np.float64).ravel()
        linear = np.asarray(linear, dtype=np.float64).ravel()
        return Fit(centers, alpha.copy(), alpha.copy(), linear.copy(), linear.copy(),
                   float(constant), float(constant))

    @property
    def alpha(self):
        """Midpoint coefficients (exact for an exact fit)."""
        return 0.5 * (self.alpha_lo + self.alpha_hi)

    @property
    def linear(self):
        return 0.5 * (self.linear_lo + self.linear_hi)

    @property
    def constant(self):
        return 0.5 * (self.const_lo + self.const_hi)

    @property
    def is_exact(self):
        return (np.array_equal(self.alpha_lo, self.alpha_hi)
                and np.array_equal(self.linear_lo, self.linear_hi)
                and self.const_lo == self.const_hi)


# ── FP64 evaluation ──────────────────────────────────────────────────

def evaluate(fit: Fit, x, hessian=False):
    """Evaluate value, gradient and optionally Hessian of the fit in float64.

    Each center's Hessian contribution is defined as zero at r = 0, its
    continuous limit.

    Returns:
        ``(f, g)`` or ``(f, g, H)`` with shapes (m,), (m, 3), (m, 3, 3).
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    v = x[:, None, :] - fit.centers[None, :, :]           # (m, n, 3)
    r = np.sqrt(np.einsum('mnk,mnk->mn', v, v))
    a = fit.alpha
    with np.errstate(all='ignore'):  # spurious macOS Accelerate matmul warnings
        f = (r * r * r) @ a + x @ fit.linear + fit.constant
    if not np.all(np.isfinite(f)):
        raise FloatingPointError('non-finite field value')
    g = fit.linear[None, :] + 3.0 * np.einsum('mn,mnk->mk', r * a[None, :], v)
    if not hessian:
        return f, g
    with np.errstate(invalid='ignore', divide='ignore'):
        inv_r = np.where(r > 0, 1.0 / r, 0.0)
    H = 3.0 * (np.einsum('mn->m', r * a)[:, None, None] * np.eye(3)
               + np.einsum('mn,mnk,mnl->mkl', a * inv_r, v, v))
    return f, g, H


# ── geometry helpers ────────────────────────────────────────────────

def box_center_radius(lo, hi):
    """Return the center x0 (float64) and an upward-rounded half-diagonal rho.

    x0 = (lo + hi) / 2 is rounded, so rho is measured from the rounded x0 to the
    farthest corner rather than assumed to be half the diagonal.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    x0 = 0.5 * (lo + hi)
    e = np.maximum(iv.up(x0 - lo), iv.up(hi - x0))
    rho = iv.up(np.sqrt(iv.sum_upper(iv.up(e * e), axis=-1)))
    return x0, rho


def far_dist2_up(lo, hi, pts):
    """Upward bound on max_{x in box} |x - p|^2, shape (m, n)."""
    # |fl(a - b)| is within half an ulp of |a - b|, so one ulp up encloses it.
    d = np.maximum(np.abs(hi[:, None, :] - pts[None]), np.abs(lo[:, None, :] - pts[None]))
    d = iv.up(d)
    return iv.up(iv.up(d[..., 0] * d[..., 0]) + iv.up(iv.up(d[..., 1] * d[..., 1])
                                                      + iv.up(d[..., 2] * d[..., 2])))


def near_far_dist(lo, hi, pts):
    """Outward bounds (dmin_lo, dmax_hi) of the distance from a box to points, (m, n)."""
    below = lo[:, None, :] - pts[None]
    above = pts[None] - hi[:, None, :]
    gap = np.maximum(np.maximum(below, above), 0.0)
    gap = np.maximum(iv.down(gap), 0.0)
    g2 = iv.down(iv.down(iv.down(gap[..., 0] * gap[..., 0]) + iv.down(gap[..., 1] * gap[..., 1]))
                 + iv.down(gap[..., 2] * gap[..., 2]))
    dmin = np.maximum(iv.down(np.sqrt(np.maximum(g2, 0.0))), 0.0)
    dmax = iv.up(np.sqrt(far_dist2_up(lo, hi, pts)))
    return dmin, dmax


# ── per-fit constants ───────────────────────────────────────────────

@dataclass
class FitConstants:
    """Box-independent quantities for B1/B2/B3 (certified unless noted)."""

    linear_norm_up: float
    abs_alpha_up: np.ndarray       # (n,) upper bounds of |alpha_i|
    A_up: float                    # upper bound of sum |alpha_i|
    centroid: np.ndarray           # (3,) reference point c (exact float64 by choice)
    S_up: float                    # upper bound of sum |alpha_i| |q_i - c|^2
    m0: tuple                      # interval of sum alpha_i
    m1: list                       # 3 intervals of sum alpha_i (q_i - c)
    m1_norm_up: float
    # FP64 estimates for reporting
    S_est: float
    linear_norm_est: float


def fit_constants(fit: Fit) -> FitConstants:
    """Precompute the box-independent constants of a fit."""
    abs_a = iv.mag((fit.alpha_lo, fit.alpha_hi))
    A_up = float(iv.sum_upper(abs_a))
    lin = [(fit.linear_lo[k], fit.linear_hi[k]) for k in range(3)]
    lin_norm = float(iv.norm_upper([(np.float64(a), np.float64(b)) for a, b in lin]))
    a_mid = fit.alpha
    w = np.abs(a_mid)
    if w.sum() > 0:
        c = (w[:, None] * fit.centers).sum(0) / w.sum()
    else:
        c = fit.centers.mean(0) if len(fit.centers) else np.zeros(3)
    # h_i = q_i - c (intervals), |h|^2 upward
    h = [iv.sub(iv.point(fit.centers[:, k]), iv.point(np.full(len(fit.centers), c[k])))
         for k in range(3)]
    h2_up = iv.up(iv.up(iv.sqr(h[0])[1] + iv.sqr(h[1])[1]) + iv.sqr(h[2])[1])
    S_up = float(iv.sum_upper(iv.up(abs_a * h2_up)))
    alpha_iv = (fit.alpha_lo, fit.alpha_hi)
    m0 = iv.isum(alpha_iv)
    m1 = [iv.isum(iv.mul(alpha_iv, h[k])) for k in range(3)]
    m1_norm = float(iv.norm_upper(m1))
    hh = fit.centers - c
    S_est = float(np.sum(w * np.einsum('nk,nk->n', hh, hh)))
    return FitConstants(lin_norm, abs_a, A_up, c, S_up,
                        (float(m0[0]), float(m0[1])), [(float(a), float(b)) for a, b in m1],
                        m1_norm, S_est, float(np.linalg.norm(fit.linear)))


# ── bounds: FP64 estimates ──────────────────────────────────────────

def b0_estimate(fit, lo, hi):
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    q = fit.centers
    d = np.maximum(np.abs(hi[:, None, :] - q[None]), np.abs(lo[:, None, :] - q[None]))
    R2 = np.einsum('mnk,mnk->mn', d, d)
    return np.linalg.norm(fit.linear) + 3.0 * R2 @ np.abs(fit.alpha)


def b1_estimate(fit, consts=None):
    """Idealized moment bound (valid only if moments vanish exactly)."""
    k = consts or fit_constants(fit)
    return k.linear_norm_est + 3.0 * k.S_est


def b3_estimate(fit, lo, hi):
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    x0 = 0.5 * (lo + hi)
    rho = 0.5 * np.linalg.norm(hi - lo, axis=1)
    _, g, H = evaluate(fit, x0, hessian=True)
    Hn = np.linalg.norm(H, ord=2, axis=(1, 2))
    return np.linalg.norm(g, axis=1) + rho * Hn + 3.0 * np.abs(fit.alpha).sum() * rho ** 2


# ── bounds: certified ───────────────────────────────────────────────

def b0_cert(fit, lo, hi, consts=None):
    """Certified B0 for each box, shape (m,)."""
    k = consts or fit_constants(fit)
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    out = np.empty(len(lo))
    step = max(1, CHUNK_ELEMS // max(len(fit.centers), 1))
    for s in range(0, len(lo), step):
        R2 = far_dist2_up(lo[s:s + step], hi[s:s + step], fit.centers)
        t = iv.sum_upper(iv.up(R2 * k.abs_alpha_up[None, :]), axis=1)
        out[s:s + step] = iv.up(k.linear_norm_up + iv.up(3.0 * t))
    return out


def b2_cert(fit, lo, hi, consts=None):
    """Certified residual-aware moment bound B2 for each box, shape (m,)."""
    k = consts or fit_constants(fit)
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    Z = iv.up(np.sqrt(far_dist2_up(lo, hi, k.centroid[None, :])[:, 0]))
    m0 = iv.mag(k.m0)
    t1 = iv.up(3.0 * k.S_up)
    t2 = iv.up(3.0 * iv.up(m0 * iv.up(Z * Z)))
    t3 = iv.up(6.0 * iv.up(k.m1_norm_up * Z))
    return iv.up(iv.up(iv.up(k.linear_norm_up + t1) + t2) + t3)


def b2_estimate(fit, lo, hi, consts=None):
    k = consts or fit_constants(fit)
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    x0 = 0.5 * (lo + hi)
    Z = np.linalg.norm(np.maximum(np.abs(lo - k.centroid), np.abs(hi - k.centroid)), axis=1)
    a = fit.alpha
    m0 = a.sum()
    m1 = np.linalg.norm((a[:, None] * (fit.centers - k.centroid)).sum(0))
    return k.linear_norm_est + 3.0 * k.S_est + 3.0 * abs(m0) * Z ** 2 + 6.0 * m1 * Z


def local_derivs_cert(fit, x0):
    """Enclose value, gradient and Hessian of the exact fit at points x0.

    Returns:
        dict with interval ``f`` (m,), list ``g`` of 3 intervals, and dict
        ``H[(k, l)]`` of 6 intervals (k <= l), each component shape (m,).
    """
    x0 = np.atleast_2d(np.asarray(x0, dtype=np.float64))
    m, n = len(x0), len(fit.centers)
    step = max(1, CHUNK_ELEMS // max(n, 1))
    f_lo, f_hi = np.empty(m), np.empty(m)
    g = [(np.empty(m), np.empty(m)) for _ in range(3)]
    keys = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    H = {key: (np.empty(m), np.empty(m)) for key in keys}
    al = (fit.alpha_lo[None, :], fit.alpha_hi[None, :])
    for s in range(0, m, step):
        X = x0[s:s + step]
        v = [iv.sub(iv.point(np.broadcast_to(X[:, k:k + 1], (len(X), n))),
                    iv.point(np.broadcast_to(fit.centers[None, :, k], (len(X), n))))
             for k in range(3)]
        sq = [iv.sqr(vk) for vk in v]
        r2 = iv.add(iv.add(sq[0], sq[1]), sq[2])
        r = iv.sqrt(r2)
        ar = iv.mul(al, r)                                   # alpha r
        # value: sum alpha r^3 + p.x + c
        val = iv.isum(iv.mul(ar, r2), axis=1)
        lin = (np.full(len(X), fit.const_lo), np.full(len(X), fit.const_hi))
        for k in range(3):
            lin = iv.add(lin, iv.mul((np.full(len(X), fit.linear_lo[k]), np.full(len(X), fit.linear_hi[k])),
                                     iv.point(X[:, k])))
        fv = iv.add(val, lin)
        f_lo[s:s + step], f_hi[s:s + step] = fv
        for k in range(3):
            gk = iv.scale(iv.isum(iv.mul(ar, v[k]), axis=1), 3.0)
            gk = iv.add(gk, (np.full(len(X), fit.linear_lo[k]), np.full(len(X), fit.linear_hi[k])))
            g[k][0][s:s + step], g[k][1][s:s + step] = gk
        # Hessian: 3 sum alpha (r delta_kl + v_k v_l / r); |v_k v_l / r| <= r,
        # and v_k^2 / r in [0, r]; zero contribution at r = 0.
        pos = r[0] > 0
        safe_r = (np.where(pos, r[0], 1.0), np.where(pos, r[1], 1.0))
        for (k, l) in keys:
            prod = iv.sqr(v[k]) if k == l else iv.mul(v[k], v[l])
            q = iv.div(prod, safe_r)
            cap_lo = np.zeros_like(r[1]) if k == l else -r[1]
            q = (np.where(pos, np.maximum(q[0], cap_lo), cap_lo),
                 np.where(pos, np.minimum(q[1], r[1]), r[1]))
            term = iv.add(q, r) if k == l else q
            hk = iv.scale(iv.isum(iv.mul(al, term), axis=1), 3.0)
            H[(k, l)][0][s:s + step], H[(k, l)][1][s:s + step] = hk
    return {'f': (f_lo, f_hi), 'g': g, 'H': H}


def hess_norm_up(H):
    """Upper bound on the spectral norm of a symmetric Hessian given entry intervals."""
    mags = {key: iv.mag(val) for key, val in H.items()}
    def e(k, l):
        return mags[(min(k, l), max(k, l))]
    fro = np.zeros_like(mags[(0, 0)])
    for k in range(3):
        for l in range(3):
            fro = iv.up(fro + iv.up(e(k, l) * e(k, l)))
    fro = iv.up(np.sqrt(fro))
    row = np.maximum.reduce([iv.up(iv.up(e(k, 0) + e(k, 1)) + e(k, 2)) for k in range(3)])
    return np.minimum(fro, row)


def b3_cert(fit, lo, hi, consts=None, derivs=None):
    """Certified cell Taylor bound B3 for each box, shape (m,)."""
    k = consts or fit_constants(fit)
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    x0, rho = box_center_radius(lo, hi)
    d = derivs or local_derivs_cert(fit, x0)
    gn = iv.norm_upper(d['g'])
    Hn = hess_norm_up(d['H'])
    return iv.up(iv.up(gn + iv.up(rho * Hn)) + iv.up(3.0 * iv.up(k.A_up * iv.up(rho * rho))))


def value_range_cert(fit, lo, hi, consts=None, derivs=None):
    """Enclose the range of the fit over each box by a second-order Taylor bound.

    ``|f(x) - f(x0) - g.h - h'Hh/2| <= A rho^3`` because the Hessian of each
    cubic term is 6-Lipschitz. Returns (lower, upper) arrays.
    """
    k = consts or fit_constants(fit)
    lo, hi = np.atleast_2d(lo), np.atleast_2d(hi)
    x0, rho = box_center_radius(lo, hi)
    d = derivs or local_derivs_cert(fit, x0)
    gn = iv.norm_upper(d['g'])
    Hn = hess_norm_up(d['H'])
    rad = iv.up(iv.up(iv.up(rho * gn) + iv.up(0.5 * iv.up(Hn * iv.up(rho * rho))))
                + iv.up(k.A_up * iv.up(rho * iv.up(rho * rho))))
    f = d['f']
    return iv.down(f[0] - rad), iv.up(f[1] + rad)


def local_bounds_cert(fit, lo, hi, consts=None, derivs=None):
    """All certified local bounds and their minimum, as a dict of (m,) arrays."""
    k = consts or fit_constants(fit)
    out = {'B0': b0_cert(fit, lo, hi, k), 'B2': b2_cert(fit, lo, hi, k),
           'B3': b3_cert(fit, lo, hi, k, derivs)}
    out['min'] = np.minimum(np.minimum(out['B0'], out['B2']), out['B3'])
    return out
