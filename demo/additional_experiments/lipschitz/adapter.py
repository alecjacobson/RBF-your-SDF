"""Frozen snapshot of a fitted PUInterpolator and an independent reference evaluator.

The snapshot is taken through the read-only ``export_lipschitz_state()`` binding
and holds every number the native evaluator uses. ``evaluate`` reproduces the
native rules (sphere partition):

* support of patch j: closed ball ``|x - xi_j|^2 <= R_j^2``
* raw weight ``v_j = psi(|x - xi_j| / R_j)``, ``psi(t) = (1-t)^4 (4t+1)`` on [0,1]
* if ``W = sum v_j > 0``: ``F = sum v_j f_j / W`` and
  ``grad F = (sum v_j grad f_j + sum f_j grad v_j - F sum grad v_j) / W``
* else (fallback): ``F = f_n`` for the patch n whose support is nearest,
  measured as ``max(|x - xi| - R, 0)``.

All quantities are in the normalized frame of the C++ fit. The Python wrapper
maps physical coordinates by ``x' = (x - center)/scale`` and returns
``F(x) = scale F'(x')`` so physical gradients equal normalized gradients and
Lipschitz constants are dimensionless and identical in both frames.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .bounds import Fit

SNAPSHOT_VERSION = 1

BRANCH_BLEND = 0
BRANCH_FALLBACK = 1


@dataclass
class Snapshot:
    """A frozen PU reconstruction in normalized coordinates."""

    patch_centers: np.ndarray        # (P, 3)
    patch_radii: np.ndarray          # (P,)
    fits: list                       # P exact Fit objects
    policy: dict
    transform_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    transform_scale: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def num_patches(self):
        return len(self.fits)

    def digest(self):
        """Return a SHA-256 over every stored number and the evaluator policy."""
        h = hashlib.sha256()
        h.update(json.dumps({'v': SNAPSHOT_VERSION, 'policy': self.policy}, sort_keys=True).encode())
        for a in (self.patch_centers, self.patch_radii, self.transform_center,
                  np.array([self.transform_scale])):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        for f in self.fits:
            for a in (f.centers, f.alpha_lo, f.linear_lo, np.array([f.const_lo])):
                h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        return h.hexdigest()

    # ── persistence ──────────────────────────────────────────────────
    def save(self, path):
        arrays = {'patch_centers': self.patch_centers, 'patch_radii': self.patch_radii,
                  'transform_center': self.transform_center,
                  'transform_scale': np.array(self.transform_scale),
                  'sizes': np.array([len(f.centers) for f in self.fits])}
        arrays['centers'] = np.concatenate([f.centers for f in self.fits])
        arrays['alpha'] = np.concatenate([f.alpha for f in self.fits])
        arrays['linear'] = np.stack([f.linear for f in self.fits])
        arrays['constant'] = np.array([f.constant for f in self.fits])
        arrays['json'] = np.array(json.dumps({'version': SNAPSHOT_VERSION, 'policy': self.policy,
                                              'meta': self.meta, 'sha256': self.digest()}))
        np.savez_compressed(path, **arrays)

    @staticmethod
    def load(path):
        z = np.load(path)
        info = json.loads(str(z['json']))
        if info['version'] != SNAPSHOT_VERSION:
            raise ValueError(f"snapshot version {info['version']} != {SNAPSHOT_VERSION}")
        offs = np.concatenate([[0], np.cumsum(z['sizes'])])
        fits = [Fit.exact(z['centers'][a:b], z['alpha'][a:b], z['linear'][i], z['constant'][i])
                for i, (a, b) in enumerate(zip(offs[:-1], offs[1:]))]
        s = Snapshot(z['patch_centers'], z['patch_radii'], fits, info['policy'],
                     z['transform_center'], float(z['transform_scale']), info['meta'])
        if s.digest() != info['sha256']:
            raise ValueError("snapshot hash mismatch")
        return s


def snapshot_from_native(interpolator, meta=None):
    """Export a Snapshot from a native PUInterpolator or the Python ScaledInterpolator."""
    center, scale = np.zeros(3), 1.0
    native = interpolator
    if hasattr(interpolator, 'interpolator') and hasattr(interpolator, 'scale'):
        native = interpolator.interpolator
        center, scale = np.asarray(interpolator.center, dtype=np.float64), float(interpolator.scale)
    state = native.export_lipschitz_state()
    pol = dict(state['policy'])
    if pol['partition'] != 'sphere':
        raise NotImplementedError(
            "only the sphere partition (the Python default) is modelled; the box "
            "partition uses tensor-product weights whose bounds are not implemented")
    fits, cs, rs = [], [], []
    for p in state['patches']:
        if p['kernel'] != 'cubic':
            raise NotImplementedError(f"kernel {p['kernel']!r} is not modelled")
        he = np.asarray(p['half_ext'])
        if not (he[0] == he[1] == he[2]):
            raise ValueError("sphere patch with unequal half extents")
        fits.append(Fit.exact(p['centers'], p['alpha'], p['linear'], p['constant']))
        cs.append(p['center'])
        rs.append(he[0])
    m = dict(meta or {})
    m['regularization'] = sorted({float(p['reg']) for p in state['patches']})
    return Snapshot(np.array(cs, dtype=np.float64), np.array(rs, dtype=np.float64), fits,
                    pol, center, scale, m)


# ── reference evaluator ─────────────────────────────────────────────

def wendland(t):
    """psi(t) = (1-t)^4 (4t+1) on [0,1], zero beyond."""
    t = np.clip(t, 0.0, 1.0)
    s = 1.0 - t
    return s ** 4 * (4.0 * t + 1.0)


def wendland_dt(t):
    """d psi / dt = -20 t (1-t)^3 on [0,1], zero beyond."""
    t = np.clip(t, 0.0, 1.0)
    return -20.0 * t * (1.0 - t) ** 3


def _local_value_grad(fit, x):
    v = x[:, None, :] - fit.centers[None]
    r = np.sqrt(np.einsum('mnk,mnk->mn', v, v))
    a = fit.alpha
    with np.errstate(all='ignore'):  # spurious macOS Accelerate matmul warnings
        f = (r * r * r) @ a + x @ fit.linear + fit.constant
    if not np.all(np.isfinite(f)):
        raise FloatingPointError('non-finite field value')
    g = fit.linear[None] + 3.0 * np.einsum('mn,mnk->mk', r * a[None], v)
    return f, g


def nearest_support_patch(snap, x):
    """Index of the patch whose support is nearest (ties: lowest index) and the gap count."""
    d = np.linalg.norm(x[:, None, :] - snap.patch_centers[None], axis=2) - snap.patch_radii[None]
    d = np.maximum(d, 0.0)
    idx = np.argmin(d, axis=1)
    best = d[np.arange(len(x)), idx]
    ties = (d == best[:, None]).sum(1) > 1
    return idx, ties


def evaluate(snap: Snapshot, x, return_info=False):
    """Evaluate the frozen PU field (normalized frame) with the native branch rules.

    Returns:
        ``(F, gradF)`` or, with ``return_info``, also a dict holding the raw
        weight sum ``W``, active patch count, branch label per point, and
        whether a nearest-patch tie occurred.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    M = len(x)
    W = np.zeros(M)
    Vw = np.zeros(M)
    G = np.zeros((M, 3))
    A = np.zeros((M, 3))
    B = np.zeros((M, 3))
    nact = np.zeros(M, dtype=np.int64)
    tree = cKDTree(x)
    for j, fit in enumerate(snap.fits):
        c, R = snap.patch_centers[j], snap.patch_radii[j]
        idx = np.asarray(tree.query_ball_point(c, R * (1 + 1e-12) + 1e-300), dtype=np.int64)
        if len(idx) == 0:
            continue
        d = x[idx] - c
        r2 = np.einsum('mk,mk->m', d, d)
        idx = idx[r2 <= R * R]
        if len(idx) == 0:
            continue
        d = x[idx] - c
        r = np.sqrt(np.einsum('mk,mk->m', d, d))
        t = np.clip(r / R, 0.0, 1.0)
        w = wendland(t)
        with np.errstate(invalid='ignore', divide='ignore'):
            dw = np.where(((r > 1e-12) & (t < 1.0))[:, None],
                          (wendland_dt(t) / (R * r))[:, None] * d, 0.0)
        f, g = _local_value_grad(fit, x[idx])
        W[idx] += w
        Vw[idx] += w * f
        G[idx] += w[:, None] * g
        A[idx] += f[:, None] * dw
        B[idx] += dw
        nact[idx] += 1
    F = np.zeros(M)
    grad = np.zeros((M, 3))
    cov = W > 0
    F[cov] = Vw[cov] / W[cov]
    grad[cov] = (G[cov] + A[cov] - F[cov, None] * B[cov]) / W[cov, None]
    branch = np.where(cov, BRANCH_BLEND, BRANCH_FALLBACK)
    ties = np.zeros(M, dtype=bool)
    if (~cov).any():
        unc = np.flatnonzero(~cov)
        near, tie = nearest_support_patch(snap, x[unc])
        ties[unc] = tie
        for j in np.unique(near):
            sel = unc[near == j]
            f, g = _local_value_grad(snap.fits[j], x[sel])
            F[sel] = f
            grad[sel] = g
    if return_info:
        return F, grad, {'W': W, 'n_active': nact, 'branch': branch, 'tie': ties}
    return F, grad


def to_normalized(snap, x_phys):
    return (np.asarray(x_phys, dtype=np.float64) - snap.transform_center) / snap.transform_scale
