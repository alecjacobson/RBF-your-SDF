"""Adaptive box hierarchies: certificate statuses and conservative cell rejection.

A box is *rejected* only if the analytic reconstructed field provably has no
zero in the closed box. Rejection rules (all strict, outward rounded):

* ``B0``:   ``mig(F(x0)) > L_B0 * rho``   (B0 local bounds through the blend bound)
* ``best``: ``mig(F(x0)) > L_best * rho`` (valid minimum, disagreement refinement)
* ``hull``: every fit that can contribute on the box has the same strict sign
  throughout the box (F is a convex combination of them wherever W > 0)
* ``best_hull``: either of ``best`` or ``hull``

All rules require status CERTIFIED or FALLBACK_FIXED. Unresolved boxes are
subdivided until ``max_depth`` and then kept, labelled unresolved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import interval as iv
from . import pu_bounds as pub

METHODS = ('B0', 'best', 'hull', 'best_hull')


def regular_cells(dom_lo, dom_hi, n):
    """Return (lo, hi, ijk) of an n^3 regular grid of cells over the domain."""
    dom_lo = np.asarray(dom_lo, float); dom_hi = np.asarray(dom_hi, float)
    n = np.broadcast_to(np.asarray(n), (3,)).astype(int)
    axes = [np.linspace(dom_lo[k], dom_hi[k], n[k] + 1) for k in range(3)]
    I, J, K = np.meshgrid(np.arange(n[0]), np.arange(n[1]), np.arange(n[2]), indexing='ij')
    ijk = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
    lo = np.stack([axes[k][ijk[:, k]] for k in range(3)], 1)
    hi = np.stack([axes[k][ijk[:, k] + 1] for k in range(3)], 1)
    return lo, hi, ijk


def rejects(out, method):
    """Boolean mask of boxes whose certificate proves F has no zero."""
    ok = (out['status'] == pub.CERTIFIED) | (out['status'] == pub.FALLBACK_FIXED)
    d0 = iv.mig((out['F0_lo'], out['F0_hi']))
    d0 = np.where(np.isfinite(out['F0_lo']) & np.isfinite(out['F0_hi']), d0, 0.0)
    def lip(L):
        with np.errstate(invalid='ignore', over='ignore'):
            return ok & np.isfinite(L) & (d0 > iv.up(L * out['rho']))
    hull = ok & (out['hull_sign'] != 0)
    if method == 'B0':
        return lip(out['L_B0'])
    if method == 'best':
        return lip(out['L_best'])
    if method == 'hull':
        return hull
    if method == 'best_hull':
        return lip(out['L_best']) | hull
    raise ValueError(f"unknown method {method!r}")


@dataclass
class PruneResult:
    method: str
    leaf_n: np.ndarray                      # leaf grid resolution per axis
    leaf_label: np.ndarray                  # (nx,ny,nz) 0 retained, 1 rejected, 2 unresolved
    reject_level: np.ndarray                # level at which a leaf's ancestor was rejected (-1)
    n_boxes_certified: int
    cost_bound_terms: int
    time_certify: float
    per_level: list = field(default_factory=list)


def _certify_cached(cert, lo, hi, keys, level, cache):
    """Certify boxes, reusing results of earlier traversals with other methods.

    Returns (status, {method: reject mask}, seconds attributed to these boxes,
    kernel-term cost attributed to these boxes). The attributed time is the
    measured certification time of the call that produced each box, split
    evenly over its boxes, so a cached traversal reports the time it would
    have taken standalone.
    """
    if cache is None:
        cache = {}
    ent = cache.setdefault(level, {'keys': np.zeros(0, np.int64), 'status': np.zeros(0, np.int8),
                                   'rej': {m: np.zeros(0, bool) for m in METHODS},
                                   'sec': np.zeros(0), 'cost': np.zeros(0)})
    have = np.isin(keys, ent['keys'])
    miss = ~have
    if miss.any():
        t = time.perf_counter()
        out = cert.certify(lo[miss], hi[miss])
        dt = time.perf_counter() - t
        nm = int(miss.sum())
        newk = keys[miss]
        ent['keys'] = np.concatenate([ent['keys'], newk])
        ent['status'] = np.concatenate([ent['status'], out['status']])
        for m in METHODS:
            ent['rej'][m] = np.concatenate([ent['rej'][m], rejects(out, m)])
        ent['sec'] = np.concatenate([ent['sec'], np.full(nm, dt / nm)])
        ent['cost'] = np.concatenate([ent['cost'], np.full(nm, out['cost_bound_terms'] / nm)])
        o = np.argsort(ent['keys'])
        for k in ('keys', 'status', 'sec', 'cost'):
            ent[k] = ent[k][o]
        for m in METHODS:
            ent['rej'][m] = ent['rej'][m][o]
    pos = np.searchsorted(ent['keys'], keys)
    return (ent['status'][pos], {m: ent['rej'][m][pos] for m in METHODS},
            float(ent['sec'][pos].sum()), float(ent['cost'][pos].sum()))


def adaptive_prune(cert, dom_lo, dom_hi, root_n, max_depth, method, time_budget=np.inf, cache=None):
    """Traverse an octree over the domain and reject boxes with a certificate.

    Returns a PruneResult on the leaf grid ``root_n * 2**max_depth`` per axis.
    Boxes that are still pending when the time budget expires are labelled
    unresolved at whatever level they reached. Pass the same ``cache`` dict
    to traversals of different methods to certify each box once;
    ``time_certify`` still reports the standalone cost.
    """
    root_n = np.broadcast_to(np.asarray(root_n), (3,)).astype(int)
    leaf_n = root_n * 2 ** max_depth
    dom_lo = np.asarray(dom_lo, float); dom_hi = np.asarray(dom_hi, float)
    label = np.zeros(tuple(leaf_n), dtype=np.int8)
    rlevel = np.full(tuple(leaf_n), -1, dtype=np.int8)
    _, _, ijk = regular_cells(dom_lo, dom_hi, root_n)
    level = 0
    cost = 0
    nbox = 0
    t_cert = 0.0
    per_level = []
    while len(ijk):
        n_l = root_n * 2 ** level
        h = (dom_hi - dom_lo) / n_l
        lo = dom_lo + ijk * h
        hi = dom_lo + (ijk + 1) * h
        hi = np.where(ijk + 1 == n_l, dom_hi, hi)
        if t_cert > time_budget:
            _paint(label, ijk, 2 ** (max_depth - level), 2)
            per_level.append({'level': level, 'boxes': len(ijk), 'timed_out': True})
            break
        keys = np.ravel_multi_index(ijk.T, tuple(n_l))
        status, rej_all, sec, c = _certify_cached(cert, lo, hi, keys, level, cache)
        t_cert += sec
        cost += c
        nbox += len(lo)
        rej = rej_all[method]
        span = 2 ** (max_depth - level)
        _paint(label, ijk[rej], span, 1)
        _paint(rlevel, ijk[rej], span, level)
        rest = ijk[~rej]
        st = status[~rej]
        per_level.append({'level': level, 'boxes': int(len(ijk)), 'rejected': int(rej.sum()),
                          'unresolved': int((status == pub.UNRESOLVED).sum())})
        if level == max_depth:
            _paint(label, rest[st == pub.UNRESOLVED], 1, 2)
            break
        child = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        ijk = (2 * rest[:, None, :] + child[None]).reshape(-1, 3)
        level += 1
    return PruneResult(method, leaf_n, label, rlevel, nbox, int(cost), t_cert, per_level)


def _paint(arr, ijk, span, val):
    if len(ijk) == 0:
        return
    off = np.stack(np.meshgrid(*[np.arange(span)] * 3, indexing='ij'), -1).reshape(-1, 3)
    idx = (ijk[:, None, :] * span + off[None]).reshape(-1, 3)
    arr[idx[:, 0], idx[:, 1], idx[:, 2]] = val


def corner_points(dom_lo, dom_hi, leaf_n, mask=None):
    """Grid corner points of all leaf cells, or only of cells where ``mask`` is set.

    Returns (points, flat corner indices) with corners on the (n+1)^3 lattice.
    """
    leaf_n = np.asarray(leaf_n)
    axes = [np.linspace(dom_lo[k], dom_hi[k], leaf_n[k] + 1) for k in range(3)]
    if mask is None:
        I, J, K = np.meshgrid(*[np.arange(n + 1) for n in leaf_n], indexing='ij')
        idx = np.stack([I.ravel(), J.ravel(), K.ravel()], 1)
    else:
        cells = np.argwhere(mask)
        off = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])
        idx = (cells[:, None, :] + off[None]).reshape(-1, 3)
        flat = np.ravel_multi_index(idx.T, tuple(leaf_n + 1))
        flat = np.unique(flat)
        idx = np.stack(np.unravel_index(flat, tuple(leaf_n + 1)), 1)
    pts = np.stack([axes[k][idx[:, k]] for k in range(3)], 1)
    return pts, np.ravel_multi_index(idx.T, tuple(leaf_n + 1))


def sign_change_violations(values, leaf_n, rejected):
    """Count sign-changing lattice edges that touch a rejected leaf cell.

    ``values`` is the (n+1)^3 corner lattice of field values. A sign change
    along an edge of a continuous field implies a zero on that edge, which
    lies in every closed cell containing the edge, so none may be rejected.
    """
    V = values.reshape(tuple(np.asarray(leaf_n) + 1))
    nx, ny, nz = leaf_n
    R = np.pad(rejected.astype(bool), 1)            # pad: index shift +1
    viol = 0
    total = 0
    for axis in range(3):
        a = np.take(V, range(V.shape[axis] - 1), axis=axis)
        b = np.take(V, range(1, V.shape[axis]), axis=axis)
        sc = (a > 0) != (b > 0)
        sc &= (a != 0) | (b != 0)
        total += int(sc.sum())
        e = np.argwhere(sc)
        # cells containing an edge along `axis` at lattice (i,j,k): the edge
        # spans cell index e[axis]; in the two other axes the cell index is
        # e[o] or e[o]-1.
        others = [o for o in range(3) if o != axis]
        hit = np.zeros(len(e), bool)
        for d1 in (0, -1):
            for d2 in (0, -1):
                c = e.copy()
                c[:, others[0]] += d1
                c[:, others[1]] += d2
                hit |= R[c[:, 0] + 1, c[:, 1] + 1, c[:, 2] + 1]
        viol += int(hit.sum())
    return viol, total
