"""Certified gradient and value bounds for the blended partition-of-unity field on boxes.

For a box B with positive certified weight sum ``W_min`` (so the blend branch
is taken everywhere in B):

    |grad F| <= max_j L_j + (1 / W_min) sum_j G_j max_k Delta_jk(B)

where j, k range over patches whose closed support meets B, ``L_j`` bounds
``|grad f_j|`` on B, ``G_j`` bounds ``|grad v_j|`` on B and ``Delta_jk`` bounds
``|f_j - f_k|`` on B (only pairs whose supports intersect are needed: where
``v_j(x) > 0`` the field ``F(x)`` is a convex combination of ``f_k(x)`` with
``v_k(x) > 0``).

Box statuses:

* ``CERTIFIED``: ``W_min > 0``; the blend formula holds on all of B.
* ``FALLBACK_FIXED``: B misses every support and one patch is provably the
  nearest support everywhere in B, so ``F = f_n`` on B.
* ``UNRESOLVED``: B may contain both blended and fallback points, or the
  fallback patch may change inside B. Nothing is claimed; subdivide.

Every quantity in the returned certificates is outward rounded.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import interval as iv
from .bounds import (box_center_radius, fit_constants, local_derivs_cert, b0_cert, b2_cert,
                     hess_norm_up, CHUNK_ELEMS)

CERTIFIED = 0
FALLBACK_FIXED = 1
UNRESOLVED = 2
STATUS_NAMES = {CERTIFIED: 'certified', FALLBACK_FIXED: 'fallback_fixed', UNRESOLVED: 'unresolved'}

KEYS6 = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
G_MAX = 135.0 / 64.0          # max_t 20 t (1-t)^3, at t = 1/4


def _pair_near_far(lo, hi, pts):
    """Outward (dmin, dmax) between paired boxes and points, all shape (n, 3)."""
    gap = np.maximum(np.maximum(lo - pts, pts - hi), 0.0)
    gap = np.maximum(iv.down(gap), 0.0)
    g2 = iv.down(iv.down(iv.down(gap[:, 0] * gap[:, 0]) + iv.down(gap[:, 1] * gap[:, 1]))
                 + iv.down(gap[:, 2] * gap[:, 2]))
    dmin = np.maximum(iv.down(np.sqrt(np.maximum(g2, 0.0))), 0.0)
    d = iv.up(np.maximum(np.abs(hi - pts), np.abs(lo - pts)))
    d2 = iv.up(iv.up(iv.up(d[:, 0] * d[:, 0]) + iv.up(d[:, 1] * d[:, 1])) + iv.up(d[:, 2] * d[:, 2]))
    return dmin, iv.up(np.sqrt(d2))


def _psi_interval(t):
    """Enclose psi(t) = (1-t)^4 (4t+1) for a point/interval t inside [0, 1]."""
    s = iv.sub(iv.point(np.ones_like(t[0])), t)
    s2 = iv.sqr(s)
    s4 = iv.sqr(s2)
    return iv.mul(s4, iv.add(iv.scale(t, 4.0), iv.point(np.ones_like(t[0]))))


def psi_lower_at(t_hi):
    """Lower bound of psi(t) for every t <= t_hi (psi is decreasing)."""
    t_hi = np.asarray(t_hi, dtype=np.float64)
    inside = t_hi < 1.0
    tc = np.where(inside, np.maximum(t_hi, 0.0), 0.0)
    val = np.maximum(_psi_interval(iv.point(tc))[0], 0.0)
    return np.where(inside, val, 0.0)


def psi_upper_at(t_lo):
    """Upper bound of psi(t) for every t >= t_lo."""
    t_lo = np.asarray(t_lo, dtype=np.float64)
    inside = t_lo < 1.0
    tc = np.where(inside, np.maximum(t_lo, 0.0), 0.0)
    return np.where(inside, _psi_interval(iv.point(tc))[1], 0.0)


def grad_weight_upper(t_lo, t_hi, R):
    """Upper bound of |grad v| = 20 t (1-t)^3 / R over radial parameter t in [t_lo, t_hi]."""
    t_lo = np.maximum(np.asarray(t_lo, dtype=np.float64), 0.0)
    t_hi = np.minimum(np.asarray(t_hi, dtype=np.float64), 1.0)
    live = t_lo < 1.0
    tstar = np.clip(0.25, t_lo, np.maximum(t_hi, t_lo))
    tstar = np.where(live, tstar, 0.0)
    t = iv.point(tstar)
    s = iv.sub(iv.point(np.ones_like(tstar)), t)
    g = iv.scale(iv.mul(t, iv.mul(iv.sqr(s), s)), 20.0)
    gmax = np.minimum(g[1], iv.up(G_MAX))
    return np.where(live, iv.up(gmax / iv.down(R)), 0.0)


def _segment_sum(x, starts, n_max, lower):
    """Rigorous lower/upper bound of segment sums of float64 values (reduceat)."""
    if len(x) == 0:
        return np.zeros(0)
    s = np.add.reduceat(x, starts)
    a = np.add.reduceat(np.abs(x), starts)
    g = iv.gamma_up(max(n_max - 1, 0))
    err = iv.up(iv.up(g * iv.up(a * iv.up(1.0 + g))) + n_max * 5e-324)
    return iv.down(s - err) if lower else iv.up(s + err)


class PUCertifier:
    """Precomputed patch data and box certification for one Snapshot."""

    def __init__(self, snap):
        self.snap = snap
        self.P = snap.num_patches
        self.centers = snap.patch_centers
        self.R = snap.patch_radii
        self.Rmax = float(self.R.max())
        self.tree = cKDTree(self.centers)
        self.consts = [fit_constants(f) for f in snap.fits]
        self.n_centers = np.array([len(f.centers) for f in snap.fits])
        self._build_pair_table()

    # ── patch pair constants ─────────────────────────────────────────
    def _build_pair_table(self):
        """A_jk = upper bound of sum |alpha| of the merged difference fit f_j - f_k.

        Centers that coincide exactly (bitwise) across patches are merged; the
        merged coefficient |a - b| is rounded up. Only pairs whose supports
        intersect are stored.
        """
        allc = np.concatenate([f.centers for f in self.snap.fits])
        _, inv = np.unique(allc, axis=0, return_inverse=True)
        inv = inv.ravel()
        offs = np.concatenate([[0], np.cumsum(self.n_centers)])
        ids, alph = [], []
        for j, f in enumerate(self.snap.fits):
            i = inv[offs[j]:offs[j + 1]]
            o = np.argsort(i, kind='stable')
            if len(np.unique(i)) != len(i):
                raise ValueError(f"patch {j} has repeated centers; merge rule undefined")
            ids.append(i[o])
            alph.append(f.alpha[o])
        pairs = self.tree.query_pairs(2 * self.Rmax * (1 + 1e-9), output_type='ndarray')
        keys, vals, shared_frac, ident = [], [], [], []
        for j, k in pairs:
            dist = np.linalg.norm(self.centers[j] - self.centers[k])
            if dist > (self.R[j] + self.R[k]) * (1 + 1e-9):
                continue
            _, ij, ik = np.intersect1d(ids[j], ids[k], assume_unique=True, return_indices=True)
            mj = np.ones(len(ids[j]), bool); mj[ij] = False
            mk = np.ones(len(ids[k]), bool); mk[ik] = False
            diff = iv.up(np.abs(alph[j][ij] - alph[k][ik]))
            parts = np.concatenate([diff, np.abs(alph[j][mj]), np.abs(alph[k][mk])])
            A = float(iv.sum_upper(parts))
            fj, fk = self.snap.fits[j], self.snap.fits[k]
            same = (not mj.any() and not mk.any() and np.array_equal(alph[j], alph[k])
                    and np.array_equal(fj.linear, fk.linear) and fj.constant == fk.constant)
            keys += [j * self.P + k, k * self.P + j]
            vals += [A, A]
            ident += [same, same]
            shared_frac.append(len(ij) / max(min(len(ids[j]), len(ids[k])), 1))
        o = np.argsort(keys)
        self.pair_keys = np.asarray(keys, dtype=np.int64)[o]
        self.pair_A = np.asarray(vals, dtype=np.float64)[o]
        self.pair_identical = np.asarray(ident, dtype=bool)[o]
        self.pair_shared_fraction = np.asarray(shared_frac)

    def pair_lookup(self, j, k):
        """Return (A_jk, supports_intersect, identical_fits) for arrays of patch indices."""
        key = j.astype(np.int64) * self.P + k
        pos = np.searchsorted(self.pair_keys, key)
        pos = np.minimum(pos, max(len(self.pair_keys) - 1, 0))
        hit = (len(self.pair_keys) > 0) & (self.pair_keys[pos] == key) if len(self.pair_keys) else np.zeros(len(key), bool)
        if not len(self.pair_keys):
            return np.zeros(len(key)), hit, np.zeros(len(key), bool)
        return np.where(hit, self.pair_A[pos], 0.0), hit, hit & self.pair_identical[pos]

    # ── candidate enumeration ────────────────────────────────────────
    def active_pairs(self, lo, hi):
        """All (box, patch) pairs whose closed support meets the box, with distances."""
        x0, rho = box_center_radius(lo, hi)
        r = (rho + self.Rmax) * (1 + 1e-9) + 1e-12
        lists = self.tree.query_ball_point(x0, r)
        cnt = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
        b = np.repeat(np.arange(len(lo)), cnt)
        p = np.fromiter((i for l in lists for i in l), dtype=np.int64, count=int(cnt.sum()))
        dmin, dmax = _pair_near_far(lo[b], hi[b], self.centers[p])
        keep = dmin <= self.R[p]
        b, p, dmin, dmax = b[keep], p[keep], dmin[keep], dmax[keep]
        o = np.lexsort((p, b))
        return x0, rho, b[o], p[o], dmin[o], dmax[o]

    def _fallback_patch(self, x0, rho, lo, hi, k=16):
        """Patch that is provably the nearest support on each box, else -1."""
        k = min(k, self.P)
        dd, ii = self.tree.query(x0, k=k)
        dd, ii = np.atleast_2d(dd), np.atleast_2d(ii)
        m = len(x0)
        b = np.repeat(np.arange(m), k)
        dmin, dmax = _pair_near_far(lo[b], hi[b], self.centers[ii.ravel()])
        Rk = self.R[ii.ravel()]
        smin = iv.down(dmin - Rk).reshape(m, k)
        smax = iv.up(dmax - Rk).reshape(m, k)
        j = np.argmin(smax, axis=1)
        best = smax[np.arange(m), j]
        others = smin.copy()
        others[np.arange(m), j] = np.inf
        # patches beyond the k queried centers: |x - xi| >= dd_k - rho for x in B
        beyond = iv.down(iv.down(dd[:, -1] - rho) - self.Rmax) if k < self.P else np.full(m, np.inf)
        ok = best < np.minimum(others.min(axis=1), beyond)
        return np.where(ok, ii[np.arange(m), j], -1)

    # ── main entry ───────────────────────────────────────────────────
    def certify(self, lo, hi, variants=True):
        """Certify a batch of boxes.

        Returns:
            dict of per-box arrays: ``status``, ``rho``, ``W_min``, ``n_active``,
            gradient bounds ``L_best`` (minimum of ``L_pu`` and
            ``L_interval``), ``L_pu`` (blend formula with disagreement
            refinement), ``L_interval`` (direct interval evaluation of the
            quotient derivative), ``L_B0`` (B0 propagated through the blend bound),
            ``L_simple`` (minimum local bounds with the simple disagreement
            bound), the deliberately incomplete diagnostic ``L_local_only``,
            the blend contributions, the center value enclosure ``F0`` and
            the hull sign certificate ``hull_sign`` (+1/-1 if every
            contributing fit is strictly positive/negative on the box, else 0),
            plus a ``cost`` count of kernel-term evaluations.
        """
        lo = np.ascontiguousarray(lo, dtype=np.float64)
        hi = np.ascontiguousarray(hi, dtype=np.float64)
        m = len(lo)
        x0, rho, b, p, dmin, dmax = self.active_pairs(lo, hi)
        npair = len(b)
        out = {k: np.zeros(m) for k in ('W_min', 'W_max', 'L_best', 'L_pu', 'L_interval', 'L_B0', 'L_simple', 'L_local_only',
                                         'blend_best', 'blend_B0', 'blend_simple', 'L_B1_ideal',
                                         'L_local_B0', 'L_local_B2', 'L_local_B3')}
        out['rho'] = rho
        out['n_active'] = np.bincount(b, minlength=m)
        out['F0_lo'] = np.full(m, -np.inf)
        out['F0_hi'] = np.full(m, np.inf)
        out['hull_sign'] = np.zeros(m, dtype=np.int8)
        out['status'] = np.full(m, UNRESOLVED, dtype=np.int8)
        out['fallback_patch'] = np.full(m, -1)

        # ---- per-pair local quantities, grouped by patch ----------------
        Rp = self.R[p]
        t_lo = np.maximum(iv.down(dmin / Rp), 0.0)
        t_hi = iv.up(dmax / Rp)
        psi_lo = psi_lower_at(t_hi)
        Gp = grad_weight_upper(t_lo, t_hi, Rp)
        B0 = np.empty(npair); B2 = np.empty(npair); B3 = np.empty(npair)
        f0 = (np.empty(npair), np.empty(npair))
        gpair = [(np.empty(npair), np.empty(npair)) for _ in range(3)]
        Hpair = {key: (np.empty(npair), np.empty(npair)) for key in KEYS6}
        vr = (np.empty(npair), np.empty(npair))
        A_p = np.array([c.A_up for c in self.consts])[p]
        cost = 0
        order = np.argsort(p, kind='stable')
        ps, starts = np.unique(p[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        for j, s, e in zip(ps, starts, ends):
            sel = order[s:e]
            fit, k = self.snap.fits[j], self.consts[j]
            bl, bh = lo[b[sel]], hi[b[sel]]
            d = local_derivs_cert(fit, x0[b[sel]])
            cost += len(sel) * len(fit.centers)
            B0[sel] = b0_cert(fit, bl, bh, k)
            B2[sel] = b2_cert(fit, bl, bh, k)
            f0[0][sel], f0[1][sel] = d['f']
            for q in range(3):
                gpair[q][0][sel], gpair[q][1][sel] = d['g'][q]
            for key in KEYS6:
                Hpair[key][0][sel], Hpair[key][1][sel] = d['H'][key]
        rp = rho[b]
        gn = iv.norm_upper(gpair)
        Hn = hess_norm_up(Hpair)
        rr = iv.up(rp * rp)
        B3 = iv.up(iv.up(gn + iv.up(rp * Hn)) + iv.up(3.0 * iv.up(A_p * rr)))
        rad = iv.up(iv.up(iv.up(rp * gn) + iv.up(0.5 * iv.up(Hn * rr))) + iv.up(A_p * iv.up(rp * rr)))
        vr = (iv.down(f0[0] - rad), iv.up(f0[1] + rad))
        Lmin = np.minimum(np.minimum(B0, B2), B3)
        B1 = np.array([c.linear_norm_est + 3.0 * c.S_est for c in self.consts])[p]

        # ---- per-box aggregation ----------------------------------------
        if npair:
            bstart = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
            boxes = b[bstart]
            nmax = int(out['n_active'].max())
            out['W_min'][boxes] = np.maximum(_segment_sum(psi_lo, bstart, nmax, lower=True), 0.0)
            for name, arr in (('L_local_only', Lmin), ('L_local_B0', B0), ('L_local_B2', B2),
                              ('L_local_B3', B3), ('L_B1_ideal', B1)):
                out[name][boxes] = np.maximum.reduceat(arr, bstart)
            # hull sign: every contributing fit strictly one-signed on the box
            pos = np.minimum.reduceat((vr[0] > 0).astype(np.int8), bstart)
            neg = np.minimum.reduceat((vr[1] < 0).astype(np.int8), bstart)
            out['hull_sign'][boxes] = pos - neg
            # center value enclosure F(x0): weights at x0
            self._center_values(x0, b, p, f0, bstart, boxes, nmax, out)
            self._blend_terms(b, p, rho, f0, gpair, Hpair, B0, Lmin, vr, Gp, bstart, out)
            self._interval_quotient(b, p, rho, gpair, gn, Hn, A_p, vr, Gp, psi_lo, dmin, bstart,
                                    boxes, nmax, out)

        covered = out['W_min'] > 0
        out['status'][covered] = CERTIFIED
        out['L_pu'] = np.where(covered, iv.up(out['L_local_only'] + out['blend_best']), np.inf)
        out['L_interval'] = np.where(covered, out['L_interval'], np.inf)
        out['L_best'] = np.minimum(out['L_pu'], out['L_interval'])
        out['L_B0'] = np.where(covered, iv.up(out['L_local_B0'] + out['blend_B0']), np.inf)
        out['L_simple'] = np.where(covered, iv.up(out['L_local_only'] + out['blend_simple']), np.inf)
        out['hull_sign'][~covered] = 0

        # ---- boxes missing every support: fixed fallback? ----------------
        empty = np.flatnonzero(out['n_active'] == 0)
        if len(empty):
            fb = self._fallback_patch(x0[empty], rho[empty], lo[empty], hi[empty])
            okb = empty[fb >= 0]
            out['fallback_patch'][empty] = fb
            for j in np.unique(fb[fb >= 0]):
                sel = okb[fb[fb >= 0] == j]
                fit, k = self.snap.fits[j], self.consts[j]
                d = local_derivs_cert(fit, x0[sel])
                cost += len(sel) * len(fit.centers)
                L0 = b0_cert(fit, lo[sel], hi[sel], k)
                L2 = b2_cert(fit, lo[sel], hi[sel], k)
                gn_ = iv.norm_upper(d['g']); Hn_ = hess_norm_up(d['H'])
                r_ = rho[sel]; rr_ = iv.up(r_ * r_)
                L3 = iv.up(iv.up(gn_ + iv.up(r_ * Hn_)) + iv.up(3.0 * iv.up(k.A_up * rr_)))
                rad_ = iv.up(iv.up(iv.up(r_ * gn_) + iv.up(0.5 * iv.up(Hn_ * rr_)))
                             + iv.up(k.A_up * iv.up(r_ * rr_)))
                Lm = np.minimum(np.minimum(L0, L2), L3)
                out['status'][sel] = FALLBACK_FIXED
                out['L_best'][sel] = Lm
                out['L_pu'][sel] = Lm
                out['L_interval'][sel] = Lm
                out['L_simple'][sel] = Lm
                out['L_local_only'][sel] = Lm
                out['L_B0'][sel] = L0
                out['F0_lo'][sel], out['F0_hi'][sel] = d['f']
                lo_r, hi_r = iv.down(d['f'][0] - rad_), iv.up(d['f'][1] + rad_)
                out['hull_sign'][sel] = (lo_r > 0).astype(np.int8) - (hi_r < 0).astype(np.int8)
        out['cost_bound_terms'] = cost
        return out

    def _interval_quotient(self, b, p, rho, gpair, gn, Hn, A_p, vr, Gp, psi_lo, dmin, bstart,
                           boxes, nmax, out):
        """Direct interval evaluation of grad F = (sum v grad f + sum (f - F) grad v) / W.

        Every factor is replaced by its enclosure over the whole box, so the
        result suffers from interval dependency (the same v, f and F appear in
        numerator and denominator); it is reported as an independent comparison.
        """
        rp = rho[b]
        Rp = self.R[p]
        # gradient of f_j anywhere in the box: g(x0) +- rho (|H(x0)| + 3 A rho)
        dev = iv.up(rp * iv.up(Hn + iv.up(3.0 * iv.up(A_p * rp))))
        t_lo = np.maximum(iv.down(dmin / Rp), 0.0)
        v_hi = psi_upper_at(t_lo)
        v = (psi_lo, v_hi)
        W_max = _segment_sum(v_hi, bstart, nmax, lower=False)
        out['W_max'][boxes] = W_max
        Fl = np.minimum.reduceat(vr[0], bstart)
        Fh = np.maximum.reduceat(vr[1], bstart)
        nb = np.repeat(np.arange(len(bstart)), np.diff(np.append(bstart, len(b))))
        diff_mag = iv.mag(iv.sub(vr, (Fl[nb], Fh[nb])))
        t2 = iv.up(Gp * diff_mag)                          # |(f_j - F) d v_j / dx_k| <= G_j |f_j - F|
        t2sum = _segment_sum(t2, bstart, nmax, lower=False)
        comps = []
        Wm = out['W_min'][boxes]
        for k in range(3):
            gk = (iv.down(gpair[k][0] - dev), iv.up(gpair[k][1] + dev))
            prod = iv.mul(v, gk)
            n_lo = iv.down(_segment_sum(prod[0], bstart, nmax, lower=True) - t2sum)
            n_hi = iv.up(_segment_sum(prod[1], bstart, nmax, lower=False) + t2sum)
            pos = Wm > 0
            q_lo = np.full(len(boxes), -np.inf); q_hi = np.full(len(boxes), np.inf)
            if pos.any():
                q = iv.div((n_lo[pos], n_hi[pos]), (Wm[pos], W_max[pos]))
                q_lo[pos], q_hi[pos] = q
            comps.append((q_lo, q_hi))
        with np.errstate(invalid='ignore', over='ignore'):
            out['L_interval'][boxes] = iv.norm_upper(comps)

    def _center_values(self, x0, b, p, f0, bstart, boxes, nmax, out):
        """Enclose F(x0) for boxes with W(x0) > 0 (convex hull intersected with the quotient)."""
        c = self.centers[p]
        X = x0[b]
        comps = [iv.sub(iv.point(X[:, k]), iv.point(c[:, k])) for k in range(3)]
        r2 = iv.add(iv.add(iv.sqr(comps[0]), iv.sqr(comps[1])), iv.sqr(comps[2]))
        r = iv.sqrt(r2)
        Rp = self.R[p]
        t = (np.maximum(iv.down(r[0] / Rp), 0.0), iv.up(r[1] / Rp))
        v = (psi_lower_at(t[1]), psi_upper_at(t[0]))
        vf = iv.mul(v, f0)
        num_lo = _segment_sum(vf[0], bstart, nmax, True)
        num_hi = _segment_sum(vf[1], bstart, nmax, False)
        den_lo = _segment_sum(v[0], bstart, nmax, True)
        den_hi = _segment_sum(v[1], bstart, nmax, False)
        live = v[1] > 0
        hlo = np.minimum.reduceat(np.where(live, f0[0], np.inf), bstart)
        hhi = np.maximum.reduceat(np.where(live, f0[1], -np.inf), bstart)
        lo_, hi_ = hlo, hhi
        ok = den_lo > 0
        if ok.any():
            q = iv.div((num_lo[ok], num_hi[ok]), (den_lo[ok], den_hi[ok]))
            lo_ = lo_.copy(); hi_ = hi_.copy()
            lo_[ok] = np.maximum(lo_[ok], q[0])
            hi_[ok] = np.minimum(hi_[ok], q[1])
        # Valid only where W(x0) > 0, which W_min > 0 implies; others stay infinite.
        good = (den_lo > 0) | (out['W_min'][boxes] > 0)
        out['F0_lo'][boxes[good]] = lo_[good]
        out['F0_hi'][boxes[good]] = hi_[good]

    def _blend_terms(self, b, p, rho, f0, gpair, Hpair, B0, Lmin, vr, Gp, bstart, out):
        """Sum_j G_j max_k Delta_jk / W_min for three disagreement bounds."""
        counts = np.diff(np.append(bstart, len(b)))
        sq = counts.astype(np.int64) ** 2
        # chunk boxes so the (j, k) combination arrays stay bounded
        limit = 4_000_000
        cut = [0]
        acc = 0
        for i, s in enumerate(sq):
            if acc + s > limit and acc > 0:
                cut.append(i); acc = 0
            acc += s
        cut.append(len(bstart))
        for c0, c1 in zip(cut[:-1], cut[1:]):
            self._blend_chunk(b, p, rho, f0, gpair, Hpair, B0, Lmin, vr, Gp,
                              bstart[c0:c1], counts[c0:c1], out)

    def _blend_chunk(self, b, p, rho, f0, gpair, Hpair, B0, Lmin, vr, Gp, bst, cnt, out):
        # combos (a, c) of pair rows within each box, a != c
        rep = np.repeat(cnt, cnt)                                   # per row a: count of its box
        rows = np.concatenate([np.arange(s, s + n) for s, n in zip(bst, cnt)])
        a = np.repeat(rows, rep)
        box_start_a = np.repeat(np.repeat(bst, cnt), rep)
        within = np.arange(len(a)) - np.repeat(np.cumsum(rep) - rep, rep)
        c = box_start_a + within
        keep = a != c
        A_jk, inter, same = self.pair_lookup(p[a], p[c])
        # Identical fits contribute exactly zero disagreement; drop them.
        keep &= inter & ~same
        a, c, A_jk = a[keep], c[keep], A_jk[keep]
        if len(a) == 0:
            return
        bx = b[a]
        rp = rho[bx]
        d0 = iv.mag(iv.sub((f0[0][a], f0[1][a]), (f0[0][c], f0[1][c])))
        D_B0 = iv.up(d0 + iv.up(rp * iv.up(B0[a] + B0[c])))
        D_simple = iv.up(d0 + iv.up(rp * iv.up(Lmin[a] + Lmin[c])))
        D_range = np.maximum(iv.up(vr[1][a] - vr[0][c]), iv.up(vr[1][c] - vr[0][a]))
        gd = [iv.sub((g[0][a], g[1][a]), (g[0][c], g[1][c])) for g in gpair]
        Hd = {key: iv.sub((h[0][a], h[1][a]), (h[0][c], h[1][c])) for key, h in Hpair.items()}
        rr = iv.up(rp * rp)
        D_taylor = iv.up(iv.up(iv.up(d0 + iv.up(rp * iv.norm_upper(gd)))
                               + iv.up(0.5 * iv.up(hess_norm_up(Hd) * rr)))
                         + iv.up(A_jk * iv.up(rp * rr)))
        D_best = np.minimum(np.minimum(D_simple, D_range), D_taylor)
        # max over c for each a (a is sorted ascending)
        astart = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
        rows_a = a[astart]
        bx_a = b[rows_a]
        Wm = out['W_min'][bx_a]
        for name, D in (('blend_B0', D_B0), ('blend_simple', D_simple), ('blend_best', D_best)):
            mx = np.maximum.reduceat(D, astart)
            term = iv.up(Gp[rows_a] * mx)
            # sum terms per box (upward), then divide by W_min (downward bounded)
            bs = np.flatnonzero(np.r_[True, bx_a[1:] != bx_a[:-1]])
            nmax = int(np.diff(np.append(bs, len(bx_a))).max())
            tot = _segment_sum(term, bs, nmax, lower=False)
            bb = bx_a[bs]
            with np.errstate(divide='ignore'):
                val = np.where(Wm[bs] > 0, iv.up(tot / np.where(Wm[bs] > 0, Wm[bs], 1.0)), np.inf)
            out[name][bb] = val
