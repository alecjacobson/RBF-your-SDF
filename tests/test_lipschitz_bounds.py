"""Mathematical regression tests for the Lipschitz-bound experiment.

Run with ``uv run python -m unittest tests/test_lipschitz_bounds.py``.
"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demo' / 'additional_experiments'))

from lipschitz import interval as iv                       # noqa: E402
from lipschitz.bounds import (Fit, evaluate, fit_constants, b0_cert, b2_cert, b3_cert,  # noqa: E402
                              b1_estimate, local_derivs_cert, value_range_cert)
from lipschitz.adapter import Snapshot, evaluate as pu_evaluate      # noqa: E402
from lipschitz import pu_bounds as pub                     # noqa: E402
from lipschitz.certify import (adaptive_prune, corner_points, sign_change_violations,  # noqa: E402
                               METHODS)


def box(center, half):
    c = np.atleast_2d(np.asarray(center, dtype=np.float64))
    h = np.broadcast_to(np.asarray(half, dtype=np.float64), c.shape)
    return c - h, c + h


def sample_box(lo, hi, n, rng):
    return lo + (hi - lo) * rng.random((n, 3))


def const_fit(value, center):
    return Fit.exact(np.atleast_2d(center), [0.0], [0, 0, 0], value)


def make_snapshot(centers, radii, fits):
    return Snapshot(np.asarray(centers, float), np.asarray(radii, float), fits,
                    {'partition': 'sphere'})


class TestInterval(unittest.TestCase):
    def test_enclosures_contain_exact_rational_results(self):
        """Check add/mul/sqr/sum enclosures against exact rational arithmetic."""
        rng = np.random.default_rng(0)
        a = rng.normal(size=200) * 10.0 ** rng.integers(-8, 8, 200)
        b = rng.normal(size=200) * 10.0 ** rng.integers(-8, 8, 200)
        A, B = iv.point(a), iv.point(b)
        for op, exact in ((iv.add, lambda x, y: x + y), (iv.sub, lambda x, y: x - y),
                          (iv.mul, lambda x, y: x * y)):
            lo, hi = op(A, B)
            for i in range(len(a)):
                e = exact(Fraction(a[i]), Fraction(b[i]))
                self.assertLessEqual(Fraction(lo[i]), e)
                self.assertGreaterEqual(Fraction(hi[i]), e)
        lo, hi = iv.isum(A, axis=0)
        e = sum(Fraction(x) for x in a)
        self.assertLessEqual(Fraction(float(lo)), e)
        self.assertGreaterEqual(Fraction(float(hi)), e)

    def test_sqrt_encloses(self):
        """Check that sqrt enclosure brackets the exact root (squared bounds)."""
        x = np.array([2.0, 3.0, 1e-300, 7.5e10])
        lo, hi = iv.sqrt(iv.point(x))
        for i in range(len(x)):
            self.assertLessEqual(Fraction(lo[i]) ** 2, Fraction(x[i]))
            self.assertGreaterEqual(Fraction(hi[i]) ** 2, Fraction(x[i]))


class TestLocalBounds(unittest.TestCase):
    def test_affine_field_bound_equals_affine_gradient(self):
        """Return the affine gradient norm for all bounds when radial coefficients vanish."""
        q = np.random.default_rng(1).normal(size=(5, 3))
        fit = Fit.exact(q, np.zeros(5), [3.0, 4.0, 0.0], 1.0)
        lo, hi = box([0.3, 0.1, -0.2], 0.25)
        for bnd in (b0_cert(fit, lo, hi), b2_cert(fit, lo, hi), b3_cert(fit, lo, hi)):
            self.assertAlmostEqual(float(bnd[0]), 5.0, places=12)
            self.assertGreaterEqual(float(bnd[0]), 5.0)

    def test_collinear_moment_bound_is_sharp(self):
        """Give moment bound 6 for centers -e,0,e with coefficients 1,-2,1 and attain it."""
        e = np.array([0.0, 0.6, 0.8])
        fit = Fit.exact(np.stack([-e, 0 * e, e]), [1.0, -2.0, 1.0], [0, 0, 0], 0.0)
        self.assertAlmostEqual(b1_estimate(fit), 6.0, places=12)
        lo, hi = box(5 * e, 0.5)
        self.assertAlmostEqual(float(b2_cert(fit, lo, hi)[0]), 6.0, places=10)
        _, g = evaluate(fit, np.stack([2 * e, 7 * e, 30 * e]))
        np.testing.assert_allclose(np.linalg.norm(g, axis=1), 6.0, rtol=1e-12)

    def test_perturbed_moments_break_b1_but_not_b2(self):
        """Show B1 underestimates when moments are nonzero while B2 stays valid on a box."""
        e = np.array([1.0, 0.0, 0.0])
        fit = Fit.exact(np.stack([-e, 0 * e, e]), [1.0, -2.0, 1.0 + 1e-3], [0, 0, 0], 0.0)
        far = 200 * e
        lo, hi = box(far, 1.0)
        _, g = evaluate(fit, far[None])
        self.assertGreater(np.linalg.norm(g), b1_estimate(fit))
        self.assertGreaterEqual(float(b2_cert(fit, lo, hi)[0]), np.linalg.norm(g))

    def test_query_at_center_is_finite(self):
        """Evaluate gradient and Hessian at a center without division by zero."""
        q = np.random.default_rng(2).normal(size=(6, 3))
        fit = Fit.exact(q, np.random.default_rng(3).normal(size=6), [0.1, 0, 0], 0.0)
        f, g, H = evaluate(fit, q[:1], hessian=True)
        self.assertTrue(np.all(np.isfinite(H)) and np.all(np.isfinite(g)))
        d = local_derivs_cert(fit, q[:1])
        for val in d['H'].values():
            self.assertTrue(np.all(np.isfinite(val[0])) and np.all(np.isfinite(val[1])))
        lo, hi = box(q[0], 1e-3)
        self.assertTrue(np.isfinite(b3_cert(fit, lo, hi)[0]))

    def test_shrinking_boxes_b3_converges_to_center_gradient(self):
        """Converge B3 to |grad f(x0)| as the box shrinks."""
        rng = np.random.default_rng(4)
        q = rng.normal(size=(20, 3))
        fit = Fit.exact(q, rng.normal(size=20), rng.normal(size=3), 0.0)
        x0 = np.array([0.1, -0.3, 0.2])
        _, g = evaluate(fit, x0[None])
        gn = np.linalg.norm(g)
        prev = np.inf
        for h in (1e-1, 1e-2, 1e-3, 1e-5):
            lo, hi = box(x0, h)
            val = float(b3_cert(fit, lo, hi)[0])
            self.assertGreaterEqual(val, gn)
            self.assertLess(val, prev)
            prev = val
        self.assertLess(prev - gn, 1e-3 * gn)

    def test_bounds_dominate_sampled_gradients(self):
        """Keep every certified bound above sampled gradient norms (random stress)."""
        rng = np.random.default_rng(5)
        for trial in range(20):
            n = rng.integers(4, 30)
            q = rng.normal(size=(n, 3))
            fit = Fit.exact(q, rng.normal(size=n), rng.normal(size=3), rng.normal())
            c = rng.normal(size=3)
            lo, hi = box(c, rng.uniform(0.01, 1.0, 3))
            pts = sample_box(lo, hi, 500, rng)
            f, g = evaluate(fit, pts)
            obs = np.linalg.norm(g, axis=1).max()
            k = fit_constants(fit)
            for bnd in (b0_cert(fit, lo, hi, k), b2_cert(fit, lo, hi, k), b3_cert(fit, lo, hi, k)):
                self.assertGreaterEqual(float(bnd[0]), obs)
            vlo, vhi = value_range_cert(fit, lo, hi, k)
            self.assertLessEqual(float(vlo[0]), f.min())
            self.assertGreaterEqual(float(vhi[0]), f.max())

    def test_physical_rescaling_preserves_bounds(self):
        """Keep the dimensionless bound when space and values scale together."""
        rng = np.random.default_rng(6)
        q = rng.normal(size=(10, 3))
        a = rng.normal(size=10)
        a -= a.mean()
        p = rng.normal(size=3)
        fit = Fit.exact(q, a, p, 0.3)
        s = 8.0     # power of two: scaled coefficients stay exactly representable
        fit_s = Fit.exact(s * q, a / s ** 2, p, 0.3 * s)
        lo, hi = box([0.2, 0.1, 0.0], 0.3)
        for fn in (b0_cert, b2_cert, b3_cert):
            np.testing.assert_allclose(fn(fit, lo, hi), fn(fit_s, s * lo, s * hi), rtol=1e-12)


class TestBlendBounds(unittest.TestCase):
    def test_different_constant_patches_need_blend_term(self):
        """Bound the varying blend of two constant patches although each local bound is zero."""
        snap = make_snapshot([[-0.5, 0, 0], [0.5, 0, 0]], [1.0, 1.0],
                             [const_fit(0.0, [-0.5, 0, 0]), const_fit(1.0, [0.5, 0, 0])])
        cert = pub.PUCertifier(snap)
        lo, hi = box([0.0, 0.0, 0.0], 0.1)
        out = cert.certify(lo, hi)
        self.assertEqual(out['status'][0], pub.CERTIFIED)
        self.assertLess(out['L_local_only'][0], 1e-100)   # zero up to outward rounding
        pts = sample_box(lo, hi, 2000, np.random.default_rng(7))
        _, g = pu_evaluate(snap, pts)
        obs = np.linalg.norm(g, axis=1).max()
        self.assertGreater(obs, 0.5)
        for name in ('L_best', 'L_pu', 'L_interval', 'L_B0', 'L_simple'):
            self.assertGreaterEqual(out[name][0], obs)

    def test_identical_patches_cancel_blend_term(self):
        """Give zero blend contribution for identical fits, even with small denominators."""
        rng = np.random.default_rng(8)
        q = rng.normal(size=(8, 3)) * 0.3
        a = rng.normal(size=8)
        fit = Fit.exact(q, a, [0.2, 0, 0], 0.1)
        fit2 = Fit.exact(q.copy(), a.copy(), [0.2, 0, 0], 0.1)
        snap = make_snapshot([[-0.5, 0, 0], [0.5, 0, 0]], [1.0, 1.0], [fit, fit2])
        cert = pub.PUCertifier(snap)
        lo, hi = box([0.0, 0.7, 0.0], 0.05)      # near both support boundaries: small W
        out = cert.certify(lo, hi)
        self.assertEqual(out['status'][0], pub.CERTIFIED)
        self.assertLess(out['W_min'][0], 1e-3)
        self.assertEqual(out['blend_best'][0], 0.0)
        # equal up to the final outward rounding of max_j L_j + 0
        self.assertLessEqual(out['L_best'][0], np.nextafter(out['L_local_only'][0], np.inf))

    def test_partial_support_overlap_is_enumerated(self):
        """Enumerate a patch whose support only clips the box corner."""
        snap = make_snapshot([[0, 0, 0], [1.9, 0, 0]], [1.0, 1.0],
                             [const_fit(0.0, [0, 0, 0]), const_fit(1.0, [1.9, 0, 0])])
        cert = pub.PUCertifier(snap)
        lo, hi = np.array([[0.5, -0.1, -0.1]]), np.array([[0.95, 0.1, 0.1]])
        _, _, b, p, _, _ = cert.active_pairs(lo, hi)
        self.assertEqual(sorted(p.tolist()), [0, 1])

    def test_uncertain_coverage_is_unresolved(self):
        """Mark a box straddling the edge of coverage unresolved instead of certifying it."""
        snap = make_snapshot([[0, 0, 0]], [1.0], [const_fit(0.0, [0, 0, 0])])
        cert = pub.PUCertifier(snap)
        lo, hi = box([1.0, 0, 0], 0.1)
        out = cert.certify(lo, hi)
        self.assertEqual(out['status'][0], pub.UNRESOLVED)
        self.assertEqual(out['hull_sign'][0], 0)

    def test_nearest_patch_jump_blocks_certificate(self):
        """Detect the fallback jump between disjoint constant patches and refuse to certify across it."""
        snap = make_snapshot([[-2, 0, 0], [2, 0, 0]], [1.0, 1.0],
                             [const_fit(-1.0, [-2, 0, 0]), const_fit(1.0, [2, 0, 0])])
        F, _, info = pu_evaluate(snap, np.array([[-1e-9, 0, 0], [1e-9, 0, 0]]), return_info=True)
        self.assertTrue(np.all(info['branch'] == 1))
        self.assertAlmostEqual(F[1] - F[0], 2.0)
        cert = pub.PUCertifier(snap)
        out = cert.certify(*box([0, 0, 0], 0.2))
        self.assertEqual(out['status'][0], pub.UNRESOLVED)
        out = cert.certify(*box([0.5, 0, 0], 0.2))
        self.assertEqual(out['status'][0], pub.FALLBACK_FIXED)
        self.assertEqual(out['fallback_patch'][0], 1)

    def test_random_blends_dominated_by_bounds(self):
        """Keep PU bounds and value enclosures above sampled values in random blends."""
        rng = np.random.default_rng(9)
        for trial in range(6):
            P = 6
            cs = rng.uniform(-1, 1, (P, 3))
            fits = []
            shared = rng.normal(size=(12, 3)) * 0.5
            for j in range(P):
                q = np.concatenate([shared, rng.normal(size=(4, 3)) * 0.5])
                fits.append(Fit.exact(q, rng.normal(size=16) * 0.1, rng.normal(size=3), rng.normal()))
            snap = make_snapshot(cs, rng.uniform(1.0, 1.6, P), fits)
            cert = pub.PUCertifier(snap)
            c = rng.uniform(-0.5, 0.5, (10, 3))
            lo, hi = box(c, 0.08)
            out = cert.certify(lo, hi)
            for i in np.flatnonzero(out['status'] == pub.CERTIFIED):
                pts = sample_box(lo[i], hi[i], 400, rng)
                F, g = pu_evaluate(snap, pts)
                obs = np.linalg.norm(g, axis=1).max()
                for name in ('L_best', 'L_pu', 'L_interval', 'L_B0', 'L_simple'):
                    self.assertGreaterEqual(out[name][i], obs)
                F0, _ = pu_evaluate(snap, (0.5 * (lo[i] + hi[i]))[None])
                self.assertLessEqual(out['F0_lo'][i], F0[0])
                self.assertGreaterEqual(out['F0_hi'][i], F0[0])
                if out['hull_sign'][i] > 0:
                    self.assertGreater(F.min(), 0)
                if out['hull_sign'][i] < 0:
                    self.assertLess(F.max(), 0)


class TestPruning(unittest.TestCase):
    def test_pruning_never_rejects_cells_cut_by_the_zero_set(self):
        """Reject only cells away from a known planar zero set, for every method."""
        q = np.array([[0.0, 0, 0], [0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]])
        a = np.array([0.5, -0.2, -0.2, -0.1])
        fits = [Fit.exact(q, a * 0.0, [1.0, 0.2, 0.0], -0.13) for _ in range(2)]
        fits.append(Fit.exact(q, a * 0.0, [1.0, 0.2, 0.0], -0.13))
        snap = make_snapshot([[-0.3, 0, 0], [0.3, 0, 0], [0, 0.3, 0]], [1.2, 1.2, 1.2], fits)
        cert = pub.PUCertifier(snap)
        lo, hi = np.full(3, -0.5), np.full(3, 0.5)
        cache = {}
        for method in METHODS:
            pr = adaptive_prune(cert, lo, hi, 2, 3, method, cache=cache)
            rejected = pr.leaf_label == 1
            self.assertGreater(rejected.mean(), 0.5)
            pts, _ = corner_points(lo, hi, pr.leaf_n)
            vals = pts @ np.array([1.0, 0.2, 0.0]) - 0.13
            viol, total = sign_change_violations(vals, pr.leaf_n, rejected)
            self.assertGreater(total, 0)
            self.assertEqual(viol, 0)
        # the checker itself must flag a rejection that crosses the zero set
        self.assertGreater(sign_change_violations(vals, pr.leaf_n, np.ones(pr.leaf_label.shape, bool))[0], 0)


class TestWitnesses(unittest.TestCase):
    def test_fallback_switch_jump_is_certified(self):
        """Certify the jump of the nearest-patch fallback between two disjoint constant patches."""
        from lipschitz.witness import verify_switch
        snap = make_snapshot([[-2, 0, 0], [2, 0, 0]], [1.0, 1.0],
                             [const_fit(-1.0, [-2, 0, 0]), const_fit(1.0, [2, 0, 0])])
        v = verify_switch(snap, {'point': [0.0, 0.0, 0.0], 'patches': [0, 1]})
        self.assertTrue(v['certified'])
        self.assertAlmostEqual(v['jump_lower_bound'], 2.0, places=9)
        self.assertLessEqual(v['jump_lower_bound'], 2.0)

    def test_union_boundary_corner_limits_depend_on_direction(self):
        """Certify direction-dependent limits where two support spheres meet outside all others."""
        from lipschitz.witness import verify_corner
        snap = make_snapshot([[-0.5, 0, 0], [0.5, 0, 0]], [1.0, 1.0],
                             [const_fit(0.0, [-0.5, 0, 0]), const_fit(1.0, [0.5, 0, 0])])
        p = np.array([0.0, np.sqrt(0.75), 0.0])
        v = verify_corner(snap, {'point': p, 'patches': [0, 1]})
        self.assertTrue(v['certified'])
        # numerical limits along both directions match the analytic prediction
        for lam, u in zip(v['lambda'], np.asarray(v['directions'])):
            F, _ = pu_evaluate(snap, (p + 1e-6 * u)[None])
            self.assertAlmostEqual(F[0], 1.0 / (lam + 1), places=4)
        self.assertGreater(v['limit_gap_lower_bound'], 0.5)


if __name__ == '__main__':
    unittest.main()
