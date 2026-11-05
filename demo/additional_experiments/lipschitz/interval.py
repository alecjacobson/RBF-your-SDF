"""Vectorized outward-rounded interval arithmetic on float64 numpy arrays.

An interval is a pair ``(lo, hi)`` of equally shaped float64 arrays. Every
elementary operation is computed in round-to-nearest and then widened by one
ulp outward with ``np.nextafter``. IEEE 754 guarantees that ``+ - * / sqrt``
are correctly rounded (error at most half an ulp, including in the subnormal
range), so the widened result encloses the exact real result.

Reductions (``isum``) do not widen per addition. Instead they use the a priori
bound for floating-point summation in any order (Higham, *Accuracy and
Stability of Numerical Algorithms*, 2nd ed., eq. 4.4 and the remark after it):
``|fl(sum x) - sum x| <= gamma_{n-1} * sum |x|`` with
``gamma_k = k u / (1 - k u)`` and ``u = 2**-53``. The bound itself is evaluated
with upward-rounded operations. This is a proved error bound, not an ad hoc
``1 + eps`` inflation.

Only the operations this experiment needs are implemented. All functions
assume round-to-nearest mode (the numpy default) and finite inputs; results
containing NaN or inf are rejected by ``check_finite``.
"""

from __future__ import annotations

import numpy as np

U = 2.0 ** -53
INF = np.inf


def down(x):
    """Return the next float64 toward -inf."""
    return np.nextafter(x, -INF)


def up(x):
    """Return the next float64 toward +inf."""
    return np.nextafter(x, INF)


def point(x):
    """Return the degenerate interval [x, x] for exact float64 values x."""
    x = np.asarray(x, dtype=np.float64)
    return x.copy(), x.copy()


def add(a, b):
    """Enclose a + b."""
    return down(a[0] + b[0]), up(a[1] + b[1])


def sub(a, b):
    """Enclose a - b."""
    return down(a[0] - b[1]), up(a[1] - b[0])


def neg(a):
    """Return -a (exact)."""
    return -a[1], -a[0]


def mul(a, b):
    """Enclose a * b for general sign intervals."""
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    lo = np.minimum(np.minimum(p[0], p[1]), np.minimum(p[2], p[3]))
    hi = np.maximum(np.maximum(p[0], p[1]), np.maximum(p[2], p[3]))
    return down(lo), up(hi)


def scale(a, s):
    """Enclose s * a for an exact float64 scalar or array s."""
    s = np.asarray(s, dtype=np.float64)
    lo = np.where(s >= 0, a[0] * s, a[1] * s)
    hi = np.where(s >= 0, a[1] * s, a[0] * s)
    return down(lo), up(hi)


def sqr(a):
    """Enclose a**2 (tight for intervals containing zero)."""
    lo2, hi2 = a[0] * a[0], a[1] * a[1]
    hi = np.maximum(lo2, hi2)
    lo = np.where((a[0] <= 0) & (a[1] >= 0), 0.0, np.minimum(lo2, hi2))
    return np.maximum(down(lo), 0.0), up(hi)


def sqrt(a):
    """Enclose sqrt(a) for a >= 0 (the lower end is clamped at zero)."""
    return np.maximum(down(np.sqrt(np.maximum(a[0], 0.0))), 0.0), up(np.sqrt(np.maximum(a[1], 0.0)))


def div(a, b):
    """Enclose a / b for intervals b with strictly positive lower bound."""
    if np.any(np.asarray(b[0]) <= 0):
        raise ValueError("div requires a strictly positive denominator interval")
    q = (a[0] / b[0], a[0] / b[1], a[1] / b[0], a[1] / b[1])
    lo = np.minimum(np.minimum(q[0], q[1]), np.minimum(q[2], q[3]))
    hi = np.maximum(np.maximum(q[0], q[1]), np.maximum(q[2], q[3]))
    return down(lo), up(hi)


def gamma_up(n):
    """Upper bound on gamma_n = n u / (1 - n u)."""
    n = float(n)
    if n * U >= 0.5:
        raise ValueError("summation too long for the gamma_n bound")
    return up(up(n * U) / down(1.0 - n * U))


def _sum_err(x, axis, n):
    # gamma_{n-1} * sum|x|, where sum|x| itself is overestimated by (1+gamma)
    # and the product rounded up. Plus n subnormal units for safety at underflow.
    s = np.sum(np.abs(x), axis=axis)
    g = gamma_up(max(n - 1, 0))
    s_up = up(s * up(1.0 + g))
    return up(up(g * s_up) + n * 5e-324)


def isum(a, axis=-1):
    """Enclose the exact sum of an interval array along ``axis``."""
    n = a[0].shape[axis]
    lo = np.sum(a[0], axis=axis)
    hi = np.sum(a[1], axis=axis)
    return down(lo - _sum_err(a[0], axis, n)), up(hi + _sum_err(a[1], axis, n))


def sum_upper(x, axis=-1):
    """Upper bound on the exact sum of nonnegative float64 values."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    return up(np.sum(x, axis=axis) + _sum_err(x, axis, n))


def mag(a):
    """Upper bound of |x| over the interval (exact)."""
    return np.maximum(np.abs(a[0]), np.abs(a[1]))


def mig(a):
    """Lower bound of |x| over the interval (exact)."""
    return np.where((a[0] <= 0) & (a[1] >= 0), 0.0, np.minimum(np.abs(a[0]), np.abs(a[1])))


def norm_upper(comps):
    """Upper bound of the Euclidean norm of a vector given component intervals.

    ``comps`` is a list of intervals; returns a float64 array.
    """
    s = np.zeros_like(np.asarray(comps[0][0]))
    for c in comps:
        m = mag(c)
        s = up(s + up(m * m))
    return up(np.sqrt(s))


def mul_up(a, b):
    """Upward-rounded product of nonnegative float64 values."""
    return up(np.asarray(a) * np.asarray(b))


def add_up(a, b):
    """Upward-rounded sum of float64 values."""
    return up(np.asarray(a) + np.asarray(b))


def contains(a, x):
    """Whether interval a contains x elementwise."""
    return (a[0] <= x) & (x <= a[1])


def check_finite(*arrays):
    """Raise if any bound is NaN or infinite (an enclosure would be void)."""
    for x in arrays:
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("non-finite value in interval computation")
