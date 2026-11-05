// Certified bounds for the partition-of-unity field on axis-aligned boxes.
//
// C++ counterpart of demo/additional_experiments/lipschitz/{bounds,pu_bounds}.py
// (Lipschitz experiment). Every quantity returned is an enclosure of the exact
// analytic field defined by the stored float64 coefficients (treated as exact
// reals) and the evaluator rules in pu_interpolator.cpp:
//
//   * O(1) compositions use std::nextafter after every operation (outward).
//   * Sums over a fit's centers are computed in round-to-nearest together with
//     the sum of term magnitudes, and enclosed by the a priori bound
//     |fl(sum) - sum| <= gamma_{n+K} * sum|terms| (Higham, ASNA 2nd ed., ch. 3-4),
//     where K covers the roundings inside each term (<= 12 here; 20 is used).
//
// The disagreement bound uses min(simple, value-range) only; the Python
// reference additionally uses a Taylor bound on merged difference fits, so its
// L_best can be smaller. Only the sphere partition is supported.

#include "pu_interpolator.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace sdf {
namespace {

constexpr double kU = 0x1p-53;
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr int kTermRoundings = 20;

inline double up(double x) { return std::nextafter(x, kInf); }
inline double dn(double x) { return std::nextafter(x, -kInf); }

/// Upper bound on gamma_k = k u / (1 - k u).
inline double gamma_up(long long k) {
    double ku = up(double(k) * kU);
    if (ku >= 0.5) throw std::runtime_error("certify_boxes: summation too long");
    return up(ku / dn(1.0 - ku));
}

/// Round-to-nearest accumulator with its magnitude sum, enclosed at the end.
struct Acc {
    double s = 0.0, a = 0.0;
    void add(double t) { s += t; a += std::abs(t); }
    void enclose(long long n, double& lo, double& hi) const {
        double g = gamma_up(n + kTermRoundings);
        double e = up(up(g * up(a * up(1.0 + g))) + double(n + 1) * DBL_TRUE_MIN * 4);
        lo = dn(s - e);
        hi = up(s + e);
    }
};

struct Ival { double lo, hi; };
inline Ival iadd(Ival a, Ival b) { return {dn(a.lo + b.lo), up(a.hi + b.hi)}; }
inline Ival isub(Ival a, Ival b) { return {dn(a.lo - b.hi), up(a.hi - b.lo)}; }
inline Ival imul(Ival a, Ival b) {
    double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    double lo = p[0], hi = p[0];
    for (double v : p) { lo = std::min(lo, v); hi = std::max(hi, v); }
    return {dn(lo), up(hi)};
}
inline double imag(Ival a) { return std::max(std::abs(a.lo), std::abs(a.hi)); }
inline double imig(Ival a) {
    if (a.lo <= 0.0 && a.hi >= 0.0) return 0.0;
    return std::min(std::abs(a.lo), std::abs(a.hi));
}

/// psi(t) = (1-t)^4 (4t+1) evaluated with all-downward (lower=true) or
/// all-upward rounding; valid for t in [0, 1] where every factor is >= 0.
inline double psi_round(double t, bool lower) {
    if (t >= 1.0) return 0.0;
    t = std::max(t, 0.0);
    auto r = [lower](double x) { return lower ? dn(x) : up(x); };
    double s = std::max(r(1.0 - t), 0.0);
    double s2 = r(s * s), s4 = r(s2 * s2);
    double q = r(4.0 * t + 1.0);
    return std::max(r(s4 * q), 0.0);
}

/// Upper bound of 20 t (1-t)^3 / R over t in [tlo, thi].
inline double grad_weight_up(double tlo, double thi, double R) {
    tlo = std::max(tlo, 0.0);
    if (tlo >= 1.0) return 0.0;
    thi = std::min(thi, 1.0);
    double t = std::min(std::max(0.25, tlo), std::max(thi, tlo));
    double s = up(1.0 - t);
    double g = up(20.0 * up(t * up(s * up(s * s))));
    g = std::min(g, up(135.0 / 64.0));
    return up(g / dn(R));
}

/// Outward (dmin, dmax) between a box and a point.
inline void near_far(const double* lo, const double* hi, const Eigen::Vector3d& c,
                     double& dmin, double& dmax) {
    double g2 = 0.0, f2 = 0.0;
    for (int k = 0; k < 3; k++) {
        double gap = std::max({lo[k] - c(k), c(k) - hi[k], 0.0});
        gap = std::max(dn(gap), 0.0);
        g2 = dn(g2 + dn(gap * gap));
        double far = up(std::max(std::abs(hi[k] - c(k)), std::abs(lo[k] - c(k))));
        f2 = up(f2 + up(far * far));
    }
    dmin = std::max(dn(std::sqrt(std::max(g2, 0.0))), 0.0);
    dmax = up(std::sqrt(f2));
}

/// Spectral-norm bound of a symmetric 3x3 matrix from entry magnitudes (00,01,02,11,12,22).
inline double sym_norm_up(const double h[6]) {
    double fro = up(up(up(h[0]*h[0]) + up(h[3]*h[3])) + up(h[5]*h[5]));
    fro = up(fro + up(2.0 * up(up(up(h[1]*h[1]) + up(h[2]*h[2])) + up(h[4]*h[4]))));
    fro = up(std::sqrt(fro));
    double r0 = up(up(h[0] + h[1]) + h[2]), r1 = up(up(h[1] + h[3]) + h[4]), r2 = up(up(h[2] + h[4]) + h[5]);
    return std::min(fro, std::max({r0, r1, r2}));
}

struct LocalCert {
    Ival f;          // value at x0
    Ival g[3];       // gradient at x0
    Ival H[6];       // Hessian entries at x0 (00,01,02,11,12,22)
    double gn_up;    // |grad f(x0)|
    double hn_up;    // spectral norm bound of the Hessian at x0
    double B0, B2, B3, Lmin;
    Ival range;      // value range over the box
};

}  // namespace

// ── per-patch constants ────────────────────────────────────────────

void PUInterpolator::prepare_certify() const {
    std::lock_guard<std::mutex> lock(certify_mutex_);
    if (certify_ready_) return;
    {
        if (use_box_)
            throw std::runtime_error("certify_boxes: only the sphere partition is supported");
        const int P = (int)patches_.size();
        patch_const_.resize(P);
        for (int j = 0; j < P; j++) {
            const auto& fit = *patches_[j].interp;
            const Eigen::MatrixXd& q = fit.centers();
            const Eigen::VectorXd& a = fit.alpha();
            const int n = (int)a.size();
            PatchConst pc;
            Acc A;
            for (int i = 0; i < n; i++) A.add(std::abs(a(i)));
            double lo, hi;
            A.enclose(n, lo, hi);
            pc.A_up = hi;
            double l2 = 0.0;
            for (int k = 0; k < 3; k++) { double p = fit.affine_linear()(k); l2 = up(l2 + up(p * p)); }
            pc.lin_norm_up = up(std::sqrt(l2));
            double wsum = 0.0;
            Eigen::Vector3d c = Eigen::Vector3d::Zero();
            for (int i = 0; i < n; i++) { wsum += std::abs(a(i)); c += std::abs(a(i)) * q.row(i).transpose(); }
            c = wsum > 0 ? Eigen::Vector3d(c / wsum) : Eigen::Vector3d(q.colwise().mean().transpose());
            pc.centroid = c;   // any exact float64 point is a valid reference
            Acc S, m0, m1[3];
            for (int i = 0; i < n; i++) {
                double h2 = 0.0;
                for (int k = 0; k < 3; k++) {
                    double h = q(i, k) - c(k);
                    h2 += h * h;
                    m1[k].add(a(i) * h);
                }
                S.add(std::abs(a(i)) * h2);
                m0.add(a(i));
            }
            S.enclose(n, lo, hi); pc.S_up = hi;
            m0.enclose(n, lo, hi); pc.m0_mag_up = std::max(std::abs(lo), std::abs(hi));
            double mm = 0.0;
            for (int k = 0; k < 3; k++) {
                m1[k].enclose(n, lo, hi);
                double v = std::max(std::abs(lo), std::abs(hi));
                mm = up(mm + up(v * v));
            }
            pc.m1_norm_up = up(std::sqrt(mm));
            patch_const_[j] = pc;
        }
        // A_jk for merged difference fits f_j - f_k: centers that coincide
        // bitwise are merged, |a_j - a_k| rounded up. Only pairs whose
        // supports intersect are stored.
        std::map<std::array<double, 3>, int> ids;
        std::vector<std::vector<std::pair<int, double>>> lists(P);
        for (int j = 0; j < P; j++) {
            const auto& fit = *patches_[j].interp;
            for (int i = 0; i < (int)fit.alpha().size(); i++) {
                std::array<double, 3> key{fit.centers()(i, 0), fit.centers()(i, 1), fit.centers()(i, 2)};
                auto it = ids.emplace(key, (int)ids.size()).first;
                lists[j].push_back({it->second, fit.alpha()(i)});
            }
            std::sort(lists[j].begin(), lists[j].end());
        }
        pair_A_.clear();
        pair_identical_.clear();
        for (int j = 0; j < P; j++) {
            for (int k = j + 1; k < P; k++) {
                const double dc = (patches_[j].center - patches_[k].center).squaredNorm();
                const double Rs = patches_[j].half_ext(0) + patches_[k].half_ext(0);
                if (dc > Rs * Rs * (1.0 + 1e-9)) continue;
                const auto &Lj = lists[j], &Lk = lists[k];
                Acc acc;
                size_t a = 0, b = 0;
                long long n = 0;
                bool same = Lj.size() == Lk.size();
                while (a < Lj.size() || b < Lk.size()) {
                    if (b == Lk.size() || (a < Lj.size() && Lj[a].first < Lk[b].first)) {
                        acc.add(std::abs(Lj[a++].second)); same = false;
                    } else if (a == Lj.size() || Lk[b].first < Lj[a].first) {
                        acc.add(std::abs(Lk[b++].second)); same = false;
                    } else {
                        if (Lj[a].second != Lk[b].second) same = false;
                        acc.add(up(std::abs(Lj[a++].second - Lk[b++].second)));
                    }
                    n++;
                }
                const auto &fj = *patches_[j].interp, &fk = *patches_[k].interp;
                same = same && fj.affine_linear() == fk.affine_linear()
                            && fj.affine_constant() == fk.affine_constant();
                double lo, hi;
                acc.enclose(n, lo, hi);
                const long long key = (long long)j * P + k;
                pair_A_[key] = hi;
                if (same) pair_identical_.insert(key);
            }
        }
    }
    certify_ready_ = true;
}

// ── certify_boxes ───────────────────────────────────────────────────

PUInterpolator::BoxCertificates PUInterpolator::certify_boxes(
    const Eigen::MatrixXd& lo, const Eigen::MatrixXd& hi, bool hull_only) const
{
    if (lo.rows() != hi.rows() || lo.cols() != 3 || hi.cols() != 3)
        throw std::invalid_argument("certify_boxes: lo and hi must be (m, 3)");
    prepare_certify();
    const int m = (int)lo.rows();
    const int P = (int)patches_.size();
    BoxCertificates out;
    out.status.setConstant(m, 2);
    out.n_active.setZero(m);
    out.hull_sign.setZero(m);
    out.fallback_patch.setConstant(m, -1);
    out.rho.setZero(m);
    out.W_min.setZero(m);
    out.F0_lo.setConstant(m, -kInf);
    out.F0_hi.setConstant(m, kInf);
    out.L_best.setConstant(m, kInf);
    out.L_local.setConstant(m, kInf);
    out.L_B0.setConstant(m, kInf);
    long long cost = 0;

    #pragma omp parallel for schedule(dynamic, 16) reduction(+:cost)
    for (int b = 0; b < m; b++) {
        double blo[3], bhi[3], x0a[3];
        for (int k = 0; k < 3; k++) { blo[k] = lo(b, k); bhi[k] = hi(b, k); x0a[k] = 0.5 * (blo[k] + bhi[k]); }
        Eigen::Vector3d x0(x0a[0], x0a[1], x0a[2]);
        double e2 = 0.0;
        for (int k = 0; k < 3; k++) {
            double e = std::max(up(x0a[k] - blo[k]), up(bhi[k] - x0a[k]));
            e2 = up(e2 + up(e * e));
        }
        const double rho = up(std::sqrt(e2));
        const double rr = up(rho * rho);
        out.rho(b) = rho;

        // local certificate of patch j on this box
        auto local = [&](int j) {
            const auto& fit = *patches_[j].interp;
            const auto& pc = patch_const_[j];
            const Eigen::MatrixXd& q = fit.centers();
            const Eigen::VectorXd& a = fit.alpha();
            const int n = (int)a.size();
            Acc F, G[3], H[6], B0s;
            static const int KL[6][2] = {{0,0},{0,1},{0,2},{1,1},{1,2},{2,2}};
            double tiny = 0.0;   // contributions of centers whose r underflowed
            for (int i = 0; i < n; i++) {
                double v[3], r2 = 0.0;
                for (int k = 0; k < 3; k++) {
                    v[k] = x0a[k] - q(i, k);
                    r2 += v[k] * v[k];
                }
                const double ai = a(i);
                if (!hull_only) {                       // B0 needs the far-corner distances
                    double R2 = 0.0;
                    for (int k = 0; k < 3; k++) {
                        double far = std::max(std::abs(bhi[k] - q(i, k)), std::abs(blo[k] - q(i, k)));
                        R2 += far * far;
                    }
                    B0s.add(std::abs(ai) * R2);
                }
                const double r = std::sqrt(r2);
                F.add(ai * r * r2);
                const double ar = ai * r;
                for (int k = 0; k < 3; k++) G[k].add(ar * v[k]);
                if (r > 0.0) {
                    const double ir = 1.0 / r;
                    for (int t = 0; t < 6; t++) {
                        int k = KL[t][0], l = KL[t][1];
                        H[t].add(ai * ((k == l ? r : 0.0) + v[k] * v[l] * ir));
                    }
                } else if (v[0] != 0.0 || v[1] != 0.0 || v[2] != 0.0) {
                    tiny += 2.0 * std::abs(ai) * DBL_MIN;   // |term| <= 2|a| r, r < sqrt(DBL_MIN)
                }
            }
            LocalCert L;
            double flo, fhi;
            F.enclose(n, flo, fhi);
            Ival lin{fit.affine_constant(), fit.affine_constant()};
            for (int k = 0; k < 3; k++)
                lin = iadd(lin, imul({fit.affine_linear()(k), fit.affine_linear()(k)}, {x0a[k], x0a[k]}));
            L.f = iadd({flo, fhi}, lin);
            double g2 = 0.0;
            for (int k = 0; k < 3; k++) {
                double glo, ghi;
                G[k].enclose(n, glo, ghi);
                Ival gk{dn(3.0 * glo), up(3.0 * ghi)};
                double p = fit.affine_linear()(k);
                L.g[k] = iadd(gk, {p, p});
                double mg = imag(L.g[k]);
                g2 = up(g2 + up(mg * mg));
            }
            L.gn_up = up(std::sqrt(g2));
            double h[6];
            for (int t = 0; t < 6; t++) {
                double hlo, hhi;
                H[t].enclose(n, hlo, hhi);
                L.H[t] = {dn(3.0 * dn(hlo - tiny)), up(3.0 * up(hhi + tiny))};
                h[t] = imag(L.H[t]);
            }
            L.hn_up = sym_norm_up(h);   // |H|_2 <= min(Frobenius, max row sum)
            double blo0, bhi0;
            B0s.enclose(n, blo0, bhi0);
            L.B0 = hull_only ? kInf : up(pc.lin_norm_up + up(3.0 * bhi0));
            double Z2 = 0.0;
            for (int k = 0; k < 3; k++) {
                double far = up(std::max(std::abs(bhi[k] - pc.centroid(k)), std::abs(blo[k] - pc.centroid(k))));
                Z2 = up(Z2 + up(far * far));
            }
            const double Z = up(std::sqrt(Z2));
            L.B2 = up(up(up(pc.lin_norm_up + up(3.0 * pc.S_up)) + up(3.0 * up(pc.m0_mag_up * up(Z * Z))))
                      + up(6.0 * up(pc.m1_norm_up * Z)));
            L.B3 = up(up(L.gn_up + up(rho * L.hn_up)) + up(3.0 * up(pc.A_up * rr)));
            L.Lmin = std::min({L.B0, L.B2, L.B3});
            const double rad = up(up(up(rho * L.gn_up) + up(0.5 * up(L.hn_up * rr))) + up(pc.A_up * up(rho * rr)));
            L.range = {dn(L.f.lo - rad), up(L.f.hi + rad)};
            return L;
        };

        // conservative enumeration of supports meeting the box
        std::vector<int> act;
        std::vector<double> dminv, dmaxv;
        {
            int stack[128], sp = 0;
            if (!patch_bvh_nodes_.empty()) stack[sp++] = 0;
            while (sp > 0) {
                const auto& nd = patch_bvh_nodes_[stack[--sp]];
                bool overlap = true;
                for (int k = 0; k < 3; k++)
                    if (bhi[k] < nd.lo[k] || blo[k] > nd.hi[k]) overlap = false;
                if (!overlap) continue;
                if (nd.left == -1) {
                    for (int t = nd.leaf_start; t < nd.leaf_start + nd.leaf_count; t++) {
                        int j = patch_bvh_leaves_[t];
                        double dmin, dmax;
                        near_far(blo, bhi, patches_[j].center, dmin, dmax);
                        if (dmin <= patches_[j].half_ext(0)) {
                            act.push_back(j); dminv.push_back(dmin); dmaxv.push_back(dmax);
                        }
                    }
                } else {
                    stack[sp++] = nd.left;
                    stack[sp++] = nd.right;
                }
            }
        }
        const int K = (int)act.size();
        out.n_active(b) = K;

        if (K == 0) {
            // fixed fallback: one patch provably nearest to every point of the box
            int best = -1;
            double best_smax = kInf, second_smin = kInf;
            std::vector<double> smin(P), smax(P);
            for (int j = 0; j < P; j++) {
                double dmin, dmax;
                near_far(blo, bhi, patches_[j].center, dmin, dmax);
                const double R = patches_[j].half_ext(0);
                smin[j] = dn(dmin - R);
                smax[j] = up(dmax - R);
                if (smax[j] < best_smax) { best_smax = smax[j]; best = j; }
            }
            for (int j = 0; j < P; j++) if (j != best) second_smin = std::min(second_smin, smin[j]);
            if (best >= 0 && best_smax < second_smin) {
                LocalCert L = local(best);
                cost += patches_[best].interp->centers().rows();
                out.status(b) = 1;
                out.fallback_patch(b) = best;
                out.L_best(b) = L.Lmin; out.L_local(b) = L.Lmin; out.L_B0(b) = L.B0;
                out.F0_lo(b) = L.f.lo; out.F0_hi(b) = L.f.hi;
                out.hull_sign(b) = (L.range.lo > 0) ? 1 : (L.range.hi < 0 ? -1 : 0);
            }
            continue;
        }

        std::vector<LocalCert> loc(K);
        std::vector<double> Gj(K);
        double Wmin = 0.0, Lloc = 0.0, LlocB0 = 0.0;
        bool allpos = true, allneg = true;
        Ival num{0.0, 0.0}, den{0.0, 0.0};
        double hull_lo = kInf, hull_hi = -kInf;
        for (int t = 0; t < K; t++) {
            const int j = act[t];
            const double R = patches_[j].half_ext(0);
            loc[t] = local(j);
            cost += patches_[j].interp->centers().rows();
            const double tlo = std::max(dn(dminv[t] / R), 0.0), thi = up(dmaxv[t] / R);
            Wmin = dn(Wmin + psi_round(thi, true));
            Gj[t] = grad_weight_up(tlo, thi, R);
            Lloc = std::max(Lloc, loc[t].Lmin);
            LlocB0 = std::max(LlocB0, loc[t].B0);
            allpos &= loc[t].range.lo > 0.0;
            allneg &= loc[t].range.hi < 0.0;
            // weight at x0
            double r2lo = 0.0, r2hi = 0.0;
            for (int k = 0; k < 3; k++) {
                double d = x0a[k] - patches_[j].center(k);
                double dlo = dn(d), dhi = up(d);
                double m_lo = (dlo <= 0 && dhi >= 0) ? 0.0 : std::min(std::abs(dlo), std::abs(dhi));
                double m_hi = std::max(std::abs(dlo), std::abs(dhi));
                r2lo = dn(r2lo + dn(m_lo * m_lo));
                r2hi = up(r2hi + up(m_hi * m_hi));
            }
            const double rlo = std::max(dn(std::sqrt(r2lo)), 0.0), rhi = up(std::sqrt(r2hi));
            Ival v{psi_round(up(rhi / R), true), psi_round(std::max(dn(rlo / R), 0.0), false)};
            num = iadd(num, imul(v, loc[t].f));
            den = iadd(den, v);
            if (v.hi > 0.0) { hull_lo = std::min(hull_lo, loc[t].f.lo); hull_hi = std::max(hull_hi, loc[t].f.hi); }
        }
        out.W_min(b) = std::max(Wmin, 0.0);
        if (!(Wmin > 0.0)) continue;          // unresolved: possible fallback inside the box

        out.status(b) = 0;
        out.hull_sign(b) = allpos ? 1 : (allneg ? -1 : 0);
        if (hull_only) continue;               // no center value, no Lipschitz bound
        double F0lo = hull_lo, F0hi = hull_hi;
        if (den.lo > 0.0) {
            double qs[4] = {num.lo / den.lo, num.lo / den.hi, num.hi / den.lo, num.hi / den.hi};
            double qlo = qs[0], qhi = qs[0];
            for (double v : qs) { qlo = std::min(qlo, v); qhi = std::max(qhi, v); }
            F0lo = std::max(F0lo, dn(qlo));
            F0hi = std::min(F0hi, up(qhi));
        }
        out.F0_lo(b) = F0lo; out.F0_hi(b) = F0hi;

        // blend term: sum_j G_j max_k Delta_jk / W_min, over k whose support meets j's
        double blend = 0.0, blendB0 = 0.0;
        for (int s = 0; s < K; s++) {
            double mx = 0.0, mxB0 = 0.0;
            const int j = act[s];
            for (int t = 0; t < K; t++) {
                if (t == s) continue;
                const int k = act[t];
                double dc = 0.0;
                for (int c = 0; c < 3; c++) {
                    double d = patches_[j].center(c) - patches_[k].center(c);
                    dc += d * d;
                }
                const double Rs = patches_[j].half_ext(0) + patches_[k].half_ext(0);
                if (dc > Rs * Rs * (1.0 + 1e-9)) continue;   // supports disjoint
                const long long key = (long long)std::min(j, k) * P + std::max(j, k);
                if (pair_identical_.count(key)) continue;    // identical fits: zero disagreement
                const auto itA = pair_A_.find(key);
                const double d0 = imag(isub(loc[s].f, loc[t].f));
                const double Dsimple = up(d0 + up(rho * up(loc[s].Lmin + loc[t].Lmin)));
                const double DB0 = up(d0 + up(rho * up(loc[s].B0 + loc[t].B0)));
                const double Drange = std::max(up(loc[s].range.hi - loc[t].range.lo),
                                               up(loc[t].range.hi - loc[s].range.lo));
                double Dtaylor = kInf;
                if (itA != pair_A_.end()) {
                    double gd2 = 0.0, hd[6];
                    for (int c2 = 0; c2 < 3; c2++) {
                        double v = imag(isub(loc[s].g[c2], loc[t].g[c2]));
                        gd2 = up(gd2 + up(v * v));
                    }
                    for (int c2 = 0; c2 < 6; c2++) hd[c2] = imag(isub(loc[s].H[c2], loc[t].H[c2]));
                    Dtaylor = up(up(up(d0 + up(rho * up(std::sqrt(gd2))))
                                    + up(0.5 * up(sym_norm_up(hd) * rr)))
                                 + up(itA->second * up(rho * rr)));
                }
                mx = std::max(mx, std::min({Dsimple, Drange, Dtaylor}));
                mxB0 = std::max(mxB0, DB0);
            }
            blend = up(blend + up(Gj[s] * mx));
            blendB0 = up(blendB0 + up(Gj[s] * mxB0));
        }
        out.L_local(b) = Lloc;
        out.L_best(b) = up(Lloc + up(blend / Wmin));
        out.L_B0(b) = up(LlocB0 + up(blendB0 / Wmin));
    }
    out.cost_terms = cost;
    return out;
}

}  // namespace sdf
