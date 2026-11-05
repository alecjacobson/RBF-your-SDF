Build a reproducible experiment on top of [CocoYard/RBF-your-SDF](https://github.com/CocoYard/RBF-your-SDF) to determine whether its reconstructed implicit function admits bounds that are both mathematically defensible and useful for conservative spatial queries. Freeze a fitted reconstruction, compare several bounds on its gradient, and measure safe rejection of cells that cannot intersect its zero set.

The primary question is practical: can cancellation and spatial subdivision produce substantially tighter bounds than summing the magnitudes of individual cubic RBF terms? The experiment must include partition-of-unity weight derivatives, finite-precision moment residuals, and any evaluator branch that can break continuity. A small observed gradient is evidence about tightness, not a certificate.

# Scope and success criteria

The target is the real-valued analytic field defined by the stored coefficients and evaluator rules on a specified finite rectangular domain. Use the existing reconstruction algorithm and its final coefficients as the baseline. Initially implement a CPU reference and instrumentation in the existing Python and C++ project; a Warp port is outside this experiment.

Deliver three distinct results:

1. **Validity:** a proved bound and a numerically enclosed implementation, or a precise reason certification fails for a region.

2. **Tightness:** distributions of certified upper bounds versus sampled and optimized lower bounds on the largest gradient norm.

3. **Utility:** cell rejection rate, field evaluations saved, preprocessing cost, and total runtime relative to the same query without pruning.

Success does not require a Lipschitz constant near one. A negative result is useful if it identifies coefficient cancellation, patch disagreement, weak coverage, or discontinuous fallback as the limiting factor. Bounds describe the reconstructed zero set; they do not certify agreement with the unknown original surface, Marching Cubes topology, or mesh Hausdorff error.

# Repository integration and initial audit

The README exposes `rbfyoursdf.main_algorithm(points, values, max_iters=10)`, returns an object with `result.interpolator`, and demonstrates `extract_surface`. It identifies `cpp/` as the implementation core and `demo/SDF_to_surface_3D.py` as the example driver. The README points to tag `v1.0.0` for paper reproduction. The current build configuration uses scikit-build-core and pybind11. [Repository](https://github.com/CocoYard/RBF-your-SDF), [build configuration](https://github.com/CocoYard/RBF-your-SDF/blob/master/pyproject.toml).

**Implementation boundary:** the public README and build configuration were accessible when preparing this design; the implementation files and clone were blocked in this environment. The hooks below are proposed interfaces. Their mapping to internal class names, source files, and current behavior is the first implementation milestone. Do not assume that the current branch implements every detail of the paper unchanged.

Create an experiment branch, record the resolved commit SHA, and inspect repository instructions before editing. Use that frozen revision for every comparison. Start from the current implementation; compare the paper tag only as a separately identified variant if behavior differs.

Audit and record:

| Area | Required facts |
| - | - |
| Local RBF | Kernel and its scaling; affine coefficient layout; actual centers including tangent points and extra conditioning constraints; solver precision and regularization |
| Coordinates | Every translation, spatial scaling, value scaling, and conversion to physical coordinates |
| Patch evaluator | Raw weights, radii, centers, normalized-weight denominator, support inclusion rules, and any pruning thresholds |
| Gradient | Whether the exposed derivative differentiates the full blended value, including weight derivatives |
| Branches | Rules for missing coverage, small denominators, nearest-patch selection, clipping, invalid values, and query-dependent center selection |
| Reconstruction | Deduplication, degenerate-patch repair, iteration count, and ability to snapshot the final fitted state |

Keep fitting and queries separate. All bound comparisons use exactly the same stored reconstruction. Bounds do not differentiate through the fitting procedure or tangent-point optimization.

# Mathematical model

For a single local fit, use the following physical-coordinate representation. The centers include every constraint in that fit.

```math
f(x)=a+b^\top x+\sum_{i=1}^{m}\alpha_i\|x-q_i\|^3.
```

The paper specifies an affine-augmented cubic RBF with zero zeroth and first moments, and blends local fits with normalized compact Wendland weights. These model choices must be checked against the pinned implementation. [Paper, Section 3.2 and Appendix B](https://arxiv.org/html/2609.06209v3).

```math
\sum_i\alpha_i=0,\qquad \sum_i\alpha_i q_i=0.
```

For a closed convex box B, define its center x₀ and half-diagonal ρ. Any upper bound on the gradient norm throughout B is a Lipschitz constant on B:

```math
\rho=\max_{x\in B}\|x-x_0\|,\qquad
\sup_{x\in B}\|\nabla f(x)\|\le L_B
\Longrightarrow |f(x)-f(y)|\le L_B\|x-y\|\quad(x,y\in B).
```

The formulas below are proposed derivations for this experiment, not Lipschitz guarantees claimed by the paper.

## Derivatives

Use analytic derivatives of the frozen fit. Define each center contribution to the Hessian as zero at rᵢ = 0, its continuous limit.

```math
v_i=x-q_i,\quad r_i=\|v_i\|,
\qquad \nabla f(x)=b+3\sum_i\alpha_i r_i v_i,
```

```math
\nabla^2 f(x)=3\sum_i\alpha_i
\left(r_i I+\frac{v_i v_i^\top}{r_i}\right).
```

Verify the implementation's gradient against the derivative of its value evaluator before treating it as a lower-bound oracle. At smooth query locations, use central differences at several step sizes and an independent analytic evaluator; neither finite differences nor dense sampling certifies an upper bound.

# Bounds for an individual RBF

Implement all four variants below. Preserve their individual values in results even when selecting their minimum.

## B0 Direct triangle bound

Let Rᵢ(B) be the greatest distance from qᵢ to the box, computable from the farthest coordinatewise endpoints.

```math
L_{\mathrm{triangle}}(B)
=\|b\|+3\sum_i|\alpha_i|R_i(B)^2.
```

This is the reference upper bound. It exposes the suspected problem: large query-to-center distances and large coefficients remain even when the sum cancels almost completely.

## B1 Moment cancellation in exact arithmetic

For any reference point c, exact moment constraints give a bound independent of query position:

```math
L_{\mathrm{moment}}
=\|b\|+3\sum_i|\alpha_i|\|q_i-c\|^2.
```

Choose c as the coefficient-magnitude-weighted centroid to minimize the expression. When all radial coefficients vanish, use the affine constant directly.

```math
c=\frac{\sum_i|\alpha_i|q_i}{\sum_i|\alpha_i|}.
```

Proof: set g(z) = ‖z‖z. Its first-order Taylor remainder obeys

```math
g(z-h)=g(z)-Dg(z)h+R(z,h),\qquad
\|R(z,h)\|\le\|h\|^2.
```

Use z = x − c and h = qᵢ − c. Summing with coefficients αᵢ cancels the constant and linear terms. Multiplying the remaining bound by three gives B1. The remainder estimate also holds when the segment passes through the origin, because g is continuously differentiable with Lipschitz derivative.

This removes query distance, but not coefficient magnitude or the spatial spread of the fit. A patch fit can also include centers outside its weight support; use the exported centers, never a presumed support-radius bound on all centers.

## B2 Moment bound with stored coefficient residuals

Rounded coefficients generally do not have exactly zero moments. Treat the stored numbers as exact real coefficients and enclose their residuals:

```math
m_0=\sum_i\alpha_i,\qquad
m_1=\sum_i\alpha_i(q_i-c),\qquad
Z(B,c)=\max_{x\in B}\|x-c\|.
```

The same expansion yields

```math
L_{\mathrm{residual}}(B)
=\|b\|+3\sum_i|\alpha_i|\|q_i-c\|^2
+3|m_0|Z(B,c)^2+6\|m_1\|Z(B,c).
```

This is the production form of the moment bound. Include outward upper bounds for both residuals; a solver tolerance or a residual rounded to zero is insufficient. B1 may be plotted as an idealized diagnostic, but must not be used for certified pruning unless its exact-moment premise is established.

Nonzero moments can cause unbounded growth outside the experimental domain. Do not label B2 a global bound on all of space. Reprojecting coefficients onto a moment nullspace changes the field and belongs in a separate ablation with interpolation and geometry errors measured.

## B3 Cell-local Taylor bound

Let A be the sum of coefficient magnitudes. The cubic kernel has a Hessian with Lipschitz constant six, so the fitted Hessian has Lipschitz constant at most 6A. Therefore

```math
A=\sum_i|\alpha_i|,\qquad
L_{\mathrm{Taylor}}(B)
=\|\nabla f(x_0)\|+\rho\|\nabla^2 f(x_0)\|_2+3A\rho^2.
```

This retains cancellation in both derivatives evaluated at the box center. It requires no exact moment constraints and converges to the center gradient norm as the box shrinks. Use enclosed derivatives and an upper bound on the Hessian spectral norm; the Frobenius norm is a simple valid first implementation. A better spectral enclosure is an optional optimization.

For completeness, the constant follows by differentiating g along a direction v. With u = z/‖z‖ and t = uᵀv,

```math
D^2g(z)[v,v]=2t v+(\|v\|^2-t^2)u,
\qquad \|D^2g(z)[v,v]\|\le2\|v\|^2.
```

Symmetry of the third derivative gives the Hessian Lipschitz bound for the cubic kernel. Integrating along a segment gives the Taylor remainder above.

Use the minimum of the applicable enclosed B0, B2, and B3 bounds on each box. A bound valid only on B must never authorize a step beyond B.

# Bounds for the blended field

## Exact derivative and patch disagreement

Write raw weights as vⱼ, their sum as W, and normalized weights as wⱼ. On a region with positive W:

```math
F(x)=\frac{\sum_j v_j(x)f_j(x)}{W(x)},\qquad
W(x)=\sum_jv_j(x),\qquad w_j(x)=v_j(x)/W(x).
```

```math
\nabla F(x)=\sum_jw_j(x)\nabla f_j(x)
+\frac{1}{W(x)}\sum_j\bigl(f_j(x)-F(x)\bigr)\nabla v_j(x).
```

Bounding only the gradients of the individual fits omits the second term. In particular, blending two different constant functions can have a nonzero gradient even though both local Lipschitz constants are zero.

For a box B, conservatively enumerate every patch whose support intersects B. Let Lⱼ bound the gradient of its local fit throughout B, Gⱼ bound its raw-weight gradient, and Δⱼₖ bound the absolute disagreement of two local fits throughout B. If a positive lower bound Wₘᵢₙ is certified, then

```math
L_{\mathrm{PU}}(B)
=\max_j L_j
+\frac{1}{W_{\min}}
\sum_j G_j\max_k\Delta_{jk}(B)
```

is a valid bound. Set Δⱼⱼ to zero exactly. This follows because F is a convex combination of local values and hence each discrepancy from F is bounded by the corresponding maximum pairwise discrepancy.

The first implementation may use

```math
\Delta_{jk}(B)
\le |f_j(x_0)-f_k(x_0)|+\rho(L_j+L_k).
```

Improve this by applying B3 to the difference function fⱼ − fₖ and replacing Lⱼ + Lₖ with a bound on its gradient. Subtract the affine coefficients and combine radial terms at identical centers; cancellation in shared constraints can matter substantially. Do not approximately merge distinct centers without an additional error bound.

If exactly one patch can contribute and its weight stays positive on B, simplify F to that local fit. If several fits are provably identical, their blending term is also zero. These cases avoid artificial blowup from a small denominator.

## Certified weight bounds

For the paper's weight, define ψ outside its support as zero:

```math
v_j(x)=\psi\!\left(\frac{\|x-\xi_j\|}{R_j}\right),\qquad
\psi(t)=(1-t)^4(4t+1)\quad(0\le t<1).
```

For a box, compute minimum and maximum distances dₘᵢₙ and dₘₐₓ to each patch center. Monotonicity gives

```math
W_{\min}(B)=\sum_j\psi\!\left(\frac{d_{\max,j}(B)}{R_j}\right),
\qquad
G_j(B)\le\frac{135}{64R_j}.
```

The gradient constant follows from maximizing 20t(1 − t)³ on the unit interval. Tighten Gⱼ by maximizing this expression only over the box's radial interval; test the interval endpoints and t = 1/4 if included. Include support-boundary intersections correctly.

A zero Wₘᵢₙ does not prove an uncovered point: different patches may cover different parts of the box. Subdivide, or use stronger interval bounds. Mark a box unresolved if the budget expires. Never replace Wₘᵢₙ by a sampled minimum or silently clamp it upward. If the implementation switches to a fallback below a positive denominator threshold, certify that the whole box stays above that threshold; mathematical positivity alone is insufficient. Apply the same branch audit to per-weight cutoffs.

Also prototype direct interval evaluation of the full quotient derivative as a comparison. Keep any validated minimum of independent bounds; report interval dependency inflation and cost.

## Coverage and evaluator transitions

Appendix B describes selecting a local fit when a query falls outside all patch supports. Its statement about visible continuity is not a continuity theorem. The implementation audit must determine whether this or an epsilon-based variant is present. [Paper, Appendix B](https://arxiv.org/html/2609.06209v3).

Maintain separate statuses for:

- A smooth covered box with a certified denominator and gradient bound.

- A box proved to use one fixed fallback formula throughout, with that formula bounded independently.

- A box containing an unresolved support, threshold, or fallback transition.

- A box with a demonstrated jump or invalid evaluator state.

Probe suspected interfaces with pairs approaching from opposite sides and plot value differences and difference quotients against separation. A persistent nonzero limiting jump would rule out a finite Lipschitz constant across that interface. Finite numerical probes alone neither prove continuity nor establish a mathematical jump; preserve a reproducible witness and seek analytic branch-limit verification.

The unchanged baseline must retain unresolved transition cells. It may claim a domain-wide constant only if a finite partition covers the entire convex domain, every cell has a valid bound, and the field is continuous across interfaces. The maximum of bounds over disconnected covered regions does not certify arbitrary pairs spanning a gap.

If coverage prevents useful certification, add an explicitly separate variant with expanded or added supports, refitted local interpolants, and certified positive coverage. Newly active patches must interpolate all constraints in their supports if interpolation is to be preserved. Merely widening weights can violate that property. Measure the resulting field and zero-set changes; do not count this as a bound on the unchanged baseline.

# Numerical certificate contract

Use three explicit labels in every table and saved result:

| Label | Meaning |
| - | - |
| Observed lower bound | A sampled or optimized gradient norm, or a difference quotient within a verified smooth region |
| Analytic estimate | A valid real-arithmetic formula evaluated in ordinary floating point without a rounding enclosure |
| Certified upper bound | A formula whose arithmetic, branch assumptions, and domain coverage have all been enclosed |

Implement ordinary FP64 estimates first, then an outward-rounded interval reference for certification. High precision alone and multiplying a result by 1 + ε do not provide a certificate. The certified path must enclose distance extrema, coefficient sums, residuals, derivative values, norms, denominator lower bounds, and the final bound.

Treat exported floating-point coefficients as exact real numbers for the analytic-field certificate. This avoids requiring an interval linear solver for an unknown exact fit. Separately enclose error in evaluating that field with the production implementation. The finite-precision program itself is not being asserted to be a Lipschitz map on all real inputs.

Use physical coordinates consistently. If a local representation uses u = T(x − t) and physical output is s times the local output plus an offset, transform derivatives and bounds by the chain rule:

```math
\nabla_x F=sT^\top\nabla_u f,\qquad
L_x\le |s|\|T\|_2L_u.
```

For anisotropic transforms, use an enclosing transformed box or another rigorously enclosed domain; do not reuse a scalar radius without adjustment. Record units and normalization in every run.

# Proposed instrumentation and experiment layout

Add a minimal read-only export from the fitted C++ object, exposed through pybind11. Avoid changing the fitting algorithm in the primary experiment.

| Export | Contents |
| - | - |
| Fit snapshot | Centers, radial and affine coefficients, kernel parameters, all coordinate transforms, and precision |
| Patch geometry | Centers, support radii, memberships, extra conditioning constraints, and overlap search metadata |
| Evaluator policy | Denominator thresholds, support cutoffs, fallback and clipping rules |
| Query diagnostics | Active patch IDs, raw weights, denominator, branch taken, value, and native gradient when available |
| Fit diagnostics | Interpolation residuals, moment residuals, counts, solver status, and available conditioning estimates |

The following names are proposed additions, not existing APIs:

| Proposed component | Responsibility |
| - | - |
| `demo/additional_experiments/lipschitz/adapter.py` | Export snapshots and reproduce the frozen evaluator |
| `.../bounds.py` | B0–B3, derivative evaluation, and difference-fit bounds |
| `.../pu_bounds.py` | Support enumeration, coverage, weights, and blend bounds |
| `.../certify.py` | Interval arithmetic, adaptive boxes, and certificate statuses |
| `.../run.py` | Reproducible configurations, fitting, queries, and timing |
| `.../report.py` | Tables, scientific plots, and failure witnesses |
| `tests/test_lipschitz_bounds.py` | Mathematical regression cases and integration checks |

Expose a proposed `export_lipschitz_state()` hook only after identifying the actual C++ owner. Give each snapshot a version and hash. Build a Python reference evaluator from the snapshot and compare its values and derivatives with the native implementation at interior, overlap, support-boundary, and fallback locations before running bound experiments.

Stream boxes and queries in batches. Store compact certificates and selected failure witnesses, not every dense intermediate array. Cache patch constants, interval coefficients, and spatial indices independently of query-grid resolution.

# Experimental protocol

## Inputs and reconstruction variants

Use a staged suite rather than a full Cartesian product.

| Stage | Inputs | Sampling and purpose |
| - | - | - |
| Algebra tests | Constructed local fits and synthetic blends | Exact identities, sharp constants, residual growth, and discontinuities |
| Analytic pilot | Sphere, box, torus, thin spherical shell | Exact SDF samples on 12³, 24³, and 40³ grids; smooth, sharp, medial-axis, and thin-feature cases |
| Repository pilot | `examples/bunny.obj` | Follow the existing sampling driver and freeze normalization |
| Scaling | A small manifested subset of available meshes | 24³, 40³, and 64³ input grids; 100³ only if measured cost permits |
| Stress | Scattered points, jitter, clustered and near-coplanar samples | Test conditioning, support coverage, and denominator sensitivity |

Use exact formulas for analytic inputs and preserve the generator. Do not silently treat an arbitrary CSG scalar field as an exact SDF. Identify noisy and truncated samples as separate non-exact-SDF stress tests, with magnitudes in normalized distance units.

Run the final fitted output first. Save optional iteration snapshots at initialization and selected refinement iterations to determine whether tangent constraints improve or worsen certifiability. Compare a single global RBF only on small cases, with an explicit total-constraint budget; it is a diagnostic for separating RBF and blending effects.

Start with upstream patch parameters. Then vary overlap and patch size one at a time. Any support refitting or moment reprojecting variant must receive a separate field hash and reconstruction-quality report.

## Query regions

Choose and record one finite domain per reconstruction: the sample bounding box for the primary experiment and a modest expanded box as a separate extrapolation stress test. Never pool their statistics.

Evaluate bounds on cells of regular 32³, 64³, and 128³ cell grids and on an adaptive subdivision with a fixed maximum depth and time budget. Treat those as query-cell counts, distinct from input sample counts. Larger grids are an optional scaling run.

Stratify point queries over the whole domain, near the reconstructed zero set, overlaps, small denominators, support boundaries, medial-axis neighborhoods, and fallback transitions. Define the near-zero stratum using the same frozen evaluator and declared threshold for every bound method; analytic inputs may additionally use true-distance bands for reporting.

Estimate gradient maxima with dense deterministic probes plus several bounded optimization starts per selected cell. Use finite-difference optimization or independent analytic derivatives as appropriate. These produce lower bounds only. Include pairwise difference quotients within smooth boxes as a separate diagnostic.

## Comparisons and metrics

Compare B0, idealized B1, residual-aware B2, B3, and their valid minimum on local fits. For the full field compare the corresponding PU bounds, the disagreement refinement, direct interval derivatives, and adaptive subdivision. Plot the patch-gradient maximum without blend derivatives only as a deliberately incomplete diagnostic, never as an upper-bound method.

Report:

- Per-patch coefficient magnitude, center spread, affine gradient, moment residuals, and interpolation error.

- Per-cell certificate status, bound, observed lower bound, denominator enclosure, overlap count, and separate local-fit and blending contributions.

- Median, p90, p99, and maximum upper/lower ratios, with zero lower bounds and unresolved cells reported separately. This ratio is an observed certificate gap, not the true overestimation factor.

- Fractions of certified cells by count and volume, plus near-zero coverage. Do not hide uncovered or timed-out cells by reporting only successful cases.

- Fit, export, precomputation, certification, and query time; peak memory; batch size; hardware; compiler; threading; and arithmetic mode.

- Scaling under translations and isotropic changes of coordinates and distance units. A consistently scaled SDF-like field should retain the same dimensionless gradient bound.

Use repeated timings after warmup and report medians. Keep expensive interval preprocessing separate and show its amortization over repeated queries. Direct evaluation is a necessary runtime baseline: fewer field evaluations do not guarantee a faster method.

# Application experiment with conservative cell rejection

For each box B, evaluate an interval enclosure of F at its center. Let d₀ be zero if that interval contains zero, otherwise its distance to zero. A certified bound permits rejection when

```math
d_0>L_B\rho.
```

Then the analytic reconstructed field cannot have a zero anywhere in B. Use a strict inequality, outward arithmetic, and the evaluation-error enclosure. Retain every unresolved cell. Handle a zero bound as a constant-field case.

Run the following comparison on the same hierarchy and target leaf resolution:

1. Traverse every target cell without certified pruning.

2. Prune using B0 propagated through the full blend bound.

3. Prune using the valid minimum with disagreement refinement and subdivision.

Measure rejected volume, surviving cells, field evaluations, bound evaluations, preprocessing, and total time. Produce slices showing rejected, retained, and unresolved cells. Check rejected regions against independent high-precision probes and all reference sign-changing grid edges; these are bug-finding checks in addition to the analytic certificate.

Surface extraction may use the retained cells, but the guarantee is only that pruning discards no zero of the analytic field. A zero set can still lie entirely inside a retained leaf without a corner sign change. Dense Marching Cubes is a reference output, not a proof of complete surface discovery.

An optional ray experiment can use the local value-to-bound ratio, capped at the distance to the boundary of the certified region. Require segment containment and account for value error. Do not use an instantaneous gradient norm as a safe step or claim that a local bound applies beyond its certified box.

# Verification cases

These tests protect the mathematical claims and are required before performance measurements.

| Case | Expected result |
| - | - |
| Affine field | All radial coefficients zero; local bound equals the affine gradient norm |
| Collinear cubic combination | For unit vector e, centers at −e, 0, e with coefficients 1, −2, 1 and zero affine part give moment bound 6 and attain gradient norm 6 on the exterior axis |
| Perturbed moments | Deliberately nonzero moments invalidate the exact global claim; B2 remains valid on each finite box |
| Query at a center | Gradient and Hessian use continuous limits without division by zero |
| Shrinking boxes | B3 approaches the center gradient norm within arithmetic enclosures |
| Different constant patches | Individual gradient bounds are zero while the blended field varies |
| Identical patches | Weight derivative contribution cancels, including near small positive denominators |
| Partial support overlap | Conservative support enumeration includes every patch that can contribute anywhere in the box |
| Uncertain coverage | A zero denominator lower bound triggers subdivision or an unresolved status |
| Nearest-patch jump | A synthetic constant-patch fallback produces a known discontinuity and blocks a cross-interface certificate |
| Physical rescaling | Consistent spatial and value scaling preserves the appropriate dimensionless bound |
| Reference versus native | Exported field values agree within evaluated error; derivative discrepancies are investigated before use |

The collinear test constructs coefficients directly and is not a solvability test for a three-dimensional interpolation system. Add randomized stress tests for each inequality, but do not use random success as the proof.

# Deliverables and implementation order

1. **Audit and snapshot:** pin the revision, document evaluator rules, add read-only export, and verify the independent evaluator. Stop claiming compatibility if the actual kernel or evaluator differs; adapt the formulas first.

2. **Local bounds:** implement analytic derivatives, B0–B3, residual diagnostics, and the algebra tests. Produce one local-bound plot on synthetic and bunny fits.

3. **Full-field bounds:** implement coverage classification, weight and disagreement bounds, transition probes, and interval certification. Include all unresolved cases in outputs.

4. **Pilot application:** run conservative cell rejection and measure net runtime and memory, not just bounds or evaluation counts.

5. **Scaling and report:** expand the manifested suite only after validity checks pass. Add a repaired-coverage variant only if baseline limitations justify it.

Save a run manifest containing the commit, configuration, input hashes, normalization, seed, dependency and compiler versions, hardware, field snapshot hash, and command. Save machine-readable patch and cell metrics, compact failure witnesses, and a human-readable report. Plot bound gaps versus cell size, blending contribution versus denominator, certification coverage, rejection rate, and runtime amortization. No figures should imply measured results before runs exist.

Correctness acceptance requires all algebra tests, agreement with the frozen evaluator, enclosed arithmetic for every claimed certificate, and zero known violations. An unresolved or discontinuous region must remain explicitly unresolved, not receive a finite constant by heuristic. Practical acceptance is demonstrated by useful rejection at competitive total cost; if that fails, the report should identify which bound term or evaluator behavior dominates and whether spatial refinement changes the conclusion.

# References

- Cheng and Gingold, **RBF Your SDF: Radial Basis Function Interpolation of Signed Distance Fields with Implied Tangent Points**, arXiv:2609.06209v3. [Paper](https://arxiv.org/html/2609.06209v3). Section 3.2 defines the local interpolant; Appendix B defines the blend and describes gap handling.

- **CocoYard/RBF-your-SDF**. [Repository and public API](https://github.com/CocoYard/RBF-your-SDF). Resolve and record the implementation commit before running the experiment.

- **Repository build configuration**. [pyproject.toml](https://github.com/CocoYard/RBF-your-SDF/blob/master/pyproject.toml). Integration context for the Python package and C++ extension.
