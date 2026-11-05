#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <optional>
#include "main_algorithm.h"
#include "duchon_interpolator.h"
#include "pu_interpolator.h"
#include "visibility.h"

namespace py = pybind11;

PYBIND11_MODULE(rbfyoursdf_cpp, m) {
    m.doc() = "C++ implementation of SDF gradient optimization";

    // ── Tolerance ───────────────────────────────────────────────────
    py::class_<sdf::Tolerance>(m, "Tolerance")
        .def(py::init<>())
        .def_readwrite("clamp_radius_ratio", &sdf::Tolerance::clamp_radius_ratio)
        .def_readwrite("clamp_sdf_tol", &sdf::Tolerance::clamp_sdf_tol)
        .def_readwrite("float_tol", &sdf::Tolerance::float_tol)
        .def_readwrite("angle_tol", &sdf::Tolerance::angle_tol);

    // ── Options ─────────────────────────────────────────────────────
    py::class_<sdf::Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("name", &sdf::Options::name)
        .def_readwrite("grid_len", &sdf::Options::grid_len)
        .def_readwrite("max_iters", &sdf::Options::max_iters)
        .def_readwrite("clamp", &sdf::Options::clamp)
        .def_readwrite("turn_off_short_arcs", &sdf::Options::turn_off_short_arcs)
        .def_readwrite("export_projections", &sdf::Options::export_projections)
        .def_readwrite("export_short_arcs",  &sdf::Options::export_short_arcs)
        .def_readwrite("verbose", &sdf::Options::verbose)
        .def_readwrite("reg", &sdf::Options::reg)
        .def_readwrite("interpolator_type", &sdf::Options::interpolator_type)
        .def_readwrite("interp_partition", &sdf::Options::interp_partition)
        .def_readwrite("interp_overlap", &sdf::Options::interp_overlap)
        .def_readwrite("pair_local", &sdf::Options::pair_local)
        .def_readwrite("pu_dedup_tol", &sdf::Options::pu_dedup_tol)
        .def_readwrite("tolerance", &sdf::Options::tolerance)
        .def_readwrite("gt_gradients", &sdf::Options::gt_gradients)
        .def_readwrite("iter_gradient_finding", &sdf::Options::iter_gradient_finding)
        .def_readwrite("grad_optimizer", &sdf::Options::grad_optimizer)
        .def_readwrite("lr", &sdf::Options::lr)
        .def_readwrite("optim_steps", &sdf::Options::optim_steps)
        .def_readwrite("degen_tol", &sdf::Options::degen_tol)
        // The short-arc candidates that survived filter_degenerate_pts, i.e.
        // exactly the zero-valued points the second RBF fit was given. Only
        // populated after main_algorithm has run on this Options object.
        .def_property_readonly("degenerate_points", [](const sdf::Options& o) {
            std::vector<int> idx;
            idx.reserve(o.degenerate_pts.size());
            for (const auto& kv : o.degenerate_pts)
                if (!kv.second.empty()) idx.push_back(kv.first);
            std::sort(idx.begin(), idx.end());  // hash order is not reproducible
            Eigen::VectorXi I((int)idx.size());
            Eigen::MatrixXd P((int)idx.size(), 3);
            for (int i = 0; i < (int)idx.size(); i++) {
                I(i) = idx[i];
                P.row(i) = o.degenerate_pts.at(idx[i])[0].transpose();
            }
            return py::make_tuple(I, P);
        }, "(indices, positions) of the surviving short-arc candidates.")
        // Every short-arc midpoint compute_exposed_batch produced, before
        // filter_degenerate_pts ran: one row per candidate, a sphere may own
        // several. Also only populated after main_algorithm.
        .def_property_readonly("short_arc_candidates", [](const sdf::Options& o) {
            const auto& si = o.batch.point_sphere_idx;
            Eigen::VectorXi I((int)si.size());
            for (int i = 0; i < (int)si.size(); i++) I(i) = si[i];
            return py::make_tuple(I, o.batch.point_positions);
        }, "(sphere_idx, positions) of every short-arc midpoint, pre-filter.");

    // ── Interpolators ───────────────────────────────────────────────
    auto extract_surface_wrapper = [](const sdf::Interpolator& self,
                                      const Eigen::Vector3d& bbox_min,
                                      const Eigen::Vector3d& bbox_max,
                                      int nx, int ny, int nz,
                                      double iso,
                                      int chunk_size,
                                      bool lipschitz_postfix,
                                      bool use_dual_contouring) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        self.extract_surface(bbox_min, bbox_max, nx, ny, nz, iso, V, F,
                             chunk_size, lipschitz_postfix,
                             use_dual_contouring);
        return py::make_tuple(V, F);
    };

    py::class_<sdf::Interpolator, std::shared_ptr<sdf::Interpolator>>(m, "Interpolator")
        .def("predict", &sdf::Interpolator::predict,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def("predict_gradients", &sdf::Interpolator::predict_gradients,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def_property("verbose",
             &sdf::Interpolator::verbose,
             &sdf::Interpolator::set_verbose)
        .def("extract_surface", extract_surface_wrapper,
             py::arg("bbox_min"), py::arg("bbox_max"),
             py::arg("nx"), py::arg("ny"), py::arg("nz"),
             py::arg("iso") = 0.0, py::arg("chunk_size") = 5000,
             py::arg("lipschitz_postfix") = true,
             py::arg("use_dual_contouring") = false,
             "Extract an isosurface from the implicit field. "
             "If use_dual_contouring=True, uses gradient-aware Dual Contouring "
             "(QEF per cell, sharp-feature preserving); otherwise uses libigl "
             "marching cubes. Returns (V, F) as numpy arrays.");

    // Helper lambda for fit() with optional pointer args
    auto fit_wrapper = [](sdf::Interpolator& self,
                          const Eigen::MatrixXd& points,
                          const Eigen::VectorXd& values,
                          std::optional<Eigen::MatrixXd> gradients,
                          std::optional<Eigen::VectorXi> mask) {
        const Eigen::MatrixXd* g = gradients ? &*gradients : nullptr;
        const Eigen::VectorXi* m = mask ? &*mask : nullptr;
        self.fit(points, values, g, m);
    };

    py::class_<sdf::DuchonInterpolator, sdf::Interpolator, std::shared_ptr<sdf::DuchonInterpolator>>(m, "DuchonInterpolator")
        .def(py::init<const std::string&>(), py::arg("kernel") = "cubic")
        .def("fit", [&fit_wrapper](sdf::DuchonInterpolator& self,
                                    const Eigen::MatrixXd& points,
                                    const Eigen::VectorXd& values,
                                    std::optional<Eigen::MatrixXd> gradients,
                                    std::optional<Eigen::VectorXi> mask) {
            fit_wrapper(self, points, values, gradients, mask);
        }, py::arg("points"), py::arg("values"),
           py::arg("gradients") = py::none(), py::arg("mask") = py::none())
        .def("predict", &sdf::DuchonInterpolator::predict,
             py::arg("x_new"), py::arg("chunk_size") = 500)
        .def("predict_gradients", &sdf::DuchonInterpolator::predict_gradients,
             py::arg("x_new"), py::arg("chunk_size") = 500)
        .def("is_trained", &sdf::DuchonInterpolator::is_trained);

    py::class_<sdf::PUInterpolator, sdf::Interpolator, std::shared_ptr<sdf::PUInterpolator>>(m, "PUInterpolator")
           .def(py::init<const std::string&, double, int, int, double, const std::string&, bool, bool>(),
             py::arg("kernel") = "cubic", py::arg("overlap") = 0.25,
             py::arg("min_points") = 10, py::arg("max_points") = 200,
               py::arg("reg") = 1e-5,
             py::arg("partition") = "box",
             py::arg("verbose") = true,
             py::arg("pair_local") = true)
        .def("fit", [&fit_wrapper](sdf::PUInterpolator& self,
                                    const Eigen::MatrixXd& points,
                                    const Eigen::VectorXd& values,
                                    std::optional<Eigen::MatrixXd> gradients,
                                    std::optional<Eigen::VectorXi> mask) {
            fit_wrapper(self, points, values, gradients, mask);
        }, py::arg("points"), py::arg("values"),
           py::arg("gradients") = py::none(), py::arg("mask") = py::none())
        .def("predict", &sdf::PUInterpolator::predict,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def("predict_gradients", &sdf::PUInterpolator::predict_gradients,
             py::arg("x_new"), py::arg("chunk_size") = 5000)
        .def("is_trained", &sdf::PUInterpolator::is_trained)
        .def("predict_with_gradients", [](const sdf::PUInterpolator& self,
                                          const Eigen::MatrixXd& x) {
            Eigen::VectorXd v; Eigen::MatrixXd g;
            self.predict_with_gradients(x, v, g);
            return py::make_tuple(v, g);
        }, py::arg("x_new"))
        .def("patches_containing", &sdf::PUInterpolator::patches_containing, py::arg("pt"))
        .def("nearest_patch", &sdf::PUInterpolator::nearest_patch, py::arg("pt"))
        .def("export_lipschitz_state", [](const sdf::PUInterpolator& self) {
            // Read-only snapshot of every quantity the evaluator uses. Version
            // bumps whenever the evaluator rules below change.
            py::list patches;
            for (int p = 0; p < self.num_patches(); p++) {
                const auto& f = self.patch_fit(p);
                py::dict d;
                d["center"] = Eigen::Vector3d(self.patch_center(p));
                d["half_ext"] = Eigen::Vector3d(self.patch_half_ext(p));
                d["centers"] = f.centers();
                d["alpha"] = f.alpha();
                d["linear"] = f.affine_linear();
                d["constant"] = f.affine_constant();
                d["kernel"] = f.kernel_type();
                d["reg"] = f.regularization();
                patches.append(d);
            }
            py::dict policy;
            policy["partition"] = self.uses_box_partition() ? "box" : "sphere";
            policy["weight"] = "wendland_c2_(1-t)^4(4t+1)";
            policy["support_test"] = self.uses_box_partition()
                ? "all(|x-c| <= h)" : "|x-c|^2 <= R^2";
            policy["denominator_rule"] = "blend if W > 0 else nearest-support patch raw fit";
            policy["nearest_metric"] = "max(|x-c|-R,0) (sphere) / AABB distance (box)";
            policy["overlap"] = self.overlap();
            policy["dedup_tol"] = self.dedup_tol();
            py::dict out;
            out["version"] = 1;
            out["type"] = "PUInterpolator";
            out["policy"] = policy;
            out["patches"] = patches;
            return out;
        });

    // ── MainResult ──────────────────────────────────────────────────
    py::class_<sdf::MainResult>(m, "MainResult")
        .def_readonly("projections", &sdf::MainResult::projections)
        .def_readonly("visibility_mask", &sdf::MainResult::visibility_mask)
        .def_readonly("interpolator", &sdf::MainResult::interpolator);

    // ── Functions ───────────────────────────────────────────────────
    m.def("are_points_visible",
          static_cast<Eigen::VectorXi (*)(
              const Eigen::MatrixXd&, const Eigen::MatrixXd&,
              const Eigen::VectorXd&, double)>(&sdf::are_points_visible),
          py::arg("query_points"), py::arg("sdf_points"),
          py::arg("sdf_values"), py::arg("epsilon") = 1e-8);

    m.def("has_openmp", []() {
#ifdef USE_OPENMP
        return true;
#else
        return false;
#endif
    }, "Whether the module was built with OpenMP (multithreaded).");

    m.def("main_algorithm", &sdf::main_algorithm,
          py::arg("sdf_points"), py::arg("sdf_values"), py::arg("options"),
          "Run the full SDF gradient optimization pipeline.\n"
          "Returns MainResult with projections and visibility_mask.");
}
