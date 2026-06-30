// Generic native (C++/Eigen) differential inverse kinematics SQP loop.
//
// This module is model-independent. It dlopen's a CasADi-generated evaluator
// (q -> [holonomic_constraints, holonomic_constraints_jacobian], both dense)
// and runs the whole frame/Newton loop natively, so the per-iteration
// Python<->CasADi/NumPy boundary cost (the dominant cost of the pure-Python
// "dik" solver) disappears.
//
// At each Newton iteration it solves the equality-constrained least-squares QP
//     min_dq  1/2 dq^T H dq + g^T dq   s.t.   Kh dq = -phih
// in closed form via the KKT system
//     [ H   Kh^T ] [dq]   [ -g   ]
//     [ Kh   0   ] [l ] = [ -phih]

#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <Eigen/Dense>

namespace nb = nanobind;

// Must match the types used by CasADi's generated C code (defaults).
typedef long long int casadi_int;
typedef double casadi_real;

typedef int (*eval_t)(const casadi_real **, casadi_real **, casadi_int *, casadi_real *, int);
typedef int (*work_t)(casadi_int *, casadi_int *, casadi_int *, casadi_int *);
typedef void (*refcount_t)(void);
typedef int (*checkout_t)(void);
typedef void (*release_t)(int);

namespace {

template <typename T>
T load_symbol(void *handle, const std::string &name, bool required) {
    dlerror();  // clear
    void *sym = dlsym(handle, name.c_str());
    if (required && sym == nullptr) {
        throw std::runtime_error("dik_native: missing symbol '" + name + "': " + std::string(dlerror()));
    }
    return reinterpret_cast<T>(sym);
}

}  // namespace

// Returns (Qopt [nq x nf], success [nf]).
static std::tuple<Eigen::MatrixXd, std::vector<bool>> solve_native(
    const std::string &lib_path,
    const std::string &fname,
    const Eigen::MatrixXd &Jm,       // (3*nb_markers, nq)   constant marker jacobian
    const Eigen::MatrixXd &H,        // (nq, nq)             constant QP hessian
    const Eigen::MatrixXd &markers,  // (3*nb_markers, nf)   each column = frame markers, flattened 'F'
    const Eigen::MatrixXd &Qinit,    // (nq, nf)             initial guess per frame
    int nc,                          // number of holonomic constraints
    int max_iter,
    double constraint_eps,
    double step_eps,
    double objective_eps,
    bool warm_start) {              // init each frame from the previous frame solution
    const int nq = static_cast<int>(H.rows());
    const int nf = static_cast<int>(markers.cols());

    void *handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        throw std::runtime_error("dik_native: cannot dlopen '" + lib_path + "': " + std::string(dlerror()));
    }

    auto eval = load_symbol<eval_t>(handle, fname, true);
    auto work = load_symbol<work_t>(handle, fname + "_work", true);
    auto incref = load_symbol<refcount_t>(handle, fname + "_incref", false);
    auto decref = load_symbol<refcount_t>(handle, fname + "_decref", false);
    auto checkout = load_symbol<checkout_t>(handle, fname + "_checkout", false);
    auto release = load_symbol<release_t>(handle, fname + "_release", false);

    if (incref) incref();
    int mem = checkout ? checkout() : 0;

    // CasADi work vectors
    casadi_int sz_arg = 0, sz_res = 0, sz_iw = 0, sz_w = 0;
    work(&sz_arg, &sz_res, &sz_iw, &sz_w);
    std::vector<const casadi_real *> arg(std::max<casadi_int>(sz_arg, 1));
    std::vector<casadi_real *> res(std::max<casadi_int>(sz_res, 2));
    std::vector<casadi_int> iw(std::max<casadi_int>(sz_iw, 1));
    std::vector<casadi_real> w(std::max<casadi_int>(sz_w, 1));

    Eigen::MatrixXd Qopt(nq, nf);
    std::vector<bool> success(nf, false);

    // Buffers reused across iterations.
    Eigen::VectorXd q(nq);
    Eigen::VectorXd phih(nc);
    Eigen::MatrixXd Kh(nc, nq);  // dense, column-major (matches CasADi densified output)
    Eigen::VectorXd md(markers.rows());
    Eigen::VectorXd g(nq);

    // KKT system; the top-left block H is constant. Buffers and the LU solver are
    // reused across all iterations/frames to avoid per-iteration heap allocations.
    // A dense PartialPivLU of the full (nq+nc) KKT proved fastest and most accurate
    // here; the range-space method (factor H once, Schur complement) is both slower
    // -- H^{-1} Kh^T is a large solve every iteration -- and less accurate, because
    // H = Jm^T Jm + 1e-8 I is rank-deficient/ill-conditioned.
    Eigen::MatrixXd KKT(nq + nc, nq + nc);
    KKT.setZero();
    KKT.topLeftCorner(nq, nq) = H;
    Eigen::VectorXd rhs(nq + nc);
    Eigen::VectorXd sol(nq + nc);
    Eigen::PartialPivLU<Eigen::MatrixXd> lu(nq + nc);
    const Eigen::MatrixXd JmT = Jm.transpose();

    for (int f = 0; f < nf; ++f) {
        // warm-chain: keep q from the previous frame as the initial guess (smooth motion)
        if (!(warm_start && f > 0)) q = Qinit.col(f);
        const auto m_f = markers.col(f);
        double previous_objective = std::numeric_limits<double>::infinity();
        bool converged = false;

        for (int it = 0; it < max_iter; ++it) {
            md.noalias() = Jm * q;
            md += m_f;

            arg[0] = q.data();
            res[0] = phih.data();
            res[1] = Kh.data();
            eval(arg.data(), res.data(), iw.data(), w.data(), mem);

            const double constraint_norm = phih.norm();
            const double objective = 0.5 * md.squaredNorm();
            const bool constraints_ok = constraint_norm < constraint_eps;
            if (constraints_ok && std::abs(previous_objective - objective) < objective_eps) {
                converged = true;
                break;
            }

            g.noalias() = JmT * md;
            KKT.topRightCorner(nq, nc) = Kh.transpose();
            KKT.bottomLeftCorner(nc, nq) = Kh;
            rhs.head(nq) = -g;
            rhs.tail(nc) = -phih;

            lu.compute(KKT);
            sol.noalias() = lu.solve(rhs);
            auto dq = sol.head(nq);
            q += dq;
            previous_objective = objective;

            if (constraints_ok && dq.norm() < step_eps) {
                converged = true;
                break;
            }
        }

        // final constraint evaluation for the convergence flag
        arg[0] = q.data();
        res[0] = phih.data();
        res[1] = Kh.data();
        eval(arg.data(), res.data(), iw.data(), w.data(), mem);

        Qopt.col(f) = q;
        success[f] = converged || phih.norm() < constraint_eps;
    }

    if (release) release(mem);
    if (decref) decref();
    dlclose(handle);

    return {Qopt, success};
}

NB_MODULE(dik_native, m) {
    m.doc() = "Native (C++/Eigen) differential inverse kinematics SQP loop.";
    m.def("solve_native", &solve_native, nb::arg("lib_path"), nb::arg("fname"), nb::arg("Jm"), nb::arg("H"),
          nb::arg("markers"), nb::arg("Qinit"), nb::arg("nc"), nb::arg("max_iter"), nb::arg("constraint_eps"),
          nb::arg("step_eps"), nb::arg("objective_eps"), nb::arg("warm_start") = false);
}
