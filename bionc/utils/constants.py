import numpy as np

SQP_IK_VALUES = {
    "beta": 0.8,  # default value
    "c1": 0.0001,  # default value
    # "hessian_approximation": "exact",
    "hessian_approximation": "limited-memory",  # faster but might fail to converge
    "lbfgs_memory": 10,
    "max_iter": 50,
    "max_iter_ls": 3,
    "merit_memory": 4,
    "print_header": False,
    "print_time": True,
    "qpsol": "qpoases",
    "tol_du": 0.1,
    "tol_pr": 0.1,
    "qpsol_options": {"error_on_fail": False},
}

IPOPT_IK_VALUES = {
    "ipopt.hessian_approximation": "exact",  # recommended
    "ipopt.warm_start_init_point": "no",
    "ipopt.print_level": 0,
    "ipopt.print_timing_statistics": "no",
}

PROXQP_DIK_VALUES = {
    "max_iter": 100,
    "eps": 1e-6,
    "constraint_eps": 1e-6,
    "step_eps": 1e-8,
    "objective_eps": 1e-12,
    "regularization": 1e-8,
    "max_delta_q": np.inf,
    "proxqp_eps_abs": 1e-8,
    "proxqp_max_iter": 1000,
    "proxqp_update_preconditioner": False,
    "use_casadi_dik_evaluators": True,
    "verbose": False,
}
