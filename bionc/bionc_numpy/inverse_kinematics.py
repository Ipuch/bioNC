import numpy as np
from casadi import vertcat, horzcat, MX, Function, sum1
from typing import Callable

from .enums import InitialGuessModeType
from .time_series_utils import TimeSeriesUtils
from ..bionc_casadi import NaturalCoordinates, SegmentNaturalCoordinates
from ..bionc_numpy.natural_coordinates import NaturalCoordinates as NaturalCoordinatesNumpy
from ..protocols.biomechanical_model import GenericBiomechanicalModel as BiomechanicalModel
from ..utils import constants
from ..utils.c3d_ik_exporter import C3DInverseKinematicsExporter
from ..utils.casadi_utils import _mx_to_sx, _solve_nlp, sarrus
from ..utils.heatmap_helpers import (
    check_format_experimental_heatmaps,
    compute_total_confidence,
)
from ..utils.heatmap_timeseries_helpers import HeatmapTimeseriesHelpers, subset_of_technical_markers
from ..utils.markers_check_import import load_and_validate_markers


class InverseKinematics:
    """
    Inverse kinematics solver also known as Multibody Kinematics Optimization (MKO)

    Attributes
    ----------
    model : BiomechanicalModel
        The model considered (bionc.numpy)
    experimental_markers : np.ndarray | str
        The experimental markers (3xNxM numpy array), or a path to a c3d file
    experimental_heatmaps : dict[str, np.ndarray]
        The experimental heatmaps, composed of two arrays and one float :
            * camera_parameters (3 x 4 x nb_cameras numpy array),
            * gaussian_parameters (5 x M x N x nb_cameras numpy array):
                - gaussian_parameters[0:2, :, :, :] is an array of the position (x, y) of the center of the gaussian.
                - gaussian_parameters[2:4, :, :, :] is an array of the standard deviation (x, y) of the gaussian.
                - gaussian_parameters[4,:,:,:] is an array of the magnitude of the gaussian.
    Q_init : np.ndarray
        The initial guess for the inverse kinematics computed from the experimental markers
    Qopt : np.ndarray
        The optimal solution of the inverse kinematics
    nb_frames : int
        The number of frames of the experimental markers
    nb_markers : int
        The number of markers of the experimental markers
    success_optim : list[bool]
        The success of convergence for each frame
    _frame_per_frame : bool
        If True, the inverse kinematics is solved frame per frame, otherwise it is solved for the whole motion
    _model_mx : BiomechanicalModel
        The model considered (bionc.casadi)
    _Q_sym : MX
        The symbolic variable of the inverse kinematics
    _vert_Q_sym : MX
        The symbolic variable of the inverse kinematics in a vertical format, useful when dealing with multiple frames at once
    _markers_sym : MX
        The symbolic variable of the experimental markers for one frame
    _objective_function : Callable
        The objective function to minimize

    Methods
    -------
    solve()
        Solves the inverse kinematics
    _declare_sym_Q()
        Declares the symbolic variables of the inverse kinematics
    _objective(Q: MX)
        builds the objective function to minimize
    _constraints(Q: MX)
        builds the constraints to satisfy
    animate()
        Animates the solution of the inverse kinematics
    _active_direct_frame_constraints()
        builds the constraints to ensure that the determinant of the matrix [u, v, w] is positive
    """

    def __init__(
        self,
        model: BiomechanicalModel,
        experimental_markers: np.ndarray | str = None,
        experimental_heatmaps: dict[str, np.ndarray] = None,
        active_direct_frame_constraints: bool = False,
        use_sx: bool = True,
    ):
        """
        Parameters
        ----------
        model : BiomechanicalModel
            The model considered (bionc.numpy)
        experimental_markers : np.ndarray | str
            The experimental markers (3 x nb_markers x nb_frames numpy array), or a path to a c3d file
        experimental_heatmaps : dict[str, np.ndarray]
            The experimental heatmaps, composed of two arrays and one float :
                * camera_parameters (3 x 4 x nb_cameras numpy array),
                * gaussian_parameters (5 x nb_markers x nb_frames x nb_cameras numpy array):
                    - gaussian_parameters[0:2, :, :, :] is an array of the position (x, y) of the center of the gaussian.
                    - gaussian_parameters[2:4, :, :, :] is an array of the standard deviation (x, y) of the gaussian.
                    - gaussian_parameters[4,:,:,:] is an array of the magnitude of the gaussian.
        active_direct_frame_constraints : bool
            If True, the direct frame constraints are active, otherwise they are not.
            It ensures that rigid body constraints lead to positive determinants or the matrix [u, v, w].
        use_sx : bool
            If True, the symbolic variables are SX, otherwise they are MX (SX are faster but take more memory)
        """

        self._validate_input(experimental_markers, experimental_heatmaps)
        self._initialize_attributes(
            model,
            experimental_markers,
            experimental_heatmaps,
            active_direct_frame_constraints,
            use_sx,
        )
        self._setup_optimization_problem()

    def _validate_input(self, experimental_markers, experimental_heatmaps):
        if experimental_markers is None and experimental_heatmaps is None:
            raise ValueError("Please provide experimental data, either marker or heatmap data")
        if experimental_markers is not None and experimental_heatmaps is not None:
            raise ValueError("Please choose between marker data and heatmap data")
        if experimental_heatmaps is not None:
            check_format_experimental_heatmaps(experimental_heatmaps)

    def _initialize_attributes(
        self,
        model,
        experimental_markers,
        experimental_heatmaps,
        active_direct_frame_constraints,
        use_sx,
    ):
        if not isinstance(model, BiomechanicalModel):
            raise ValueError("model must be a BiomechanicalModel")

        self.model = model
        self._model_mx = model.to_mx()
        self._active_direct_frame_constraints = active_direct_frame_constraints
        self.use_sx = use_sx

        self.experimental_markers = (
            load_and_validate_markers(experimental_markers) if experimental_markers is not None else None
        )
        self.experimental_heatmaps = experimental_heatmaps

        self._initialize_dimensions()
        self._initialize_heatmap_attributes()

        self.Qopt = None
        self.segment_determinants = None
        self.success_optim = []

    def _initialize_heatmap_attributes(self):
        if self.experimental_heatmaps:
            self.gaussian_parameters = np.reshape(
                self.experimental_heatmaps["gaussian_parameters"],
                (5 * self.nb_markers, self.nb_frames, self.nb_cameras),
            )
            self.camera_parameters = np.reshape(
                self.experimental_heatmaps["camera_parameters"], (3 * 4, self.nb_cameras)
            )
        else:
            self.gaussian_parameters = None
            self.camera_parameters = None

    def _initialize_dimensions(self):
        if self.experimental_markers is not None:
            self.nb_markers = self.experimental_markers.shape[1]
            self.nb_frames = self.experimental_markers.shape[2]
            self.nb_cameras = 0
        elif self.experimental_heatmaps is not None:
            self.nb_markers = self.experimental_heatmaps["gaussian_parameters"].shape[1]
            self.nb_frames = self.experimental_heatmaps["gaussian_parameters"].shape[2]
            self.nb_cameras = self.experimental_heatmaps["gaussian_parameters"].shape[3]

    def _setup_optimization_problem(self):
        self._Q_sym, self._vert_Q_sym = self._declare_sym_Q()
        self._setup_symbolic_variables()
        self.objective_sym = self._create_objective_function()
        self._objective_function = None
        self._update_objective_function()

    def _setup_symbolic_variables(self):
        if self.experimental_markers is not None:
            self._markers_sym = MX.sym("markers", (3, self.nb_markers))
            self._camera_parameters_sym = MX.sym("camera_param", (0, 0))
            self._gaussian_parameters_sym = MX.sym("gaussian_param", (0, 0))
        else:
            self._markers_sym = MX.sym("markers", (0, 0))
            self._camera_parameters_sym = MX.sym("cam_param", (3 * 4, self.nb_cameras))
            self._gaussian_parameters_sym = MX.sym("gaussian_param", (5 * self.nb_markers, self.nb_cameras))

    def _create_objective_function(self):
        if self.experimental_markers is not None:
            return [self._objective_minimize_marker_distance(self._Q_sym, self._markers_sym)]
        else:
            return [
                self._objective_maximize_confidence(
                    self._Q_sym, self._camera_parameters_sym, self._gaussian_parameters_sym
                )
            ]

    def _update_objective_function(self):
        """
        This method updates the objective function of the inverse kinematics problem. It is called each time a new
        objective is added to the inverse kinematics problem.
        """
        self._objective_function = Function(
            "objective_function",
            [
                self._Q_sym,
                self._markers_sym,
                self._camera_parameters_sym,
                self._gaussian_parameters_sym,
            ],
            [sum1(vertcat(*self.objective_sym))],
        ).expand()

    def add_objective(self, objective_function: Callable):
        """

        This method adds an extra objective to the inverse kinematics problem. The objective function has to be a
        CasADi Function with the following signature: [objective_sym] = objective_function(Q_sym, markers_sym)

        Parameters
        ----------
        objective_function: Callable[[MX, MX], MX]
            The objective function to add to the inverse kinematics problem
        """

        if isinstance(objective_function, Callable):
            # check the number of inputs of the objective function
            if objective_function.n_in() != 2:
                raise ValueError(
                    "objective_function must be a CasADi Function with the following signature: "
                    "[objective_sym] = objective_function(Q_sym, markers_sym)"
                )
            # check the number of outputs of the objective function
            if objective_function.n_out() != 1:
                raise ValueError(
                    "objective_function must be a CasADi Function with the following signature: "
                    "[objective_sym] = objective_function(Q_sym, markers_sym)"
                    "with an output of shape (1, 1)"
                )

            symbolic_objective = objective_function(self._Q_sym, self._markers_sym)
        else:
            raise TypeError(
                "objective_function must be a callable, i.e. a CasADi Function with the"
                "following signature: objective_sym = objective_function(Q_sym, markers_sym). "
                "It should be build from the following lines of code: \n"
                "from casadi import Function \n"
                "objective_function = Function('objective_function', "
                "[Q_sym, markers_sym], [objective_sym])"
            )

        self.objective_sym.append(symbolic_objective)
        self._update_objective_function()

    def solve(
        self,
        Q_init: np.ndarray | NaturalCoordinates = None,
        initial_guess_mode: InitialGuessModeType = InitialGuessModeType.FROM_CURRENT_MARKERS,
        method: str = "ipopt",
        options: dict = None,
    ) -> np.ndarray:
        """
        Solves the inverse kinematics

        Parameters
        ----------
        Q_init  : np.ndarray | NaturalCoordinates (optionnal)
            The initial guess for the inverse kinematics computed from the experimental markers.
            Expected when initial_guess_mode_type is USER_PROVIDED (for all frames)
            or USER_PROVIDED_FIRST_FRAME_ONLY (one frame only then)
        initial_guess_mode : InitialGuessModeType
            The type of initialization, by default InitialGuessModeType.FROM_CURRENT_MARKERS
        method : str
            The method to use to solve the inverse kinematics (ipopt, sqpmethod, dik, ...)
            Use "dik" for QP-based differential inverse kinematics with proxsuite/proxQP.
        options : dict
            The options to pass to the solver

        Returns
        -------
        np.ndarray
            The optimal solution of the inverse kinematics
        """

        options = self._get_solver_options(method, options)
        if method == "dik":
            self._validate_dik_problem()
        Q_init = self._get_initial_guess(Q_init, initial_guess_mode)

        if method == "dik":
            Qopt = self._solve_frame_per_frame_dik(Q_init, initial_guess_mode, options)
        else:
            Qopt = self._solve_frame_per_frame(Q_init, initial_guess_mode, method, options)

        self.Qopt = Qopt.reshape((12 * self.model.nb_segments, self.nb_frames))
        self.check_segment_determinants()
        return Qopt

    def _get_solver_options(self, method: str, options: dict) -> dict:
        default_options = {
            "sqpmethod": constants.SQP_IK_VALUES,
            "ipopt": constants.IPOPT_IK_VALUES,
            "dik": {
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
            },
        }
        if options is None:
            if method not in default_options:
                raise ValueError("method must be one of the following str: 'sqpmethod', 'ipopt' or 'dik'")
            return default_options[method]
        if method == "dik":
            if method not in default_options:
                raise ValueError("method must be one of the following str: 'sqpmethod', 'ipopt' or 'dik'")
            return {**default_options[method], **options}
        return options

    def _validate_dik_problem(self):
        if self.experimental_markers is None:
            raise ValueError('method="dik" only supports marker-based inverse kinematics.')
        if self.experimental_heatmaps is not None:
            raise ValueError('method="dik" does not support heatmap-based inverse kinematics.')
        if self._active_direct_frame_constraints:
            raise ValueError('method="dik" does not support active_direct_frame_constraints.')
        if len(self.objective_sym) > 1:
            raise ValueError('method="dik" only supports the default marker objective.')

    def _get_initial_guess(
        self,
        Q_init: np.ndarray | NaturalCoordinates,
        initial_guess_mode: InitialGuessModeType,
    ) -> np.ndarray | NaturalCoordinates:
        if initial_guess_mode in (
            InitialGuessModeType.FROM_CURRENT_MARKERS,
            InitialGuessModeType.FROM_FIRST_FRAME_MARKERS,
        ):
            return self._compute_Q_init_from_markers(initial_guess_mode)
        return self._validate_user_provided_Q_init(Q_init, initial_guess_mode)

    def _compute_Q_init_from_markers(self, initial_guess_mode: InitialGuessModeType) -> np.ndarray:
        if self.experimental_markers is None:
            raise ValueError("Please provide experimental_markers in order to initialize the optimization")
        if self.experimental_heatmaps is not None:
            raise ValueError("Q_init cannot be computed from markers using heatmap data")

        frame_slice = (
            slice(0, 1) if initial_guess_mode == InitialGuessModeType.FROM_FIRST_FRAME_MARKERS else slice(None)
        )
        return self.model.Q_from_markers(self.experimental_markers[:, :, frame_slice])

    def _validate_user_provided_Q_init(
        self,
        Q_init: np.ndarray | NaturalCoordinates,
        initial_guess_mode: InitialGuessModeType,
    ) -> np.ndarray | NaturalCoordinates:
        if Q_init is None:
            raise ValueError("Please provide Q_init if you want to use InitialGuessModeType.USER_PROVIDED.")

        if initial_guess_mode == InitialGuessModeType.USER_PROVIDED:
            if Q_init.shape[1] != self.nb_frames:
                raise ValueError(
                    f"Q_init.shape\\[1\\] must equal the number of frames ({self.nb_frames}). Currently, Q_init.shape\\[1\\] = {Q_init.shape[1]}."
                )
        elif initial_guess_mode == InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY:
            if Q_init.shape[1] != 1:
                raise ValueError("Provide only the first frame of Q_init. Currently, Q_init.shape\\[1\\] = 2")

        return Q_init

    def _solve_frame_per_frame(
        self,
        Q_init: np.ndarray | NaturalCoordinates,
        initial_guess_mode: InitialGuessModeType,
        method: str,
        options: dict,
    ) -> np.ndarray:
        Qopt = np.zeros((12 * self.model.nb_segments, self.nb_frames))
        nlp = self._setup_nlp()
        self.objective_function = np.zeros(self.nb_frames)

        for f in range(self.nb_frames):
            nlp["f"] = self._get_objective_for_frame(f)
            r, success = self._solve_single_frame(method, nlp, Q_init[:, f], options)
            Qopt[:, f : f + 1] = r["x"].toarray()
            self.objective_function[f] = r["f"]
            Q_init = self._update_initial_guess(Q_init, Qopt, initial_guess_mode, f)

        return Qopt

    def _solve_frame_per_frame_dik(
        self,
        Q_init: np.ndarray | NaturalCoordinates,
        initial_guess_mode: InitialGuessModeType,
        options: dict,
    ) -> np.ndarray:
        """
        Solves marker-based inverse kinematics frame per frame with differential IK.

        At each frame, this iteratively solves the linearized QP:
            min_dQ 1/2 ||Phi_m(Q) + K_m dQ||^2
            s.t.   K_h dQ = -Phi_h(Q)
        where Phi_m are marker defects and Phi_h are holonomic constraints.
        """
        proxsuite = self._check_proxsuite_available()

        Qopt = np.zeros((12 * self.model.nb_segments, self.nb_frames))
        self.objective_function = np.zeros(self.nb_frames)

        max_iter = options["max_iter"]
        constraint_eps = options.get("constraint_eps", options["eps"])
        step_eps = options.get("step_eps", options["eps"])
        objective_eps = options.get("objective_eps", options["eps"])

        marker_jacobian = np.ascontiguousarray(self.model.markers_constraints_jacobian(only_technical=True))
        marker_jacobian_transpose = np.ascontiguousarray(marker_jacobian.T)
        hessian = np.ascontiguousarray(
            marker_jacobian_transpose @ marker_jacobian + options["regularization"] * np.eye(marker_jacobian.shape[1])
        )
        dik_evaluator = self._setup_dik_evaluator() if options["use_casadi_dik_evaluators"] else None
        qp = self._setup_dik_qp(proxsuite, hessian, self.model.nb_holonomic_constraints, options)

        for f in range(self.nb_frames):
            q_current = np.asarray(Q_init[:, f], dtype=float).reshape(-1)
            converged = False
            previous_objective = np.inf

            for iteration in range(max_iter):
                marker_defects = self._dik_marker_defects(marker_jacobian, q_current, f)
                holonomic_constraints, holonomic_jacobian = self._dik_holonomic_constraints(q_current, dik_evaluator)

                final_constraint_norm = np.linalg.norm(holonomic_constraints)
                marker_objective = 0.5 * marker_defects.T @ marker_defects
                constraints_ok = final_constraint_norm < constraint_eps
                if constraints_ok and abs(previous_objective - marker_objective) < objective_eps:
                    converged = True
                    print("converged at iteration ", iteration)
                    break

                delta_q, success = self._solve_dik_qp(
                    qp,
                    marker_jacobian_transpose,
                    marker_defects,
                    holonomic_jacobian,
                    -holonomic_constraints,
                    options,
                    f,
                    iteration,
                )
                if not success:
                    break

                q_current = q_current + delta_q
                previous_objective = marker_objective

                if constraints_ok and np.linalg.norm(delta_q) < step_eps:
                    converged = True
                    break

            marker_defects = self._dik_marker_defects(marker_jacobian, q_current, f)
            holonomic_constraints, _ = self._dik_holonomic_constraints(q_current, dik_evaluator)
            final_constraint_norm = np.linalg.norm(holonomic_constraints)

            Qopt[:, f : f + 1] = q_current[:, np.newaxis]
            self.objective_function[f] = 0.5 * marker_defects.T @ marker_defects
            self.success_optim.append(converged or final_constraint_norm < constraint_eps)
            Q_init = self._update_initial_guess(Q_init, Qopt, initial_guess_mode, f)

        return Qopt

    @staticmethod
    def _check_proxsuite_available():
        try:
            import proxsuite
        except ImportError as exc:
            raise ImportError(
                'proxsuite is required to use method="dik". Install it with: pip install proxsuite'
            ) from exc
        return proxsuite

    def _setup_dik_evaluator(self) -> Function:
        Q = NaturalCoordinates(self._Q_sym)
        return Function(
            "dik_evaluator",
            [self._Q_sym],
            [
                self._model_mx.holonomic_constraints(Q),
                self._model_mx.holonomic_constraints_jacobian(Q),
            ],
        ).expand()

    def _dik_marker_defects(self, marker_jacobian: np.ndarray, q_current: np.ndarray, frame: int) -> np.ndarray:
        experimental_markers = self.experimental_markers[:, :, frame].flatten("F")
        return np.ascontiguousarray(experimental_markers + marker_jacobian @ q_current)

    def _dik_holonomic_constraints(
        self,
        q_current: np.ndarray,
        dik_evaluator: Function | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if dik_evaluator is not None:
            constraints, jacobian = dik_evaluator(q_current)
            return np.array(constraints, dtype=float).reshape(-1), np.ascontiguousarray(np.array(jacobian, dtype=float))

        Q_current = NaturalCoordinatesNumpy(q_current[:, np.newaxis])
        return (
            self.model.holonomic_constraints(Q_current).reshape(-1),
            np.ascontiguousarray(self.model.holonomic_constraints_jacobian(Q_current)),
        )

    @staticmethod
    def _setup_dik_qp(proxsuite, hessian: np.ndarray, nb_constraints: int, options: dict):
        nb_q = hessian.shape[0]
        max_delta_q = options["max_delta_q"]
        has_delta_bounds = max_delta_q is not None and np.isfinite(max_delta_q)
        nb_inequality = nb_q if has_delta_bounds else 0
        C = np.ascontiguousarray(np.eye(nb_q)) if has_delta_bounds else None
        l = -max_delta_q * np.ones(nb_q) if has_delta_bounds else None
        u = max_delta_q * np.ones(nb_q) if has_delta_bounds else None

        qp = proxsuite.proxqp.dense.QP(nb_q, nb_constraints, nb_inequality)
        qp.settings.eps_abs = options["proxqp_eps_abs"]
        qp.settings.max_iter = options["proxqp_max_iter"]
        qp.settings.verbose = options["verbose"]
        qp.init(
            hessian,
            np.zeros(nb_q),
            np.zeros((nb_constraints, nb_q)),
            np.zeros(nb_constraints),
            C,
            l,
            u,
        )
        return qp

    def _solve_dik_qp(
        self,
        qp,
        marker_jacobian_transpose: np.ndarray,
        marker_defects: np.ndarray,
        constraint_jacobian: np.ndarray,
        constraint_rhs: np.ndarray,
        options: dict,
        frame: int,
        iteration: int,
    ) -> tuple[np.ndarray, bool]:
        import proxsuite

        qp.update(
            g=np.ascontiguousarray(marker_jacobian_transpose @ marker_defects),
            A=np.ascontiguousarray(constraint_jacobian),
            b=np.ascontiguousarray(constraint_rhs),
            update_preconditioner=options["proxqp_update_preconditioner"],
        )
        qp.solve()

        success = qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED
        if not success and options["verbose"]:
            print(f"Warning: proxQP failed at frame {frame}, iteration {iteration}: {qp.results.info.status}")

        return qp.results.x, success

    def _setup_nlp(self) -> dict:
        constraints = self._constraints(self._Q_sym)
        if self._active_direct_frame_constraints:
            constraints = vertcat(constraints, self._direct_frame_constraints(self._Q_sym))

        nlp = {
            "x": self._vert_Q_sym,
            "g": _mx_to_sx(constraints, [self._vert_Q_sym]) if self.use_sx else constraints,
        }
        return nlp

    def _get_objective_for_frame(self, frame: int) -> MX:
        return self._objective_function(
            self._Q_sym,
            [] if self.experimental_markers is None else self.experimental_markers[:, :, frame],
            [] if self.experimental_heatmaps is None else self.camera_parameters,
            [] if self.experimental_heatmaps is None else self.gaussian_parameters[:, frame, :],
        )

    def _update_initial_guess(
        self,
        Q_init: np.ndarray | NaturalCoordinates,
        Qopt: np.ndarray | NaturalCoordinates,
        initial_guess_mode: InitialGuessModeType,
        frame: int,
    ):
        """Updates the initial guess for the next frame when solving frame per frame"""
        if (
            initial_guess_mode
            in (
                InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY,
                InitialGuessModeType.FROM_FIRST_FRAME_MARKERS,
            )
            and frame < self.nb_frames - 1
        ):
            Q_init = np.append(Q_init, Qopt[:, frame : frame + 1], axis=1)
        return Q_init

    def _solve_single_frame(self, method: str, nlp: dict, Q_init_frame: np.ndarray, options: dict) -> tuple:
        lbg, ubg = self._get_constraint_bounds()
        r, success = _solve_nlp(method, nlp, Q_init_frame, lbg, ubg, options)
        self.success_optim.append(success)
        return r, success

    def _get_constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lbg = np.zeros(self.model.nb_holonomic_constraints)
        ubg = np.zeros(self.model.nb_holonomic_constraints)
        if self._active_direct_frame_constraints:
            lbg = np.concatenate((lbg, np.zeros(self.model.nb_segments)))
            ubg = np.concatenate((ubg, np.full(self.model.nb_segments, np.inf)))
        return lbg, ubg

    def _declare_sym_Q(self) -> tuple[MX, MX]:
        """Declares the symbolic variables for the natural coordinates and handle single frame"""
        Q_sym = []
        f = 0
        Q_f_sym = []
        for ii in range(self.model.nb_segments):
            Q_f_sym.append(SegmentNaturalCoordinates.sym(f"_{ii}_{f}"))
        Q_sym.append(vertcat(*Q_f_sym))
        Q = horzcat(*Q_sym)
        vert_Q = vertcat(*Q_sym)
        return Q, vert_Q

    def _objective_minimize_marker_distance(self, Q, experimental_markers) -> MX:
        """
        Computes the objective function that minimizes marker distance and handles single frame

        Returns
        -------
        MX
            The objective function that minimizes the distance between the experimental markers and the model markers
        """
        f = 0
        Q_f = NaturalCoordinates(Q[:, f])
        xp_markers = (
            experimental_markers[:3, :, f] if isinstance(experimental_markers, np.ndarray) else experimental_markers
        )
        phim = self._model_mx.markers_constraints(xp_markers, Q_f, only_technical=True)

        return 1 / 2 * phim.T @ phim

    def _objective_maximize_confidence(self, Q, camera_parameters, gaussian_parameters) -> MX:
        """
        Computes the objective function that maximizes confidence value of the model keypoints
        Does not handle multi frames

        Returns
        -------
        MX
            The objective function that maximizes the confidence value of the model keypoints
        """
        Q_f = NaturalCoordinates(Q)
        all_marker_position = self._model_mx.markers(Q_f)
        marker_positions = subset_of_technical_markers(self._model_mx, all_marker_position)
        total_confidence = compute_total_confidence(marker_positions, camera_parameters, gaussian_parameters)

        return 1 / total_confidence

    def _constraints(self, Q) -> MX:
        """Computes the constraints and handle single frame or multi frames"""
        phir = []
        phik = []
        f = 0
        Q_f = NaturalCoordinates(Q[:, f])
        phir.append(self._model_mx.rigid_body_constraints(Q_f))
        phik.append(self._model_mx.joint_constraints(Q_f))
        return vertcat(*phir, *phik)

    def _direct_frame_constraints(self, Q):
        """Computes the direct frame constraints and handle single frame or multi frames"""
        direct_frame_constraints = []
        f = 0
        Q_f = NaturalCoordinates(Q[:, f])
        for ii in range(self.model.nb_segments):
            u, v, w = Q_f.vector(ii).to_uvw()
            direct_frame_constraints.append(sarrus(horzcat(u, v, w)))
        return vertcat(*direct_frame_constraints)

    def check_segment_determinants(self):
        """Checks the determinant of each segment frame with the optimal solution"""
        self.segment_determinants = np.zeros((self.model.nb_segments, self.nb_frames))
        for i in range(0, self.Qopt.shape[1]):
            Qi = NaturalCoordinatesNumpy(self.Qopt)[:, i : i + 1]
            for s in range(0, self.model.nb_segments):
                u, v, w = Qi.vector(s).to_uvw()
                matrix = np.concatenate((u[:, np.newaxis], v[:, np.newaxis], w[:, np.newaxis]), axis=1)
                self.segment_determinants[s, i] = np.linalg.det(matrix)
                if self.segment_determinants[s, i] < 0:
                    print(f"Warning: frame {i} segment {s} has a negative determinant")

    def sol(self):
        """
        Create and return a dict that contains the output of each optimization.
        Return
        ------
        self.output: dict[str, np.ndarray | list[str]]
            - 'marker_residuals_norm' : np.ndarray
                Norm of the residuals for each marker
            - 'marker_residuals_xyz' : np.ndarray
                Residuals of the marker on all axis
            - 'total_marker_residuals' : np.ndarray
                Residuals of all marker for each frame
            - 'max_marker_distance' : list[str]
                A list of the marker with the highest residual for each frame
            - 'joint_residuals' : np.ndarray
                Joint constraint residual for each joint and each frame
            - 'total_joint_residuals' : list[str]
                Global joint constraint residual for each frame
            - 'max_joint_violation' : list[str]
                A list of the joint with the highest residual for each frame
            - 'rigidity_residuals' : np.ndarray
                Residuals of the rigidity constraint for each segment and each frame
            - 'total_rigidity_residuals' : np.ndarray
                Global rigidity constraint residual for each frame
            - 'max_rigidbody_violation' : list[str]
                A list of the segment with the highest rigidity residual for each frame
            - 'success' : list[bool]
                A list of boolean indicating for each frame of the sucess of the optimization.

        """

        joint_residuals = TimeSeriesUtils.joint_constraints(self.model, self.Qopt)
        rigidity_residuals = TimeSeriesUtils.rigid_body_constraints(self.model, self.Qopt)
        segment_rigidity_residuals = np.reshape(
            rigidity_residuals, (6, self.model.nb_segments, self.nb_frames), order="F"
        )
        segment_rigidity_residual_norm = np.sqrt(np.sum(segment_rigidity_residuals**2, axis=0))

        # Global will correspond to the squared sum of all the specific residuals
        total_joint_residuals = TimeSeriesUtils.total_joint_constraints(self.model, self.Qopt)
        total_rigidity_residuals = TimeSeriesUtils.total_rigid_body_constraints(self.model, self.Qopt)

        ind_max_rigidy_error = np.argmax(segment_rigidity_residual_norm, axis=0)
        ind_max_joint_constraint_error = np.argmax(joint_residuals, axis=0)

        # Create a list of marker, segment and joint from the indices
        max_rigidbody_violation = [self.model.segment_names[ind_max] for ind_max in ind_max_rigidy_error]
        max_joint_violation = [
            self.model.joint_names[self.model.joint_constraints_indices[ind_max]]
            for ind_max in ind_max_joint_constraint_error
        ]

        total_euler_angles = TimeSeriesUtils.total_euler_angles(self.model, self.Qopt)

        self.output = dict(
            objective_function=self.objective_function,
            success=self.success_optim,
            joint_residuals=joint_residuals,
            total_joint_residuals=total_joint_residuals,
            max_joint_violation=max_joint_violation,
            rigidity_residuals=rigidity_residuals,
            segment_rigidity_residuals=segment_rigidity_residuals,
            segment_rigidity_residual_norm=segment_rigidity_residual_norm,
            total_rigidity_residuals=total_rigidity_residuals,
            max_rigidbody_violation=max_rigidbody_violation,
            total_euler_angles=total_euler_angles,
        )

        if self.experimental_markers is not None:
            marker_residuals_xyz = TimeSeriesUtils.marker_constraints_xyz(
                self.model, self.Qopt, self.experimental_markers
            )
            marker_residuals_norm = np.sqrt(np.sum(marker_residuals_xyz**2, axis=0))

            total_marker_residuals = TimeSeriesUtils.total_marker_constraints(
                self.model, self.Qopt, self.experimental_markers
            )
            ind_max_marker_distance = np.argmax(marker_residuals_norm, axis=0)
            max_marker_distance = [self.model.marker_names_technical[ind_max] for ind_max in ind_max_marker_distance]

            self.output["marker_residuals_xyz"] = marker_residuals_xyz
            self.output["marker_residuals_norm"] = marker_residuals_norm
            self.output["total_marker_residuals"] = total_marker_residuals
            self.output["max_marker_distance"] = max_marker_distance

        if self.experimental_heatmaps is not None:
            frame_total_confidence = HeatmapTimeseriesHelpers.total_confidence(
                self.model, self.Qopt, self.camera_parameters, self.gaussian_parameters
            )
            heatmap_confidence_3d = HeatmapTimeseriesHelpers.total_confidence_for_all_markers(
                self.model, self.Qopt, self.camera_parameters, self.gaussian_parameters
            )
            heatmap_confidence_2d = HeatmapTimeseriesHelpers.total_confidence_for_all_markers_on_each_camera(
                self.model, self.Qopt, self.camera_parameters, self.gaussian_parameters
            )

            self.output["total_heatmap_confidence"] = frame_total_confidence
            self.output["heatmap_confidence_3d"] = heatmap_confidence_3d  # 3d [Nb_markers, N_frame],
            self.output["heatmap_confidence_2d"] = heatmap_confidence_2d  # [Nb_markers, Nb_camera, N_frame],

        return self.output

    def export_in_c3d(self, filename, unit="m"):
        c3d_export = C3DInverseKinematicsExporter(model=self.model, filename=filename)
        c3d_export.add_natural_coordinate(self.Qopt, unit=unit)
        c3d_export.add_technical_markers(self.Qopt, unit=unit)
        c3d_export.export()
