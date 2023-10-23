from typing import Callable

from casadi import vertcat, horzcat, MX, nlpsol, SX, Function, sum1, dot, exp
import numpy as np
from pyomeca import Markers

from ..bionc_casadi import NaturalCoordinates, SegmentNaturalCoordinates
from ..protocols.biomechanical_model import GenericBiomechanicalModel as BiomechanicalModel
from ..bionc_numpy.natural_coordinates import NaturalCoordinates as NaturalCoordinatesNumpy


def _mx_to_sx(mx: MX, symbolics: list[MX]) -> SX:
    """
    Converts a MX to a SX

    Parameters
    ----------
    mx : MX
        The MX to convert
    symbolics : list[MX]
        The symbolics to use

    Returns
    -------
    The converted SX
    """
    f = Function("f", symbolics, [mx]).expand()
    return f(*symbolics)


def _solve_nlp(method: str, nlp: dict, Q_init: np.ndarray, lbg: np.ndarray, ubg: np.ndarray, options: dict):
    """
    Solves a nonlinear program with CasADi

    Parameters
    ----------
    method : str
        The method to use to solve the NLP (ipopt, sqpmethod, ...)
    nlp : dict
        The NLP to solve
    Q_init : np.ndarray
        The initial guess
    lbg : np.ndarray
        The lower bound of the constraints
    ubg : np.ndarray
        The upper bound of the constraints
    options : dict
        The options to pass to the solver

    Returns
    -------
    The output of the solver
    """
    S = nlpsol("InverseKinematics", method, nlp, options)
    r = S(x0=Q_init, lbg=lbg, ubg=ubg)

    if S.stats()["success"] is False:
        print("Inverse Kinematics failed to converge")

    return r


def sarrus(matrix: MX):
    """
    Computes the determinant of a 3x3 matrix using the Sarrus rule

    Parameters
    ----------
    matrix : MX
        The matrix to compute the determinant of

    Returns
    -------
    The determinant of the matrix
    """
    return (
        matrix[0, 0] * matrix[1, 1] * matrix[2, 2]
        + matrix[0, 1] * matrix[1, 2] * matrix[2, 0]
        + matrix[0, 2] * matrix[1, 0] * matrix[2, 1]
        - matrix[0, 0] * matrix[1, 2] * matrix[2, 1]
        - matrix[0, 1] * matrix[1, 0] * matrix[2, 2]
        - matrix[0, 2] * matrix[1, 1] * matrix[2, 0]
    )


def _compute_phim_hmp(X, cal, ratio, amp, pos, sigma):
    phim = amp[0] * exp(
        -(
            (
                ((ratio * (dot(X, cal[1, 0:3]) + cal[1, 3]) / (dot(X, cal[2, 0:3]) + cal[2, 3]) - pos[0]) ** 2)
                / (2 * sigma[0] ** 2)
            )
            + (
                ((ratio * (dot(X, cal[0, 0:3]) + cal[0, 3]) / (dot(X, cal[2, 0:3]) + cal[2, 3]) - pos[1]) ** 2)
                / (2 * sigma[1] ** 2)
            )
        )
    )
    return phim


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
        The experimental heatmaps, composed of two arrays and one float : camera_parameters (3 x 4 x nb_cameras numpy array), gaussian_parameters (5 x M x N x nb_cameras numpy array)
    Q_init : np.ndarray
        The initial guess for the inverse kinematics computed from the experimental markers
    Qopt : np.ndarray
        The optimal solution of the inverse kinematics
    nb_frames : int
        The number of frames of the experimental markers
    nb_markers : int
        The number of markers of the experimental markers
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
        experimental_heatmaps: dict[str, np.ndarray] | str = None,
        Q_init: np.ndarray | NaturalCoordinates = None,
        solve_frame_per_frame: bool = True,
        active_direct_frame_constraints: bool = False,
        use_sx: bool = True,
    ):
        """
        Parameters
        ----------
        model : BiomechanicalModel
            The model considered (bionc.numpy)
        experimental_markers : np.ndarray | str
            The experimental markers (3xNxM numpy array), or a path to a c3d file
        experimental_heatmaps : dict[str, np.ndarray]
            The experimental heatmaps, composed of two arrays and one float : camera_parameters (3 x 4 x nb_cameras numpy array), gaussian_parameters (5 x M x N x nb_cameras numpy array)
        Q_init : np.ndarray | NaturalCoordinates
            The initial guess for the inverse kinematics computed from the experimental markers
        solve_frame_per_frame : bool
            If True, the inverse kinematics is solved frame per frame, otherwise it is solved for the whole motion
        active_direct_frame_constraints : bool
            If True, the direct frame constraints are active, otherwise they are not.
            It ensures that rigid body constraints lead to positive determinants or the matrix [u, v, w].
        use_sx : bool
            If True, the symbolic variables are SX, otherwise they are MX (SX are faster but take more memory)
        """

        self._frame_per_frame = solve_frame_per_frame
        self._active_direct_frame_constraints = active_direct_frame_constraints
        self.use_sx = use_sx

        if not isinstance(model, BiomechanicalModel):
            raise ValueError("model must be a BiomechanicalModel")
        self.model = model
        self._model_mx = model.to_mx()

        if experimental_markers is None and experimental_heatmaps is None:
            raise ValueError("Please feed experimental data, either marker or heatmap data")
        if experimental_markers is not None and experimental_heatmaps is not None:
            raise ValueError("Please choose between marker data and heatmap data")
        
        if experimental_markers is not None:
            if isinstance(experimental_markers, str):
                self.experimental_markers = Markers.from_c3d(experimental_markers).to_numpy()
            if isinstance(experimental_markers, np.ndarray):
                if (
                    experimental_markers.shape[0] != 3
                    or experimental_markers.shape[1] < 1
                    or len(experimental_markers.shape) < 3
                ):
                    raise ValueError("experimental_markers must be a 3xNxM numpy array")
                self.experimental_markers = experimental_markers
            self.nb_markers = self.experimental_markers.shape[1]
            self.nb_frames = self.experimental_markers.shape[2]

            self.nb_cameras = 0
            self.experimental_heatmaps = None
            self.gaussian_parameters = None
            self.camera_parameters = None

            self._markers_sym = MX.sym("markers", (3, self.nb_markers))
            self.objective_sym = [self._objective_minimize_marker_distance(self._Q_sym, self._markers_sym)]
       
        if experimental_heatmaps is not None:
            if isinstance(experimental_heatmaps, dict):
                if solve_frame_per_frame is False:
                    raise ValueError("NotImplementedError")
                else:
                    self.experimental_heatmaps = experimental_heatmaps
                    self.gaussian_parameters = experimental_heatmaps["gaussian_parameters"]
                    self.camera_parameters = experimental_heatmaps["camera_parameters"]
                    self.ratio = experimental_heatmaps["ratio"] # to be deleted
                    self.nb_markers = self.experimental_heatmaps["gaussian_parameters"].shape[1]
                    self.nb_cameras = self.experimental_heatmaps["gaussian_parameters"].shape[3]

                    self.experimental_markers = None

                    self._camera_parameters_sym = MX.sym("cam_param", (3, 4*self.nb_cameras))
                    self._gaussian_parameters_sym = MX.sym("gaussian_param", (5, self.nb_markers*self.nb_cameras))
                    self._ratio_sym = MX.sym("ratio")
                    self.objective_sym = [self._objective_maximize_confidence(self._Q_sym, self._camera_parameters_sym, self._gaussian_parameters_sym, self._ratio_sym)]

            else:
                raise ValueError("Please provide experimental_heatmaps as a dictionnary")

        if Q_init is None and self.experimental_markers is not None:
            self.Q_init = self.model.Q_from_markers(self.experimental_markers[:, :, :])
        elif Q_init is None and self.experimental_heatmaps is not None:
            raise ValueError("Q_init must be provided for heatmap analysis")
        else:
            self.Q_init = Q_init

        self.Qopt = None
        self.segment_determinants = None
        self._Q_sym, self._vert_Q_sym = self._declare_sym_Q()

        self._objective_function = None
        self._update_objective_function()

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
                self._ratio_sym,
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

    def solve(self, method: str = "ipopt", options: dict = None) -> np.ndarray:
        """
        Solves the inverse kinematics

        Parameters
        ----------
        method : str
            The method to use to solve the NLP (ipopt, sqpmethod, ...)
        options : dict
            The options to pass to the solver

        Returns
        -------
        np.ndarray
            The optimal solution of the inverse kinematics
        """

        if method == "sqpmethod":
            if options is None:
                options = {
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
        elif method == "ipopt":
            if options is None:
                options = {
                    "ipopt.hessian_approximation": "exact",  # recommended
                    "ipopt.warm_start_init_point": "no",
                    "ipopt.print_level": 0,
                    "ipopt.print_timing_statistics": "no",
                }
        else:
            raise ValueError("method must be one of the following str: 'sqpmethod' or 'ipopt'")

        if self._frame_per_frame:
            Qopt = np.zeros((12 * self.model.nb_segments, self.nb_frames))
            lbg = np.zeros(self.model.nb_holonomic_constraints)
            ubg = np.zeros(self.model.nb_holonomic_constraints)
            constraints = self._constraints(self._Q_sym)
            if self._active_direct_frame_constraints:
                constraints = vertcat(constraints, self._direct_frame_constraints(self._Q_sym))
                lbg = np.concatenate((lbg, np.zeros(self.model.nb_segments)))
                # upper bounds infinity
                ubg = np.concatenate((ubg, np.full(self.model.nb_segments, np.inf)))
            nlp = dict(
                x=self._vert_Q_sym,
                g=_mx_to_sx(constraints, [self._vert_Q_sym]) if self.use_sx else constraints,
            )
            for f in range(self.nb_frames):
                objective = self._objective_function(
                    self._Q_sym,
                    self.experimental_markers[:, :, f],
                    self.camera_parameters,
                    self.gaussian_parameters[:, :, f, :],
                    self.ratio,
                )

                nlp["f"] = _mx_to_sx(objective, [self._vert_Q_sym]) if self.use_sx else objective
                Q_init = self.Q_init[:, f : f + 1]
                r = _solve_nlp(method, nlp, Q_init, lbg, ubg, options)
                Qopt[:, f : f + 1] = r["x"].toarray()
        else:
            constraints = self._constraints(self._Q_sym)
            if self._active_direct_frame_constraints:
                constraints = vertcat(constraints, self._direct_frame_constraints(self._Q_sym))
            if self.experimental_markers is not None:
                objective = self._objective_minimize_marker_distance(self._Q_sym, self.experimental_markers)
            else:
                raise ValueError("NotImplementedError")
            nlp = dict(
                x=self._vert_Q_sym,
                f=_mx_to_sx(objective, [self._vert_Q_sym]) if self.use_sx else objective,
                g=_mx_to_sx(constraints, [self._vert_Q_sym]) if self.use_sx else constraints,
            )
            Q_init = self.Q_init.reshape((12 * self.model.nb_segments * self.nb_frames, 1))
            lbg = np.zeros(self.model.nb_holonomic_constraints * self.nb_frames)
            ubg = np.zeros(self.model.nb_holonomic_constraints * self.nb_frames)
            if self._active_direct_frame_constraints:
                lbg = np.concatenate((lbg, np.zeros(self.model.nb_segments * self.nb_frames)))
                ubg = np.concatenate((ubg, np.full(self.model.nb_segments * self.nb_frames, np.inf)))
            r = _solve_nlp(method, nlp, Q_init, lbg, ubg, options)
            Qopt = r["x"].reshape((12 * self.model.nb_segments, self.nb_frames)).toarray()

        self.Qopt = Qopt.reshape((12 * self.model.nb_segments, self.nb_frames))

        self.check_segment_determinants()

        return Qopt

    def _declare_sym_Q(self) -> tuple[MX, MX]:
        """Declares the symbolic variables for the natural coordinates and handle single frame or multi frames"""
        Q_sym = []
        nb_frames = 1 if self._frame_per_frame else self.nb_frames
        for f in range(nb_frames):
            Q_f_sym = []
            for ii in range(self.model.nb_segments):
                Q_f_sym.append(SegmentNaturalCoordinates.sym(f"_{ii}_{f}"))
            Q_sym.append(vertcat(*Q_f_sym))
        Q = horzcat(*Q_sym)
        vert_Q = vertcat(*Q_sym)
        return Q, vert_Q

    def _objective_minimize_marker_distance(self, Q, experimental_markers) -> MX:
        """
        Computes the objective function that minimizes marker distance and handles single frame or multi frames

        Returns
        -------
        MX
            The objective function that minimizes the distance between the experimental markers and the model markers
        """
        error_m = 0
        nb_frames = 1 if self._frame_per_frame else self.nb_frames
        for f in range(nb_frames):
            Q_f = NaturalCoordinates(Q[:, f])
            xp_markers = (
                experimental_markers[:3, :, f] if isinstance(experimental_markers, np.ndarray) else experimental_markers
            )
            phim = self._model_mx.markers_constraints(xp_markers, Q_f, only_technical=True)
            error_m += 1 / 2 * phim.T @ phim
        return error_m

    def _objective_maximize_confidence(self, Q, camera_parameters, gaussian_parameters, ratio) -> MX:
        """
        Computes the objective function that maximizes confidence value of the model keypoints
        Does not handle multi frames !

        Returns
        -------
        MX
            The objective function that maximizes the confidence value of the model keypoints
        """
        error_m = 0
        Q_f = NaturalCoordinates(Q)
        for s in range(len(self.model.segment_names)):
            n = len(self.model.segments[self.model.segment_names[s]]._markers)
            for m in range(n):
                N = self._model_mx.segments[self.model.segment_names[s]]._markers[m].interpolation_matrix
                X = N @ Q_f[12 * s : 12 * (s + 1)]
                for c in range(camera_parameters.shape[2]):
                    cal = camera_parameters[:, :, c]
                    pos = gaussian_parameters[
                        0:2, name2kp[self._model_mx.segments[self.model.segment_names[s]]._markers[m].name], :, c
                    ]
                    sig = gaussian_parameters[
                        2:4, name2kp[self._model_mx.segments[self.model.segment_names[s]]._markers[m].name], :, c
                    ]
                    amp = gaussian_parameters[
                        4, name2kp[self._model_mx.segments[self.model.segment_names[s]]._markers[m].name], :, c
                    ]
                    error_m += _compute_phim_hmp(X, cal, ratio, amp, pos, sig)

        return 1 / error_m

    def _constraints(self, Q) -> MX:
        """Computes the constraints and handle single frame or multi frames"""
        nb_frames = 1 if self._frame_per_frame else self.nb_frames
        phir = []
        phik = []
        for f in range(nb_frames):
            Q_f = NaturalCoordinates(Q[:, f])
            phir.append(self._model_mx.rigid_body_constraints(Q_f))
            phik.append(self._model_mx.joint_constraints(Q_f))
        return vertcat(*phir, *phik)

    def _direct_frame_constraints(self, Q):
        """Computes the direct frame constraints and handle single frame or multi frames"""
        nb_frames = 1 if self._frame_per_frame else self.nb_frames
        direct_frame_constraints = []
        for f in range(nb_frames):
            Q_f = NaturalCoordinates(Q[:, f])
            for ii in range(self.model.nb_segments):
                u, v, w = Q_f.vector(ii).to_uvw()
                direct_frame_constraints.append(sarrus(horzcat(u, v, w)))
        return vertcat(*direct_frame_constraints)

    def check_segment_determinants(self):
        """Checks the determinant of each segment frame"""
        self.segment_determinants = np.zeros((self.model.nb_segments, self.nb_frames))
        for i in range(0, self.Qopt.shape[1]):
            Qi = NaturalCoordinatesNumpy(self.Qopt)[:, i : i + 1]
            for s in range(0, self.model.nb_segments):
                u, v, w = Qi.vector(s).to_uvw()
                matrix = np.concatenate((u[:, np.newaxis], v[:, np.newaxis], w[:, np.newaxis]), axis=1)
                self.segment_determinants[s, i] = np.linalg.det(matrix)
                if self.segment_determinants[s, i] < 0:
                    print(f"Warning: frame {i} segment {s} has a negative determinant")

    # todo: def sol() -> dict that returns the details of the inverse kinematics such as all the metrics, etc...
    #     def sol(self):
    #             """
    #             Create and return a dict that contains the output of each optimization.
    #             Return
    #             ------
    #             self.output: dict()
    #                 The output of least_square function, such as number of iteration per frames,
    #                 and the marker with highest residual
    #             """
    #             residuals_xyz = np.zeros((self.nb_markers * self.nb_dim, self.nb_frames))
    #             residuals = np.zeros((self.nb_markers, self.nb_frames))
    #             for f in range(self.nb_frames):
    #                 #  residuals_xyz must contains position for each markers on axis x, y and z
    #                 #  (or less depending on number of dimensions)
    #             self.output = dict(
    #                 residuals=residuals,
    #                 residuals_xyz=residuals_xyz,
    #                 max_marker=[self.marker_names[i] for i in np.argmax(residuals, axis=0)],
    #                 message=[sol.message for sol in self.list_sol],
    #                 status=[sol.status for sol in self.list_sol],
    #                 success=[sol.success for sol in self.list_sol],
    #             )
    #             return self.output
