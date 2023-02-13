from casadi import vertcat, horzcat, MX, Function, nlpsol
import numpy as np
from pyomeca import Markers

from ..bionc_casadi import (
    NaturalCoordinates,
    SegmentNaturalCoordinates,
)

from ..protocols.biomechanical_model import GenericBiomechanicalModel as BiomechanicalModel


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
    S = nlpsol(
        "InverseKinematics",
        method,
        nlp,
        options,
    )
    r = S(x0=Q_init, lbg=lbg, ubg=ubg)
    return r


class InverseKinematics:
    """
    Inverse kinematics solver also known as Multibody Kinematics Optimization (MKO)

    Attributes
    ----------
    model : BiomechanicalModel
        The model considered (bionc.numpy)
    experimental_markers : np.ndarray | str
        The experimental markers (3xNxM numpy array), or a path to a c3d file
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
    """

    def __init__(
        self,
        model: BiomechanicalModel,
        experimental_markers: np.ndarray | str,
        solve_frame_per_frame: bool = True,
    ):
        self._frame_per_frame = solve_frame_per_frame

        if not isinstance(model, BiomechanicalModel):
            raise ValueError("model must be a BiomechanicalModel")
        self.model = model
        self._model_mx = model.to_mx()

        if isinstance(experimental_markers, str):
            self.experimental_markers = Markers.from_c3d(experimental_markers).to_numpy()
        elif isinstance(experimental_markers, np.ndarray):
            if (
                experimental_markers.shape[0] != 3
                or experimental_markers.shape[1] < 1
                or len(experimental_markers.shape) < 3
            ):
                raise ValueError("experimental_markers must be a 3xNxM numpy array")
            self.experimental_markers = experimental_markers
        else:
            raise ValueError("experimental_markers must be a numpy array or a path to a c3d file")

        self.Q_init = self.model.Q_from_markers(self.experimental_markers[:, :, :])
        self.Qopt = None

        self.nb_frames = self.experimental_markers.shape[2]
        self.nb_markers = self.experimental_markers.shape[1]

        self._Q_sym, self._vert_Q_sym = self._declare_sym_Q()
        self._markers_sym = MX.sym("markers", (3, self.nb_markers))

        objective_sym = self._objective(self._Q_sym, self._markers_sym)
        self._objective_function = Function(
            "objective_function", [self._Q_sym, self._markers_sym], [objective_sym]
        ).expand()

    def solve(
        self,
        method: str = "ipopt",
        options: dict = None,
    ) -> np.ndarray:
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
            raise ValueError("method must be 'sqpmethod' or 'ipopt'")

        if self._frame_per_frame:
            Qopt = np.zeros((12 * self.model.nb_segments, self.nb_frames))
            lbg = np.zeros(self.model.nb_holonomic_constraints)
            ubg = np.zeros(self.model.nb_holonomic_constraints)
            constraints = self._constraints(self._Q_sym)
            nlp = dict(
                x=self._vert_Q_sym,
                g=constraints,
            )
            for f in range(self.nb_frames):
                nlp["f"] = self._objective_function(self._Q_sym, self.experimental_markers[:, :, f])
                Q_init = self.Q_init[:, f : f + 1]
                r = _solve_nlp(method, nlp, Q_init, lbg, ubg, options)
                Qopt[:, f : f + 1] = r["x"].toarray()
        else:
            constraints = self._constraints(self._Q_sym)
            objective = self._objective(self._Q_sym, self.experimental_markers)
            nlp = dict(
                x=self._vert_Q_sym,
                f=objective,
                g=constraints,
            )
            Q_init = self.Q_init.reshape((12 * self.model.nb_segments * self.nb_frames, 1))
            lbg = np.zeros(self.model.nb_holonomic_constraints * self.nb_frames)
            ubg = np.zeros(self.model.nb_holonomic_constraints * self.nb_frames)
            r = _solve_nlp(method, nlp, Q_init, lbg, ubg, options)
            Qopt = r["x"].reshape((12 * self.model.nb_segments, self.nb_frames)).toarray()

        self.Qopt = Qopt.reshape((12 * self.model.nb_segments, self.nb_frames))

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

    def _objective(self, Q, experimental_markers) -> MX:
        """Computes the objective function and handle single frame or multi frames"""
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

    # todo: def sol() -> dict that returns the details of the inverse kinematics such as all the metrics, etc...
