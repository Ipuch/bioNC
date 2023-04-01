import numpy as np
from typing import Union
from .natural_vector import NaturalVector
from ..utils.enums import NaturalAxis


class SegmentNaturalCoordinates(np.ndarray):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array: Union[np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """

        obj = np.asarray(input_array).view(cls)

        if obj.shape.__len__() == 1:
            obj = obj[:, np.newaxis]

        return obj

    @classmethod
    def from_components(
        cls,
        u: Union[np.ndarray, list] = None,
        rp: Union[np.ndarray, list] = None,
        rd: Union[np.ndarray, list] = None,
        w: Union[np.ndarray, list] = None,
    ):
        """
        Constructor of the class from the components of the natural coordinates
        """

        if u is None:
            raise ValueError("u must be a numpy array (3x1) or a list of 3 elements")
        if rp is None:
            raise ValueError("rp must be a numpy array (3x1) or a list of 3 elements")
        if rd is None:
            raise ValueError("rd must be a numpy array (3x1) or a list of 3 elements")
        if w is None:
            raise ValueError("w must be a numpy array (3x1) or a list of 3 elements")

        if not isinstance(u, np.ndarray):
            u = np.array(u)
        if not isinstance(rp, np.ndarray):
            rp = np.array(rp)
        if not isinstance(rd, np.ndarray):
            rd = np.array(rd)
        if not isinstance(w, np.ndarray):
            w = np.array(w)

        if u.shape[0] != 3:
            raise ValueError("u must be a 3x1 numpy array")
        if rp.shape[0] != 3:
            raise ValueError("rp must be a 3x1 numpy array")
        if rd.shape[0] != 3:
            raise ValueError("rd must be a 3x1 numpy array")
        if w.shape[0] != 3:
            raise ValueError("v must be a 3x1 numpy array")

        input_array = np.concatenate((u, rp, rd, w), axis=0)

        if input_array.shape.__len__() == 1:
            input_array = input_array[:, np.newaxis]

        return cls(input_array)

    def to_array(self):
        return np.array(self).squeeze()

    @property
    def u(self):
        return self[0:3, :].to_array()

    @property
    def rp(self):
        return self[3:6, :].to_array()

    @property
    def rd(self):
        return self[6:9, :].to_array()

    @property
    def w(self):
        return self[9:12, :].to_array()

    @property
    def v(self):
        return self.rp - self.rd

    @property
    def vector(self):
        return self.to_array()

    def to_components(self):
        return self.u, self.rp, self.rd, self.w

    def to_uvw(self):
        return self.u, self.v, self.w

    def to_natural_vector(self, vector: np.ndarray) -> NaturalVector | np.ndarray:
        """
        This function converts a vector expressed in the global coordinate system
        to a vector expressed in a non-orthogonal coordinate system associated to the segment coordinates.

        Parameters
        ----------
        vector: np.ndarray
            The vector expressed in the global coordinate system (3x1) or (3xN)

        Returns
        -------
        np.ndarray
            The vector expressed in the non-orthogonal coordinate system (rp, u, v, w)

        """
        if self.shape[1] > 1:
            return self.vnop_array(vector - self.rp, self.u, self.v, self.w)  # not satisfied yet of this
        else:
            return NaturalVector(self.vnop_array(vector - self.rp, self.u, self.v, self.w))

    @staticmethod
    def vnop_array(V: np.ndarray, e1: np.ndarray, e2: np.ndarray, e3: np.ndarray) -> np.ndarray:
        """
        This function converts a vector expressed in the global coordinate system
        to a vector expressed in a non-orthogonal coordinate system.

        Parameters
        ----------
        V: np.ndarray
            The vector expressed in the global coordinate system
        e1: np.ndarray
            The first vector of the non-orthogonal coordinate system, usually the u-axis
        e2: np.ndarray
            The second vector of the non-orthogonal coordinate system, usually the v-axis
        e3: np.ndarray
            The third vector of the non-orthogonal coordinate system, usually the w-axis

        Returns
        -------
        vnop: np.ndarray
            The vector expressed in the non-orthogonal coordinate system

        """

        if V.shape[0] != 3:
            raise ValueError("The vector must be expressed in 3D.")
        if len(V.shape) == 1:
            V = V[:, np.newaxis]

        if e1.shape[0] != 3:
            raise ValueError("The first vector of the non-orthogonal coordinate system must be expressed in 3D.")
        if len(e1.shape) == 1:
            e1 = e1[:, np.newaxis]
        if e2.shape[0] != 3:
            raise ValueError("The second vector of the non-orthogonal coordinate system must be expressed in 3D.")
        if len(e2.shape) == 1:
            e2 = e2[:, np.newaxis]
        if e3.shape[0] != 3:
            raise ValueError("The third vector of the non-orthogonal coordinate system must be expressed in 3D.")
        if len(e3.shape) == 1:
            e3 = e3[:, np.newaxis]

        vnop = np.zeros(V.shape)

        vnop[0, :] = np.sum(np.cross(e2, e3, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
        vnop[1, :] = np.sum(np.cross(e3, e1, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
        vnop[2, :] = np.sum(np.cross(e1, e2, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)

        return vnop

    def axis(self, axis: Union[str, NaturalAxis]) -> np.ndarray:
        """
        This function returns the axis of the segment.

        Parameters
        ----------
        axis: str
            The axis to return (u, v, w)

        Returns
        -------
        np.ndarray
            The axis of the segment

        """
        if axis == "u" or axis == NaturalAxis.U:
            return self.u
        elif axis == "v" or axis == NaturalAxis.V:
            return self.v
        elif axis == "w" or axis == NaturalAxis.W:
            return self.w
        else:
            raise ValueError("The axis must be u, v or w")


class NaturalCoordinates(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        """
        Create a new instance of the class.
        """
        if input_array.shape.__len__() == 1:
            input_array = input_array[:, np.newaxis]
        return np.asarray(input_array).view(cls)

    @classmethod
    def from_qi(cls, tuple_of_Q: tuple):
        """
        Constructor of the class.
        """
        if not isinstance(tuple_of_Q, tuple):
            raise ValueError("tuple_of_Q must be a tuple of SegmentNaturalCoordinates")

        for Q in tuple_of_Q:
            if not isinstance(Q, SegmentNaturalCoordinates):
                raise ValueError("tuple_of_Q must be a tuple of SegmentNaturalCoordinates")

        input_array = np.concatenate(tuple_of_Q, axis=0)
        return cls(input_array)

    def to_array(self):
        return np.array(self).squeeze()

    def nb_qi(self):
        return self.shape[0] // 12

    def u(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx, :].to_array()

    def rp(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx, :].to_array()

    def rd(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx, :].to_array()

    def w(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx, :].to_array()

    def v(self, segment_idx: int):
        return self.rp(segment_idx) - self.rd(segment_idx)

    def vector(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)
        return SegmentNaturalCoordinates(self[array_idx, :].to_array())


class ExternalForce:
    def __init__(self, application_point_in_local: np.ndarray, external_forces: np.ndarray):
        self.application_point_in_local = application_point_in_local
        self.external_forces = external_forces

    @classmethod
    def from_components(cls, application_point_in_local: np.ndarray, force: np.ndarray, torque: np.ndarray):
        """
        This function creates an external force from its components.

        Parameters
        ----------
        application_point_in_local : np.ndarray
            The application point of the force in the natural coordinate system of the segment
        force
            The force vector in the global coordinate system
        torque
            The torque vector in the global coordinate system

        Returns
        -------
        ExternalForce

        """

        return cls(application_point_in_local, np.concatenate((torque, force)))

    @property
    def force(self) -> np.ndarray:
        return self.external_forces[3:6]

    @property
    def torque(self) -> np.ndarray:
        return self.external_forces[0:3]

    @staticmethod
    def compute_pseudo_interpolation_matrix(Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        Return the force moment transformation matrix

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The natural coordinates of the segment

        Returns
        -------
        np.ndarray
            The force moment transformation matrix
        """
        # default we apply force at the proximal point

        left_interpolation_matrix = np.zeros((12, 3))

        left_interpolation_matrix[9:12, 0] = Qi.u
        left_interpolation_matrix[0:3, 1] = Qi.v
        left_interpolation_matrix[3:6, 2] = -Qi.w
        left_interpolation_matrix[6:9, 2] = Qi.w

        # Matrix of lever arms for forces equivalent to moment at proximal endpoint, denoted Bstar
        lever_arm_force_matrix = np.zeros((3, 3))

        lever_arm_force_matrix[:, 0] = np.cross(Qi.w, Qi.u)
        lever_arm_force_matrix[:, 1] = np.cross(Qi.u, Qi.v)
        lever_arm_force_matrix[:, 2] = np.cross(-Qi.v, Qi.w)

        return (left_interpolation_matrix @ np.linalg.inv(lever_arm_force_matrix)).T

    def to_natural_force(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        Apply external forces to the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Segment natural coordinates

        Returns
        -------
        np.ndarray
            The external forces adequately transformed for the equation of motion in natural coordinates

        """

        pseudo_interpolation_matrix = self.compute_pseudo_interpolation_matrix(Qi)
        point_interpolation_matrix = NaturalVector(self.application_point_in_local).interpolate()
        application_point_in_global = np.array(point_interpolation_matrix @ Qi).squeeze()

        fext = point_interpolation_matrix.T @ self.force
        fext += pseudo_interpolation_matrix.T @ self.torque

        # Bour's formula to transport the moment from the application point to the proximal point
        # fext += pseudo_interpolation_matrix.T @ np.cross(application_point_in_global - Qi.rp, self.force)

        return np.array(fext)


class ExternalForceList:
    """
    This class is made to handle all the external forces of each segment, if none are provided, it will be an empty list.
    All segment forces are expressed in natural coordinates to be added to the equation of motion as:

    Q @ Qddot + K^T @ lambda = Weight + f_ext

    Attributes
    ----------
    external_forces : list
        List of ExternalForces

    Examples
    --------
    >>> from bionc import ExternalForceList, ExternalForce
    >>> import numpy as np
    >>> f_ext = ExternalForceList.empty_from_nb_segment(2)
    >>> segment_force = ExternalForce(force=np.array([0,1,1.1]), torque=np.zeros(3), application_point_in_local=np.array([0,0.5,0]))
    >>> f_ext.add_external_force(segment_index=0, external_force=segment_force)
    """

    def __init__(self, external_forces: list[list[ExternalForce, ...]] = None):
        if external_forces is None:
            raise ValueError(
                "f_ext must be a list of ExternalForces, or use the classmethod"
                "NaturalExternalForceList.empty_from_nb_segment(nb_segment)"
            )
        self.external_forces = external_forces

    @property
    def nb_segments(self) -> int:
        return len(self.external_forces)

    @classmethod
    def empty_from_nb_segment(cls, nb_segment: int):
        """
        Create an empty NaturalExternalForceList from the model size
        """
        return cls(external_forces=[[] for _ in range(nb_segment)])

    def segment_external_forces(self, segment_index: int) -> list[ExternalForce]:
        return self.external_forces[segment_index]

    def add_external_force(self, segment_index: int, external_force: ExternalForce):
        self.external_forces[segment_index].append(external_force)

    def to_natural_external_forces(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        Converts and sums all the segment natural external forces to the full vector of natural external forces

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the model
        """

        if len(self.external_forces) != Q.nb_qi():
            raise ValueError(
                "The number of segment in the model and the number of segment in the external forces must be the same"
            )

        natural_external_forces = np.zeros((12 * Q.nb_qi(), 1))
        for segment_index, segment_external_forces in enumerate(self.external_forces):
            segment_natural_external_forces = np.zeros((12, 1))
            slice_index = slice(segment_index * 12, (segment_index + 1) * 12)
            for external_force in segment_external_forces:
                segment_natural_external_forces += external_force.to_natural_force(Q.vector(segment_index))[
                    :, np.newaxis
                ]
            natural_external_forces[slice_index, 0:1] = segment_natural_external_forces

        return natural_external_forces

    def __iter__(self):
        return iter(self.external_forces)

    def __len__(self):
        return len(self.external_forces)
