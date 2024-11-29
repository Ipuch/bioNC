from typing import Union

import numpy as np

from .external_force_global import ExternalForceInGlobal
from .external_force_global_local_point import ExternalForceGlobalLocalPoint
from .external_force_global_on_proximal import ExternalForceInGlobalOnProximal
from .external_force_in_local import ExternalForceInLocal
from .natural_coordinates import NaturalCoordinates

ExternalForce = Union[
    ExternalForceInGlobal, ExternalForceInGlobalOnProximal, ExternalForceGlobalLocalPoint, ExternalForceInLocal
]


class ExternalForceSet:
    """
    This class is made to handle all the external forces of each segment, if none are provided, it will be an empty list.
    All segment forces are expressed in natural coordinates to be added to the equation of motion as:

    Q @ Qddot + K^T @ lambda = Weight + f_ext

    Attributes
    ----------
    external_forces : list
        List of ExternalForces for each segment

    Methods
    -------
    add_external_force(segment_index, external_force)
        This function adds an external force to the list of external forces.
    empty_from_nb_segment(nb_segment)
        This function creates an empty ExternalForceSet from the number of segments.
    to_natural_external_forces(Q)
        This function returns the external forces in the natural coordinate format.
    segment_external_forces(segment_index)
        This function returns the external forces of a segment.
    nb_segments
        This function returns the number of segments.

    Examples to revised
    --------
    from bionc import ExternalForceSet, ExternalForce
    import numpy as np
    f_ext = ExternalForceSet.empty_from_nb_segment(2)
    segment_force = ExternalForce(force=np.array([0,1,1.1]), torque=np.zeros(3), application_point_in_local=np.array([0,0.5,0]))
    f_ext.add_external_force(segment_index=0, external_force=segment_force)
    """

    def __init__(self, external_forces: list[list[ExternalForce, ...]] = None):
        if external_forces is None:
            raise ValueError(
                "f_ext must be a list of ExternalForces, or use the classmethod"
                "NaturalExternalForceSet.empty_from_nb_segment(nb_segment)"
            )
        self.external_forces = external_forces

    @property
    def nb_segments(self) -> int:
        """Returns the number of segments"""
        return len(self.external_forces)

    @classmethod
    def empty_from_nb_segment(cls, nb_segment: int):
        """
        Create an empty NaturalExternalForceSet from the model size
        """
        return cls(external_forces=[[] for _ in range(nb_segment)])

    def segment_external_forces(self, segment_index: int) -> list[ExternalForce]:
        """Returns the external forces of the segment"""
        return self.external_forces[segment_index]

    def add_in_global(self, segment_index: int, external_force: np.ndarray, point_in_global: np.ndarray = None):
        """
        Add an external force to the segment

        Parameters
        ----------
        segment_index: int
            The index of the segment
        external_force:
            The external force to add [6 x 1], (torque, force)
        point_in_global: np.ndarray
            The point in global coordinates [3 x 1]
        """
        self.external_forces[segment_index].append(
            ExternalForceInGlobal(
                application_point_in_global=point_in_global if point_in_global is not None else np.zeros((3, 1)),
                external_forces=external_force,
            )
        )

    def add_in_global_local_point(
        self, segment_index: int, external_force: np.ndarray, point_in_local: np.ndarray = None
    ):
        """
        Add an external force to the segment

        Parameters
        ----------
        segment_index: int
            The index of the segment
        external_force:
            The external force to add [6 x 1], (torque, force)
        point_in_local: np.ndarray
            The point in global coordinates [3 x 1]
        """
        self.external_forces[segment_index].append(
            ExternalForceGlobalLocalPoint(
                application_point_in_local=point_in_local if point_in_local is not None else np.zeros((3, 1)),
                external_forces=external_force,
            )
        )

    def add_in_local(
        self,
        segment_index: int,
        external_force: np.ndarray,
        point_in_local: np.ndarray = None,
        transformation_matrix=None,
    ):
        """
        Add an external force to the segment

        Parameters
        ----------
        segment_index: int
            The index of the segment
        external_force:
            The external force in local cartesian frame to add [6 x 1], (torque, force)
        point_in_local: np.ndarray
            The point in global coordinates [3 x 1]
        """
        self.external_forces[segment_index].append(
            ExternalForceInLocal(
                application_point_in_local=point_in_local if point_in_local is not None else np.zeros((3, 1)),
                external_forces=external_force,
                transformation_matrix=transformation_matrix,
            )
        )

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
            segment_forces_on_proximal = []
            for external_force in segment_external_forces:
                segment_forces_on_proximal += external_force.transport_on_proximal(Q.vector(segment_index))

            segment_natural_external_forces = np.zeros((12, 1))
            for external_force in segment_forces_on_proximal:
                segment_natural_external_forces += external_force.to_generalized_natural_force(Q.vector(segment_index))[
                    :, np.newaxis
                ]

            slice_index = slice(segment_index * 12, (segment_index + 1) * 12)
            natural_external_forces[slice_index, 0:1] = segment_natural_external_forces

        return natural_external_forces

    def __iter__(self):
        return iter(self.external_forces)

    def __len__(self):
        return len(self.external_forces)


# class ExternalForceSet:
#     """
#     This class is made to handle all the external forces of each segment, if none are provided, it will be an empty list.
#     All segment forces are expressed in natural coordinates to be added to the equation of motion as:
#
#     Q @ Qddot + K^T @ lambda = Weight + f_ext
#
#     Attributes
#     ----------
#     external_forces : list
#         List of ExternalForces for each segment
#
#     Methods
#     -------
#     add_external_force(segment_index, external_force)
#         This function adds an external force to the list of external forces.
#     empty_from_nb_segment(nb_segment)
#         This function creates an empty ExternalForceSet from the number of segments.
#     to_natural_external_forces(Q)
#         This function returns the external forces in the natural coordinate format.
#     segment_external_forces(segment_index)
#         This function returns the external forces of a segment.
#     nb_segments
#         This function returns the number of segments.
#
#     Examples
#     --------
#     >>> from bionc import ExternalForceSet, ExternalForce
#     >>> import numpy as np
#     >>> f_ext = ExternalForceSet.empty_from_nb_segment(2)
#     >>> segment_force = ExternalForce(force=np.array([0,1,1.1]), torque=np.zeros(3), application_point_in_local=np.array([0,0.5,0]))
#     >>> f_ext.add_external_force(segment_index=0, external_force=segment_force)
#     """
#
#     def __init__(self, external_forces: list[list[ExternalForce, ...]] = None):
#         if external_forces is None:
#             raise ValueError(
#                 "f_ext must be a list of ExternalForces, or use the classmethod"
#                 "NaturalExternalForceSet.empty_from_nb_segment(nb_segment)"
#             )
#         self.external_forces = external_forces
#
#     @property
#     def nb_segments(self) -> int:
#         """Returns the number of segments"""
#         return len(self.external_forces)
#
#     @classmethod
#     def empty_from_nb_segment(cls, nb_segment: int):
#         """
#         Create an empty NaturalExternalForceSet from the model size
#         """
#         return cls(external_forces=[[] for _ in range(nb_segment)])
#
#     def segment_external_forces(self, segment_index: int) -> list[ExternalForce]:
#         """Returns the external forces of the segment"""
#         return self.external_forces[segment_index]
#
#     def add_external_force(self, segment_index: int, external_force: ExternalForce):
#         """
#         Add an external force to the segment
#
#         Parameters
#         ----------
#         segment_index: int
#             The index of the segment
#         external_force:
#             The external force to add
#         """
#         self.external_forces[segment_index].append(external_force)
#
#     def to_natural_external_forces(self, Q: NaturalCoordinates) -> np.ndarray:
#         """
#         Converts and sums all the segment natural external forces to the full vector of natural external forces
#
#         Parameters
#         ----------
#         Q : NaturalCoordinates
#             The natural coordinates of the model
#         """
#
#         if len(self.external_forces) != Q.nb_qi():
#             raise ValueError(
#                 "The number of segment in the model and the number of segment in the external forces must be the same"
#             )
#
#         natural_external_forces = np.zeros((12 * Q.nb_qi(), 1))
#         for segment_index, segment_external_forces in enumerate(self.external_forces):
#             segment_natural_external_forces = np.zeros((12, 1))
#             slice_index = slice(segment_index * 12, (segment_index + 1) * 12)
#             for external_force in segment_external_forces:
#                 segment_natural_external_forces += external_force.to_natural_force(Q.vector(segment_index))[
#                     :, np.newaxis
#                 ]
#             natural_external_forces[slice_index, 0:1] = segment_natural_external_forces
#
#         return natural_external_forces
#
#     def to_segment_natural_external_forces(self, Q: NaturalCoordinates, segment_index: int) -> np.ndarray:
#         """
#         Converts and sums all the segment natural external forces to the full vector of natural external forces
#         for one segment
#
#         Parameters
#         ----------
#         Q : NaturalCoordinates
#             The natural coordinates of the model
#         segment_index: int
#             The index of the segment
#
#         Returns
#         -------
#         segment_natural_external_forces: np.ndarray
#         """
#
#         if len(self.external_forces) != Q.nb_qi():
#             raise ValueError(
#                 "The number of segment in the model and the number of segment in the external forces must be the same"
#             )
#
#         if segment_index >= len(self.external_forces):
#             raise ValueError("The segment index is out of range")
#
#         segment_natural_external_forces = np.zeros((12, 1))
#         for external_force in self.external_forces[segment_index]:
#             segment_natural_external_forces += external_force.to_natural_force(Q.vector(segment_index))[:, np.newaxis]
#
#         return segment_natural_external_forces
#
#     def __iter__(self):
#         return iter(self.external_forces)
#
#     def __len__(self):
#         return len(self.external_forces)
