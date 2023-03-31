# import numpy as np
# from bionc import SegmentNaturalCoordinates, NaturalCoordinates
# from bionc.bionc_numpy import NaturalVector


# class ExternalForce:
#     def __init__(self,
#                  application_point_in_local: np.ndarray,
#                  external_forces: np.ndarray):
#         self.application_point_in_local = application_point_in_local
#         self.external_forces = external_forces
#
#     @classmethod
#     def from_components(cls, application_point_in_local: np.ndarray, force: np.ndarray, torque: np.ndarray):
#         return cls(application_point_in_local, np.concatenate((torque, force)))
#
#     @property
#     def force(self):
#         return self.external_forces[3:6]
#
#     @property
#     def torque(self):
#         return self.external_forces[0:3]
#
#     @staticmethod
#     def compute_pseudo_interpolation_matrix(Qi: SegmentNaturalCoordinates):
#         """
#         Return the force moment transformation matrix
#         """
#         # default we apply force at the proximal point
#
#         left_interpolation_matrix = np.zeros((12, 3))
#
#         left_interpolation_matrix[9:12, 0] = Qi.u
#         left_interpolation_matrix[0:3, 1] = Qi.v
#         left_interpolation_matrix[3:6, 2] = -Qi.w
#         left_interpolation_matrix[6:9, 2] = Qi.w
#
#         # Matrix of lever arms for forces equivalent to moment at proximal endpoint, denoted Bstar
#         lever_arm_force_matrix = np.zeros((3, 3))
#
#         lever_arm_force_matrix[:, 0] = np.cross(Qi.w, Qi.u)
#         lever_arm_force_matrix[:, 1] = np.cross(Qi.u, Qi.v)
#         lever_arm_force_matrix[:, 2] = np.cross(-Qi.v, Qi.w)
#
#         return (left_interpolation_matrix @ lever_arm_force_matrix).T
#
#     def to_natural_force(self, Qi: SegmentNaturalCoordinates):
#         """
#         Apply external forces to the segment
#
#         Parameters
#         ----------
#         Qi: SegmentNaturalCoordinates
#             Segment natural coordinates
#
#         Returns
#         -------
#
#         """
#
#         pseudo_interpolation_matrix = self.compute_pseudo_interpolation_matrix(Qi)
#
#         fext = NaturalVector.proximal().interpolate().T @ self.force
#         fext += pseudo_interpolation_matrix.T @ self.torque
#         fext += pseudo_interpolation_matrix.T @ np.cross(self.application_point_in_local - Qi.rp, self.force)
#
#         return np.array(fext)
#
#
# class ExternalForceList:
#     """
#     This class is made to handle all the external forces of each segment, if none are provided, it will be an empty list.
#     All segment forces are expressed in natural coordinates to be added to the equation of motion as:
#
#     Q @ Qddot + K^T @ lambda = Weight + f_ext
#
#     Attributes
#     ----------
#     external_forces : list
#         List of ExternalForces
#
#     Examples
#     --------
#     >>> from bionc import ExternalForceList, ExternalForce
#     >>> import numpy as np
#     >>> f_ext = ExternalForceList.empty_from_nb_segment(2)
#     >>> segment_force = ExternalForce(np.array([0,1,1.1]), np.zeros(12,1))
#     >>> f_ext.add_external_force(segment_index=0, external_force=segment_force)
#     """
#
#     def __init__(self, external_forces: list[list[ExternalForce, ...]] = None):
#
#         if external_forces is None:
#             raise ValueError("f_ext must be a list of ExternalForces, or use the classmethod"
#                              "NaturalExternalForceList.empty_from_nb_segment(nb_segment)")
#         self.external_forces = external_forces
#
#     @property
#     def nb_segments(self) -> int:
#         return len(self.external_forces)
#
#     @classmethod
#     def empty_from_nb_segment(cls, nb_segment: int):
#         """
#         Create an empty NaturalExternalForceList from the model size
#         """
#         return cls(external_forces=[[] for _ in range(nb_segment)])
#
#     def segment_external_forces(self, segment_index: int) -> list[ExternalForce]:
#         return self.external_forces[segment_index]
#
#     def add_external_force(self, segment_index: int, external_force: ExternalForce):
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
#         if len(self.external_forces) != Q.nb_qi:
#             raise ValueError(
#                 "The number of segment in the model and the number of segment in the external forces must be the same")
#
#         natural_external_forces = np.zeros((12 * Q.nb_qi, 1))
#         for segment_index, segment_external_forces in enumerate(self.external_forces):
#             segment_natural_external_forces = np.zeros((12, 1))
#             slice_index = slice(segment_index * 12, (segment_index + 1) * 12)
#             for external_force in segment_external_forces:
#                 segment_natural_external_forces += external_force.to_natural_force(Q.vector(segment_index))
#             natural_external_forces[slice_index] = segment_natural_external_forces
#
#         return natural_external_forces
#
#     def __iter__(self):
#         return iter(self.external_forces)
#
#     def __len__(self):
#         return len(self.external_forces)


#
#
# def cartesian_force_to_natural_force_from_interpolation_matrix(cartesian_force: np.ndarray, interpolation_matrix: InterpolationMatrix):
#     """
#     Convert the cartesian force to the natural force
#     """
#     if cartesian_force.shape[0] != 3:
#         raise ValueError("cartesian_force must be a 3x1 numpy array")
#     if cartesian_force.shape.__len__() == 1:
#         cartesian_force = cartesian_force[:, np.newaxis]
#
#     if isinstance(interpolation_matrix, np.ndarray):
#         interpolation_matrix = InterpolationMatrix(interpolation_matrix)
#
#     return np.array(interpolation_matrix @ cartesian_force)
#
# def cartesian_force_to_natural_force_from_segment(cartesian_force: np.ndarray, segment_index: int, Q: NaturalCoordinates):
#     """
#     Convert the cartesian force to the natural force
#     """
#     if cartesian_force.shape[0] != 3:
#         raise ValueError("cartesian_force must be a 3x1 numpy array")
#     if cartesian_force.shape.__len__() == 1:
#         cartesian_force = cartesian_force[:, np.newaxis]
#
#     Qi = Q.vector(segment_idx=segment_index)
#     # default we apply force at the proximal point
#     interpolation_matrix = NaturalVector.proximal().interpolate()
#
#     return np.array(segment.interpolation_matrix @ cartesian_force)
#
# def cartesian
#
#
# # There should be several format
# # one in cartesian coordinates
# # one in natural coordinates
# # capitaliser pour créer des efforts generalizee ?
#
# # class SegmentNaturalExternalForce(np.ndarray):
# #     """
# #     This class is made to handle External Forces of a Segment in natural coordinates format
# #     """
# #
# #     def __new__(cls, input_array: np.ndarray | list | tuple):
# #         """
# #         Create a new instance of the class.
# #         """
# #
# #         obj = np.asarray(input_array).view(cls)
# #
# #         if obj.shape[0] != 12:
# #             raise ValueError("input_array must be a 12x1 numpy array")
# #
# #         if obj.shape.__len__() == 1:
# #             obj = obj[:, np.newaxis]
# #
# #         return obj
# #
# #     @classmethod
# #     def from_cartesian_components(
# #         cls,
# #         force: np.ndarray | list = None,
# #         torque: np.ndarray | list = None,
# #     ):
# #         """
# #         Constructor of the class from the components of the natural coordinates
# #
# #         Parameters
# #         ----------
# #         force : np.ndarray | list
# #             Force vector in R0
# #         torque : np.ndarray | list
# #             Torque vector in R0
# #         # todo : application_point : NaturalVector
# #         #     Application point in R0
# #         # todo : segment : int
# #         #     Segment to which the external force is applied
# #         # todo : segment_application_point : NaturalVector
# #         #     Application point in the segment frame
# #
# #         Returns
# #         -------
# #         ExternalForces
# #         """
# #
# #         if force is None:
# #             raise ValueError("force must be a numpy array (3x1) or a list of 3 elements")
# #         if torque is None:
# #             raise ValueError("torque must be a numpy array (3x1) or a list of 3 elements")
# #
# #         if not isinstance(force, np.ndarray):
# #             force = np.array(force)
# #         if not isinstance(torque, np.ndarray):
# #             torque = np.array(torque)
# #
# #         if force.shape[0] != 3:
# #             raise ValueError("force must be a 3x1 numpy array")
# #         if torque.shape[0] != 3:
# #             raise ValueError("torque must be a 3x1 numpy array")
# #
# #         return cls(np.concatenate((force, torque), axis=0))
#
# class SegmentExternalForce:
#     """
#     This class is made to handle External Forces of a Segment in natural coordinates format
#     """
#
#     def __init__(self,
#                  cartesian_force: np.ndarray | list = None,
#                  cartesian_torque: np.ndarray | list = None,
#                  segment_index: int = None,
#                  application_point: NaturalVector = None,
#                     ):
#         self.cartesian_force = cartesian_force
#         self.cartesian_torque = cartesian_torque
#         self.segment_index = segment_index
#         self.application_point = application_point
#
#     def to_natural_force(self, Q: NaturalCoordinates):
#         """
#         Convert the cartesian force to the natural force
#         """
#         interpolation_matrix_proximal_point = NaturalVector.proximal().interpolate()
#         force = interpolation_matrix_proximal_point @ self.cartesian_force
#
#
#         return np.array(interpolation_matrix @ cartesian_force)
#
# def pseudo_interpolation_matrix(Qi: SegmentNaturalCoordinates):
#     """
#     Return the pseudo interpolation matrix
#     """
#     pseudo_interpolation_matrix = np.zeros((3,12))
#
#     pseudo_interpolation_matrix[0, 9:12] = Qi.u
#     pseudo_interpolation_matrix[1, 0:3] = Qi.v
#     pseudo_interpolation_matrix[2, 3:6] = -Qi.w
#     pseudo_interpolation_matrix[2, 6:3] = Qi.w
#
#     return pseudo_interpolation_matrix
#
# def force_moment_transformation_matrix(Qi: SegmentNaturalCoordinates):
#     """
#     Return the force moment transformation matrix
#     """
#     # default we apply force at the proximal point
#     force_moment_transformation_matrix = np.zeros((3,12))
#
#     force_moment_transformation_matrix[:, 0] = np.cross(Qi.w, Qi.u)
#     force_moment_transformation_matrix[:, 1] = np.cross(Qi.u, Qi.v)
#     force_moment_transformation_matrix[:, 2] = np.cross(-Qi.v, Qi.w)
#
#     return force_moment_transformation_matrix
#
# def apply_external_forces(Qi: SegmentNaturalCoordinates, external_forces: np.ndarray, application_point_in_local:np.ndarray):
#     """
#     Apply external forces to the segment
#
#     Parameters
#     ----------
#     Qi: SegmentNaturalCoordinates
#         Segment natural coordinates
#
#     external_forces: np.ndarray
#         External forces in cartesian coordinates
#
#     Returns
#     -------
#
#     """
#     torque = external_forces[3:6]
#     force = external_forces[0:3]
#
#     pseudo_interpolation_matrix = pseudo_interpolation_matrix(Qi)
#     force_moment_transformation_matrix = force_moment_transformation_matrix(Qi)
#
#     fext = pseudo_interpolation_matrix.T @ force
#     fext += force_moment_transformation_matrix.T @ torque
#     fext += force_moment_transformation_matrix.T @ np.cross(application_point_in_local - Qi.rp, force)
#
#     return fext
#
#
# class NaturalExternalForces(np.ndarray):
#     """
#     This class is made to handle External Forces for all segments in natural coordinates format
#     """
#
#     def __new__(cls, input_array: np.ndarray):
#         """
#         Create a new instance of the class.
#         """
#
#         obj = np.asarray(input_array).view(input_array)
#
#         if obj.shape.__len__() == 1:
#             obj = obj[:, np.newaxis]
#
#         if obj.shape[0] % 12 != 0:
#             raise ValueError("input_array must be a [12xN, 1] numpy array")
#
#         return obj
#
#     @classmethod
#     def from_segment_natural_forces(cls, segment_natural_forces: list | tuple):
#         """
#         Constructor of the class from the components of the natural coordinates
#
#         Parameters
#         ----------
#         segment_natural_forces : list
#             List of segment natural forces
#
#         """
#         if not isinstance(segment_natural_forces, tuple | list):
#             raise ValueError("tuple_of_Q must be a tuple of SegmentNaturalCoordinates")
#
#         for Q in segment_natural_forces:
#             if not isinstance(Q, SegmentNaturalExternalForce):
#                 raise ValueError("tuple_of_Q must be a tuple of SegmentNaturalCoordinates")
#
#         input_array = np.concatenate(segment_natural_forces, axis=0)
#         return cls(input_array)
#
#     def to_array(self):
#         return np.array(self).squeeze()
#
#     def vector(self, segment_idx: int) -> SegmentNaturalExternalForce:
#         """
#         Return the vector of the external forces
#         """
#         return SegmentNaturalExternalForce(self[segment_idx * 12 : (segment_idx + 1) * 12, :].to_array())
#
# # @classmethod
# # def from_components(
# #     cls,
# #     force: np.ndarray | list = None,
# #     torque: np.ndarray | list = None,
# #     application_point: NaturalVector = None,
# #     segment: int = None,
# #     segment_application_point: NaturalVector = None,
# # ):
# #     """
# #     Constructor of the class from the components of the natural coordinates
# #
# #     Parameters
# #     ----------
# #     force : np.ndarray | list
# #         Force vector in R0
# #     torque : np.ndarray | list
# #         Torque vector in R0
# #     application_point : NaturalVector
# #         Application point in R0
# #     segment : int
# #         Segment to which the external force is applied
# #     segment_application_point : NaturalVector
# #         Application point in the segment frame
# #
# #     Returns
# #     -------
# #     ExternalForces
# #     """
# #
# #     if force is None:
# #         raise ValueError("force must be a numpy array (3x1) or a list of 3 elements")
# #     if torque is None:
# #         raise ValueError("torque must be a numpy array (3x1) or a list of 3 elements")
# #
# #     if not isinstance(force, np.ndarray):
# #         force = np.array(force)
# #     if not isinstance(torque, np.ndarray):
# #         torque = np.array(torque)
# #
# #     if force.shape[0] != 3:
# #         raise ValueError("force must be a 3x1 numpy array")
# #     if torque.shape[0] != 3:
# #         raise ValueError("torque must be a 3x1 numpy array")
# #
# #     external_force =
# #
# #     return cls(np.concatenate((force, torque), axis=0))
#
#
#
#
# #
#
#
#
#
# cartesian_fext = CartesianExternalForces.from_world(
#     force=[0, 0, 0],
#     torque=[0, 0, 0],
#     application_point=CartesianVector.from_components([0, 0, 0]),
# )
# cartesian_fext.in_
# segment = 1,
# segment_application_point = CartesianVector.from_components([0, 0, 0]),
#
#
# segment_external_force = SegmentNaturalExternalForces.from_cartesian(
#     cartesian_fext,
#     segment=1,
#     segment_application_point: Marker| other| ...= ...,
#     Qi=...,
# )
#
# # doesn't' look comaptible ... :/
# external_forces = NaturalExternalForces()
# external_forces.add()
# external_forces.element(1) -> SegmentNaturalExternalForces
# external_forces.in_segement(2) -> tuple[SegmentNaturalExternalForces, ...]
# external_forces.to_equations()
