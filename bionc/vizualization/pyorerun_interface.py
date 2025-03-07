import numpy as np

from pyorerun.biorbd_components.model_display_options import DisplayModelOptions
from ..bionc_numpy import NaturalVector, NaturalCoordinates
from ..bionc_numpy.joints import Joint
from ..protocols.biomechanical_model import GenericBiomechanicalModel


class BioncModelNoMesh:
    """
    A class to handle a bioNC model and its transformations
    """

    def __init__(self, model: GenericBiomechanicalModel, options=None):
        self.model = model
        self.options: DisplayModelOptions = options if options is not None else DisplayModelOptions()

    @property
    def name(self):
        return "Natural Coordinate Model"

    @property
    def marker_names(self) -> tuple[str, ...]:
        return tuple(self.model.marker_names_technical)

    def markers(self, q: np.ndarray) -> np.ndarray:
        """
        Returns a [N_markers x 3] array containing the position of each marker in the global reference frame
        """
        # slicing only the technical markers
        name_technical = self.model._markers.names_technical
        name = self.model.marker_names

        idx_technical = [name.index(n) for n in name_technical]

        return self.model.markers(q).T[:, idx_technical, :]

    @property
    def nb_markers(self) -> int:
        return self.model.nb_markers_technical

    @property
    def segment_names(self) -> tuple[str, ...]:
        return self.model.segment_names

    @property
    def nb_segments(self) -> int:
        return self.model.nb_segments

    @property
    def segments(self) -> tuple["BioncSegment", ...]:
        return tuple(BioncSegment(s, i) for i, s in enumerate(self.model.segments.values()))

    @property
    def segment_names_with_mass(self) -> tuple[str, ...]:
        return tuple(
            [segment.name for segment in self.model.segments.values() if segment.mass is not None and segment.mass > 0]
        )

    def segment_homogeneous_matrices_in_global(self, q: np.ndarray, segment_index: int) -> np.ndarray:
        """
        Returns a biorbd object containing the roto-translation matrix of the segment in the global reference frame.
        This is useful if you want to interact with biorbd directly later on.
        """
        segment = self.model.segments.segment_from_index(segment_index)
        idx = segment.coordinates_slice
        qi = q[idx]
        transform = segment.segment_coordinates_system(qi)

        return transform

    def centers_of_mass(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the center of mass in the global reference frame
        """
        Q = NaturalCoordinates(q[:, None])
        return self.model.center_of_mass_position(Q).squeeze().T

    @property
    def nb_ligaments(self) -> int:
        """
        Returns the number of ligaments
        """
        count = 0
        for j, joint in enumerate(self.model.joints.values()):
            if isinstance(joint, Joint.ConstantLength):
                count += 1

        return count

    @property
    def ligament_names(self) -> tuple[str, ...]:
        """
        Returns the names of the ligaments
        """
        return tuple([joint.name for joint in self.model.joints.values() if isinstance(joint, Joint.ConstantLength)])

    def ligament_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        """
        strips = []
        for joint in self.model.joints.values():
            if isinstance(joint, Joint.ConstantLength):
                q_p = q[joint.parent.coordinates_slice]
                q_c = q[joint.child.coordinates_slice]
                origin = joint.parent_point.position_in_global(q_p)
                insert = joint.child_point.position_in_global(q_c)
                strips.append([origin, insert])

        return strips

    @property
    def nb_muscles(self) -> int:
        """
        Returns the number of ligaments
        """
        return 0

    @property
    def muscle_names(self) -> tuple[str, ...]:
        """
        Returns the names of the ligaments
        """
        return tuple()

    def muscle_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        """
        return list()

    @property
    def nb_q(self) -> int:
        return self.model.nb_Q

    @property
    def dof_names(self) -> tuple[str, ...]:
        return self.model.dof_names

    @property
    def q_ranges(self) -> tuple[tuple[float, float], ...]:
        pass

    @property
    def gravity(self) -> np.ndarray:
        pass

    @property
    def has_mesh(self) -> bool:
        return False

    @property
    def has_meshlines(self) -> bool:
        return True

    @property
    def has_soft_contacts(self) -> bool:
        return False

    def soft_contacts(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the soft contacts spheres in the global reference frame
        """
        pass

    @property
    def soft_contacts_names(self) -> tuple[str, ...]:
        """
        Returns the names of the soft contacts
        """
        pass

    @property
    def soft_contact_radii(self) -> tuple[float, ...]:
        """
        Returns the radii of the soft contacts
        """
        pass

    @property
    def meshlines(self) -> list[np.ndarray]:
        meshes = []
        for s in self.segments:
            p = s.segment.compute_transformation_matrix().T @ NaturalVector.proximal()
            d = s.segment.compute_transformation_matrix().T @ NaturalVector.distal()

            meshes += [np.array([p, d])]

        return meshes


class BioncSegment:
    """
    An interface to simplify the access to a segment of a biorbd model
    """

    def __init__(self, segment, index):
        self.segment = segment
        self._index: int = index

    @property
    def name(self) -> str:
        return self.segment.name

    @property
    def id(self) -> int:
        return self._index

    @property
    def has_mesh(self) -> bool:
        return False

    @property
    def has_meshlines(self) -> bool:
        return True

    @property
    def mesh_path(self) -> str:
        raise NotImplemented("This segment does not have a mesh")
