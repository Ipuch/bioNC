import numpy as np

from pyorerun.biorbd_components.model_display_options import DisplayModelOptions
from ..protocols.biomechanical_model import GenericBiomechanicalModel


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
        return False

    @property
    def mesh_path(self) -> str:
        raise NotImplemented("This segment does not have a mesh")


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
    def segments(self) -> tuple[BioncSegment, ...]:
        return tuple(BioncSegment(s, i) for i, s in enumerate(self.model.segments.values()))

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

    def center_of_mass(self, q: np.ndarray) -> np.ndarray:
        """
        Returns the position of the center of mass in the global reference frame
        """
        pass

    @property
    def nb_ligaments(self) -> int:
        """
        Returns the number of ligaments
        """
        return 0

    @property
    def ligament_names(self) -> tuple[str, ...]:
        """
        Returns the names of the ligaments
        """
        return tuple()

    def ligament_strips(self, q: np.ndarray) -> list[list[np.ndarray]]:
        """
        Returns the position of the ligaments in the global reference frame
        """

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
        return False

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
