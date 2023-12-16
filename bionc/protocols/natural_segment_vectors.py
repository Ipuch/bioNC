from abc import ABC

from .natural_markers import AbstractSegmentNaturalVector


class AbstractNaturalSegmentVectors(ABC):
    """
    Abstract class used to define anatomical segment based on natural vectors.

    This class provides an interface for adding vectors to a segment, retrieving the number,
    retrieving the names of the markers, and performing operations related to the markers' positions,
    constraints, and jacobian.

    Methods
    -------
    add()
        This function adds a marker to the segment
    nb_vectors
        This function returns the number of vectors of the segment
    vector_names
        This function returns the names of the vectors of the segment
    __iter__
        This function returns an iterator for the markers of the segment

    Attributes
    ----------
    _vectors : list[NaturalMarker]
        The list of vectors of the segment
    """

    def __init__(self):
        self._vectors = []

    def add(self, marker: AbstractSegmentNaturalVector):
        """
        Add a new marker to the segment

        Parameters
        ----------
        marker
            The marker to add
        """
        self._vectors.append(marker)

    @property
    def nb_vectors(self) -> int:
        """
        Get the number of markers in the segment.

        Returns
        -------
        int
            The number of markers in the segment.
        """
        return len(self._vectors)

    @property
    def vector_names(self) -> list[str]:
        """
        Get the names of the markers in the segment.

        Returns
        -------
        list[str]
            The names of the markers in the segment.
        """
        return [vector.name for vector in self._vectors]

    def __iter__(self):
        """
        Get an iterator for the markers in the segment.

        Returns
        -------
        iterator
            An iterator for the markers in the segment.
        """
        return iter(self._vectors)
