from ..protocols.natural_segment_vectors import AbstractNaturalSegmentVectors


class NaturalSegmentVectors(AbstractNaturalSegmentVectors):
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
        super().__init__()