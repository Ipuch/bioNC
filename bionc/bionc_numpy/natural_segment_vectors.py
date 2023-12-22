from ..protocols import AbstractNaturalSegmentVectors


class NaturalSegmentVectors(AbstractNaturalSegmentVectors):
    """
    This class provides an interface for adding vectors to a segment, retrieving the number,
    retrieving the names of the markers, and performing operations related to the vectors' positions, or more
    """

    def __init__(self):
        super().__init__()
