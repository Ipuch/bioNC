from .biomechanical_model_joints import GenericBiomechanicalModelJoints
from .biomechanical_model_segments import GenericBiomechanicalModelSegments


class GenericBiomechanicalModelTree:
    """
    This is an abstract base class that provides the basic structure and methods for all biomechanical models.
    It contains the segments and the joints of the model. The implemented methods are not specific to numpy or casadi.

    Attributes
    ----------
    segments : dict
        A dictionary containing the segments of the model. The keys are the names of the segments and the values are the corresponding segment objects.
    joints : dict
        A dictionary containing the joints of the model. The keys are the names of the joints and the values are the corresponding joint objects.

    Methods
    -------
    children(self, segment: str | int) -> list[int]
        Returns the children of the given segment.
    parents(self, segment: str | int) -> list[int]
        Returns the parents of the given segment.
    segment_subtrees(self) -> list[list[int]]
        Returns the subtrees of the segments.
    segment_subtree(self, segment: str | int) -> list[int]
        Returns the subtree of the given segment.
    starting_depth_first_search(self) -> list[bool]
        Returns the segments in a depth first search order, starting from the ground segment.
    _depth_first_search(self, segment_index, visited_segments=None) -> list[bool]
        Returns the segments in a depth first search order.
    """

    def __init__(
        self,
        segments: GenericBiomechanicalModelSegments = None,
        joints: GenericBiomechanicalModelJoints = None,
    ):
        self.segments = segments
        self.joints = joints

    def children(self, segment_id: str | int) -> list[int]:
        """
        This function returns the children of the given segment

        Parameters
        ----------
        segment_id : int | str
            The segment index for which the children are returned, works also with the segment name

        Returns
        -------
        list[int]
            The children of the given segment
        """
        if isinstance(segment_id, str):
            segment_id = self.segments[segment_id].index

        segment = self.segments.segment_from_index(segment_id)
        joints_with_a_parent = [joint for joint in self.joints.values() if joint.parent is not None]
        joints_with_the_given_parent = [joint for joint in joints_with_a_parent if joint.parent.name == segment.name]

        return [joint.child.index for joint in joints_with_the_given_parent]

    def parents(self, segment_id: int | str) -> list[int]:
        """
        This function returns the parents of the given segment

        Parameters
        ----------
        segment_id : int | str
            The segment for which the parents are returned

        Returns
        -------
        list[int]
            The parents of the given segment
        """
        if isinstance(segment_id, str):
            segment_id = self.segments[segment_id].index

        segment = self.segments.segment_from_index(segment_id)

        return [joint.parent.index for joint in self.joints.values() if joint.child == segment]

    def segment_subtrees(self) -> list[list[int]]:
        """
        This function returns the subtrees of the segments

        Returns
        -------
        list[list[int]]
            The subtrees of the segments
        """
        subtrees = []
        for segment in self.segments.values():
            subtrees.append(self.segment_subtree(segment.index))
        return subtrees

    def segment_subtree(self, segment_id: int | str) -> list[int]:
        """
        This function returns the subtree of the given segment

        Parameters
        ----------
        segment_id : int | str
            The segment for which the subtree is returned, works also with the segment name

        Returns
        -------
        list[int]
            The subtree of the given segment
        """
        if isinstance(segment_id, str):
            segment_id = self.segments[segment_id].index

        segment = self.segments.segment_from_index(segment_id)

        subtree = [segment.index]
        for child_idx in self.children(segment.index):
            subtree += self.segment_subtree(child_idx)
        return subtree

    def starting_depth_first_search(self, segment_index=0) -> list[bool]:
        """
        This function returns the segments in a depth first search order, starting from the ground segment.

        todo: generalize to any number of segments with no parent.

        Returns
        -------
        list[bool, ...
            The segments in a depth first search order
        """
        visited_segments = [False for _ in range(self.segments.nb_segments)]

        return self._depth_first_search(segment_index, visited_segments)

    def _depth_first_search(self, segment_index, visited_segments=None) -> list[bool]:
        """
        This function returns the segments in a depth first search order.

        todo: generalize to any number of segments with no parent.

        Parameters
        ----------
        segment_index: int
            The index of the segment to start the search from
        visited_segments: list[Segment]
            The segments already visited

        Returns
        -------
        list[bool, ...
            The segments in a depth first search order
        """

        visited_segments[segment_index] = True
        for child_index in self.children(segment_index):
            if visited_segments[child_index]:
                raise RuntimeError("The model contain closed loops, we cannot use this algorithm")
            if not visited_segments[child_index]:
                visited_segments = self._depth_first_search(child_index, visited_segments)

        return visited_segments
