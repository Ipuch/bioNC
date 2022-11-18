from abc import ABC, abstractmethod
from typing import List, Union
from .natural_segment import AbstractNaturalSegment
from .biomechanical_model import AbstractBiomechanicalModel


class Segment(object):
    """
    An optimization variable and the indices to find this variable in its state or control vector

    Attributes
    ----------
    name: str
        The name of the variable
    natural_segment: AbstractNaturalSegment
        The MX variable associated with this variable
    index: range
        The indices to find the natural segment
    segment_list: SegmentList
        The parent that added this entry

    Methods
    -------
    __new__()
        The method which returns the natural segment
    """

    def __init__(self,
                 name: str,
                 index: [range, list],
                 natural_segment: AbstractNaturalSegment,
                 segment_list=None,
                 ):
        """
        Parameters
        ----------

        """
        self.name: str = name
        self.natural_segment: AbstractNaturalSegment = natural_segment
        self.index: [range, list] = index
        self.segment_list: SegmentList = segment_list

    def __new__(cls, name, index, natural_segment: AbstractNaturalSegment, segment_list=None):
        inst = super().__new__(cls)
        inst.natural_segment = natural_segment
        return inst.natural_segment


class SegmentList:
    """
        A list of OptimizationVariable

        Attributes
        ----------
        segments: list
            Each of the segment separated

        Methods
        -------
        __getitem__(self, item: Union[int, str])
            Get a specific segment in the list, whether by name or by index
        __setitem__(self, name, natural_segment: AbstractNaturalSegment)
            Add a new segment to the list
        __len__(self)
            The number of segment in the list
        __contains__(self, item)
            Allow for the use of the "in" keyword
        keys(self)
            Get the keys of the list
        __iter__(self)
            Allow for the list to be used in a for loop
        __next__(self)
            Get the next phase of the option list
        """

    def __init__(self):
        self.segments: list = []

    def __getitem__(self, item: Union[int, str, list, range]):
        """
        Get a specific segment in the list, whether by name or by index

        Parameters
        ----------
        item: Union[int, str]
            The index or name of the element to return

        Returns
        -------
        The specific segment in the list
        """
        if isinstance(item, int):
            return self.segments[item]
        elif isinstance(item, str):
            for segment in self.segments:
                if segment.name == item:
                    return segment
            raise ValueError(f"Segment {item} not found")
        elif isinstance(item, list):
            return [self.segments[i] for i in item]
        elif isinstance(item, range):
            return [self.segments[i] for i in item]
        else:
            raise ValueError(f"Invalid type {type(item)}")

    def __setitem__(self, name, natural_segment: AbstractNaturalSegment):
        """
        Add a new segment to the list

        Parameters
        ----------
        name: str
            The name of the segment
        natural_segment: AbstractNaturalSegment
            The natural segment to add
        """
        index = len(self.segments) + 1
        segment = Segment(name=name,
                          index=index,
                          natural_segment=natural_segment,
                          segment_list=self,
                          )

        self.segments.append(segment)

    def __len__(self):
        """
        The number of segment in the list

        Returns
        -------
        The number of segment in the list
        """
        return len(self.segments)

    def __contains__(self, item):
        """
        Allow for the use of the "in" keyword
        """
        return item in self.segments

    def keys(self):
        """
        Get the keys of the list

        Returns
        -------
        The keys of the list

        """
        return [segment.name for segment in self.segments]

    def __iter__(self):
        """
        Allow for the list to be used in a for loop

        Returns
        -------
        A reference to self
        """
        self._iter_idx = 0
        return self

    def __next__(self):
        """
        Get the next phase of the option list

        Returns
        -------
        The next phase of the option list
        """

        self._iter_idx += 1
        if self._iter_idx > len(self):
            raise StopIteration
        return self[self._iter_idx - 1].name
