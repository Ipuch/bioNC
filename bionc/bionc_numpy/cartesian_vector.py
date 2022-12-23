import numpy as np
from ..utils.enums import CartesianAxis


class CartesianVector(np.ndarray):
    """
    Class used to create a natural vector, a vector that is expressed in the natural coordinate system of a segment
    """

    def __new__(cls, input_array: np.ndarray | list | tuple):
        """
        Create a new instance of the class.
        """

        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        size1 = input_array.shape[0]
        size2 = input_array.shape[1] if input_array.shape.__len__() == 2 else 1

        if size1 != 3:
            raise ValueError("The input array must have 3 elements")

        if size2 != 1:
            raise ValueError("The position must be a 3d vector with only one column")

        obj = np.asarray(input_array).view(cls)

        return obj

    @classmethod
    def axis(cls, axis: CartesianAxis):
        if axis == CartesianAxis.X:
            return cls(np.array([1, 0, 0]))
        elif axis == CartesianAxis.Y:
            return cls(np.array([0, 1, 0]))
        elif axis == CartesianAxis.Z:
            return cls(np.array([0, 0, 1]))
