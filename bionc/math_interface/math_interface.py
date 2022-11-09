from casadi import MX, tril, tril2symm
from casadi.casadi import MX as MX_type
import casadi as cas
from numpy import ndarray
import numpy as np


def zeros(*args, instance_type: type) -> ndarray | MX:
    if instance_type == MX_type:
        return MX.zeros(*args)
    elif instance_type == ndarray:
        return np.zeros(args)


def eye(n, instance_type: type) -> ndarray | MX:
    if instance_type == MX_type:
        return MX.eye(n)
    elif instance_type == ndarray:
        return np.eye(n)


def array(data, instance_type: type) -> ndarray | MX:
    if instance_type == MX_type:
        return MX(data)
    elif instance_type == ndarray:
        return np.array(data)


def symmetrize_upp(A, instance_type: type) -> ndarray | MX:
    """
    This function symmetrizes a matrix by copying the upper triangular part 
    to the lower triangular part conserving the diagonal. 
    """
    if instance_type == ndarray:
        return np.tril(A) + np.tril(A, -1).T
    elif instance_type == MX_type:
        return tril2symm(tril(A))


def vertcat(*args, instance_type: type) -> ndarray | MX:
    if instance_type == ndarray:
        return np.vstack(args)
    elif instance_type == MX_type:
        return cas.vertcat(*args)


def horzcat(*args, instance_type: type) -> ndarray | MX:
    if instance_type == ndarray:
        return np.hstack(args)
    elif instance_type == MX_type:
        return cas.horzcat(*args)
