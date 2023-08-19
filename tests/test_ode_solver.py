import numpy as np

from bionc import RK4


def test_rk4():
    # 1. Test with a simple function dy/dt = y
    def simple_function(t, y):
        return y + t

    y0 = np.array([1.0])
    t = np.linspace(0, 1, 10)
    expected = np.array([[1.        , 1.12392674, 1.27547487, 1.45789044, 1.67480095,
        1.9302602 , 2.22879841, 2.57547816, 2.97595701, 3.43655736]])
    y = RK4(t, simple_function, y0)
    assert np.allclose(y[0], expected, rtol=1e-5)

    # 2. Test with normalization
    y0 = np.array([1.0, 0.0])
    normalized_indices = ((1,), )
    expected = np.array([[1.        , 1.12392674, 1.27547487, 1.45789044, 1.67480095,
        1.9302602 , 2.22879841, 2.57547816, 2.97595701, 3.43655736],
       [0.        , 1.        , 1.        , 1.        , 1.        ,
        1.        , 1.        , 1.        , 1.        , 1.        ]])
    y = RK4(t, simple_function, y0, normalize_idx=normalized_indices)
    assert np.allclose(y, expected, rtol=1e-5)
