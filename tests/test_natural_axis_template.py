import numpy as np


def test_natural_axis_template_normalized_cross_product():
    from bionc.model_creation.natural_axis_template import AxisTemplate

    vec1 = np.zeros((4, 2))
    vec2 = np.zeros((4, 2))
    cross_vec = np.zeros((4, 2))
    vec1[0, :] = 1
    vec2[1, :] = 1
    cross_vec[2, :] = 1
    normalized_cross_product = AxisTemplate.normalized_cross_product(None, None, vec1, vec2)

    assert (normalized_cross_product == cross_vec).all()
