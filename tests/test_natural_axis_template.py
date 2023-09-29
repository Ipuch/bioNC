import numpy as np


def test_natural_axis_template_normalized_cross_product():
    from bionc.model_creation.natural_axis_template import AxisTemplate

    vec1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    vec2 = np.array([[5, 4], [7, 8], [2, 3], [9, 1]])

    vec1[0, :] = 1
    vec2[1, :] = 1

    normalized_cross_product = AxisTemplate.normalized_cross_product(None, None, vec1, vec2)
    expected_value = np.array(
        [
            [0.037113480951260276, 0.22645540682891915],
            [0.8536100618789862, 0.7925939239012171],
            [-0.5195887333176439, -0.5661385170722979],
            [0.0, 0.0],
        ]
    )

    np.testing.assert_array_almost_equal(expected_value, normalized_cross_product)
