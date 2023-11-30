from casadi import dot, exp


def _projection(model_markers, camera_calibration_matrix, axis):
    """
    Projects a point on the camera output in either x or y direction

    Parameters
    ----------
    model_markers : MX
        [3 x 1] symbolic expression. Represents the position of the 3D point in global reference frame, is also known as model keypoints in OpenPose for example
    camera_calibration_matrix : MX
        [3 x 4] symbolic expression. Represents the calibration matrix of the considered camera.
    axis : int
        Is equal to 0 or 1 according to if the projection is given in x axis (1) or y axis (0)
    """
    if axis > 1:
        raise ValueError("Please set axis to 0 or 1")
    numerator = dot(model_markers, camera_calibration_matrix[axis, 0:3].T) + camera_calibration_matrix[axis, 3]
    denominator = dot(model_markers, camera_calibration_matrix[2, 0:3].T) + camera_calibration_matrix[2, 3]
    marker_projected = numerator / denominator
    return marker_projected


def _compute_confidence_value_for_one_heatmap(
    model_markers, camera_calibration_matrix, gaussian_magnitude, gaussian_center, gaussian_standard_deviation
):
    """
    Computes the confidence value of one 3D point associated with one camera in the case of 2D heatmaps computations

    Parameters
    ----------
    model_markers : MX
        [3 x 1] symbolic expression. Represents the position of the 3D point in global reference frame, is also known as model keypoints in OpenPose for example
    camera_calibration_matrix : MX
        [3 x 4] symbolic expression. Represents the calibration matrix of the considered camera.
    gaussian_magnitude : MX
        [1 x 1] symbolic expression. Represents the amplitude of the gaussian considered.
    gaussian_center : MX
        [2 x 1] symbolic expression. Represents the position of the center of the gaussian considered along x and y directions.
    gaussian_standard_deviation : MX
        [2 x 1] symbolic expression. Represents the standard deviation of the gaussian considered along x and y directions.
    """
    marker_projected_on_x = _projection(model_markers, camera_calibration_matrix, axis=1)
    marker_projected_on_y = _projection(model_markers, camera_calibration_matrix, axis=0)

    x_exponent = gaussian_exponent(marker_projected_on_x, gaussian_center[0], gaussian_standard_deviation[0])
    y_exponent = gaussian_exponent(marker_projected_on_y, gaussian_center[1], gaussian_standard_deviation[1])

    confidence_value = gaussian_magnitude[0] * exp(-(x_exponent + y_exponent))
    return confidence_value


def gaussian_exponent(value, expected_value, standard_deviation):
    """
    Computes the exponent of a gaussian function

    Parameters
    ----------
    value : MX
        [1 x 1] symbolic expression. Represents the value of the variable of the gaussian function.
    expected_value : MX
        [1 x 1] symbolic expression. Represents the expected value of the variable of the gaussian function.
    standard_deviation : MX
        [1 x 1] symbolic expression. Represents the standard deviation of the variable of the gaussian function.
    """
    return ((value - expected_value) ** 2) / (2 * standard_deviation ** 2)
