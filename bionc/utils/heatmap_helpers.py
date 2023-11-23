from casadi import dot, exp


def _projection(model_markers, camera_calibration_matrix, x, y):
    """
    Projects a point on the camera output in either x or y direction

    Parameters
    ----------
    model_markers : MX
        [3 x 1] symbolic expression. Represents the position of the 3D point in global reference frame, is also known as model keypoints in OpenPose for example
    camera_calibration_matrix : MX
        [3 x 4] symbolic expression. Represents the calibration matrix of the considered camera.
    x : boolean
        True to obtain the result of the projection in x direction
    y : boolean
        True to obtain the result of the projection in x direction
    """
    if x:
        marker_projected = (
            dot(model_markers, camera_calibration_matrix[1, 0:3].T) + camera_calibration_matrix[1, 3]
        ) / (dot(model_markers, camera_calibration_matrix[2, 0:3].T) + camera_calibration_matrix[2, 3])

    if y:
        marker_projected = (
            dot(model_markers, camera_calibration_matrix[0, 0:3].T) + camera_calibration_matrix[0, 3]
        ) / (dot(model_markers, camera_calibration_matrix[2, 0:3].T) + camera_calibration_matrix[2, 3])

    if x and y:
        raise ValueError("Project in only one direction please, either x or y")

    if not x and not y:
        raise ValueError("Project in one direction please, either x or y")
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
    marker_projected_on_x = _projection(model_markers, camera_calibration_matrix, x=True, y=False)
    marker_projected_on_y = _projection(model_markers, camera_calibration_matrix, x=False, y=True)

    x_xg_sigx = ((marker_projected_on_x - gaussian_center[0]) ** 2) / (2 * gaussian_standard_deviation[0] ** 2)
    y_yg_sigy = ((marker_projected_on_y - gaussian_center[1]) ** 2) / (2 * gaussian_standard_deviation[1] ** 2)

    confidence_value = gaussian_magnitude[0] * exp(-(x_xg_sigx + y_yg_sigy))
    return confidence_value
