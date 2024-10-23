import numpy as np
import pytest
import re
from pyomeca import Markers

from bionc import InverseKinematics
from bionc.bionc_numpy.enums import InitialGuessModeType
from .utils import TestUtils


def test_user_provided():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data(c3d_filename)
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()
    Q_initialize = model.Q_from_markers(markers[:, :, :])
    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
    )
    Qsolve = ik.solve(Q_init=Q_initialize, initial_guess_mode=InitialGuessModeType.USER_PROVIDED, method="ipopt")

    TestUtils.assert_equal(
        ik.Qopt,
        np.array(
            [
                [-9.87874506e-01, -9.89595666e-01],
                [-6.99890796e-02, -6.62605396e-02],
                [-1.38583870e-01, -1.27710451e-01],
                [3.82041718e-01, 3.96302523e-01],
                [1.23305382e00, 1.23633424e00],
                [9.43367244e-01, 9.43138900e-01],
                [2.52735753e-01, 2.66241862e-01],
                [1.30342522e00, 1.30693351e00],
                [8.45182622e-01, 8.46121614e-01],
                [-3.24870340e-03, 1.05201696e-03],
                [9.99944013e-01, 9.99976166e-01],
                [1.00706401e-02, 6.82361027e-03],
                [-9.95064243e-01, -9.96258143e-01],
                [4.68750858e-02, 4.07229278e-02],
                [-8.74635846e-02, -7.62322421e-02],
                [2.52735753e-01, 2.66241862e-01],
                [1.30342522e00, 1.30693351e00],
                [8.45182622e-01, 8.46121614e-01],
                [2.89347351e-01, 2.98049419e-01],
                [1.34885069e00, 1.34955902e00],
                [4.53001468e-01, 4.53207772e-01],
                [7.99023863e-03, 7.50077931e-03],
                [9.16383796e-01, 9.19448214e-01],
                [4.00221057e-01, 3.93139568e-01],
                [-9.52186272e-01, -9.57501310e-01],
                [2.87573319e-01, 2.76197563e-01],
                [-1.03164393e-01, -8.31032478e-02],
                [2.89347351e-01, 2.98049419e-01],
                [1.34885069e00, 1.34955902e00],
                [4.53001468e-01, 4.53207772e-01],
                [3.41870189e-01, 3.42330099e-01],
                [1.39876990e00, 1.39869513e00],
                [1.07377449e-01, 1.06319604e-01],
                [2.96557339e-01, 2.83554846e-01],
                [9.51152942e-01, 9.54146984e-01],
                [-8.58010929e-02, -9.59175901e-02],
                [-9.50770300e-01, -9.53296033e-01],
                [1.92809971e-01, 1.92197512e-01],
                [-2.42611114e-01, -2.32995256e-01],
                [3.41870189e-01, 3.42330099e-01],
                [1.39876990e00, 1.39869513e00],
                [1.07377449e-01, 1.06319604e-01],
                [2.29918419e-01, 2.29616479e-01],
                [1.43078459e00, 1.43075439e00],
                [3.16439395e-02, 3.17438206e-02],
                [6.49591207e-02, 6.38231069e-02],
                [9.97815590e-01, 9.97864804e-01],
                [1.20150461e-02, 1.38724266e-02],
            ]
        ),
    )


def test_user_provided_first_frame_only():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data(c3d_filename)
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()
    Q_initialize = model.Q_from_markers(markers[:, :, :])
    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
    )
    ik.solve(
        Q_init=Q_initialize[:, 0:1],
        initial_guess_mode=InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY,
        method="ipopt",
    )

    TestUtils.assert_equal(
        ik.Qopt,
        np.array(
            [
                [-9.87874506e-01, -9.89595666e-01],
                [-6.99890796e-02, -6.62605396e-02],
                [-1.38583870e-01, -1.27710451e-01],
                [3.82041718e-01, 3.96302523e-01],
                [1.23305382e00, 1.23633424e00],
                [9.43367244e-01, 9.43138900e-01],
                [2.52735753e-01, 2.66241862e-01],
                [1.30342522e00, 1.30693351e00],
                [8.45182622e-01, 8.46121614e-01],
                [-3.24870340e-03, 1.05201696e-03],
                [9.99944013e-01, 9.99976166e-01],
                [1.00706401e-02, 6.82361018e-03],
                [-9.95064243e-01, -9.96258148e-01],
                [4.68750858e-02, 4.07229294e-02],
                [-8.74635846e-02, -7.62322424e-02],
                [2.52735753e-01, 2.66241862e-01],
                [1.30342522e00, 1.30693351e00],
                [8.45182622e-01, 8.46121614e-01],
                [2.89347351e-01, 2.98049419e-01],
                [1.34885069e00, 1.34955902e00],
                [4.53001468e-01, 4.53207772e-01],
                [7.99023863e-03, 7.50078089e-03],
                [9.16383796e-01, 9.19448218e-01],
                [4.00221057e-01, 3.93139569e-01],
                [-9.52186272e-01, -9.57501312e-01],
                [2.87573319e-01, 2.76197561e-01],
                [-1.03164393e-01, -8.31032485e-02],
                [2.89347351e-01, 2.98049419e-01],
                [1.34885069e00, 1.34955902e00],
                [4.53001468e-01, 4.53207772e-01],
                [3.41870189e-01, 3.42330099e-01],
                [1.39876990e00, 1.39869513e00],
                [1.07377449e-01, 1.06319604e-01],
                [2.96557339e-01, 2.83554843e-01],
                [9.51152942e-01, 9.54146987e-01],
                [-8.58010929e-02, -9.59175903e-02],
                [-9.50770300e-01, -9.53296033e-01],
                [1.92809971e-01, 1.92197512e-01],
                [-2.42611114e-01, -2.32995256e-01],
                [3.41870189e-01, 3.42330099e-01],
                [1.39876990e00, 1.39869513e00],
                [1.07377449e-01, 1.06319604e-01],
                [2.29918419e-01, 2.29616479e-01],
                [1.43078459e00, 1.43075439e00],
                [3.16439395e-02, 3.17438207e-02],
                [6.49591207e-02, 6.38231068e-02],
                [9.97815590e-01, 9.97864804e-01],
                [1.20150461e-02, 1.38724270e-02],
            ]
        ),
    )


def test_from_current_markers():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data(c3d_filename)
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()

    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
    )
    ik.solve(initial_guess_mode=InitialGuessModeType.FROM_CURRENT_MARKERS, method="ipopt")

    TestUtils.assert_equal(
        ik.Qopt,
        np.array(
            [
                [-9.87874506e-01, -9.89595666e-01],
                [-6.99890796e-02, -6.62605396e-02],
                [-1.38583870e-01, -1.27710451e-01],
                [3.82041718e-01, 3.96302523e-01],
                [1.23305382e00, 1.23633424e00],
                [9.43367244e-01, 9.43138900e-01],
                [2.52735753e-01, 2.66241862e-01],
                [1.30342522e00, 1.30693351e00],
                [8.45182622e-01, 8.46121614e-01],
                [-3.24870340e-03, 1.05201696e-03],
                [9.99944013e-01, 9.99976166e-01],
                [1.00706401e-02, 6.82361027e-03],
                [-9.95064243e-01, -9.96258143e-01],
                [4.68750858e-02, 4.07229278e-02],
                [-8.74635846e-02, -7.62322421e-02],
                [2.52735753e-01, 2.66241862e-01],
                [1.30342522e00, 1.30693351e00],
                [8.45182622e-01, 8.46121614e-01],
                [2.89347351e-01, 2.98049419e-01],
                [1.34885069e00, 1.34955902e00],
                [4.53001468e-01, 4.53207772e-01],
                [7.99023863e-03, 7.50077931e-03],
                [9.16383796e-01, 9.19448214e-01],
                [4.00221057e-01, 3.93139568e-01],
                [-9.52186272e-01, -9.57501310e-01],
                [2.87573319e-01, 2.76197563e-01],
                [-1.03164393e-01, -8.31032478e-02],
                [2.89347351e-01, 2.98049419e-01],
                [1.34885069e00, 1.34955902e00],
                [4.53001468e-01, 4.53207772e-01],
                [3.41870189e-01, 3.42330099e-01],
                [1.39876990e00, 1.39869513e00],
                [1.07377449e-01, 1.06319604e-01],
                [2.96557339e-01, 2.83554846e-01],
                [9.51152942e-01, 9.54146984e-01],
                [-8.58010929e-02, -9.59175901e-02],
                [-9.50770300e-01, -9.53296033e-01],
                [1.92809971e-01, 1.92197512e-01],
                [-2.42611114e-01, -2.32995256e-01],
                [3.41870189e-01, 3.42330099e-01],
                [1.39876990e00, 1.39869513e00],
                [1.07377449e-01, 1.06319604e-01],
                [2.29918419e-01, 2.29616479e-01],
                [1.43078459e00, 1.43075439e00],
                [3.16439395e-02, 3.17438206e-02],
                [6.49591207e-02, 6.38231069e-02],
                [9.97815590e-01, 9.97864804e-01],
                [1.20150461e-02, 1.38724266e-02],
            ]
        ),
    )


def test_from_first_frame_markers():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data(c3d_filename)
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()

    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
    )
    ik.solve(initial_guess_mode=InitialGuessModeType.FROM_FIRST_FRAME_MARKERS, method="ipopt")

    TestUtils.assert_equal(
        ik.Qopt,
        np.array(
            [
                [-9.87874506e-01, -9.89595666e-01],
                [-6.99890796e-02, -6.62605396e-02],
                [-1.38583870e-01, -1.27710451e-01],
                [3.82041718e-01, 3.96302523e-01],
                [1.23305382e00, 1.23633424e00],
                [9.43367244e-01, 9.43138900e-01],
                [2.52735753e-01, 2.66241862e-01],
                [1.30342522e00, 1.30693351e00],
                [8.45182622e-01, 8.46121614e-01],
                [-3.24870340e-03, 1.05201696e-03],
                [9.99944013e-01, 9.99976166e-01],
                [1.00706401e-02, 6.82361018e-03],
                [-9.95064243e-01, -9.96258148e-01],
                [4.68750858e-02, 4.07229294e-02],
                [-8.74635846e-02, -7.62322424e-02],
                [2.52735753e-01, 2.66241862e-01],
                [1.30342522e00, 1.30693351e00],
                [8.45182622e-01, 8.46121614e-01],
                [2.89347351e-01, 2.98049419e-01],
                [1.34885069e00, 1.34955902e00],
                [4.53001468e-01, 4.53207772e-01],
                [7.99023863e-03, 7.50078089e-03],
                [9.16383796e-01, 9.19448218e-01],
                [4.00221057e-01, 3.93139569e-01],
                [-9.52186272e-01, -9.57501312e-01],
                [2.87573319e-01, 2.76197561e-01],
                [-1.03164393e-01, -8.31032485e-02],
                [2.89347351e-01, 2.98049419e-01],
                [1.34885069e00, 1.34955902e00],
                [4.53001468e-01, 4.53207772e-01],
                [3.41870189e-01, 3.42330099e-01],
                [1.39876990e00, 1.39869513e00],
                [1.07377449e-01, 1.06319604e-01],
                [2.96557339e-01, 2.83554843e-01],
                [9.51152942e-01, 9.54146987e-01],
                [-8.58010929e-02, -9.59175903e-02],
                [-9.50770300e-01, -9.53296033e-01],
                [1.92809971e-01, 1.92197512e-01],
                [-2.42611114e-01, -2.32995256e-01],
                [3.41870189e-01, 3.42330099e-01],
                [1.39876990e00, 1.39869513e00],
                [1.07377449e-01, 1.06319604e-01],
                [2.29918419e-01, 2.29616479e-01],
                [1.43078459e00, 1.43075439e00],
                [3.16439395e-02, 3.17438207e-02],
                [6.49591207e-02, 6.38231068e-02],
                [9.97815590e-01, 9.97864804e-01],
                [1.20150461e-02, 1.38724270e-02],
            ]
        ),
    )


def test_Q_init_None_User_Provided():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data(c3d_filename)
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()

    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
    )

    with pytest.raises(
        ValueError,
        match="Please provide Q_init if you want to use InitialGuessModeType.USER_PROVIDED.",
    ):
        ik.solve(initial_guess_mode=InitialGuessModeType.USER_PROVIDED, method="ipopt")


def test_Q_init_None_User_Provided_First_Frame_Only():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data(c3d_filename)
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()

    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
    )

    with pytest.raises(
        ValueError,
        match="Please provide Q_init if you want to use InitialGuessModeType.USER_PROVIDED.",
    ):
        ik.solve(initial_guess_mode=InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY, method="ipopt")


def test_Q_init_Incorrect_Shape_User_Provided():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data(c3d_filename)
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()
    Q_initialize = model.Q_from_markers(markers[:, :, 0:1])
    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
    )
    with pytest.raises(
        ValueError,
        # match=f"Please make sure Q_init, shape\\[1\\] is equal to the number of frames 2." f"",
        match=re.escape("Q_init.shape\\[1\\] must equal the number of frames (2). Currently, Q_init.shape\\[1\\] = 1."),
    ):
        ik.solve(Q_init=Q_initialize, initial_guess_mode=InitialGuessModeType.USER_PROVIDED, method="ipopt")
