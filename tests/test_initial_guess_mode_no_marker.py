import numpy as np
import pytest
from pyomeca import Markers

from bionc import InverseKinematics
from bionc.bionc_numpy.enums import InitialGuessModeType
from .utils import TestUtils

CAMERA_PARAMETERS = np.array(
    [
        [
            [-1.55093066e03, 5.49446167e02, -9.23099670e02, 8.82368835e02],
            [8.28569702e02, 1.66826160e03, -1.46722717e03, -1.23651172e03],
            [-1.01161758e02, 1.60179367e02, -2.93746948e02, -8.89093933e02],
            [7.66470215e02, 2.08185962e03, 8.99022644e02, 3.22349414e03],
        ],
        [
            [-2.51400177e02, 5.41528053e01, 1.68163223e02, -3.47867432e01],
            [-1.18579895e02, 4.64855896e02, -5.43082314e01, 8.14255676e02],
            [-1.91493835e03, -1.87241968e03, -1.92569006e03, -1.75258215e03],
            [4.01009595e03, 4.34932373e03, 4.69903662e03, 4.33119482e03],
        ],
        [
            [-6.56528711e-01, -7.11826026e-01, 6.23610318e-01, 8.27846706e-01],
            [-6.57654285e-01, 5.90441763e-01, -6.57450020e-01, 4.20727432e-01],
            [-3.69406223e-01, -3.80371124e-01, -4.22929883e-01, -3.71023208e-01],
            [2.99309301e00, 2.89029670e00, 4.35403728e00, 4.40847349e00],
        ],
    ]
)

GAUSSIAN_PARAMETERS = np.array(
    [
        [
            [
                [1.06714142e03, 1.06446062e03, 1.20639163e03, 1.25335555e03],
                [1.06076732e03, 1.05946157e03, 1.20112325e03, 1.24477715e03],
            ],
            [
                [1.07519687e03, 1.06031935e03, 1.21894286e03, 1.27235586e03],
                [1.06574716e03, 1.05415927e03, 1.21341905e03, 1.26530705e03],
            ],
            [
                [9.74476726e02, 9.86718935e02, 1.03856922e03, 1.04480510e03],
                [9.84731800e02, 9.89803578e02, 1.03535150e03, 1.04518919e03],
            ],
        ],
        [
            [
                [7.04349766e02, 3.37739531e02, 6.37834793e02, 6.38974707e02],
                [7.06629188e02, 3.41365755e02, 6.35883343e02, 6.35507461e02],
            ],
            [
                [7.21604610e02, 3.22195122e02, 6.52869764e02, 6.34039306e02],
                [7.22993457e02, 3.26963628e02, 6.50554477e02, 6.31694064e02],
            ],
            [
                [6.49882249e02, 4.53243551e02, 5.25500815e02, 5.71080676e02],
                [6.49607060e02, 4.56502104e02, 5.20017111e02, 5.72161447e02],
            ],
        ],
        [
            [
                [2.50363493e01, 2.24169721e01, 2.30618277e01, 2.35408778e01],
                [2.67769673e01, 2.22556750e01, 2.27478501e01, 2.34699293e01],
            ],
            [
                [2.47461378e01, 2.28844408e01, 2.29531535e01, 2.27401936e01],
                [2.64136446e01, 2.29784083e01, 2.24985898e01, 2.31349544e01],
            ],
            [
                [2.33452766e01, 2.20464299e01, 2.29610708e01, 2.37294105e01],
                [2.28521893e01, 2.25124043e01, 2.27653934e01, 2.33560431e01],
            ],
        ],
        [
            [
                [2.23441964e01, 2.23805730e01, 2.19398731e01, 2.19198796e01],
                [2.22907108e01, 2.23286874e01, 2.18499559e01, 2.20587541e01],
            ],
            [
                [2.26424494e01, 2.33769487e01, 2.37124480e01, 2.37290552e01],
                [2.18700256e01, 2.39457129e01, 2.39716782e01, 2.39544366e01],
            ],
            [
                [2.24923671e01, 2.24916537e01, 2.24847049e01, 2.23899861e01],
                [2.24457724e01, 2.22730900e01, 2.26367285e01, 2.26276481e01],
            ],
        ],
        [
            [
                [7.60322750e-01, 8.61716032e-01, 7.90513277e-01, 8.05084348e-01],
                [6.99700534e-01, 8.74770045e-01, 8.01184952e-01, 8.02848816e-01],
            ],
            [
                [7.49124348e-01, 8.47520888e-01, 8.13684702e-01, 8.05207789e-01],
                [6.32247686e-01, 8.62179697e-01, 8.37052286e-01, 8.07403505e-01],
            ],
            [
                [8.23556781e-01, 8.67787421e-01, 8.20179939e-01, 8.12999964e-01],
                [8.53100777e-01, 8.59496415e-01, 8.05822492e-01, 8.23351741e-01],
            ],
        ],
    ]
)

XP_HEATMAP_PARAMETERS = {
    "CAMERA_PARAMETERS": CAMERA_PARAMETERS,
    "GAUSSIAN_PARAMETERS": GAUSSIAN_PARAMETERS,
}


def test_exp_markers_none_from_current_markers():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    ik = InverseKinematics(
        model=module.model_creation_markerless(c3d_filename, False),
        experimental_heatmaps=XP_HEATMAP_PARAMETERS,
        solve_frame_per_frame=True,
    )
    with pytest.raises(
        ValueError,
        match=f"Please provide experimental_markers in order to initialize the optimization",
    ):
        ik.solve(initial_guess_mode=InitialGuessModeType.FROM_CURRENT_MARKERS, method="ipopt")


def test_exp_markers_none_from_first_frame_markers():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    ik = InverseKinematics(
        model=module.model_creation_markerless(c3d_filename, False),
        experimental_heatmaps=XP_HEATMAP_PARAMETERS,
        solve_frame_per_frame=True,
    )
    with pytest.raises(
        ValueError,
        match=f"Please provide experimental_markers in order to initialize the optimization",
    ):
        ik.solve(initial_guess_mode=InitialGuessModeType.FROM_FIRST_FRAME_MARKERS, method="ipopt")


def test_from_first_frame_markers_and_frame_per_frame_is_false():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data()
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()
    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
        solve_frame_per_frame=False,
    )
    with pytest.raises(
        ValueError,
        match=f"Please set frame_per_frame to True",
    ):
        ik.solve(initial_guess_mode=InitialGuessModeType.FROM_FIRST_FRAME_MARKERS, method="ipopt")


def test_user_provided_first_frame_only_and_frame_per_frame_is_false():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data()
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()
    Q_initialize = model.Q_from_markers(markers[:, :, 0:1])
    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
        solve_frame_per_frame=False,
    )
    with pytest.raises(
        ValueError,
        match=f"Either, set frame_per_frame == True or use InitialGuessModeType.USER_PROVIDED.",
    ):
        ik.solve(
            Q_init=Q_initialize, initial_guess_mode=InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY, method="ipopt"
        )


def test_user_provided_first_frame_only_Q_init_too_many_frames():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    c3d_filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data()
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()
    Q_initialize = model.Q_from_markers(markers[:, :, :])
    ik = InverseKinematics(
        model=model,
        experimental_markers=markers[0:3, :, :],
        solve_frame_per_frame=True,
    )

    with pytest.raises(
        ValueError,
        match=f"Please provide only the first frame of Q_init." f"Currently, Q_init.shape\\[1\\] = 2.",
    ):
        ik.solve(
            Q_init=Q_initialize, initial_guess_mode=InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY, method="ipopt"
        )
