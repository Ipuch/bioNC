import numpy as np
import pytest
from casadi import DM

from bionc import InverseKinematics, NaturalCoordinates
from bionc.algorithms.heatmap_helpers import _compute_confidence_value_for_one_heatmap, _projection
from .utils import TestUtils


def test_compute_confidence():
    right_ankle = DM([-1.3368181, -0.10489886, 0.12287701])
    right_knee = DM([-1.08386636, -0.0476904, 0.45526105])
    cam = DM(
        [
            [-1.55093066e03, 8.28569702e02, -1.01161758e02, 7.66470215e02],
            [-2.51400177e02, -1.18579895e02, -1.91493835e03, 4.01009595e03],
            [-6.56528711e-01, -6.57654285e-01, -3.69406223e-01, 2.99309301e00],
        ]
    )
    gaussian_knee = DM([9.74476726e02, 6.49882249e02, 2.33452766e01, 2.24923671e01, 8.23556781e-01])
    conf_knee = _compute_confidence_value_for_one_heatmap(
        right_knee, cam, [gaussian_knee[-1]], gaussian_knee[0:2], gaussian_knee[2:4]
    )
    print(conf_knee)
    gaussian_ankle = DM([1.06714142e03, 7.04349766e02, 2.50363493e01, 2.23441964e01, 7.60322750e-01])
    conf_ankle = _compute_confidence_value_for_one_heatmap(
        right_ankle, cam, [gaussian_ankle[-1]], gaussian_ankle[0:2], gaussian_ankle[2:4]
    )
    np.testing.assert_almost_equal(conf_knee, np.array(0.5478), decimal=1e-4)
    np.testing.assert_almost_equal(conf_ankle, np.array(0.7188), decimal=1e-4)


def test_compute_projection():
    right_knee = DM([-1.08386636, -0.0476904, 0.45526105])
    cam = DM(
        [
            [-1.55093066e03, 8.28569702e02, -1.01161758e02, 7.66470215e02],
            [-2.51400177e02, -1.18579895e02, -1.91493835e03, 4.01009595e03],
            [-6.56528711e-01, -6.57654285e-01, -3.69406223e-01, 2.99309301e00],
        ]
    )
    proj_knee_on_x = _projection(right_knee, cam, axis=1)
    proj_knee_on_y = _projection(right_knee, cam, axis=0)

    np.testing.assert_almost_equal(proj_knee_on_x, np.array(957.55), decimal=1e-2)
    np.testing.assert_almost_equal(proj_knee_on_y, np.array(661.99), decimal=1e-2)


camera_parameters = np.array(
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

gaussian_parameters = np.array(
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

experimental_heatmap_parameters = {
    "camera_parameters": camera_parameters,
    "gaussian_parameters": gaussian_parameters,
}

Q_initialize = NaturalCoordinates(
    np.array(
        [
            [0.85960984, 0.83158587],
            [-0.15412753, -0.16571733],
            [-0.48715052, -0.53009688],
            [-1.3368181, -1.33367074],
            [-0.10489886, -0.10492831],
            [0.12287701, 0.14064722],
            [-1.23256058, -1.23554468],
            [-0.13079953, -0.12957038],
            [0.02528557, 0.02747832],
            [-0.04983346, -0.01654953],
            [-0.97725584, -0.97985765],
            [0.20612533, 0.19901031],
            [0.7984484, 0.77446546],
            [-0.16296381, -0.13845085],
            [-0.5795886, -0.61728001],
            [-1.08386636, -1.07111251],
            [-0.0476904, -0.04517972],
            [0.45526105, 0.45666268],
            [-1.3368181, -1.33367074],
            [-0.10489886, -0.10492831],
            [0.12287701, 0.14064722],
            [-0.04983346, -0.01654953],
            [-0.97725584, -0.97985765],
            [0.20612533, 0.19901031],
        ]
    )
)


def test_global_heatmap_ik():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    ik = InverseKinematics(
        model=module.model_creation_markerless(c3d_filename, False),
        experimental_heatmaps=experimental_heatmap_parameters,
        solve_frame_per_frame=True,
        Q_init=Q_initialize,
    )

    ik.solve(method="ipopt")

    TestUtils.assert_equal(
        ik.Qopt,
        np.array(
            [
                [0.86003233, 0.827774],
                [-0.29151673, -0.29206467],
                [-0.41876293, -0.47904952],
                [-1.32020261, -1.31965406],
                [-0.08772465, -0.08494258],
                [0.13528834, 0.14898173],
                [-1.21514542, -1.2215382],
                [-0.13671284, -0.13414238],
                [0.04252033, 0.04900466],
                [-0.10292607, -0.0745547],
                [-0.92297668, -0.92178909],
                [0.37083725, 0.3804556],
                [0.75618075, 0.72711386],
                [-0.32122102, -0.31960291],
                [-0.57009449, -0.6075849],
                [-1.05595443, -1.0391412],
                [-0.03313965, -0.0262218],
                [0.45503453, 0.45379093],
                [-1.32020261, -1.31965406],
                [-0.08772465, -0.08494258],
                [0.13528834, 0.14898173],
                [-0.1711126, -0.1475683],
                [-0.93797416, -0.93709944],
                [0.30153765, 0.31633564],
            ]
        ),
    )


def test_error_solve_frame_per_frame():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(
        NotImplementedError,
        match=f"Not possible to solve for all frames with heatmap parameters. Please set solve_frame_per_frame=True",
    ):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=False,
        )


def test_error_Qinit_is_none():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(NotImplementedError, match=f"Not available yet, please provide Q_init"):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=None,
            solve_frame_per_frame=True,
        )


def test_error_markers_and_heatmaps():
    experimental_markers = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(ValueError, match=f"Please choose between marker data and heatmap data"):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_markers=experimental_markers,
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )


def test_error_no_markers_and_no_heatmaps():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(ValueError, match=f"Please feed experimental data, either marker or heatmap data"):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_markers=None,
            experimental_heatmaps=None,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )


def test_error_experimental_heatmaps_is_not_a_dictionnary():
    experimental_heatmap_parameters = np.array(
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

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(ValueError, match=f"Please feed experimental heatmaps as a dictionnary"):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )


def test_error_first_dim_cam_param_is_not_3():
    camera_parameters = np.array(
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
        ]
    )
    experimental_heatmap_parameters = {
        "camera_parameters": camera_parameters,
        "gaussian_parameters": gaussian_parameters,
    }

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(ValueError, match=f"First dimension of camera parameters must be 3"):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )


def test_error_len_cam_param_is_not_3():
    camera_parameters = np.array(
        [
            [-1.55093066e03, 5.49446167e02, -9.23099670e02, 8.82368835e02],
            [8.28569702e02, 1.66826160e03, -1.46722717e03, -1.23651172e03],
            [-1.01161758e02, 1.60179367e02, -2.93746948e02, -8.89093933e02],
            [7.66470215e02, 2.08185962e03, 8.99022644e02, 3.22349414e03],
        ]
    )

    experimental_heatmap_parameters = {
        "camera_parameters": camera_parameters,
        "gaussian_parameters": gaussian_parameters,
    }

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(
        ValueError,
        match=f'The number of dimensions of the NumPy array stored in experimental_heatmaps\["camera_parameters"\] must be 3 and the expected shape is 3 x 4 x nb_cameras',
    ):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )


def test_error_second_dim_cam_param_is_not_4():
    camera_parameters = np.array(
        [
            [
                [-1.55093066e03, 5.49446167e02, -9.23099670e02],
                [8.28569702e02, 1.66826160e03, -1.46722717e03],
                [7.66470215e02, 2.08185962e03, 8.99022644e02],
            ],
            [
                [-2.51400177e02, 5.41528053e01, 1.68163223e02],
                [-1.18579895e02, 4.64855896e02, -5.43082314e01],
                [4.01009595e03, 4.34932373e03, 4.69903662e03],
            ],
            [
                [-6.56528711e-01, -7.11826026e-01, 6.23610318e-01],
                [-6.57654285e-01, 5.90441763e-01, -6.57450020e-01],
                [2.99309301e00, 2.89029670e00, 4.35403728e00],
            ],
        ]
    )
    experimental_heatmap_parameters = {
        "camera_parameters": camera_parameters,
        "gaussian_parameters": gaussian_parameters,
    }

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(ValueError, match=f"Second dimension of camera parameters must be 4"):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )


def test_error_len_gaussian_param_is_not_4():
    gaussian_parameters = np.array(
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
    experimental_heatmap_parameters = {
        "camera_parameters": camera_parameters,
        "gaussian_parameters": gaussian_parameters,
    }

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(
        ValueError,
        match=f'The number of dimensions of the NumPy array stored in experimental_heatmaps\["gaussian_parameters"\] must be 4 and the expected shape is 5 x nb_markers x nb_frames x nb_cameras',
    ):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )


def test_error_first_dim_gaussian_param_is_not_5():
    gaussian_parameters = np.array(
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
        ]
    )

    experimental_heatmap_parameters = {
        "camera_parameters": camera_parameters,
        "gaussian_parameters": gaussian_parameters,
    }

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(ValueError, match=f"First dimension of gaussian parameters must be 5"):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )


def test_error_same_nb_cam_for_gaussian_and_cam_param():
    camera_parameters = np.array(
        [
            [
                [-1.55093066e03, 5.49446167e02, -9.23099670e02],
                [8.28569702e02, 1.66826160e03, -1.46722717e03],
                [-1.01161758e02, 1.60179367e02, -2.93746948e02],
                [7.66470215e02, 2.08185962e03, 8.99022644e02],
            ],
            [
                [-2.51400177e02, 5.41528053e01, 1.68163223e02],
                [-1.18579895e02, 4.64855896e02, -5.43082314e01],
                [-1.91493835e03, -1.87241968e03, -1.92569006e03],
                [4.01009595e03, 4.34932373e03, 4.69903662e03],
            ],
            [
                [-6.56528711e-01, -7.11826026e-01, 6.23610318e-01],
                [-6.57654285e-01, 5.90441763e-01, -6.57450020e-01],
                [-3.69406223e-01, -3.80371124e-01, -4.22929883e-01],
                [2.99309301e00, 2.89029670e00, 4.35403728e00],
            ],
        ]
    )

    gaussian_parameters = np.array(
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

    experimental_heatmap_parameters = {
        "camera_parameters": camera_parameters,
        "gaussian_parameters": gaussian_parameters,
    }

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/markerless_model.py")

    c3d_filename = module.generate_c3d_file()

    with pytest.raises(
        ValueError,
        match=f'Third dimension of experimental_heatmaps\["camera_parameters"\] and fourth dimension of experimental_heatmaps\["gaussian_parameters"\] should be equal. Currently we have '
        + str(experimental_heatmap_parameters["camera_parameters"].shape[2])
        + " and "
        + str(experimental_heatmap_parameters["gaussian_parameters"].shape[3]),
    ):
        InverseKinematics(
            model=module.model_creation_markerless(c3d_filename, False),
            experimental_heatmaps=experimental_heatmap_parameters,
            Q_init=Q_initialize,
            solve_frame_per_frame=True,
        )
