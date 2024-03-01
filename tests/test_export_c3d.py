import os

import ezc3d
import numpy as np
from pyomeca import Markers

from bionc.bionc_numpy.inverse_kinematics import InverseKinematics
from bionc.utils.c3d_export_utils import (
    get_points_ezc3d,
)
from .utils import TestUtils


def test_export_c3d():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")
    # Generate c3d file
    filename = module.generate_c3d_file()

    acq = ezc3d.c3d(filename)
    # delete c3d file
    os.remove(filename)
    points_data, points_name, points_ind = get_points_ezc3d(acq)
    # We test some value of the c3d file for both the marker and the natural coordinate
    TestUtils.assert_equal(
        points_data[:, points_ind["RKNE"], :],
        np.array([[0.28976313, 0.29850912], [1.40092926, 1.40187666], [0.47585389, 0.47547166]]),
    )

    TestUtils.assert_equal(
        points_data[:, points_ind["RKNI"], :],
        np.array([[0.2888405, 0.29766819], [1.29665239, 1.29736223], [0.4304646, 0.43063049]]),
    )

    TestUtils.assert_equal(
        points_data[:, points_ind["u_PELVIS"], :],
        np.array([[0.28336256, 0.29723339], [1.22604903, 1.22971387], [0.92960282, 0.93027608]]),
    )

    TestUtils.assert_equal(
        points_data[:, points_ind["rp_THIGH"], :],
        np.array([[0.25266419, 0.2663112], [1.30339712, 1.30696092], [0.84530961, 0.84599574]]),
    )
    TestUtils.assert_equal(
        points_data[:, points_ind["rd_SHANK"], :],
        np.array([[0.34153934, 0.34266457], [1.3989895, 1.39847636], [0.10725695, 0.10643581]]),
    )
    TestUtils.assert_equal(
        points_data[:, points_ind["w_FOOT"], :],
        np.array([[0.23617024, 0.23624429], [1.53057721, 1.53053023], [0.03290445, 0.03306805]]),
    )
    # We test that all the new marker are in the c3d file
    list_name_marker = [
        "RFWT_optim",
        "LFWT_optim",
        "RBWT_optim",
        "LBWT_optim",
        "RIGHT_HIP_JOINT_optim",
        "LEFT_HIP_JOINT_optim",
        "HIP_CENTER_optim",
        "RKNE_optim",
        "RKNI_optim",
        "KNEE_JOINT_optim",
        "RANE_optim",
        "RANI_optim",
        "ANKLE_JOINT_optim",
        "RHEE_optim",
        "RTARI_optim",
        "RTAR_optim",
        "u_PELVIS",
        "rp_PELVIS",
        "rd_PELVIS",
        "w_PELVIS",
        "u_THIGH",
        "rp_THIGH",
        "rd_THIGH",
        "w_THIGH",
        "u_SHANK",
        "rp_SHANK",
        "rd_SHANK",
        "w_SHANK",
        "u_FOOT",
        "rp_FOOT",
        "rd_FOOT",
        "w_FOOT",
    ]
    assert set(list_name_marker).issubset(points_name)


def test_ik_export():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")
    filename = module.generate_c3d_file()
    model = module.model_creation_from_measured_data(filename)
    markers = Markers.from_c3d(filename).to_numpy()[:3, :, :]
    MOCK_IK = InverseKinematics(model=model, experimental_markers=markers)

    MOCK_IK.Qopt = np.random.random((model.nb_Q, 2))
    MOCK_IK.export_in_c3d(filename)
