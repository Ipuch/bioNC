import os

import ezc3d
import numpy as np
from pyomeca import Markers

from bionc.bionc_numpy.inverse_kinematics import InverseKinematics
from bionc.utils.c3d_export_utils import (
    get_points_ezc3d,
)
from .utils import TestUtils


def test_get_points_ezc3d():
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
        np.array([[0.289765, 0.298507], [1.401148, 1.401658], [0.475949, 0.475378]]),
    )

    # We test that all the new marker are in the c3d file
    list_name_marker = [
        "RFWT",
        "LFWT",
        "RBWT",
        "LBWT",
        "RKNE",
        "RKNI",
        "RANE",
        "RANI",
        "RHEE",
        "RTARI",
        "RTAR",
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
