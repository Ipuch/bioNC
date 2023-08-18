import os
import numpy as np
import pytest

from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_biomech_model(bionc_type):
    bionc = TestUtils.bionc_folder()
    module_c3d = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")
    module = TestUtils.load_module(bionc + "/examples/model_creation/two_side_lower_limbs.py")

    # Generate c3d file
    filename = module_c3d.generate_c3d_file(two_side=True)
    # Generate model
    natural_model = module.model_creation_from_measured_data(filename)

    # delete c3d file
    os.remove(filename)

    if bionc_type == "casadi":
        natural_model = natural_model.to_mx()

    # Test model
    assert natural_model.nb_segments == 7
    assert natural_model.nb_markers == 30
    assert natural_model.nb_markers_technical == 18
    assert natural_model.nb_joints == 6
    assert natural_model.nb_joint_constraints == 18
    assert natural_model.nb_joint_dof == 24
    assert natural_model.nb_rigid_body_constraints == 6 * 7
    assert natural_model.nb_holonomic_constraints == 60

    assert natural_model.nb_Q == 7 * 12
    assert natural_model.nb_Qdot == 7 * 12
    assert natural_model.nb_Qddot == 7 * 12
