from .utils import TestUtils


def test_play_with_joints_constant_length():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/play_with_joints/constant_length.py")

    module.main("hanged", False)
    module.main("ready_to_swing", False)


def test_play_with_joints_two_constant_length():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/play_with_joints/two_constant_length.py")

    module.main("hanged", False)
    module.main("ready_to_swing", False)


def test_inverse_kinematics_one_frame():
    # import the lower limb model
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")
    module_ik = TestUtils.load_module(bionc + "/examples/inverse_kinematics/one_frame_inverse_kinematics.py")

    # Generate c3d file
    filename = module.generate_c3d_file()
    # Generate model
    model = module.model_creation_from_measured_data(filename)

    module_ik.main(model, filename, show_animation=False)
