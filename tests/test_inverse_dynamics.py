import numpy as np
import pytest
from casadi import MX

from bionc import (
    BiomechanicalModel,
    NaturalSegment,
    CartesianAxis,
    NaturalAxis,
    JointType,
    EulerSequence,
    TransformationMatrixType,
)
from .utils import TestUtils


def build_n_link_pendulum(nb_segments: int = 1) -> BiomechanicalModel:
    """Build a n-link pendulum model"""
    if nb_segments < 1:
        raise ValueError("The number of segment must be greater than 1")
    # Let's create a model
    model = BiomechanicalModel()
    # number of segments
    # fill the biomechanical model with the segment
    for i in range(nb_segments):
        name = f"pendulum_{i}"
        model[name] = NaturalSegment.with_cartesian_inertial_parameters(
            name=name,
            alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1,
            mass=1 * (i + 1),
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
            inertial_transformation_matrix=TransformationMatrixType.Buv,
        )
    # add a revolute joint (still experimental)
    # if you want to add a revolute joint,
    # you need to ensure that x is always orthogonal to u and v
    model._add_joint(
        dict(
            name="hinge_0",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum_0",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
            theta=[np.pi / 2, np.pi / 2],
            projection_basis=EulerSequence.XYZ,
        )
    )
    for i in range(1, nb_segments):
        model._add_joint(
            dict(
                name=f"hinge_{i}",
                joint_type=JointType.REVOLUTE,
                parent=f"pendulum_{0}",
                child=f"pendulum_{i}",
                parent_axis=[NaturalAxis.U, NaturalAxis.U],
                child_axis=[NaturalAxis.V, NaturalAxis.W],
                theta=[np.pi / 2, np.pi / 2],
                projection_basis=EulerSequence.XYZ,
            )
        )

    return model


def build_n_link_pendulum_spherical(nb_segments: int = 1) -> BiomechanicalModel:
    """Build a n-link pendulum model"""
    if nb_segments < 1:
        raise ValueError("The number of segment must be greater than 1")
    # Let's create a model
    model = BiomechanicalModel()
    # number of segments
    # fill the biomechanical model with the segment
    for i in range(nb_segments):
        name = f"pendulum_{i}"
        model[name] = NaturalSegment.with_cartesian_inertial_parameters(
            name=name,
            alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1,
            mass=1 * (i + 1),
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
            inertial_transformation_matrix=TransformationMatrixType.Buv,
        )
    # add a revolute joint (still experimental)
    # if you want to add a revolute joint,
    # you need to ensure that x is always orthogonal to u and v
    model._add_joint(
        dict(
            name="hinge_0",
            joint_type=JointType.GROUND_SPHERICAL,
            parent="GROUND",
            child="pendulum_0",
        )
    )
    for i in range(1, nb_segments):
        model._add_joint(
            dict(
                name=f"hinge_{i}",
                joint_type=JointType.SPHERICAL,
                parent=f"pendulum_{0}",
                child=f"pendulum_{i}",
            )
        )

    return model


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
@pytest.mark.parametrize(
    "configuration",
    [
        1,
        2,
        3,
    ],
)
def test_inverse_dynamics_projected(bionc_type, configuration):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
            NaturalCoordinates,
            NaturalAccelerations,
        )
    else:
        from bionc.bionc_numpy import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
            NaturalCoordinates,
            NaturalAccelerations,
        )

    nb_segments = 3

    model = build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()

    for seg in model.segments.values():
        print(seg.mass)

    if configuration == 1:
        # vertical
        tuple_of_Q = [
            SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1]),
            SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, -1, 0], rd=[0, -2, 0], w=[0, 0, 1]),
            SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, -2, 0], rd=[0, -3, 0], w=[0, 0, 1]),
        ]
    elif configuration == 2:
        # horizontal
        tuple_of_Q = [
            SegmentNaturalCoordinates.from_components(
                u=[1, 0, 0], rp=[0, 0, -i if i <= 1 else -1], rd=[0, 0, -i - 1 if i <= 1 else -2], w=[0, -1, 0]
            )
            for i in range(0, nb_segments)
        ]
    else:
        # each segment are turned by 90°
        tuple_of_Q = [
            SegmentNaturalCoordinates.from_components(
                u=[1, 0, 0],
                rp=[0, 0, 0],
                rd=[0, -1, 0],
                w=[0, 0, 1],
            ),
            SegmentNaturalCoordinates.from_components(
                u=[1, 0, 0],
                rp=[0, -1, 0],
                rd=[0, -1, 1],
                w=[0, 1, 0],
            ),
            SegmentNaturalCoordinates.from_components(
                u=[0, 1, 0],
                rp=[0, -1, 1],
                rd=[1, -1, 1],
                w=[0, 0, 1],
            ),
        ]

    Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))

    tuple_of_Qddot = [
        SegmentNaturalAccelerations.from_components(
            uddot=[0, 0, 0], rpddot=[0, 0, 0], rdddot=[0, 0, 0], wddot=[0, 0, 0]
        )
        for _ in range(0, nb_segments)
    ]

    # only to debug this test :)
    # from bionc import Viz
    #
    # v = Viz(model)
    # v.animate(Q)

    Qddot = NaturalAccelerations.from_qddoti(tuple(tuple_of_Qddot))

    torques, *_ = model.inverse_dynamics(Q, Qddot)

    if configuration == 1:
        expected_torques = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-39.24, 19.62, 29.43]])
    elif configuration == 2:
        expected_torques = np.array([[0.0, 0.0, 0.0], [-39.24, 19.62, 29.43], [0.0, 0.0, 0.0]])
    else:
        expected_torques = np.array([[0.0, 0.0, 0.0], [0.0, -19.62, 0.0], [-39.24, 0.0, 29.43]])

    projected_torques = model.express_joint_torques_in_euler_basis(Q, torques)

    TestUtils.assert_equal(
        projected_torques,
        expected_torques,
        expand=False,
    )


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_projection_of_torques(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            SegmentNaturalCoordinates,
            NaturalCoordinates,
        )
    else:
        from bionc.bionc_numpy import (
            SegmentNaturalCoordinates,
            NaturalCoordinates,
        )

    nb_segments = 3

    model = build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()
    tuple_of_Q = [
        SegmentNaturalCoordinates.from_components(
            u=[1.02, 0.03, 0.04],
            rp=[0.05, 0.06, 0.07],
            rd=[0, -1.002, 0.001],
            w=[0.001, 0.002, 1],
        ),
        SegmentNaturalCoordinates.from_components(
            u=[1.01, 0.02, 0.03],
            rp=[0.0007, -1.001, 0.002],
            rd=[0.0006, -1.003, 1.001],
            w=[0.0008, 1.004, 0.001],
        ),
        SegmentNaturalCoordinates.from_components(
            u=[0.0008, 1.002, 0.0007],
            rp=[0.0006, -1.005, 1.003],
            rd=[1.005, -1.004, 1.01],
            w=[0.0003, 0.0007, 1.009],
        ),
    ]
    Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))

    torques = np.array([[0.001, 0.04, 0.2], [0.5, 0.2, 0.1], [-39.0, 1, 2]])
    expected_torques = np.array(
        [
            [3.99988440e-02, 3.82187782e-02, 1.91215874e-01],
            [5.77998844e-01, -9.98470561e-01, 8.76828403e-02],
            [-3.89988440e01, 1.99052206e-01, 1.97464529e00],
        ]
    )
    projected_torques = model.express_joint_torques_in_euler_basis(Q, torques)
    TestUtils.assert_equal(
        projected_torques,
        expected_torques,
        expand=False,
    )


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_inverse_dynamics_segment(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
        )
    else:
        from bionc.bionc_numpy import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
        )

    nb_segments = 3
    model = build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()

    Qi = SegmentNaturalCoordinates.from_components(
        u=[1, 0.001, 0], rp=[0, 0.001, 0], rd=[0, 0, 0.001], w=[0, -1, 0.001]
    )
    Qddoti = SegmentNaturalAccelerations.from_components(
        uddot=[0.01, 0.02, 0.03], rpddot=[0.04, 0.05, 0.06], rdddot=[0.07, 0.08, 0.09], wddot=[0.010, 0.011, 0.012]
    )
    subtree_forces = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.011, 0.013])[:, np.newaxis]
    external_forces = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.011, 0.013])[
        :, np.newaxis
    ]

    if bionc_type == "casadi":
        subtree_forces = MX(subtree_forces)
        external_forces = MX(external_forces)

    gf = model.segments["pendulum_1"].inverse_dynamics(
        Qi, Qddoti, subtree_intersegmental_generalized_forces=subtree_forces, segment_external_forces=external_forces
    )

    TestUtils.assert_equal(gf[0], np.array([-0.11, -0.13, 19.47]), expand=False)
    TestUtils.assert_equal(gf[1], np.array([0.001314, 0.01794, -0.012805]), expand=False)
    TestUtils.assert_equal(
        gf[2],
        np.array([1.25000000e-03, -5.75000000e-02, -2.82757758e-02, -4.87384150e03, -1.10167643e01, -8.38213213e-03]),
        expand=False,
        decimal=5,
    )


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_inverse_dynamics(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
            NaturalCoordinates,
            NaturalAccelerations,
        )
    else:
        from bionc.bionc_numpy import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
            NaturalCoordinates,
            NaturalAccelerations,
        )

    nb_segments = 3

    model = build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()

    for seg in model.segments.values():
        print(seg.mass)

        # vertical
    tuple_of_Q = [
        SegmentNaturalCoordinates.from_components(
            u=[1, 0, 0], rp=[0, 0, -i if i <= 1 else -1], rd=[0, 0, -i - 1 if i <= 1 else -2], w=[0, -1, 0]
        )
        for i in range(0, nb_segments)
    ]

    Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))

    tuple_of_Qddot = [
        SegmentNaturalAccelerations.from_components(
            uddot=[0, 0, 0], rpddot=[0, 0, 0], rdddot=[0, 0, 0], wddot=[0, 0, 0]
        )
        for _ in range(0, nb_segments)
    ]
    Qddot = NaturalAccelerations.from_qddoti(tuple(tuple_of_Qddot))

    torques, forces, lambdas = model.inverse_dynamics(Q, Qddot)

    print(torques)
    print(forces)
    print(lambdas)

    TestUtils.assert_equal(torques, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-39.24, 19.62, 29.43]]), expand=False)

    TestUtils.assert_equal(
        forces,
        np.array(
            [
                [1.20137851e-15, -6.00689255e-16, -9.01033882e-16],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ]
        ),
        expand=False,
    )

    TestUtils.assert_equal(
        lambdas,
        np.array(
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [2.45250000e00, 4.90500000e00, 7.35750000e00],
                [-3.00344627e-16, -6.00689255e-16, -9.01033882e-16],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ]
        ),
        expand=False,
    )


def test_id_example():
    bionc = TestUtils.bionc_folder()
    module_id = TestUtils.load_module(bionc + "/examples/inverse_dynamics/three_link_pendulum.py")

    a = module_id.main("horizontal")
    assert isinstance(a, tuple)
    assert len(a) == 3

    b = module_id.main("vertical")
    assert isinstance(b, tuple)
    assert len(b) == 3

    torques = b[0]
    forces = b[1]
    lambdas = b[2]

    TestUtils.assert_equal(torques, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-29.43, 9.81, 19.62]]), expand=False)

    TestUtils.assert_equal(
        forces,
        np.array(
            [
                [9.01033882e-16, -3.00344627e-16, -6.00689255e-16],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ],
        ),
        expand=False,
    )

    TestUtils.assert_equal(
        lambdas,
        np.array(
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 2.45250000e00, 4.90500000e00],
                [0.00000000e00, -3.00344627e-16, -6.00689255e-16],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ]
        ),
        expand=False,
    )


def test_id_example_with_fext():
    bionc = TestUtils.bionc_folder()
    module_id = TestUtils.load_module(bionc + "/examples/inverse_dynamics/three_link_pendulum.py")

    b = module_id.main("vertical", with_fext=True)
    assert isinstance(b, tuple)
    assert len(b) == 3

    torques = b[0]
    forces = b[1]
    lambdas = b[2]

    TestUtils.assert_equal(
        torques,
        np.array(
            [
                [1.1000e-02, -1.0000e-02, -1.0000e-03],
                [2.2000e-02, -2.0000e-02, -2.0000e-03],
                [-2.9397e01, 9.7800e00, 1.9617e01],
            ],
        ),
        expand=False,
    )

    TestUtils.assert_equal(
        forces,
        np.array(
            [
                [0.0, -0.1013, -0.012],
                [0.0, -0.2001, -0.019],
                [0.0, -0.2995, -0.03],
            ]
        ),
        expand=False,
    )

    TestUtils.assert_equal(
        lambdas,
        np.array(
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 2.45250000e00, 4.90500000e00],
                [0.00000000e00, -3.00344627e-16, -6.00689255e-16],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ]
        ),
        expand=False,
    )
