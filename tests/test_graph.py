from bionc import BiomechanicalModel, JointType, NaturalAxis, CartesianAxis, NaturalSegment, TransformationMatrixType
import numpy as np
import pytest


def build_model() -> BiomechanicalModel:
    # Let's create a model
    model = BiomechanicalModel()
    # number of segments
    # fill the biomechanical model with the segment
    for i in range(6):
        name = f"pendulum_{i}"
        model[name] = NaturalSegment.with_cartesian_inertial_parameters(
            name=name,
            alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates an orthogonal coordinate system
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1,
            mass=1,
            center_of_mass=np.array([0, 0.1, 0]),  # in segment coordinates system
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
        )
    )

    model._add_joint(
        dict(
            name=f"hinge_{1}",
            joint_type=JointType.REVOLUTE,
            parent=f"pendulum_0",
            child=f"pendulum_1",
            parent_axis=[NaturalAxis.U, NaturalAxis.U],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    model._add_joint(
        dict(
            name=f"hinge_{2}",
            joint_type=JointType.REVOLUTE,
            parent=f"pendulum_1",
            child=f"pendulum_2",
            parent_axis=[NaturalAxis.U, NaturalAxis.U],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    model._add_joint(
        dict(
            name=f"hinge_{3}",
            joint_type=JointType.REVOLUTE,
            parent=f"pendulum_1",
            child=f"pendulum_3",
            parent_axis=[NaturalAxis.U, NaturalAxis.U],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    model._add_joint(
        dict(
            name=f"hinge_{4}",
            joint_type=JointType.REVOLUTE,
            parent=f"pendulum_0",
            child=f"pendulum_4",
            parent_axis=[NaturalAxis.U, NaturalAxis.U],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    model._add_joint(
        dict(
            name=f"hinge_{5}",
            joint_type=JointType.REVOLUTE,
            parent=f"pendulum_0",
            child=f"pendulum_5",
            parent_axis=[NaturalAxis.U, NaturalAxis.U],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    return model


def test_graph():
    model = build_model()
    # test the values of a list
    assert model.children("pendulum_0") == [1, 4, 5]
    assert model.children("pendulum_1") == [2, 3]
    assert model.children("pendulum_2") == []
    assert model.children("pendulum_3") == []
    assert model.children("pendulum_4") == []
    assert model.children("pendulum_5") == []

    assert model.children(0) == [1, 4, 5]
    assert model.children(1) == [2, 3]
    assert model.children(2) == []
    assert model.children(3) == []
    assert model.children(4) == []
    assert model.children(5) == []

    with pytest.raises(AttributeError):
        model.parents("pendulum_0")

    assert model.parents("pendulum_1") == [0]
    assert model.parents("pendulum_2") == [1]
    assert model.parents("pendulum_3") == [1]
    assert model.parents("pendulum_4") == [0]
    assert model.parents("pendulum_5") == [0]

    assert model.parents(1) == [0]
    assert model.parents(2) == [1]
    assert model.parents(3) == [1]
    assert model.parents(4) == [0]
    assert model.parents(5) == [0]

    assert model.segment_subtrees() == [[0, 1, 2, 3, 4, 5], [1, 2, 3], [2], [3], [4], [5]]
