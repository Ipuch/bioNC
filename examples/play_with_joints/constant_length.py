from bionc import (
    BiomechanicalModel,
    SegmentNaturalCoordinates,
    SegmentNaturalVelocities,
    NaturalCoordinates,
    NaturalVelocities,
    NaturalSegment,
    JointType,
    CartesianAxis,
    NaturalAxis,
    NaturalMarker,
    Viz,
)

import numpy as np

from utils import forward_integration, post_computations


def build_two_link_segment():
    # Let's create a model
    model = BiomechanicalModel()
    # number of segments
    # fill the biomechanical model with the segment
    for i in range(2):
        name = f"segment_{i}"
        model[name] = NaturalSegment(
            name=name,
            alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates an orthogonal coordinate system
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=0.8,
            mass=1,
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
        )

    model["segment_0"].add_natural_marker(
        NaturalMarker(
            name="point_A",
            parent_name="segment_0",
            position=np.array([0, -1, 0]),
            is_technical=True,
            is_anatomical=False,
        )
    )

    model["segment_1"].add_natural_marker(
        NaturalMarker(
            name="point_B",
            parent_name="segment_1",
            position=np.array([0, 0, 0]),
            is_technical=True,
            is_anatomical=False,
        )
    )

    # add a revolute joint (still experimental)
    # if you want to add a revolute joint,
    # you need to ensure that x is always orthogonal to u and v
    model._add_joint(
        dict(
            name="hinge_0",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="segment_0",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
            theta=[np.pi / 2, np.pi / 2],
        )
    )
    model._add_joint(
        dict(
            name=f"constant_length",
            joint_type=JointType.CONSTANT_LENGTH,
            parent=f"segment_0",
            child=f"segment_1",
            length=0.2,
            parent_point="point_A",
            child_point="point_B",
        )
    )
    return model


def main(initial_pose: str = "hanged"):
    model = build_two_link_segment()

    if initial_pose == "hanged":
        Q0 = SegmentNaturalCoordinates([1, 0, 0, 0, 0, 0, 0, 0, -0.8, 0, -1, 0])
        Q1 = SegmentNaturalCoordinates([1, 0, 0, 0, 0, -1, 0, 0, -1.8, 0, -1, 0])
    elif initial_pose == "ready_to_swing":
        Q0 = SegmentNaturalCoordinates([1, 0, 0, 0, 0, 0, 0, 0, -0.8, 0, -1, 0])
        Q1 = SegmentNaturalCoordinates(
            [
                1,
                0,
                0,
                0.2 * np.cos(np.pi / 4),
                0,
                -(0.8 + 0.2 * np.sin(np.pi / 4)),
                0.2 * np.cos(np.pi / 4),
                0,
                -(0.8 + 0.2 * np.sin(np.pi / 4)) - 0.8,
                0,
                -1,
                0,
            ]
        )
    Q = NaturalCoordinates.from_qi((Q0, Q1))

    # # numpy version
    # Q_test = NaturalCoordinates(np.arange(24))
    # jacobian_numpy = model.joint_constraints_jacobian(Q_test)
    #
    # model_mx = model.to_mx()
    # sym = NaturalCoordinatesMX.sym(2)
    # j_constraints_sym = model_mx.joint_constraints(sym)
    # # jacobian
    # j_jacobian_sym = jacobian(j_constraints_sym, sym)
    # j_jacobian_func = Function("j_jacobian_func", [sym], [j_jacobian_sym])
    #
    # jacobian_mx = j_jacobian_func(np.arange(24)).toarray()

    print("--------------------")
    print("INITIAL CONSTRAINTS")
    print(model.rigid_body_constraints(Q))
    print(model.joint_constraints(Q))
    print("--------------------")

    # uncomment to see the initial position
    # viz = Viz(model)
    # viz.animate(Q)

    tuple_of_Qdot = [
        SegmentNaturalVelocities.from_components(udot=[0, 0, 0], rpdot=[0, 0, 0], rddot=[0, 0, 0], wdot=[0, 0, 0])
        for i in range(0, model.nb_segments)
    ]
    Qdot = NaturalVelocities.from_qdoti(tuple(tuple_of_Qdot))

    # actual simulation
    t_final = 5  # seconds
    time_steps, all_states, dynamics = forward_integration(
        model=model,
        Q_init=Q,
        Qdot_init=Qdot,
        t_final=t_final,
        steps_per_second=100,
    )

    defects, defects_dot, joint_defects, all_lambdas = post_computations(
        model=model,
        time_steps=time_steps,
        all_states=all_states,
        dynamics=dynamics,
    )

    # plot results
    import matplotlib.pyplot as plt

    plt.figure()
    for i in range(0, model.nb_rigid_body_constraints):
        plt.plot(time_steps, defects[i, :], marker="o", label=f"defects {i}")
    plt.title("Rigid body constraints")
    plt.legend()

    plt.figure()
    for i in range(0, model.nb_joint_constraints):
        plt.plot(time_steps, joint_defects[i, :], marker="o", label=f"joint_defects {i}")
    plt.title("Joint constraints")
    plt.legend()

    plt.figure()
    for i in range(0, model.nb_holonomic_constraints):
        plt.plot(time_steps, all_lambdas[i, :], marker="o", label=f"lambda {i}")
    plt.title("Lagrange multipliers")
    plt.legend()
    plt.show()

    # animation
    viz = Viz(model)
    viz.animate(NaturalCoordinates(all_states[: (12 * model.nb_segments), :]), None)


if __name__ == "__main__":
    # main("hanged")
    main("ready_to_swing")
