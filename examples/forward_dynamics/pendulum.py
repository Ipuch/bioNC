from typing import Callable
import numpy as np

from bionc.bionc_numpy import (
    BiomechanicalModel,
    NaturalSegment,
    JointType,
    SegmentNaturalCoordinates,
    NaturalCoordinates,
    SegmentNaturalVelocities,
)
from bionc import NaturalAxis, CartesianAxis

if __name__ == "__main__":

    # Let's create a model
    model = BiomechanicalModel()
    # fill the biomechanical model with the segment
    model["box"] = NaturalSegment(
        name="box",
        alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0.1, 0]),  # in segment coordinates system
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
    )
    # add a revolute joint (still experimental)
    # if you want to add a revolute joint,
    # you need to ensure that x is always orthogonal to u and v
    model._add_joint(
        dict(
            name="hinge",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="box",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
            theta=[np.pi / 2, np.pi / 2],
        )
        )
    print(model.joints)

    model.nb_joints()
    model.nb_joint_constraints()

    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    Q = NaturalCoordinates(Qi)

    model.joint_constraints(Q)
    model.joint_constraints_jacobian(Q)


    # u as y-axis
    # Qi = SegmentNaturalCoordinates.from_components(
    #     u=[0, 1, 0], rp=[0, 0, 0], rd=[0, 0, -1], w=[1, 0, 0]
    #     )
    # u as z-axis
    # Qi = SegmentNaturalCoordinates.from_components(u=[0, 0, 1], rp=[0, 0, 0], rd=[-1, 0, 0], w=[0, 1, 0])

    # # Velocities are set to zero at t=0
    # Qidot = SegmentNaturalVelocities.from_components(
    #     udot=np.array([0, 0, 0]), rpdot=np.array([0, 0, 0]), rddot=np.array([0, 0, 0]), wdot=np.array([0, 0, 0])
    # )
    #
    # t_final = 2
    # time_steps, all_states, dynamics = drop_the_box(
    #     my_segment=my_segment,
    #     Q_init=Qi,
    #     Qdot_init=Qidot,
    #     t_final=t_final,
    # )
    #
    # defects, defects_dot, all_lambdas, center_of_mass = post_computations(my_segment, time_steps, all_states, dynamics)
    #
    # from viz import plot_series, animate_natural_segment
    #
    # # Plot the results
    # plot_series(time_steps, defects, legend="rigid_constraint")  # Phi_r
    # plot_series(time_steps, defects_dot, legend="rigid_constraint_derivative")  # Phi_r_dot
    # plot_series(time_steps, all_lambdas, legend="lagrange_multipliers")  # lambda
    #
    # # animate the motion
    # animate_natural_segment(time_steps, all_states, center_of_mass, t_final)
