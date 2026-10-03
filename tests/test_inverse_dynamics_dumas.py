"""
Reproduce the 'generalized coordinates' inverse dynamics of the lower limb gait example of
R. Dumas' "3D Kinematics and Inverse Dynamics" toolbox (v2.4.0, Main_Question_3.m / Inverse_Dynamics_GC.m)
and check that bionc gives the same ankle, knee and hip forces and moments.

The reference is a line by line port of Inverse_Dynamics_GC.m (explicit generalized mass matrix G, Kt, NPt, NCt
and N*t, distal to proximal), while bionc builds the same three segments as a model and runs its recursive
inverse dynamics. The toolbox has gravity along -Y, bionc along -Z: the data are rotated by +90° about X
before being handed to bionc, and the outputs rotated back.

bionc receives the same natural inertial parameters as the reference, so that this test checks the inverse dynamics
algorithm itself. Note that Inverse_Dynamics_GC.m fills the pseudo-inertia with the inertia tensor at the proximal
point, inv(B) I_P inv(B)^T, whereas the generalized mass matrix needs the second moment of mass
inv(B) (tr(I_P)/2 E - I_P) inv(B)^T, as Inverse_Dynamics_HM.m builds it and as bionc does from cartesian parameters.

Dataset: tests/data/dumas_lower_limb_gait (BSD license, Copyright (c) 2021, Raphael Dumas).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

from bionc import JointType, TransformationMatrixType
from bionc.bionc_numpy import (
    BiomechanicalModel,
    NaturalSegment,
    SegmentNaturalCoordinates,
    NaturalCoordinates,
    NaturalAccelerations,
    compute_transformation_matrix,
)

DATA_FOLDER = Path(__file__).parent / "data" / "dumas_lower_limb_gait"
SAMPLING_FREQUENCY = 100  # f in Main_Question_3.m
CUT_OFF_FREQUENCY = 6  # fc in Main_Question_3.m
GRAVITY_TOOLBOX = np.array([0, -9.81, 0])
# rotation of +90° about X, sends the toolbox vertical axis Y onto the bionc vertical axis Z
R_TOOLBOX_TO_BIONC = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def load_dumas_gait():
    segment = loadmat(DATA_FOLDER / "Segment.mat", struct_as_record=False)["Segment"][0]
    joint = loadmat(DATA_FOLDER / "Joint.mat", struct_as_record=False)["Joint"][0]
    # Toolbox indices: 1 = force plate (rP = centre of pressure), 2 = foot, 3 = shank, 4 = thigh, 5 = pelvis
    Q = [s.Q[:, 0, :] for s in segment]  # 12 x n
    inertia = {i: (float(segment[i].m[0, 0]), segment[i].rCs[:, 0], segment[i].Is) for i in (1, 2, 3)}
    # Force and moment of the foot on the ground, at the centre of pressure
    return Q, inertia, joint[0].F[:, 0, :], joint[0].M[:, 0, :]


def filtered_second_derivative(Q: np.ndarray) -> np.ndarray:
    """Vfilt_array3(Derive_array3(Vfilt_array3(Derive_array3(Q, dt), f, fc), dt), f, fc)"""
    dt = 1 / SAMPLING_FREQUENCY
    b, a = butter(4, CUT_OFF_FREQUENCY / (SAMPLING_FREQUENCY / 2))
    padlen = 3 * (max(len(a), len(b)) - 1)  # Matlab's filtfilt padding

    def vfilt(V):
        return filtfilt(b, a, V, axis=1, padlen=padlen)

    return vfilt(np.gradient(vfilt(np.gradient(Q, dt, axis=1)), dt, axis=1))


def mean_segment_geometry(Qi: np.ndarray) -> tuple[float, float, float, float]:
    """Mean length and alpha, beta, gamma angles, as in Inverse_Dynamics_GC.m"""
    u, rp, rd, w = Qi[0:3], Qi[3:6], Qi[6:9], Qi[9:12]
    length = np.mean(np.linalg.norm(rp - rd, axis=0))
    v = (rp - rd) / np.linalg.norm(rp - rd, axis=0)
    alpha = np.mean(np.arccos(np.sum(v * w, axis=0)))
    beta = np.mean(np.arccos(np.sum(w * u, axis=0)))
    gamma = np.mean(np.arccos(np.sum(u * v, axis=0)))
    return length, alpha, beta, gamma


def dumas_inverse_dynamics_gc(Q, inertia, Qddot, F_ground, M_ground):
    """
    Port of Inverse_Dynamics_GC.m, frame by frame.
    Returns F[i], M[i] (3 x n) for i = 1 ankle, 2 knee, 3 hip: the force and moment applied on segment i
    by segment i+1 at the proximal endpoint rP of segment i, in the toolbox ICS.
    """
    n = Q[0].shape[1]
    E33, O33 = np.eye(3), np.zeros((3, 3))
    F = {0: F_ground}
    M = {0: M_ground}

    for i in (1, 2, 3):
        m, rCs, Is = inertia[i]
        L, a, b, c = mean_segment_geometry(Q[i])
        Buv = np.array(
            [
                [1, L * np.cos(c), np.cos(b)],
                [0, L * np.sin(c), (np.cos(a) - np.cos(b) * np.cos(c)) / np.sin(c)],
                [0, 0, np.sqrt(1 - np.cos(b) ** 2 - ((np.cos(a) - np.cos(b) * np.cos(c)) / np.sin(c)) ** 2)],
            ]
        )
        inv_Buv = np.linalg.inv(Buv)
        nC = inv_Buv @ rCs
        J = inv_Buv @ (Is + m * ((rCs @ rCs) * E33 - np.outer(rCs, rCs))) @ inv_Buv.T

        G = np.block(
            [
                [J[0, 0] * E33, (m * nC[0] + J[0, 1]) * E33, -J[0, 1] * E33, J[0, 2] * E33],
                [
                    (m * nC[0] + J[0, 1]) * E33,
                    (m + 2 * m * nC[1] + J[1, 1]) * E33,
                    -(m * nC[1] + J[1, 1]) * E33,
                    (m * nC[2] + J[1, 2]) * E33,
                ],
                [-J[0, 1] * E33, -(m * nC[1] + J[1, 1]) * E33, J[1, 1] * E33, -J[1, 2] * E33],
                [J[0, 2] * E33, (m * nC[2] + J[1, 2]) * E33, -J[1, 2] * E33, J[2, 2] * E33],
            ]
        )
        NPt = np.vstack((O33, E33, O33, O33))
        NCt = np.vstack((nC[0] * E33, (1 + nC[1]) * E33, -nC[1] * E33, nC[2] * E33))

        F[i], M[i] = np.zeros((3, n)), np.zeros((3, n))
        for k in range(n):
            u, rp, rd, w = Q[i][0:3, k], Q[i][3:6, k], Q[i][6:9, k], Q[i][9:12, k]
            o = np.zeros(3)
            Kt = (
                np.array(
                    [
                        np.concatenate((2 * u, rp - rd, w, o, o, o)),
                        np.concatenate((o, u, o, 2 * (rp - rd), w, o)),
                        np.concatenate((o, -u, o, -2 * (rp - rd), -w, o)),
                        np.concatenate((o, o, u, o, rp - rd, 2 * w)),
                    ]
                )
                .reshape(4, 6, 3)
                .transpose(0, 2, 1)
                .reshape(12, 6)
            )
            Bstar = np.column_stack((np.cross(w, u), np.cross(u, rp - rd), np.cross(-(rp - rd), w)))
            Nstart = np.block(
                [
                    [o[:, None], (rp - rd)[:, None], o[:, None]],
                    [o[:, None], o[:, None], -w[:, None]],
                    [o[:, None], o[:, None], w[:, None]],
                    [u[:, None], o[:, None], o[:, None]],
                ]
            ) @ np.linalg.inv(Bstar)
            # distal neighbour: force plate (centre of pressure) for the foot, the previous segment otherwise
            rp_distal = Q[i - 1][3:6, k]
            rhs = (
                G @ Qddot[i][:, k]
                - NCt @ (m * GRAVITY_TOOLBOX)
                - NPt @ (-F[i - 1][:, k])
                - Nstart @ (-M[i - 1][:, k] + np.cross(rp_distal - rp, -F[i - 1][:, k]))
            )
            FMLambda = np.linalg.solve(np.hstack((NPt, Nstart, -Kt)), rhs)
            F[i][:, k], M[i][:, k] = FMLambda[0:3], FMLambda[3:6]

    return F, M


def dumas_natural_inertial_parameters(length, alpha, beta, gamma, mass, rCs, Is) -> tuple[np.ndarray, np.ndarray]:
    """nC = inv(Buv) rCs and the pseudo-inertia J at the proximal point, as in Inverse_Dynamics_GC.m"""
    inv_Buv = np.linalg.inv(compute_transformation_matrix(TransformationMatrixType.Buv, length, alpha, beta, gamma))
    natural_center_of_mass = inv_Buv @ rCs
    pseudo_inertia = inv_Buv @ (Is + mass * ((rCs @ rCs) * np.eye(3) - np.outer(rCs, rCs))) @ inv_Buv.T
    return natural_center_of_mass, pseudo_inertia


def build_bionc_lower_limb(Q, inertia) -> BiomechanicalModel:
    """
    thigh (root) -> shank -> foot, with the toolbox mean geometry and inertial parameters.
    The natural inertial parameters are given directly, those of Inverse_Dynamics_GC.m.
    """
    model = BiomechanicalModel()
    names = {3: "thigh", 2: "shank", 1: "foot"}
    for i in (3, 2, 1):  # bionc's inverse dynamics starts its depth first search from segment 0
        m, rCs, Is = inertia[i]
        length, alpha, beta, gamma = mean_segment_geometry(Q[i])
        natural_center_of_mass, pseudo_inertia = dumas_natural_inertial_parameters(
            length, alpha, beta, gamma, m, rCs, Is
        )
        model[names[i]] = NaturalSegment(
            name=names[i],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            length=length,
            mass=m,
            natural_center_of_mass=natural_center_of_mass,
            natural_pseudo_inertia=pseudo_inertia,
        )
    model._add_joint(dict(name="knee", joint_type=JointType.SPHERICAL, parent="thigh", child="shank"))
    model._add_joint(dict(name="ankle", joint_type=JointType.SPHERICAL, parent="shank", child="foot"))
    return model


def rotate_Q(Qi: np.ndarray) -> np.ndarray:
    return np.vstack([R_TOOLBOX_TO_BIONC @ Qi[j : j + 3] for j in range(0, 12, 3)])


def test_inverse_dynamics_reproduces_dumas_gait_example():
    Q, inertia, F_ground, M_ground = load_dumas_gait()
    Qddot = [None] + [filtered_second_derivative(Q[i]) for i in (1, 2, 3)]

    F_ref, M_ref = dumas_inverse_dynamics_gc(Q, inertia, Qddot, F_ground, M_ground)

    model = build_bionc_lower_limb(Q, inertia)
    bionc_order = (3, 2, 1)  # thigh, shank, foot
    foot_index = bionc_order.index(1)
    Q_bionc = {i: rotate_Q(Q[i]) for i in bionc_order}
    Qddot_bionc = {i: rotate_Q(Qddot[i]) for i in bionc_order}
    center_of_pressure = R_TOOLBOX_TO_BIONC @ Q[0][3:6]
    F_ground_bionc = R_TOOLBOX_TO_BIONC @ F_ground
    M_ground_bionc = R_TOOLBOX_TO_BIONC @ M_ground

    n = Q[1].shape[1]
    F_bionc = {i: np.zeros((3, n)) for i in bionc_order}
    M_bionc = {i: np.zeros((3, n)) for i in bionc_order}
    for k in range(n):
        Qk = NaturalCoordinates.from_qi(
            tuple(
                SegmentNaturalCoordinates.from_components(
                    u=Q_bionc[i][0:3, k], rp=Q_bionc[i][3:6, k], rd=Q_bionc[i][6:9, k], w=Q_bionc[i][9:12, k]
                )
                for i in bionc_order
            )
        )
        Qddotk = NaturalAccelerations(np.concatenate([Qddot_bionc[i][:, k] for i in bionc_order]))

        # the ground acts on the foot with the opposite of the foot-on-ground wrench, at the centre of pressure
        external_forces = model.external_force_set()
        external_forces.add_in_global(
            segment_index=foot_index,
            external_force=np.concatenate((-M_ground_bionc[:, k], -F_ground_bionc[:, k])),
            point_in_global=center_of_pressure[:, k],
        )

        torques, forces, _ = model.inverse_dynamics(Q=Qk, Qddot=Qddotk, external_forces=external_forces)
        for j, i in enumerate(bionc_order):
            F_bionc[i][:, k] = R_TOOLBOX_TO_BIONC.T @ forces[:, j]
            M_bionc[i][:, k] = R_TOOLBOX_TO_BIONC.T @ torques[:, j]

    for i in (1, 2, 3):  # ankle, knee, hip
        np.testing.assert_allclose(F_bionc[i], F_ref[i], rtol=0, atol=1e-8)
        np.testing.assert_allclose(M_bionc[i], M_ref[i], rtol=0, atol=1e-8)

    # sanity check on magnitudes, the hip vertical force peaks above body weight during gait (90 kg subject)
    assert np.max(np.abs(F_ref[3][1])) > 700


@pytest.mark.xfail(
    strict=True,
    reason="NaturalSegment.compute_transformation_matrix returns Buv.T, nC and J are wrong for non-orthogonal segments",
)
def test_with_cartesian_inertial_parameters_non_orthogonal_segment():
    """nC = inv(Buv) rCs and J = inv(Buv) (tr(I_P)/2 E - I_P) inv(Buv)^T, on the dataset foot"""
    Q, inertia, _, _ = load_dumas_gait()
    m, rCs, Is = inertia[1]  # foot, the least orthogonal segment of the dataset
    length, alpha, beta, gamma = mean_segment_geometry(Q[1])
    inv_Buv = np.linalg.inv(compute_transformation_matrix(TransformationMatrixType.Buv, length, alpha, beta, gamma))
    natural_center_of_mass = inv_Buv @ rCs
    inertia_at_proximal_point = Is + m * ((rCs @ rCs) * np.eye(3) - np.outer(rCs, rCs))
    second_moment = 0.5 * np.trace(inertia_at_proximal_point) * np.eye(3) - inertia_at_proximal_point
    pseudo_inertia = inv_Buv @ second_moment @ inv_Buv.T

    segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="foot",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        length=length,
        mass=m,
        center_of_mass=rCs,
        inertia=Is,
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

    np.testing.assert_allclose(np.array(segment.natural_center_of_mass).squeeze(), natural_center_of_mass, atol=1e-12)
    np.testing.assert_allclose(segment.natural_pseudo_inertia, pseudo_inertia, atol=1e-12)
