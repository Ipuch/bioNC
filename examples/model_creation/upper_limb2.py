from pathlib import Path

from bionc import (
    AxisTemplate,
    AxisFunctionTemplate,
    BiomechanicalModelTemplate,
    MarkerTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
    TRCData,
    BiomechanicalModel,
    JointType,
    EulerSequence,
    TransformationMatrixType,
)


import numpy as np


def u_thorax(ij: np.ndarray, centijc7: np.ndarray, centpxt8: np.ndarray) -> np.ndarray:
    """
    This function is a helper for the construction of the thorax frame,
    because the thorax frame rely on the previous computations to get the postero-anterior axis u,
    I had to redefine the entire sequence of computations for it.

    (rp - rd) x w / norm2((rp - rd) x w)

    It returns a (4 x Nframes) np.ndarray

    """
    rp = (ij + centijc7) / 2
    rd = centpxt8

    v = rp - rd

    w = np.ones((4, ij.shape[1]))
    cross_product = np.ones((4, ij.shape[1]))

    for i, (mk1i, mk2i, mk3i) in enumerate(zip(ij.T, centijc7.T, centpxt8.T)):
        v1 = mk2i[:3] - mk1i[:3]
        v2 = mk3i[:3] - mk1i[:3]
        w[:3, i] = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        temp = np.cross(v[:3, i], w[:3, i])
        cross_product[:3, i] = temp / np.linalg.norm(temp)

    return cross_product


def model_creation_from_measured_data(trc_filename) -> BiomechanicalModel:
    """
    Create a model from a data file and we build the biomechanical model as a template using the marker names
    """

    # Fill the kinematic chain model
    model = BiomechanicalModelTemplate()
    # de_leva = DeLevaTable(total_mass=100, sex="female")
    # Marker list: ij, centpxt8, centijc7, ts, ai, aa, gu, EpL, centelbow
    u_axis_thorax = lambda m, bio: u_thorax(m["ij"], m["centijc7"], m["centpxt8"])

    model["THORAX"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(function=u_axis_thorax),
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "centijc7", "ij"),
            distal_point="centpxt8",
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "ij", "centijc7", "centpxt8")
            ),
        )
    )

    model["THORAX"].add_marker(MarkerTemplate(name="ij", parent_name="THORAX", is_technical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="centijc7", parent_name="THORAX", is_technical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="centpxt8", parent_name="THORAX", is_technical=True))

    model["SCAPULA"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "aa", "ai", "ts")),
            proximal_point="aa",
            distal_point="ai",
            w_axis=AxisTemplate(start="ts", end="aa"),
        )
    )

    model["SCAPULA"].add_marker(MarkerTemplate(name="aa", parent_name="SCAPULA", is_technical=True))
    model["SCAPULA"].add_marker(MarkerTemplate(name="ts", parent_name="SCAPULA", is_technical=True))
    model["SCAPULA"].add_marker(MarkerTemplate(name="ai", parent_name="SCAPULA", is_technical=True))

    model["HUMERUS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "EpL", "centelbow", "gu")
            ),
            proximal_point="gu",
            distal_point="centelbow",
            w_axis=AxisTemplate(start="centelbow", end="EpL"),
        )
    )
    model["HUMERUS"].add_marker(MarkerTemplate(name="gu", parent_name="HUMERUS", is_technical=True))
    model["HUMERUS"].add_marker(MarkerTemplate(name="centelbow", parent_name="HUMERUS", is_technical=True))
    model["HUMERUS"].add_marker(MarkerTemplate(name="EpL", parent_name="HUMERUS", is_technical=True))

    model.add_joint(
        name="Freeflyer",
        joint_type=JointType.GROUND_FREE,
        parent="GROUND",
        child="THORAX",
        projection_basis=EulerSequence.XYZ,
        child_basis=TransformationMatrixType.Bvu,
    )

    model.add_joint(
        name="Scapulothoracic",
        joint_type=JointType.FREE,
        parent="THORAX",
        child="SCAPULA",
        projection_basis=EulerSequence.XYZ,
        child_basis=TransformationMatrixType.Bvu,
    )

    model.add_joint(
        name="Glenohumeral",
        joint_type=JointType.FREE,
        parent="SCAPULA",
        child="HUMERUS",
        projection_basis=EulerSequence.XYZ,
        child_basis=TransformationMatrixType.Bvu,
    )

    data = TRCData(f"{trc_filename}", first_frame=0, last_frame=200)
    natural_model = model.update(data)

    return natural_model


def main():
    trc_filename = f"{Path(__file__).parent.resolve()}/data/ABD01.trc"
    # Create the model from a c3d file and markers as template
    model = model_creation_from_measured_data(trc_filename)

    # # load experimental markers
    trc_data = TRCData(f"{trc_filename}")
    markers_xp = trc_data._get_position(marker_names=model.marker_names_technical)

    # # compute the natural coordinates
    Qxp = model.Q_from_markers(markers_xp)
    #
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from bionc.vizualization.pyorerun_natural_vectors import add_natural_vectors
    from pyorerun import PhaseRerun, PyoMarkers

    #
    # # display the experimental markers in white and the model markers in blue
    # # almost superimposed because the model is well defined on the experimental data
    prr = PhaseRerun(t_span=np.linspace(0, 1, markers_xp.shape[2]))
    model_interface = BioncModelNoMesh(model)
    pyomarker = PyoMarkers.from_trc(trc_filename) / 1000
    prr.add_animated_model(model_interface, Qxp, pyomarker)
    add_natural_vectors(prr, model, np.asarray(Qxp), scale_u=0.1, scale_v=1.0, scale_w=0.1)
    prr.rerun()

    # dump the model in a pickle format
    model.save("../models/upper_limb.nc")


if __name__ == "__main__":
    main()
