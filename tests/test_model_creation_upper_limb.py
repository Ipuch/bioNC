import numpy as np

from bionc import TRCData

from .utils import TestUtils

TRC_FILE = TestUtils.bionc_folder() + "/examples/model_creation/data/ABD01.trc"


def _load_module():
    bionc = TestUtils.bionc_folder()
    return TestUtils.load_module(bionc + "/examples/model_creation/upper_limb.py")


def test_u_thorax_axis():
    module = _load_module()

    ij = np.array([[0.0, 0.0, 0.0, 1.0]]).T
    centijc7 = np.array([[1.0, 0.0, 0.0, 1.0]]).T
    centpxt8 = np.array([[0.0, 1.0, 0.0, 1.0]]).T

    axis = module.u_thorax(ij, centijc7, centpxt8)

    assert axis.shape == (4, 1)
    # orthogonal to the (rp - rd) direction and to the frame normal, unit norm
    np.testing.assert_almost_equal(
        axis[:, 0],
        np.array([-0.89442719, -0.4472136, 0.0, 1.0]),
    )
    np.testing.assert_almost_equal(np.linalg.norm(axis[:3, 0]), 1.0)


def test_model_creation_structure():
    module = _load_module()

    model = module.model_creation_from_measured_data(TRC_FILE)

    assert model.segment_names == ["THORAX", "SCAPULA", "HUMERUS"]
    assert model.joint_names == ["Freeflyer", "Scapulothoracic", "Glenohumeral"]
    assert model.nb_segments == 3
    assert model.nb_Q == 36
    assert model.nb_markers == 9
    assert sorted(model.marker_names_technical) == sorted(
        ["ij", "centijc7", "centpxt8", "aa", "ts", "ai", "gu", "centelbow", "EpL"]
    )


def test_model_reconstructs_experimental_markers():
    module = _load_module()

    model = module.model_creation_from_measured_data(TRC_FILE)

    data = TRCData(TRC_FILE)
    markers_xp = data._get_position(marker_names=model.marker_names_technical)
    Q = model.Q_from_markers(markers_xp)

    assert Q.shape == (36, markers_xp.shape[2])
    assert np.isfinite(np.asarray(Q)).all()

    # the model is built on these markers, so it reconstructs them almost exactly
    markers_model = model.markers(Q)
    error = np.linalg.norm(markers_model[:3] - markers_xp[:3], axis=0)
    assert np.nanmax(error) < 1e-3
