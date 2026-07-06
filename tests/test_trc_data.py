import numpy as np
import pytest

from bionc.model_creation.trc_data import TRCData

from .utils import TestUtils

TRC_FILE = TestUtils.bionc_folder() + "/examples/model_creation/data/ABD01.trc"

MARKER_NAMES = ["gu", "centelbow", "EpL", "centpxt8", "centijc7", "ij", "ts", "aa", "ai"]


def test_load_defaults():
    data = TRCData(TRC_FILE)

    assert data.first_frame == 0
    # the file holds 1091 frames (0..1090)
    assert data.last_frame == 1090
    assert data.trc_data["Markers"] == MARKER_NAMES
    assert data.trc_data["Units"] == "mm"

    # values holds one squeezed (4, nb_frames) array per marker
    assert set(data.values.keys()) == set(MARKER_NAMES)
    assert data.values["gu"].shape == (4, 1091)
    assert not np.isnan(np.array(list(data.values.values()))).any()


def test_frame_bounds():
    data = TRCData(TRC_FILE, first_frame=0, last_frame=10)

    assert data.first_frame == 0
    assert data.last_frame == 10

    positions = data._get_position(("gu",))
    # (xyz+homogeneous, nb_markers, nb_frames), nb_frames = last - first + 1
    assert positions.shape == (4, 1, 11)


def test_get_position_converts_mm_to_meter():
    data = TRCData(TRC_FILE, first_frame=0, last_frame=10)

    positions = data._get_position(("gu",))
    # raw first-frame value of gu in the file is (-74.708, -22.067, 163.892) mm
    np.testing.assert_almost_equal(
        positions[:, 0, 0],
        np.array([-0.074708, -0.022067, 0.163892, 1.0]),
    )
    # homogeneous row is always 1
    np.testing.assert_array_equal(positions[-1, :, :], np.ones((1, 11)))


def test_get_position_multiple_markers_order():
    data = TRCData(TRC_FILE, first_frame=0, last_frame=5)

    positions = data._get_position(("centpxt8", "gu"))
    assert positions.shape == (4, 2, 6)

    # requested order is honoured, independent of the file order
    np.testing.assert_almost_equal(
        positions[:, 0, 0],
        np.array([-0.046816, -0.153158, -0.008669, 1.0]),
    )
    np.testing.assert_almost_equal(
        positions[:, 1, 0],
        np.array([-0.074708, -0.022067, 0.163892, 1.0]),
    )


def test_indices_in_trc():
    data = TRCData(TRC_FILE, first_frame=0, last_frame=1)

    assert data._indices_in_trc(("gu",)) == (0,)
    assert data._indices_in_trc(("ij", "ai")) == (5, 8)
    assert data._indices_in_trc(("ai", "ij")) == (8, 5)


def test_mean_marker_positions():
    data = TRCData(TRC_FILE)

    mean = data.mean_marker_positions(("ij", "ai"))
    assert mean.shape == (4,)
    np.testing.assert_almost_equal(
        mean,
        np.array([-0.0724057, -0.04099002, 0.04078963, 1.0]),
    )

    # the mean of a single marker matches its own time-average
    mean_gu = data.mean_marker_positions(("gu",))
    np.testing.assert_almost_equal(mean_gu, np.nanmean(data.values["gu"], axis=1))


def test_unknown_marker_raises():
    data = TRCData(TRC_FILE, first_frame=0, last_frame=1)

    with pytest.raises(ValueError):
        data._indices_in_trc(("not_a_marker",))
