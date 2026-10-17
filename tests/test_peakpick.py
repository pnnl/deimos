import pandas as pd
import pytest

import deimos
from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load(localfile("resources/example_data.h5"), key="ms1")


@pytest.fixture()
def bug_data():
    data = pd.read_csv(localfile("resources/deimos_bug_test_data.csv"))
    data = data[data["intensity"] > 150].copy()
    return deimos.collapse(data, keep=["mz", "drift_time"], how="sum")


@pytest.mark.parametrize(
    "dims,bins,scale_by,ref_res,scale",
    [
        (["mz", "drift_time", "retention_time"], [2.7, 0.94, 3.64], None, None, None),
        (
            ["mz", "drift_time", "retention_time"],
            [2.7, 0.94, 3.64],
            "mz",
            0.002445221,
            "drift_time",
        ),
        (["mz", "drift_time"], [2.7, 0.94], "mz", 0.002445221, "drift_time"),
        ("retention_time", 3.64, None, None, None),
    ],
)
def test_local_maxima(ms1, dims, bins, scale_by, ref_res, scale):
    # make smaller for testing
    subset = deimos.slice(ms1, by="mz", low=200, high=300)

    peaks = deimos.peakpick.local_maxima(
        subset, dims=dims, bins=bins, scale_by=scale_by, ref_res=ref_res, scale=scale
    )

    assert type(peaks) is pd.DataFrame

    for d in deimos.utils.safelist(dims) + ["intensity"]:
        assert d in peaks.columns


@pytest.mark.parametrize(
    "dims,bins,scale_by,ref_res,scale",
    [
        (
            ["mz", "drift_time", "retention_time"],
            [2.7, 0.94, 3.64],
            "mz",
            0.002445221,
            None,
        ),
        (
            ["mz", "drift_time", "retention_time"],
            [2.7, 0.94, 3.64],
            "mz",
            None,
            "drift_time",
        ),
        (
            ["mz", "drift_time", "retention_time"],
            [2.7, 0.94, 3.64],
            None,
            0.002445221,
            "drift_time",
        ),
        (["mz", "drift_time", "retention_time"], [2.7, 0.94, 3.64], "mz", None, None),
        (
            ["mz", "drift_time", "retention_time"],
            [2.7, 0.94, 3.64],
            None,
            0.002445221,
            None,
        ),
        (
            ["mz", "drift_time", "retention_time"],
            [2.7, 0.94, 3.64],
            None,
            None,
            "drift_time",
        ),
        (["mz", "retention_time"], [2.7, 0.94, 3.64], "mz", 0.002445221, None),
        (["mz", "drift_time", "retention_time"], 2.7, "mz", 0.002445221, None),
    ],
)
def test_local_maxima_fail(ms1, dims, bins, scale_by, ref_res, scale):
    with pytest.raises(ValueError):
        deimos.peakpick.local_maxima(
            ms1, dims=dims, bins=bins, scale_by=scale_by, ref_res=ref_res, scale=scale
        )


def test_persistent_homology_no_ghost_features(bug_data):
    """Weighted positions must fall near actual data points (no ghost features)."""
    peaks = deimos.peakpick.persistent_homology(
        bug_data, dims=["mz", "drift_time"], radius=[3, 3]
    )

    assert len(peaks) > 0
    assert "mz_weighted" in peaks.columns
    assert "drift_time_weighted" in peaks.columns

    for _, peak in peaks.iterrows():
        mz = peak["mz_weighted"]
        dt = peak["drift_time_weighted"]

        nearby = bug_data[
            (bug_data["mz"] >= mz - 0.01)
            & (bug_data["mz"] <= mz + 0.01)
            & (bug_data["drift_time"] >= dt - 2)
            & (bug_data["drift_time"] <= dt + 2)
        ]
        assert len(nearby) > 0, (
            f"Ghost feature at m/z={mz:.6f}, drift_time={dt:.2f}: "
            f"no raw data within tolerance"
        )


@pytest.mark.parametrize("radius", [None, [0, 0], [1, 1], [3, 3]])
def test_persistent_homology(bug_data, radius):
    """persistent_homology returns a valid DataFrame for various radius values."""
    peaks = deimos.peakpick.persistent_homology(
        bug_data, dims=["mz", "drift_time"], radius=radius
    )

    assert isinstance(peaks, pd.DataFrame)
    assert len(peaks) > 0
    for col in ["mz", "drift_time", "intensity", "persistence"]:
        assert col in peaks.columns

    if radius is not None and any(r > 0 for r in radius):
        assert "mz_weighted" in peaks.columns
        assert "drift_time_weighted" in peaks.columns
