import numpy as np
import pandas as pd
import pytest

import deimos
from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load(localfile("resources/example_data.h5"), key="ms1")


@pytest.fixture()
def ms2():
    return deimos.load(localfile("resources/example_data.h5"), key="ms2")


@pytest.mark.parametrize(
    "threshold,length", [(0, 1229849), (1e3, 20564), (1e4, 2522), (1e6, 0)]
)
def test_threshold(ms1, threshold, length):
    thresholded = deimos.threshold(ms1, by="intensity", threshold=threshold)
    assert len(thresholded.index) == length


@pytest.mark.parametrize(
    "keep,how,length",
    [
        (["mz", "drift_time"], "sum", 651076),
        (["retention_time"], "max", 41),
        (["mz", "drift_time", "retention_time"], "sum", 1958605),
    ],
)
def test_collapse(ms1, keep, how, length):
    collapsed = deimos.collapse(ms1, keep=keep, how=how)

    assert all(collapsed.columns == keep + ["intensity"])
    assert len(collapsed.index) == length


@pytest.mark.parametrize(
    "by,loc,tol,return_index",
    [
        (
            ["mz", "drift_time", "retention_time"],
            [212.0, 17.2, 4.79],
            [1.06e-3, 1.03, 0.8],
            False,
        ),
        (
            ["mz", "drift_time", "retention_time"],
            [212.0, 17.2, 4.79],
            [1.06e-3, 1.03, 0.8],
            True,
        ),
        ("mz", 212.0, 1.06e-3, False),
        ("mz", 212.0, 1.06e-3, True),
    ],
)
def test_locate(ms1, by, loc, tol, return_index):
    if return_index is True:
        subset, idx = deimos.locate(
            ms1, by=by, loc=loc, tol=tol, return_index=return_index
        )
        assert idx is not None
        assert len(idx) == len(ms1.index)
    else:
        subset = deimos.locate(ms1, by=by, loc=loc, tol=tol, return_index=return_index)

    assert subset is not None
    assert all(subset.columns == ms1.columns)

    for x, dx, dim in zip(
        deimos.utils.safelist(loc),
        deimos.utils.safelist(tol),
        deimos.utils.safelist(by),
    ):
        assert subset[dim].min() >= x - dx
        assert subset[dim].max() <= x + dx


@pytest.mark.parametrize(
    "by,loc,tol,return_index",
    [
        (
            ["mz", "drift_time", "retention_time"],
            [212.0, 17.2, 4.79],
            [1.06e-3, 1.03, 0.8],
            False,
        ),
        (
            ["mz", "drift_time", "retention_time"],
            [212.0, 17.2, 4.79],
            [1.06e-3, 1.03, 0.8],
            True,
        ),
    ],
)
def test_locate_pass_none(by, loc, tol, return_index):
    if return_index is True:
        subset, idx = deimos.locate(
            None, by=by, loc=loc, tol=tol, return_index=return_index
        )
        assert subset is None
        assert idx is None
    else:
        subset = deimos.locate(None, by=by, loc=loc, tol=tol, return_index=return_index)

        assert subset is None


@pytest.mark.parametrize(
    "by,loc,tol,return_index",
    [("mz", -212.0, 1.06e-3, False), ("mz", 212.0e3, 1.06e-3, True)],
)
def test_locate_return_none(ms1, by, loc, tol, return_index):
    if return_index is True:
        subset, idx = deimos.locate(
            ms1, by=by, loc=loc, tol=tol, return_index=return_index
        )
        assert subset is None
        assert idx is not None
        assert len(idx) == len(ms1.index)
        assert all(idx) is False
    else:
        subset = deimos.locate(ms1, by=by, loc=loc, tol=tol, return_index=return_index)

        assert subset is None


@pytest.mark.parametrize(
    "by,low,high,return_index",
    [
        (
            ["mz", "drift_time", "retention_time"],
            [211.0, 16.2, 4.00],
            [213.0, 18.2, 5.16],
            False,
        ),
        (
            ["mz", "drift_time", "retention_time"],
            [211.0, 16.2, 4.00],
            [213.0, 18.2, 5.16],
            True,
        ),
        ("mz", 211.0, 213.0, False),
        ("mz", 211.0, 213.0, True),
    ],
)
def test_slice(ms1, by, low, high, return_index):
    if return_index is True:
        subset, idx = deimos.slice(
            ms1, by=by, low=low, high=high, return_index=return_index
        )
        assert idx is not None
        assert len(idx) == len(ms1.index)
    else:
        subset = deimos.slice(ms1, by=by, low=low, high=high, return_index=return_index)

    assert subset is not None
    assert all(subset.columns == ms1.columns)

    for ll, hh, dim in zip(
        deimos.utils.safelist(low),
        deimos.utils.safelist(high),
        deimos.utils.safelist(by),
    ):
        assert subset[dim].min() >= ll
        assert subset[dim].max() <= hh


@pytest.mark.parametrize(
    "by,low,high,return_index",
    [
        (
            ["mz", "drift_time", "retention_time"],
            [211.0, 16.2, 4.00],
            [213.0, 18.2, 5.16],
            False,
        ),
        (
            ["mz", "drift_time", "retention_time"],
            [211.0, 16.2, 4.00],
            [213.0, 18.2, 5.16],
            True,
        ),
    ],
)
def test_slice_pass_none(by, low, high, return_index):
    if return_index is True:
        subset, idx = deimos.slice(
            None, by=by, low=low, high=high, return_index=return_index
        )
        assert subset is None
        assert idx is None
    else:
        subset = deimos.slice(
            None, by=by, low=low, high=high, return_index=return_index
        )

        assert subset is None


@pytest.mark.parametrize(
    "by,low,high,return_index",
    [("mz", -211.0, -213.0, False), ("mz", 211.0e3, 213.0e3, True)],
)
def test_slice_return_none(ms1, by, low, high, return_index):
    if return_index is True:
        subset, idx = deimos.slice(
            ms1, by=by, low=low, high=high, return_index=return_index
        )
        assert subset is None
        assert idx is not None
        assert len(idx) == len(ms1.index)
        assert all(idx) is False
    else:
        subset = deimos.slice(ms1, by=by, low=low, high=high, return_index=return_index)

        assert subset is None


@pytest.mark.parametrize(
    "by,loc,low,high,relative,return_index",
    [
        (
            ["mz", "drift_time", "retention_time"],
            [212.0, 17.2, 4.79],
            [-0.1, -0.03, -0.3],
            [1, 0.03, 0.3],
            [False, True, False],
            False,
        )
    ],
)
def test_locate_asym(ms1, by, loc, low, high, relative, return_index):
    if return_index is True:
        subset, idx = deimos.locate_asym(
            ms1, by=by, low=low, high=high, return_index=return_index
        )
        assert idx is not None
        assert len(idx) == len(ms1.index)
    else:
        subset = deimos.locate_asym(
            ms1, loc=loc, by=by, low=low, high=high, return_index=return_index
        )

    assert subset is not None
    assert all(subset.columns == ms1.columns)

    for x, ll, hh, rel, dim in zip(
        deimos.utils.safelist(loc),
        deimos.utils.safelist(low),
        deimos.utils.safelist(high),
        deimos.utils.safelist(relative),
        deimos.utils.safelist(by),
    ):
        if rel is True:
            assert subset[dim].min() >= x * (1 + ll)
            assert subset[dim].max() <= x * (1 + hh)
        else:
            assert subset[dim].min() >= x + ll
            assert subset[dim].max() <= x + hh


@pytest.mark.parametrize(
    "by,loc,low,high,relative,return_index",
    [
        (
            ["mz", "drift_time", "retention_time"],
            [212.0, 17.2, 4.79],
            [-0.1, -0.03, -0.3],
            [1, 0.03, 0.3],
            [False, True, False],
            False,
        )
    ],
)
def test_locate_asym_pass_none(ms1, by, loc, low, high, relative, return_index):
    if return_index is True:
        subset, idx = deimos.locate_asym(
            None,
            by=by,
            loc=loc,
            low=low,
            high=high,
            relative=relative,
            return_index=return_index,
        )
        assert subset is None
        assert idx is None
    else:
        subset = deimos.locate_asym(
            None,
            by=by,
            loc=loc,
            low=low,
            high=high,
            relative=relative,
            return_index=return_index,
        )

        assert subset is None


@pytest.mark.parametrize(
    "by,loc,low,high,relative,return_index",
    [
        (
            ["mz", "drift_time", "retention_time"],
            [212.0, 17.2, 4.79],
            [0.1, 0.03, 0.3],
            [-1, -0.03, -0.3],
            [False, True, False],
            False,
        )
    ],
)
def test_locate_asym_return_none(ms1, by, loc, low, high, relative, return_index):
    if return_index is True:
        subset, idx = deimos.locate_asym(
            None,
            by=by,
            loc=loc,
            low=low,
            high=high,
            relative=relative,
            return_index=return_index,
        )
        assert subset is None
        assert idx is not None
        assert len(idx) == len(ms1.index)
        assert all(idx) is False
    else:
        subset = deimos.locate_asym(
            None,
            by=by,
            loc=loc,
            low=low,
            high=high,
            relative=relative,
            return_index=return_index,
        )

        assert subset is None


class TestPartitions:
    @pytest.mark.parametrize(
        "split_on,size,overlap", [("mz", 1000, 0.05), ("mz", 2000, 0.5)]
    )
    def test_init(self, ms1, split_on, size, overlap):
        partitions = deimos.partition(
            ms1, split_on=split_on, size=size, overlap=overlap
        )

        assert partitions.features.equals(ms1)
        assert partitions.split_on == split_on
        assert partitions.size == size

    @pytest.mark.parametrize(
        "split_on,size,overlap", [("mz", 1000, 0.05), ("mz", 2000, 0.5)]
    )
    def test__compute_splits(self, ms1, split_on, size, overlap):
        partitions = deimos.partition(
            ms1, split_on=split_on, size=size, overlap=overlap
        )

        idx_unq = np.unique(ms1[split_on].values)

        assert len(partitions.bounds) == np.ceil(len(idx_unq) / size)
        assert len(partitions.fbounds) == np.ceil(len(idx_unq) / size)

        for i, (b, fb) in enumerate(zip(partitions.bounds, partitions.fbounds)):
            # first partition
            if i < 1:
                assert b[0] == fb[0]
                assert 2 * (b[1] - fb[1]) - overlap < 1e-3

            # middle partitions
            elif i < len(partitions.bounds) - 1:
                assert abs(2 * (fb[0] - b[0]) - overlap) < 1e-3
                assert abs(2 * (b[1] - fb[1]) - overlap) < 1e-3

            # last partition
            else:
                assert abs(2 * (fb[0] - b[0]) - overlap) < 1e-3
                assert b[1] == fb[1]

    def test_iter(self, ms1):
        partitions = deimos.partition(ms1, split_on="mz", size=2000, overlap=1)
        i = -1
        for i, part in enumerate(partitions):
            assert type(part) is pd.DataFrame
            assert all(part.columns == ms1.columns)

            lb, ub = partitions.bounds[i]

            assert part["mz"].min() >= lb
            assert part["mz"].max() <= ub

        assert i == 82

    @pytest.mark.parametrize("processes", [(1), (2)])
    def test_map(self, ms1, processes):
        partitions = deimos.partition(ms1, split_on="mz", size=2000, overlap=1)
        pres = partitions.map(
            deimos.threshold, processes=processes, by="intensity", threshold=1e3
        )
        pres = pres.sort_values(by=["mz", "drift_time", "retention_time"]).reset_index(
            drop=True
        )

        res = deimos.threshold(ms1, by="intensity", threshold=1e3)
        res = res.sort_values(by=["mz", "drift_time", "retention_time"]).reset_index(
            drop=True
        )

        assert pres.equals(res)

    @pytest.mark.parametrize("processes", [(1), (2)])
    def test_zipmap(self, ms1, ms2, processes):
        ms1 = deimos.threshold(ms1, by="intensity", threshold=1e4)
        ms2 = deimos.threshold(ms2, by="intensity", threshold=1e4)

        partitions = deimos.partition(ms1, split_on="mz", size=2000, overlap=1)

        pres_a, pres_b = partitions.zipmap(
            deimos.alignment.tolerance,
            ms2,
            processes=processes,
            dims=["mz", "drift_time", "retention_time"],
            tol=[5e-6, 0.025, 0.3],
            relative=[True, True, False],
        )

        res_a, res_b = deimos.alignment.tolerance(
            ms1,
            ms2,
            dims=["mz", "drift_time", "retention_time"],
            tol=[5e-6, 0.025, 0.3],
            relative=[True, True, False],
        )

        assert pres_a.equals(res_a)
        assert pres_b.equals(res_b)


@pytest.mark.parametrize(
    "split_on,size,overlap", [("mz", 1000, 0.05), ("mz", 2000, 0.5)]
)
def test_partition(ms1, split_on, size, overlap):
    partitions = deimos.partition(ms1, split_on=split_on, size=size, overlap=overlap)

    assert type(partitions) is deimos.subset.Partitions


class TestMultiSamplePartitions:
    def test_init(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test__compute_splits(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_iter(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_next(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError

    def test_map(self):
        with pytest.raises(NotImplementedError):
            raise NotImplementedError


def test_multi_sample_partition():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


# Tests for batch slice functionality
def test_slice_batch_mode(ms1):
    """Test slice function in batch mode with multiple slices."""
    # Define multiple slices
    low_bounds = [[211.0, 16.2, 4.00], [300.0, 20.0, 5.0], [400.0, 25.0, 6.0]]
    high_bounds = [[213.0, 18.2, 5.16], [302.0, 22.0, 6.0], [402.0, 27.0, 7.0]]
    by = ["mz", "drift_time", "retention_time"]

    results = deimos.slice(ms1, by=by, low=low_bounds, high=high_bounds)

    # Should return list of DataFrames
    assert isinstance(results, list)
    assert len(results) == 3

    # Check first result (should have data)
    assert results[0] is not None
    assert all(results[0].columns == ms1.columns)
    for i, dim in enumerate(by):
        assert results[0][dim].min() >= low_bounds[0][i]
        assert results[0][dim].max() <= high_bounds[0][i]


def test_slice_batch_mode_with_return_index(ms1):
    """Test slice function in batch mode with return_index=True."""
    low_bounds = [[211.0, 16.2], [300.0, 20.0]]
    high_bounds = [[213.0, 18.2], [302.0, 22.0]]
    by = ["mz", "drift_time"]

    results = deimos.slice(
        ms1, by=by, low=low_bounds, high=high_bounds, return_index=True
    )

    # Should return list of (DataFrame, index) tuples
    assert isinstance(results, list)
    assert len(results) == 2

    for subset, idx in results:
        if subset is not None:
            assert isinstance(subset, pd.DataFrame)
            assert isinstance(idx, np.ndarray)
            assert len(idx) == len(ms1.index)
            assert idx.dtype == np.bool_


def test_slice_batch_vs_single_consistency(ms1):
    """Verify batch mode gives same results as individual slices."""
    low_bounds = [[211.0, 16.2], [300.0, 20.0]]
    high_bounds = [[213.0, 18.2], [302.0, 22.0]]
    by = ["mz", "drift_time"]

    # Batch mode
    batch_results = deimos.slice(ms1, by=by, low=low_bounds, high=high_bounds)

    # Individual slices
    single_results = [
        deimos.slice(ms1, by=by, low=low_bounds[0], high=high_bounds[0]),
        deimos.slice(ms1, by=by, low=low_bounds[1], high=high_bounds[1]),
    ]

    # Compare results
    for batch, single in zip(batch_results, single_results):
        if batch is None and single is None:
            continue
        assert batch is not None and single is not None
        pd.testing.assert_frame_equal(
            batch.reset_index(drop=True), single.reset_index(drop=True)
        )


@pytest.fixture()
def mz_sorted():
    # Float32 m/z beside float64 columns, so bounds are compared in float64
    return pd.DataFrame(
        {
            "mz": np.array([100.0, 100.5, 101.0, 101.5, 102.0], dtype=np.float32),
            "retention_time": [1.0, 2.0, 3.0, 4.0, 5.0],
            "intensity": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=pd.Index([14, 3, 9, 0, 7]),
    )


@pytest.mark.parametrize(
    "order,index,mask",
    [
        ([0, 1, 2, 3, 4], [3, 9, 0], [False, True, True, True, False]),
        ([3, 0, 4, 2, 1], [0, 9, 3], [True, False, False, True, True]),
    ],
    ids=["sorted", "unsorted"],
)
def test_slice_includes_bounds_and_keeps_index(mz_sorted, order, index, mask):
    features = mz_sorted.iloc[order]

    result = deimos.slice(features, by="mz", low=100.5, high=101.5, return_index=True)

    assert isinstance(result, tuple)
    subset, idx = result
    assert subset is not None and idx is not None
    assert subset.index.tolist() == index
    assert idx.tolist() == mask


@pytest.mark.parametrize("high,length", [(100.0999984, 1), (100.0999985, 2)])
def test_slice_sorted_float32_compares_exact_bounds(high, length):
    # Float32 100.1 is 100.09999847..., between the two upper bounds
    features = pd.DataFrame(
        {
            "mz": np.array([100.0, 100.1, 100.2], dtype=np.float32),
            "intensity": [1.0, 2.0, 3.0],
        }
    )

    subset = deimos.slice(features, by="mz", low=99.0, high=high)

    assert isinstance(subset, pd.DataFrame)
    assert len(subset.index) == length


def test_locate_asym_relative_tolerance_on_sorted_mz():
    features = pd.DataFrame(
        {
            "mz": [499.99, 499.996, 500.0, 500.004, 500.01],
            "intensity": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    subset = deimos.locate_asym(
        features, by="mz", loc=500.0, low=-10e-6, high=10e-6, relative=True
    )

    assert isinstance(subset, pd.DataFrame)
    assert subset["mz"].tolist() == [499.996, 500.0, 500.004]


@pytest.mark.parametrize(
    "low,high", [(100.6, 100.9), (50.0, 60.0), (110.0, 120.0), (101.0, 100.0)]
)
def test_slice_sorted_empty_window(mz_sorted, low, high):
    result = deimos.slice(mz_sorted, by="mz", low=low, high=high, return_index=True)

    assert isinstance(result, tuple)
    subset, idx = result
    assert subset is None
    assert idx is not None
    assert idx.tolist() == [False] * 5


@pytest.mark.filterwarnings("error")
def test_slice_sorted_bounds_beyond_float32_range(mz_sorted):
    subset = deimos.slice(mz_sorted, by="mz", low=1e300, high=1e301)

    assert subset is None


def test_slice_skips_window_when_mz_has_nan():
    features = pd.DataFrame(
        {
            "mz": np.array([100.0, np.nan, 101.0], dtype=np.float32),
            "intensity": [1.0, 2.0, 3.0],
        }
    )

    subset = deimos.slice(features, by="mz", low=99.0, high=102.0)

    assert isinstance(subset, pd.DataFrame)
    assert subset.index.tolist() == [0, 2]


@pytest.mark.parametrize("repeats", [1, 4], ids=["3-slices", "12-slices"])
@pytest.mark.parametrize(
    "order,expected",
    [
        ([0, 1, 2, 3, 4], [[3, 9], [3, 9, 0], None]),
        ([3, 0, 4, 2, 1], [[9, 3], [0, 9, 3], None]),
    ],
    ids=["sorted", "unsorted"],
)
def test_slice_batch_windows_keep_original_order(
    mz_sorted, order, expected, repeats: int
):
    # Twelve slices are enough to sort unsorted input once instead of scanning
    features = mz_sorted.iloc[order]
    low = [[100.5, 0.0], [100.0, 1.5], [102.5, 0.0]] * repeats
    high = [[101.0, 10.0], [102.0, 4.5], [103.0, 10.0]] * repeats

    results = deimos.slice(
        features, by=["mz", "retention_time"], low=low, high=high, return_index=True
    )

    assert isinstance(results, list)
    for (subset, idx), index in zip(results, expected * repeats):
        if index is None:
            assert subset is None
        else:
            assert subset.index.tolist() == index
        assert features.index[idx].tolist() == (index or [])


@pytest.fixture()
def two_ions():
    # Ions at m/z 100 and 101 across three retention times, plus one at 100.5
    return pd.DataFrame(
        {
            "mz": [100.0, 100.0, 100.5, 101.0, 100.0, 101.0],
            "retention_time": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            "intensity": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        }
    )


@pytest.mark.parametrize("repeats", [1, 4], ids=["3-slices", "12-slices"])
@pytest.mark.parametrize("ordered", [False, True], ids=["rt-order", "mz-sorted"])
@pytest.mark.parametrize(
    "how,intensity", [("sum", [3.0, 16.0, 8.0, 32.0]), ("max", [2.0, 16.0, 8.0, 32.0])]
)
def test_collapse_slices_gives_one_profile_per_slice(
    two_ions, ordered, repeats, how, intensity
):
    features = two_ions.sort_values("mz", kind="stable") if ordered else two_ions
    low = [99.9, 100.9, 200.0] * repeats
    high = [100.1, 101.1, 201.0] * repeats

    xics = deimos.collapse_slices(
        features, by="mz", low=low, high=high, keep="retention_time", how=how
    )

    assert isinstance(xics, pd.DataFrame)
    assert xics.columns.tolist() == ["slice_idx", "retention_time", "intensity"]
    for k in range(repeats):
        part = xics[xics["slice_idx"].isin([3 * k, 3 * k + 1, 3 * k + 2])]
        assert (part["slice_idx"] - 3 * k).tolist() == [0, 0, 1, 1]
        assert part["retention_time"].tolist() == [1.0, 3.0, 2.0, 3.0]
        assert part["intensity"].tolist() == intensity


def test_collapse_slices_bounds_in_two_dimensions(two_ions):
    xics = deimos.collapse_slices(
        two_ions,
        by=["mz", "retention_time"],
        low=[[99.9, 2.0], [100.4, 0.0]],
        high=[[101.1, 3.0], [100.6, 1.0]],
        keep="mz",
    )

    assert isinstance(xics, pd.DataFrame)
    assert xics["slice_idx"].tolist() == [0, 0, 0]
    assert xics["mz"].tolist() == [100.5, 101.0, 100.0]
    assert xics["intensity"].tolist() == [4.0, 40.0, 16.0]


def test_collapse_slices_rejects_mismatched_bounds(two_ions):
    with pytest.raises(ValueError):
        deimos.collapse_slices(
            two_ions,
            by=["mz", "retention_time"],
            low=[[99.9, 0.0]],
            high=[[100.1, 1.0], [101.1, 1.0]],
            keep="retention_time",
        )


def test_collapse_slices_pass_none():
    xics = deimos.collapse_slices(
        None, by="mz", low=[99.9], high=[100.1], keep="retention_time"
    )

    assert xics is None
