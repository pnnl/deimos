import multiprocessing as mp
from collections.abc import Sequence
from functools import partial
from typing import Any, Literal, cast, overload

import dask.dataframe as dd
import numpy as np
import pandas as pd

import deimos


def threshold(features, by="intensity", threshold: float = 0):
    """
    Thresholds input :obj:`~pandas.DataFrame` using `by` keyword, greater than
    value passed to `threshold`.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : str
        Variable to threshold by.
    threshold : float
        Threshold value.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Thresholded feature coordinates.

    """

    return features.loc[features[by] > threshold, :].reset_index(drop=True)


def collapse(features, keep=None, how="sum") -> pd.DataFrame:
    """
    Collpases input data such that only specified dimensions remain, according
    to the supplied aggregation function.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    keep : str or list, optional
        Dimensions to keep during collapse operation.
        Default is ``["mz", "drift_time", "retention_time"]``.
    how : function or str
        Aggregation function for collapse operation.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Collapsed feature coordinates and aggregated
        intensities.

    """

    # Set mutable argument defaults
    if keep is None:
        keep = ["mz", "drift_time", "retention_time"]

    return cast(
        pd.DataFrame,
        features.groupby(by=keep, as_index=False, sort=False).agg({"intensity": how}),
    )


@overload
def locate(
    features, by=None, loc=None, tol=None, return_index: Literal[False] = False
) -> pd.DataFrame | None: ...
@overload
def locate(
    features, by=None, loc=None, tol=None, *, return_index: Literal[True]
) -> tuple[pd.DataFrame | None, np.ndarray | None]: ...
@overload
def locate(
    features, by=None, loc=None, tol=None, return_index: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame | None, np.ndarray | None] | None: ...
def locate(
    features,
    by=None,
    loc=None,
    tol=None,
    return_index=False,
) -> pd.DataFrame | tuple[pd.DataFrame | None, np.ndarray | None] | None:
    """
    Given a coordinate and tolerances, return a subset of the
    data. If `features` is sorted by the first dimension in `by`, only
    the rows within its tolerance are scanned.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : str or list, optional
        Dimension(s) by which to subset the data.
        Default is ``["mz", "drift_time", "retention_time"]``.
    loc : float or list, optional
        Coordinate location.
        Default is ``[0, 0, 0]``.
    tol : float or list, optional
        Tolerance in each dimension.
        Default is ``[0, 0, 0]``.
    return_index : bool
        Return boolean index of subset if True.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Subset of feature coordinates and intensities.
    :obj:`~numpy.array`
        If `return_index` is True, boolean index of subset elements,
        i.e. `features[index] = subset`.

    """

    # Set mutable argument defaults
    if by is None:
        by = ["mz", "drift_time", "retention_time"]
    if loc is None:
        loc = [0, 0, 0]
    if tol is None:
        tol = [0, 0, 0]

    # Safely cast to list
    by = deimos.utils.safelist(by)
    loc = deimos.utils.safelist(loc)
    tol = deimos.utils.safelist(tol)

    # Check dims
    deimos.utils.check_length([by, loc, tol])

    # Slice by the bounds the tolerances imply
    low = [x - dx for x, dx in zip(loc, tol)]
    high = [x + dx for x, dx in zip(loc, tol)]
    return _slice_single(features, by, low, high, return_index)


@overload
def locate_asym(
    features,
    by=None,
    loc=None,
    low=None,
    high=None,
    relative=None,
    return_index: Literal[False] = False,
) -> pd.DataFrame | None: ...
@overload
def locate_asym(
    features,
    by=None,
    loc=None,
    low=None,
    high=None,
    relative=None,
    *,
    return_index: Literal[True],
) -> tuple[pd.DataFrame | None, np.ndarray | None]: ...
@overload
def locate_asym(
    features,
    by=None,
    loc=None,
    low=None,
    high=None,
    relative=None,
    return_index: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame | None, np.ndarray | None] | None: ...
def locate_asym(
    features,
    by=None,
    loc=None,
    low=None,
    high=None,
    relative=None,
    return_index=False,
):
    """
    Given a coordinate and asymmetrical tolerances, return a subset of the
    data. If `features` is sorted by the first dimension in `by`, only
    the rows within its tolerance are scanned.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : str or list, optional
        Dimension(s) by which to subset the data.
        Default is ``["mz", "drift_time", "retention_time"]``.
    loc : float or list, optional
        Coordinate location.
        Default is ``[0, 0, 0]``.
    low : float or list, optional
        Lower tolerance(s) in each dimension.
        Default is ``[0, 0, 0]``.
    high : float or list, optional
        Upper tolerance(s) in each dimension.
        Default is ``[0, 0, 0]``.
    relative : bool or list, optional
        Whether to use relative or absolute tolerance per dimension.
        Default is ``[False, False, False]``.
    return_index : bool
        Return boolean index of subset if True.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Subset of feature coordinates and intensities.
    :obj:`~numpy.array`
        If `return_index` is True, boolean index of subset elements,
        i.e. `features[index] = subset`.

    """

    # Set mutable argument defaults
    if by is None:
        by = ["mz", "drift_time", "retention_time"]
    if loc is None:
        loc = [0, 0, 0]
    if low is None:
        low = [0, 0, 0]
    if high is None:
        high = [0, 0, 0]
    if relative is None:
        relative = [False, False, False]

    # Safely cast to list
    by = deimos.utils.safelist(by)
    loc = deimos.utils.safelist(loc)
    low = deimos.utils.safelist(low)
    high = deimos.utils.safelist(high)
    relative = deimos.utils.safelist(relative)

    # Check dims
    deimos.utils.check_length([by, loc, low, high, relative])

    # Determine bounds
    lb = []
    ub = []
    for x, lower, upper, rel in zip(loc, low, high, relative):
        if rel is True:
            lb.append(x * (1 + lower))
            ub.append(x * (1 + upper))
        else:
            lb.append(x + lower)
            ub.append(x + upper)

    return deimos.subset.slice(
        features, by=by, low=lb, high=ub, return_index=return_index
    )


# Scalar or 1D bounds produce a single slice; 2D array bounds produce a batch
@overload
def slice(
    features,
    by=None,
    low: float | Sequence[float] | None = None,
    high: float | Sequence[float] | None = None,
    return_index: Literal[False] = False,
) -> pd.DataFrame | None: ...
@overload
def slice(
    features,
    by=None,
    low: float | Sequence[float] | None = None,
    high: float | Sequence[float] | None = None,
    *,
    return_index: Literal[True],
) -> tuple[pd.DataFrame | None, np.ndarray | None]: ...
@overload
def slice(
    features, by=None, low=None, high=None, return_index: bool = False
) -> Any: ...
def slice(
    features,
    by=None,
    low=None,
    high=None,
    return_index=False,
):
    """
    Slice data by feature coordinate bounds. Efficiently handles both single and multiple slices.

    For single slices, returns a DataFrame. For multiple slices (when low/high are 2D arrays),
    returns a list of DataFrames. Uses vectorized operations for efficiency when processing
    multiple slices.

    If `features` is sorted by the first dimension in `by`, only the rows within
    that dimension's bounds are scanned, so sort once before slicing many times.
    Unsorted data is sorted internally when a batch has enough slices to
    benefit.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : str or list, optional
        Dimension(s) by which to subset the data.
        Default is ``["mz", "drift_time", "retention_time"]``.
    low : float, list, or array-like, optional
        Lower bound(s) in each dimension. For single slice: 1D array matching dimensions in `by`.
        For multiple slices: 2D array of shape (n_slices, n_dims) where each row defines the
        lower bounds for one slice.
        Default is ``[0, 0, 0]``.
    high : float, list, or array-like, optional
        Upper bound(s) in each dimension. For single slice: 1D array matching dimensions in `by`.
        For multiple slices: 2D array of shape (n_slices, n_dims) where each row defines the
        upper bounds for one slice.
        Default is ``[0, 0, 0]``.
    return_index : bool
        Return boolean index of subset if True. For multiple slices, returns list of indices.

    Returns
    -------
    :obj:`~pandas.DataFrame` or list of :obj:`~pandas.DataFrame`
        Subset(s) of feature coordinates and intensities. Returns single DataFrame for single slice,
        list of DataFrames for multiple slices. Returns None (or list containing None) when no
        data matches the bounds.
    :obj:`~numpy.array` or list of :obj:`~numpy.array`
        If `return_index` is True, boolean index of subset elements where `features[index] = subset`.
        For multiple slices, returns list of boolean arrays.

    Examples
    --------
    Single slice:

    >>> subset = slice(ms1, by='mz', low=211.0, high=213.0)

    Multiple slices (batch mode):

    >>> low_bounds = [[211.0, 16.2], [300.0, 20.0], [400.0, 25.0]]
    >>> high_bounds = [[213.0, 18.2], [302.0, 22.0], [402.0, 27.0]]
    >>> results = slice(ms1, by=['mz', 'drift_time'], low=low_bounds, high=high_bounds)
    >>> # results[0] contains data for first slice, results[1] for second, etc.

    """

    # Set mutable argument defaults
    if by is None:
        by = ["mz", "drift_time", "retention_time"]
    if low is None:
        low = [0, 0, 0]
    if high is None:
        high = [0, 0, 0]

    # Safely cast to list
    by = deimos.utils.safelist(by)
    low = deimos.utils.safelist(low)
    high = deimos.utils.safelist(high)

    # Convert to numpy arrays to check dimensions
    low_arr = np.asarray(low)
    high_arr = np.asarray(high)

    # Detect batch mode: if low/high are 2D arrays
    is_batch = low_arr.ndim == 2 or high_arr.ndim == 2

    if is_batch:
        # Batch mode: multiple slices
        return _slice_batch(features, by, low_arr, high_arr, return_index)
    else:
        # Single slice mode: original behavior
        return _slice_single(features, by, low, high, return_index)


def _slice_single(features, by, low, high, return_index):
    """
    Internal function for single slice operation (original slice behavior).
    """
    # Check dims
    deimos.utils.check_length([by, low, high])

    if features is None:
        if return_index is True:
            return None, None
        else:
            return None

    # Extend columns
    cols = features.columns
    cidx = [cols.get_loc(x) for x in by]

    # Scan only the rows a sorted first dimension allows
    start, stop = _sorted_rows(features, by, cidx, low, high)
    window = features if stop is None else features.iloc[start:stop]

    # Store index
    rindex = window.index.values

    # Subset by each dim
    features_vals = window.values
    idx = np.full(features_vals.shape[0], True, dtype=bool)
    for i, lb, ub in zip(cidx, low, high):
        idx *= (features_vals[:, i] <= ub) & (features_vals[:, i] >= lb)

    features_vals = features_vals[idx]
    rindex = rindex[idx]

    if return_index is True:
        # Map the boolean index from scanned rows to all rows
        index = np.full(len(features.index), False, dtype=bool)
        index[start:stop] = idx

        # Data found
        if features_vals.shape[0] > 0:
            return pd.DataFrame(features_vals, index=rindex, columns=cols), index

        # No data
        return None, index
    else:
        # Data found
        if features_vals.shape[0] > 0:
            return pd.DataFrame(features_vals, index=rindex, columns=cols)

        # No data
        return None


def _slice_batch(features, by, low_bounds, high_bounds, return_index):
    """
    Internal function for batch slice operation. When the data is sorted by
    the first dimension, or the batch is large enough to sort it once, each
    slice scans only the rows within its first-dimension bounds.
    """
    # Ensure 2D arrays
    if low_bounds.ndim == 1:
        low_bounds = low_bounds.reshape(1, -1)
    if high_bounds.ndim == 1:
        high_bounds = high_bounds.reshape(1, -1)

    n_slices = low_bounds.shape[0]
    n_dims = len(by)

    if low_bounds.shape[1] != n_dims or high_bounds.shape[1] != n_dims:
        raise ValueError(
            f"Bound dimensions ({low_bounds.shape[1]}, {high_bounds.shape[1]}) "
            f"must match 'by' dimensions ({n_dims})"
        )

    if low_bounds.shape[0] != high_bounds.shape[0]:
        raise ValueError(
            f"Number of low bounds ({low_bounds.shape[0]}) must match "
            f"number of high bounds ({high_bounds.shape[0]})"
        )

    if features is None:
        if return_index:
            return [(None, None)] * n_slices
        else:
            return [None] * n_slices

    # Build output dataframes from the matching rows of each slice
    output = []
    for rows in _batch_rows(features, by, low_bounds, high_bounds):
        subset = features.iloc[rows] if len(rows) > 0 else None
        if return_index:
            index = np.full(len(features.index), False, dtype=np.bool_)
            index[rows] = True
            output.append((subset, index))
        else:
            output.append(subset)

    return output


def _batch_rows(features, by, low_bounds, high_bounds):
    """
    Find the rows within each slice of a batch.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : list
        Dimensions by which to subset the data.
    low_bounds : :obj:`~numpy.array`
        Lower bounds of shape (n_slices, n_dims).
    high_bounds : :obj:`~numpy.array`
        Upper bounds of shape (n_slices, n_dims).

    Returns
    -------
    list of :obj:`~numpy.array`
        Positions of the rows within each slice, in their original order.

    """

    # Get column indices
    cols = features.columns
    cidx = np.array([cols.get_loc(x) for x in by])

    # Extract relevant columns as contiguous numpy array
    data_subset = np.ascontiguousarray(features.iloc[:, cidx].values)

    # Bound each slice to a run of rows ordered by the first dimension
    starts, stops, order = _batch_windows(data_subset, low_bounds, high_bounds)

    output = []
    for slice_idx in range(len(low_bounds)):
        start, stop = starts[slice_idx], stops[slice_idx]

        # Candidate rows: a contiguous run, or sorted positions taken from the order
        positions = None if order is None else np.sort(order[start:stop])
        if positions is None:
            candidates = data_subset[start:stop]
        else:
            candidates = data_subset[positions]

        # Apply bounds for each dimension
        mask = np.ones(len(candidates), dtype=np.bool_)
        for dim_idx in range(len(by)):
            mask &= (candidates[:, dim_idx] >= low_bounds[slice_idx, dim_idx]) & (
                candidates[:, dim_idx] <= high_bounds[slice_idx, dim_idx]
            )

        # Matching rows in their original order
        if positions is None:
            output.append(start + np.flatnonzero(mask))
        else:
            output.append(positions[mask])

    return output


def _is_sorted(values):
    """
    Check whether values are floating point and in non-decreasing order,
    which also rules out NaN. Compares in chunks of doubling size so unsorted
    input returns early.

    Parameters
    ----------
    values : :obj:`~numpy.array`
        Values to check.

    Returns
    -------
    bool
        True if `values` is floating point and sorted.

    """

    if values.dtype.kind != "f":
        return False

    start, size = 0, 1024
    while start < len(values) - 1:
        part = values[start : start + size + 1]
        if not np.all(part[1:] >= part[:-1]):
            return False
        start += size
        size *= 2

    return True


def _sorted_window(values, low, high):
    """
    Locate the run of sorted values that can fall within inclusive bounds.

    Bounds are cast to the dtype of `values` so the values are never copied,
    then widened by one unit in the last place to absorb that rounding. The
    run therefore holds every value within the bounds and possibly a few
    beyond them, which callers filter exactly.

    Parameters
    ----------
    values : :obj:`~numpy.array`
        Floating-point values in non-decreasing order.
    low : float or :obj:`~numpy.array`
        Lower bound(s).
    high : float or :obj:`~numpy.array`
        Upper bound(s).

    Returns
    -------
    start, stop : int or :obj:`~numpy.array`
        First position and one past the last position of each run.

    """

    # Bounds beyond the dtype's range become inf, which still bounds correctly
    with np.errstate(over="ignore"):
        low = np.nextafter(np.asarray(low, dtype=values.dtype), -np.inf)
        high = np.nextafter(np.asarray(high, dtype=values.dtype), np.inf)

    return (
        np.searchsorted(values, low, side="left"),
        np.searchsorted(values, high, side="right"),
    )


def _sorted_rows(features, by, cidx, low, high):
    """
    Bound the rows of a single slice using the first dimension, when its
    column is floating point and sorted.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : list
        Dimensions by which to subset the data.
    cidx : list
        Column positions of the slice dimensions.
    low : list
        Lower bound in each dimension.
    high : list
        Upper bound in each dimension.

    Returns
    -------
    start, stop : int or None
        Row positions bounding the slice, or None for both when every row
        must be scanned.

    """

    # Window only on a sorted floating-point column of a pandas data frame
    if (
        not isinstance(features, pd.DataFrame)
        or len(cidx) == 0
        or not isinstance(cidx[0], (int, np.integer))
    ):
        return None, None

    values = features[by[0]].to_numpy()
    if not _is_sorted(values):
        return None, None

    # Only plain NumPy columns convert to the same values in part and in whole
    if not all(isinstance(x, np.dtype) for x in features.dtypes):
        return None, None

    start, stop = _sorted_window(values, low[0], high[0])
    return int(start), int(stop)


def _batch_windows(data, low_bounds, high_bounds):
    """
    Bound each slice of a batch to a run of rows ordered by the first
    dimension, sorting the rows once when that pays for itself.

    Parameters
    ----------
    data : :obj:`~numpy.array`
        Coordinates of every row, one column per dimension.
    low_bounds : :obj:`~numpy.array`
        Lower bounds of shape (n_slices, n_dims).
    high_bounds : :obj:`~numpy.array`
        Upper bounds of shape (n_slices, n_dims).

    Returns
    -------
    starts, stops : :obj:`~numpy.array`
        First position and one past the last position of each run in the
        row order.
    order : :obj:`~numpy.array` or None
        Row positions in sorted order, or None when the rows are already
        sorted or every row must be scanned.

    """

    n_rows, n_dims = data.shape
    n_slices = len(low_bounds)
    scan_all = np.zeros(n_slices, dtype=int), np.full(n_slices, n_rows), None

    # Without dimensions every row is in every slice
    if n_dims == 0:
        return scan_all

    # Already sorted
    values = data[:, 0]
    if _is_sorted(values):
        starts, stops = _sorted_window(values, low_bounds[:, 0], high_bounds[:, 0])
        return starts, stops, None

    # Scan every row unless there are enough slices for one sort to cost less;
    # on 2e4 to 2e6 rows a sort costs 7 to 70 scans, below 4 * log2(n_rows)
    if values.dtype.kind != "f" or n_slices <= 4 * np.log2(max(n_rows, 2)):
        return scan_all

    order = np.argsort(values, kind="stable")
    starts, stops = _sorted_window(values[order], low_bounds[:, 0], high_bounds[:, 0])
    return starts, stops, order


class Partitions:
    """
    Generator object that will lazily build and return each partition.

    Attributes
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    split_on : str
        Dimension to partition the data.
    size : int
        Target partition size.
    overlap : float
        Amount of overlap between partitions to ameliorate edge effects.

    """

    def __init__(self, features, split_on="mz", size=1000, overlap=0.05):
        """
        Initialize :obj:`~deimos.subset.Partitions` instance.

        Parameters
        ----------
        features : :obj:`~pandas.DataFrame`
            Input feature coordinates and intensities.
        split_on : str
            Dimension to partition the data.
        size : int
            Target partition size.
        overlap : float
            Amount of overlap between partitions to ameliorate edge effects.

        """

        self.features = features
        self.split_on = split_on
        self.size = size
        self.overlap = overlap

        self._compute_splits()

    def _compute_splits(self):
        """
        Determines data splits for partitioning.

        """

        # Unique to split on
        idx = np.unique(self.features[self.split_on].values)

        # Number of partitions
        partitions = np.ceil(len(idx) / self.size)

        # Determine partition bounds
        bounds = [[x.min(), x.max()] for x in np.array_split(idx, partitions)]
        for i in range(1, len(bounds)):
            bounds[i][0] = bounds[i - 1][1] - self.overlap

        if (self.overlap > 0) & (len(bounds) > 1):
            # Functional bounds
            fbounds = []
            for i in range(len(bounds)):
                a, b = bounds[i]

                # First partition
                if i < 1:
                    b = b - self.overlap / 2

                # Middle partitions
                elif i < len(bounds) - 1:
                    a = a + self.overlap / 2
                    b = b - self.overlap / 2

                # Last partition
                else:
                    a = a + self.overlap / 2

                fbounds.append([a, b])
        else:
            fbounds = bounds

        self.bounds = bounds
        self.fbounds = fbounds

    def __iter__(self):
        """
        Yields each partition.

        Yields
        ------
        :obj:`~pandas.DataFrame`
            Partition of feature coordinates and intensities.

        """

        for a, b in self.bounds:
            yield slice(self.features, by=self.split_on, low=a, high=b)

    def map(self, func, processes=1, **kwargs):
        """
        Maps `func` to each partition, then returns the combined result,
        accounting for overlap regions.

        Parameters
        ----------
        func : function
            Function to apply to partitions.
        processes : int
            Number of parallel processes. If less than 2, a serial mapping is
            applied.
        kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            Combined result of `func` applied to partitions.

        """

        # Serial
        if processes < 2:
            result = [func(x, **kwargs) for x in self]

        # Parallel
        else:
            with mp.Pool(processes=processes) as p:
                result = list(p.imap(partial(func, **kwargs), self))

        # Reconcile overlap; scalar bounds always produce a single slice
        result = [
            cast(
                "pd.DataFrame | None", slice(result[i], by=self.split_on, low=a, high=b)
            )
            for i, (a, b) in enumerate(self.fbounds)
        ]

        # Combine non-empty partitions
        return pd.concat([x for x in result if x is not None]).reset_index(drop=True)

    def zipmap(self, func, b, processes=1, **kwargs):
        """
        Maps `func` to each partition pair resulting from the zip operation of
        `self` and `b`, then returns the combined result, accounting for
        overlap regions.

        Parameters
        ----------
        func : function
            Function to apply to zipped partitions. Must accept and return two
            :obj:`~pandas.DataFrame` instances.
        b : :obj:`~pandas.DataFrame`
            Input feature coordinates and intensities.
        processes : int
            Number of parallel processes. If less than 2, a serial mapping is
            applied.
        kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
        a, b : :obj:`~pandas.DataFrame`
            Result of `func` applied to paired partitions.

        """

        # Partition other dataset
        partitions = (
            slice(b, by=self.split_on, low=a, high=b_) for a, b_ in self.bounds
        )

        # Serial
        if processes < 2:
            result = [func(a, b_, **kwargs) for a, b_ in zip(self, partitions)]

        # Parallel
        else:
            with mp.Pool(processes=processes) as p:
                result = list(p.starmap(partial(func, **kwargs), zip(self, partitions)))

        parts_a = [x[0] for x in result]
        parts_b = [x[1] for x in result]

        # Reconcile overlap; scalar bounds always produce a single slice
        tmp = [
            cast(
                "tuple[pd.DataFrame | None, np.ndarray | None]",
                slice(parts_a[i], by=self.split_on, low=a, high=b_, return_index=True),
            )
            for i, (a, b_) in enumerate(self.fbounds)
        ]

        parts_a = [x[0] for x in tmp]
        idx = [x[1] for x in tmp]
        parts_b = [p.iloc[i, :] for p, i in zip(parts_b, idx) if i is not None]

        # Combine non-empty partitions
        return (
            pd.concat([x for x in parts_a if x is not None]),
            pd.concat(parts_b),
        )


class MultiSamplePartitions:
    """
    Generator object that will lazily build and return each partition constructed
    from multiple samples.

    Attributes
    ----------
    features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
        Input feature coordinates and intensities.
    split_on : str
        Dimension to partition the data.
    size : int
        Target partition size.
    tol : float
        Largest allowed distance between unique `split_on` observations.

    """

    def __init__(self, features, split_on="mz", size=500, tol=25e-6):
        """
        Initialize :obj:`~deimos.subset.Partitions` instance.

        Parameters
        ----------
        features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
            Input feature coordinates and intensities.
        split_on : str
            Dimension to partition the data.
        size : int
            Target partition size.
        tol : float
            Largest allowed distance between unique `split_on` observations.

        """

        self.features = features
        self.split_on = split_on
        self.size = size
        self.tol = tol

        if isinstance(features, dd.DataFrame):
            self.dask = True
        else:
            self.dask = False

        self._compute_splits()

    def _compute_splits(self):
        """
        Determines data splits for partitioning.

        """

        self.counter = 0

        if self.dask:
            idx = self.features.groupby(by=self.split_on).size().compute().sort_index()
        else:
            idx = self.features.groupby(by=self.split_on).size().sort_index()

        counts = idx.values
        idx = idx.index

        dxs = np.diff(idx) / idx[:-1]

        bins = []
        current_count = counts[0]
        current_bin = [idx[0]]
        self._counts = []

        for i, dx in zip(range(1, len(idx)), dxs):
            if (current_count + counts[i] <= self.size) or (dx <= self.tol):
                current_bin.append(idx[i])
                current_count += counts[i]

            else:
                bins.append(np.array(current_bin))
                self._counts.append(current_count)

                current_bin = [idx[i]]
                current_count = counts[i]

        # Add last unadded bin
        bins.append(np.array(current_bin))
        self._counts.append(current_count)

        self.bounds = np.array([[x.min(), x.max()] for x in bins])

    def __iter__(self):
        return self

    def __next__(self):
        if self.counter < len(self.bounds):
            low, high = self.bounds[self.counter]
            q = f"({self.split_on} >= {low}) & ({self.split_on} <= {high})"

            if self.dask:
                subset = self.features.query(q).compute()
            else:
                subset = self.features.query(q)

            self.counter += 1
            if len(subset.index) > 1:
                return subset
            else:
                return None

        raise StopIteration

    def map(self, func, processes=1, **kwargs):
        """
        Maps `func` to each partition, then returns the combined result.

        Parameters
        ----------
        func : function
            Function to apply to partitions.
        processes : int
            Number of parallel processes. If less than 2, a serial mapping is
            applied.
        kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            Combined result of `func` applied to partitions.

        """

        # Serial
        if processes < 2:
            result = [func(x, **kwargs) for x in self]

        # Parallel
        else:
            with mp.Pool(processes=processes) as p:
                result = list(p.imap(partial(func, **kwargs), self))

        # Add partition index
        for i in range(len(result)):
            if result[i] is not None:
                result[i]["partition_idx"] = i

        # Combine partitions
        return pd.concat(result, ignore_index=True)


def partition(features, split_on="mz", size=1000, overlap=0.05):
    """
    Partitions data along a given dimension.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    split_on : str
        Dimension to partition the data.
    size : int
        Target partition size.
    overlap : float
        Amount of overlap between partitions to ameliorate edge effects.

    Returns
    -------
    :obj:`~deimos.subset.Partitions`
        A generator object that will lazily build and return each partition.

    """

    return Partitions(features, split_on, size, overlap)


def multi_sample_partition(features, split_on="mz", size=500, tol=25e-6):
    """
    Partitions data along a given dimension. For use with features across
    multiple samples, e.g. in alignment.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
        Input feature coordinates and intensities.
    split_on : str
        Dimension to partition the data.
    size : int
        Target partition size.
    tol : float
        Largest allowed distance between unique `split_on` observations.

    Returns
    -------
    :obj:`~deimos.subset.Partitions`
        A generator object that will lazily build and return each partition.

    """

    return MultiSamplePartitions(features, split_on, size, tol)
