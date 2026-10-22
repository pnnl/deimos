import numpy as np
import scipy
from scipy import sparse
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVR

import deimos


def match(
    a,
    b,
    dims=None,
    tol=None,
    relative=None,
):
    """
    Identify features in `b` within tolerance of those in `a` . Matches are
    bidirectionally one-to-one by highest intensity.

    Parameters
    ----------
    a : :obj:`~pandas.DataFrame`
        First set of input feature coordinates and intensities.
    b : :obj:`~pandas.DataFrame`
        Second set of input feature coordinates and intensities.
    dims : str or list, optional
        Dimensions considered in matching.
        Default is ``["mz", "drift_time", "retention_time"]``.
    tol : float or list, optional
        Tolerance in each dimension to define a match.
        Default is ``[5e-6, 0.015, 0.3]``.
    relative : bool or list, optional
        Whether to use relative or absolute tolerances per dimension.
        Default is ``[True, True, False]``.

    Returns
    -------
    a, b : :obj:`~pandas.DataFrame`
        Features matched within tolerances. E.g., `a[i..n]`and `b[i..n]` each
        represent matched features.

    """

    # Set mutable argument defaults
    if dims is None:
        dims = ["mz", "drift_time", "retention_time"]
    if tol is None:
        tol = [5e-6, 0.015, 0.3]
    if relative is None:
        relative = [True, True, False]

    if a is None or b is None:
        return None, None

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # Check dims
    deimos.utils.check_length([dims, tol, relative])

    # Find feature pairs within tolerance
    ii, jj = _pairs_within_tolerance(a, b, dims, tol, relative)

    if len(ii) < 1:
        return None, None

    # Compute normalized 3d distance per pair
    v1 = a[dims].to_numpy() / tol
    v2 = b[dims].to_numpy() / tol
    dist3d = np.abs(v1[ii] - v2[jj]).sum(axis=1)

    # Normalize to 0-1
    mx = dist3d.max()
    if mx > 0:
        dist3d = dist3d / dist3d.max()

    # Keep only the most intense feature in `a` for each feature in `b`
    intensity = a["intensity"].to_numpy()[ii]
    maxcols = _group_max(intensity, jj, len(b.index))
    intensity = np.where(intensity == maxcols, intensity, 0)

    # Break ties by distance
    intensity = intensity - dist3d

    # Where max per feature in `a` and nonzero
    maxrows = _group_max(intensity, ii, len(a.index))
    keep = (intensity == maxrows) & (intensity > 0)
    ii = ii[keep]
    jj = jj[keep]

    # Reorder
    a = a.iloc[ii]
    b = b.iloc[jj]

    if len(a.index) < 1 or len(b.index) < 1:
        return None, None

    return a, b


def tolerance(
    a,
    b,
    dims=None,
    tol=None,
    relative=None,
):
    """
    Identify features in `b` within tolerance of those in `a`. Matches are
    potentially many-to-one.

    Parameters
    ----------
    a : :obj:`~pandas.DataFrame`
        First set of input feature coordinates and intensities.
    b : :obj:`~pandas.DataFrame`
        Second set of input feature coordinates and intensities.
    dims : str or list, optional
        Dimensions considered in matching.
        Default is ``["mz", "drift_time", "retention_time"]``.
    tol : float or list, optional
        Tolerance in each dimension to define a match.
        Default is ``[5e-6, 0.025, 0.3]``.
    relative : bool or list, optional
        Whether to use relative or absolute tolerances per dimension.
        Default is ``[True, True, False]``.

    Returns
    -------
    a, b : :obj:`~pandas.DataFrame`
        Features matched within tolerances. E.g., `a[i..n]` and `b[i..n]` each
        represent matched features.

    """

    # Set mutable argument defaults
    if dims is None:
        dims = ["mz", "drift_time", "retention_time"]
    if tol is None:
        tol = [5e-6, 0.025, 0.3]
    if relative is None:
        relative = [True, True, False]

    if a is None or b is None:
        return None, None

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # Check dims
    deimos.utils.check_length([dims, tol, relative])

    # Per-dataset indices
    ii, jj = _pairs_within_tolerance(a, b, dims, tol, relative)

    # Reorder
    a = a.iloc[ii]
    b = b.iloc[jj]

    if len(a.index) < 1 or len(b.index) < 1:
        return None, None

    return a, b


def _pairs_within_tolerance(a, b, dims, tol, relative, chunk_size=2**20):
    """
    Find all pairs of features in `a` and `b` within tolerance in every
    dimension, without forming the dense `a` by `b` distance matrix.

    Candidate pairs come from sorted windows in the dimension that yields the
    fewest of them, and each candidate is then checked in every dimension.

    Parameters
    ----------
    a : :obj:`~pandas.DataFrame`
        First set of input feature coordinates and intensities.
    b : :obj:`~pandas.DataFrame`
        Second set of input feature coordinates and intensities.
    dims : list
        Dimensions considered in matching.
    tol : list
        Tolerance in each dimension to define a match.
    relative : list
        Whether to use relative or absolute tolerances per dimension.
    chunk_size : int
        Approximate number of candidate pairs checked at once.

    Returns
    -------
    ii, jj : :obj:`~numpy.array`
        Positional indices of matched features in `a` and `b`, ordered by
        `ii`, then `jj`.

    """

    n = len(a.index)
    m = len(b.index)

    if n < 1 or m < 1:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    # Window each feature in `a` in the most selective dimension
    low, high, order = np.zeros(n, dtype=int), np.full(n, m), np.arange(m)
    for i, f in enumerate(dims):
        # Window only by finite, non-negative tolerances; the exact check
        # below still applies the others
        if not (np.isfinite(tol[i]) and tol[i] >= 0):
            continue

        window = _candidate_windows(
            a[f].to_numpy(), b[f].to_numpy(), tol[i], relative[i] is True
        )
        if (window[1] - window[0]).sum() < (high - low).sum():
            low, high, order = window

    # Gather values per dimension
    v1 = [a[f].to_numpy() for f in dims]
    v2 = [b[f].to_numpy() for f in dims]

    # Count candidates per feature in `a`, and their running total
    counts = high - low
    total = np.cumsum(counts)

    # Check candidates in chunks of whole features in `a`, each holding about
    # `chunk_size` pairs, to bound memory
    pairs = []
    start = 0
    while start < n:
        # End the chunk at the last feature that keeps it within `chunk_size`,
        # taking at least one feature
        done = total[start - 1] if start > 0 else 0
        stop = int(np.searchsorted(total, done + chunk_size, side="right"))
        stop = max(stop, start + 1)

        # Expand windows to candidate pairs, pairing each feature in `a` with
        # sorted positions `low` through `high` - 1 of `b`
        c = counts[start:stop]
        ii = np.repeat(np.arange(start, stop), c)
        offset = np.arange(len(ii)) - np.repeat(np.cumsum(c) - c, c)
        jj = order[low[ii] + offset]

        # Check tol in each dimension
        idx = np.ones(len(ii), dtype=bool)
        for i in range(len(dims)):
            # Vectors
            x1 = v1[i][ii]
            x2 = v2[i][jj]

            # Distances in float64
            d = np.abs(x1.astype(np.float64) - x2.astype(np.float64))

            if relative[i] is True:
                # Divisor
                basis = np.where(x1 == 0, x2, x1)

                # Divide, storing the result in the dtype of the inputs
                d = np.divide(d, basis, out=np.zeros_like(basis), where=basis != 0)

            # Check tol
            idx &= d <= tol[i]

        # Encode each pair as one sortable integer
        pairs.append(ii[idx] * m + jj[idx])
        start = stop

    # Order by feature in `a`, then feature in `b`
    pairs = np.sort(np.concatenate(pairs))

    return pairs // m, pairs % m


def _candidate_windows(v1, v2, tol, relative):
    """
    Bound the values of `v2` within tolerance of each value of `v1`.

    Windows are widened slightly to allow for floating point rounding, so no
    match is missed.

    Parameters
    ----------
    v1 : :obj:`~numpy.array`
        Values of the features to window.
    v2 : :obj:`~numpy.array`
        Values of the candidate features.
    tol : float
        Tolerance to define a match.
    relative : bool
        Whether `tol` is relative to `v1`.

    Returns
    -------
    low, high : :obj:`~numpy.array`
        Start and end of each window in the sorted order of `v2`.
    order : :obj:`~numpy.array`
        Indices that sort `v2`.

    """

    # Get the precision of stored relative distances, as Python floats so the
    # padding is computed in float64
    dtype = np.result_type(v1, v2)
    if not np.issubdtype(dtype, np.floating):
        dtype = np.float64
    finfo = np.finfo(dtype)
    eps = float(finfo.eps)
    tiny = float(finfo.smallest_subnormal)
    largest = float(finfo.max)

    # Sort candidates
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    order = np.argsort(v2, kind="stable")
    v2 = v2[order]

    # Span every candidate when a relative tolerance exceeds the range of the
    # dtype, as the exact check then stores it as infinity
    if relative and tol > largest:
        return np.zeros(len(v1), dtype=int), np.full(len(v1), len(v2)), order

    # Center non-finite values at zero, as their windows are emptied below
    finite = np.isfinite(v1)
    v1 = np.where(finite, v1, 0)

    # Widen relative windows by the rounding of stored distances, then all
    # windows by the rounding of their bounds
    if relative:
        width = np.abs(v1) * (tol * (1 + 2 * eps) + 2 * tiny)
    else:
        width = np.full(len(v1), float(tol))
    width = width + 8 * np.finfo(np.float64).eps * (np.abs(v1) + width)

    # Locate windows
    low = np.searchsorted(v2, v1 - width, side="left")
    high = np.searchsorted(v2, v1 + width, side="right")

    # Non-finite values are never within tolerance
    high[~finite] = low[~finite]

    # Relative distances from zero or negative values are degenerate, so
    # check those features against every candidate
    if relative:
        low[finite & (v1 <= 0)] = 0
        high[finite & (v1 <= 0)] = len(v2)

    return low, high, order


def _group_max(values, groups, n_groups):
    """
    Maximum of `values` within each group, aligned with `values`.

    Parameters
    ----------
    values : :obj:`~numpy.array`
        Values to reduce.
    groups : :obj:`~numpy.array`
        Group label of each value, from 0 to `n_groups` - 1.
    n_groups : int
        Number of group labels.

    Returns
    -------
    :obj:`~numpy.array`
        Maximum of each value's group.

    """

    # Seed each group with one of its values, then take the maximum
    maxima = np.empty(n_groups, dtype=values.dtype)
    maxima[groups] = values
    np.maximum.at(maxima, groups, values)

    return maxima[groups]


def fit_spline(a, b, align="retention_time", **kwargs):
    """
    Fit a support vector regressor to matched features.

    Parameters
    ----------
    a : :obj:`~pandas.DataFrame`
        First set of input feature coordinates and intensities.
    b : :obj:`~pandas.DataFrame`
        Second set of input feature coordinates and intensities.
    align : str
        Dimension to align.
    kwargs
        Keyword arguments for support vector regressor
        (:class:`sklearn.svm.SVR`).

    Returns
    -------
    :obj:`~scipy.interpolate.interp1d`
        Interpolated fit of the SVR result.

    """

    # Uniqueify
    x = a[align].values
    y = b[align].values
    arr = np.vstack((x, y)).T
    arr = np.unique(arr, axis=0)

    # Check kwargs
    if "kernel" in kwargs:
        kernel = kwargs.get("kernel")
    else:
        kernel = "linear"

    # Construct interpolation axis
    newx = np.linspace(arr[:, 0].min(), arr[:, 0].max(), 1000)

    # Linear kernel
    if kernel == "linear":
        reg = scipy.stats.linregress(x, y)
        newy = reg.slope * newx + reg.intercept

    # Other kernels
    else:
        # Fit
        svr = SVR(**kwargs)
        svr.fit(arr[:, 0].reshape(-1, 1), arr[:, 1])

        # Predict
        newy = svr.predict(newx.reshape(-1, 1))

    return scipy.interpolate.interp1d(
        newx, newy, kind="linear", fill_value="extrapolate"
    )


def agglomerative_clustering(
    features,
    dims=None,
    tol=None,
    relative=None,
):
    """
    Cluster features within provided linkage tolerances. Recursively merges
    the pair of clusters that minimally increases a given linkage distance.
    See :class:`sklearn.cluster.AgglomerativeClustering`.

    When ``sample_idx`` is present, same-sample pairs are treated as
    unlinkable: their precomputed distance is set above
    ``distance_threshold`` so complete linkage cannot merge two features
    from the same sample, including through parent clusters.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
        Input feature coordinates and intensities per sample. An optional
        ``sample_idx`` column identifies the source sample.
    dims : str or list, optional
        Dimensions considered in clustering.
        Default is ``["mz", "drift_time", "retention_time"]``.
    tol : float or list, optional
        Tolerance in each dimension to define maximum cluster linkage
        distance.
        Default is ``[20e-6, 0.03, 0.3]``.
    relative : bool or list, optional
        Whether to use relative or absolute tolerances per dimension.
        Default is ``[True, True, False]``.

    Returns
    -------
    features : :obj:`~pandas.DataFrame`
        Features concatenated over samples with cluster labels.

    """

    # Set mutable argument defaults
    if dims is None:
        dims = ["mz", "drift_time", "retention_time"]
    if tol is None:
        tol = [20e-6, 0.03, 0.3]
    if relative is None:
        relative = [True, True, False]

    if features is None:
        return None

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # Check dims
    deimos.utils.check_length([dims, tol, relative])

    # Copy input
    features = features.copy()

    # Compute inter-feature distances
    distances = []
    for i, d in enumerate(dims):
        # Vectors
        v1 = features[d].values.reshape(-1, 1)

        # Distances
        dist = scipy.spatial.distance.cdist(v1, v1)

        if relative[i] is True:
            # Divisor
            basis = np.repeat(v1, v1.shape[0], axis=1)
            fix = np.repeat(v1, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # Divide
            dist = np.divide(dist, basis, out=np.zeros_like(basis), where=basis != 0)

        # Check tol
        distances.append(dist / tol[i])

    # Stack distances
    distances = np.dstack(distances)

    # Max distance
    distances = np.max(distances, axis=-1)

    # Set same-sample distances above threshold
    merge_threshold = 1.0
    if "sample_idx" in features.columns:
        sample_idx = np.asarray(features["sample_idx"])
        same_sample = sample_idx[:, None] == sample_idx[None, :]
        np.fill_diagonal(same_sample, False)
        distances[same_sample] = merge_threshold + 1.0

    # Perform clustering
    try:
        # sklearn accepts None with distance_threshold, but its inferred type is int
        clustering = AgglomerativeClustering(
            n_clusters=None,  # pyright: ignore[reportArgumentType]
            linkage="complete",
            metric="precomputed",
            distance_threshold=merge_threshold,
        ).fit(distances)
        features["cluster"] = clustering.labels_

    # All data points are singleton clusters
    except ValueError:
        features["cluster"] = np.arange(len(features.index))

    return features


def merge_features(
    features,
    dims=None,
    tol=None,
    relative=None,
):
    """
    Merge features within provided tolerances.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
        Input feature coordinates and intensities per sample.
    dims : str or list, optional
        Dimensions considered in clustering.
        Default is ``["mz", "drift_time", "retention_time"]``.
    tol : float or list, optional
        Tolerance in each dimension to define maximum cluster tolerance.
        distance.
        Default is ``[20e-6, 0.03, 0.3]``.
    relative : bool or list, optional
        Whether to use relative or absolute tolerances per dimension.
        Default is ``[True, True, False]``.

    Returns
    -------
    features : :obj:`~pandas.DataFrame`
        Features concatenated over samples with cluster labels.

    """

    # Set mutable argument defaults
    if dims is None:
        dims = ["mz", "drift_time", "retention_time"]
    if tol is None:
        tol = [20e-6, 0.03, 0.3]
    if relative is None:
        relative = [True, True, False]

    if features is None:
        return None

    if len(features.index) == 0:
        return features

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # Check dims
    deimos.utils.check_length([dims, tol, relative])

    # Copy input
    features = features.copy()

    # Sort
    features = features.sort_values(by="intensity", ascending=False).reset_index(
        drop=True
    )

    # Compute inter-feature distances
    distances = None
    for i in range(len(dims)):
        # Construct k-d tree for drift time
        values = features[dims[i]].values
        tree = KDTree(values.reshape(-1, 1))

        max_tol = tol[i]
        if relative[i] is True:
            # Maximum absolute tolerance
            max_tol = tol[i] * values.max()

        # Compute sparse distance matrix
        sdm = tree.sparse_distance_matrix(tree, max_tol, output_type="coo_matrix")

        # Only consider forward case, exclude diagonal
        sdm = sparse.triu(sdm, k=1)

        # Filter relative distances
        if relative[i] is True:
            # Compute relative distances
            rel_dists = sdm.data / values[sdm.row]  # or col?

            # Indices of relative distances less than tolerance
            idx = rel_dists <= tol[i]

            # Reconstruct sparse distance matrix
            sdm = sparse.coo_matrix(
                (rel_dists[idx], (sdm.row[idx], sdm.col[idx])),
                shape=(len(values), len(values)),
            )

        # Cast as binary matrix
        sdm.data = np.ones_like(sdm.data)

        # Stack distances
        if distances is None:
            distances = sdm
        else:
            distances = distances.multiply(sdm)

    # Require at least one dimension
    if distances is None:
        raise ValueError("At least one dimension must be specified.")

    # Extract indices of within-tolerance points
    distances = distances.tocoo()
    pairs = np.stack((distances.row, distances.col), axis=1)

    # Drop within-tolerance points
    to_drop = []
    while len(pairs) > 0:
        # Find root parents and their children
        root_parents = np.setdiff1d(np.unique(pairs[:, 0]), np.unique(pairs[:, 1]))
        id_root_parents = np.isin(pairs[:, 0], root_parents)
        children_of_roots = np.unique(pairs[id_root_parents, 1])
        to_drop = np.append(to_drop, children_of_roots)

        # Set up pairs array for next iteration
        pairs = pairs[~np.isin(pairs[:, 1], to_drop), :]
        pairs = pairs[~np.isin(pairs[:, 0], to_drop), :]

    return features.reset_index(drop=True).drop(index=np.array(to_drop))
