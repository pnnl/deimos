import numpy as np
import scipy
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVR

import deimos


def match(a, b, dims=['mz', 'drift_time', 'retention_time'],
          tol=[5E-6, 0.015, 0.3], relative=[True, True, False]):
    '''
    Identify features in `b` within tolerance of those in `a` . Matches are
    bidirectionally one-to-one by highest intensity.

    Parameters
    ----------
    a : :obj:`~pandas.DataFrame`
        First set of input feature coordinates and intensities.
    b : :obj:`~pandas.DataFrame`
        Second set of input feature coordinates and intensities.
    dims : str or list
        Dimensions considered in matching.
    tol : float or list
        Tolerance in each dimension to define a match.
    relative : bool or list
        Whether to use relative or absolute tolerances per dimension.

    Returns
    -------
    a, b : :obj:`~pandas.DataFrame`
        Features matched within tolerances. E.g., `a[i..n]`and `b[i..n]` each
        represent matched features.

    Raises
    ------
    ValueError
        If `dims`, `tol`, and `relative` are not the same length.

    '''

    if a is None or b is None:
        return None, None

    # safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # check dims
    deimos.utils.check_length([dims, tol, relative])

    # compute inter-feature distances
    idx = []
    for i, f in enumerate(dims):
        # vectors
        v1 = a[f].values.reshape(-1, 1)
        v2 = b[f].values.reshape(-1, 1)

        # distances
        d = scipy.spatial.distance.cdist(v1, v2)

        if relative[i] is True:
            # divisor
            basis = np.repeat(v1, v2.shape[0], axis=1)
            fix = np.repeat(v2, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # divide
            d = np.divide(d, basis, out=np.zeros_like(basis), where=basis != 0)

        # check tol
        idx.append(d <= tol[i])

    # stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1, dtype=bool)

    # compute normalized 3d distance
    v1 = a[dims].values / tol
    v2 = b[dims].values / tol
    # v1 = (v1 - v1.min(axis=0)) / (v1.max(axis=0) - v1.min(axis=0))
    # v2 = (v2 - v1.min(axis=0)) / (v1.max(axis=0) - v1.min(axis=0))
    dist3d = scipy.spatial.distance.cdist(v1, v2, 'cityblock')
    dist3d = np.multiply(dist3d, idx)

    # normalize to 0-1
    mx = dist3d.max()
    if mx > 0:
        dist3d = dist3d / dist3d.max()

    # intensities
    intensity = np.repeat(a['intensity'].values.reshape(-1, 1),
                          b.shape[0], axis=1)
    intensity = np.multiply(intensity, idx)

    # max over dims
    maxcols = np.max(intensity, axis=0, keepdims=True)

    # zero out nonmax over dims
    intensity[intensity != maxcols] = 0

    # break ties by distance
    intensity = intensity - dist3d

    # max over clusters
    maxrows = np.max(intensity, axis=1, keepdims=True)

    # where max and nonzero
    ii, jj = np.where((intensity == maxrows) & (intensity > 0))

    # reorder
    a = a.iloc[ii]
    b = b.iloc[jj]

    if len(a.index) < 1 or len(b.index) < 1:
        return None, None

    return a, b


def tolerance(a, b, dims=['mz', 'drift_time', 'retention_time'],
              tol=[5E-6, 0.025, 0.3], relative=[True, True, False]):
    '''
    Identify features in `b` within tolerance of those in `a`. Matches are
    potentially many-to-one.

    Parameters
    ----------
    a : :obj:`~pandas.DataFrame`
        First set of input feature coordinates and intensities.
    b : :obj:`~pandas.DataFrame`
        Second set of input feature coordinates and intensities.
    dims : str or list
        Dimensions considered in matching.
    tol : float or list
        Tolerance in each dimension to define a match.
    relative : bool or list
        Whether to use relative or absolute tolerances per dimension.

    Returns
    -------
    a, b : :obj:`~pandas.DataFrame`
        Features matched within tolerances. E.g., `a[i..n]` and `b[i..n]` each
        represent matched features.

    Raises
    ------
    ValueError
        If `dims`, `tol`, and `relative` are not the same length.

    '''

    if a is None or b is None:
        return None, None

    # safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # check dims
    deimos.utils.check_length([dims, tol, relative])

    # compute inter-feature distances
    idx = []
    for i, f in enumerate(dims):
        # vectors
        v1 = a[f].values.reshape(-1, 1)
        v2 = b[f].values.reshape(-1, 1)

        # distances
        d = scipy.spatial.distance.cdist(v1, v2)

        if relative[i] is True:
            # divisor
            basis = np.repeat(v1, v2.shape[0], axis=1)
            fix = np.repeat(v2, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # divide
            d = np.divide(d, basis, out=np.zeros_like(basis), where=basis != 0)

        # check tol
        idx.append(d <= tol[i])

    # stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1, dtype=bool)

    # per-dataset indices
    ii, jj = np.where(idx > 0)

    # reorder
    a = a.iloc[ii]
    b = b.iloc[jj]

    if len(a.index) < 1 or len(b.index) < 1:
        return None, None

    return a, b


def fit_spline(a, b, align='retention_time', **kwargs):
    '''
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

    '''

    # uniqueify
    x = a[align].values
    y = b[align].values
    arr = np.vstack((x, y)).T
    arr = np.unique(arr, axis=0)

    # check kwargs
    if 'kernel' in kwargs:
        kernel = kwargs.get('kernel')
    else:
        kernel = 'linear'

    newx = np.linspace(arr[:, 0].min(), arr[:, 0].max(), 1000)

    if kernel == 'linear':
        reg = scipy.stats.linregress(x, y)
        newy = reg.slope * newx + reg.intercept

    else:
        # fit
        svr = SVR(**kwargs)
        svr.fit(arr[:, 0].reshape(-1, 1), arr[:, 1])

        # predict
        newy = svr.predict(newx.reshape(-1, 1))

    return scipy.interpolate.interp1d(newx, newy,
                                      kind='linear', fill_value='extrapolate')


def agglomerative_clustering(features,
                             dims=['mz', 'drift_time', 'retention_time'],
                             tol=[20E-6, 0.03, 0.3],
                             relative=[True, True, False]):
    '''
    Cluster features within provided linkage tolerances. Recursively merges
    the pair of clusters that minimally increases a given linkage distance.
    See :class:`sklearn.cluster.AgglomerativeClustering`.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
        Input feature coordinates and intensities per sample.
    dims : str or list
        Dimensions considered in clustering.
    tol : float or list
        Tolerance in each dimension to define maximum cluster linkage
        distance.
    relative : bool or list
        Whether to use relative or absolute tolerances per dimension.

    Returns
    -------
    features : :obj:`~pandas.DataFrame`
        Features concatenated over samples with cluster labels.

    Raises
    ------
    ValueError
        If `dims`, `tol`, and `relative` are not the same length.

    '''

    if features is None:
        return None

    # safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # check dims
    deimos.utils.check_length([dims, tol, relative])

    # copy input
    features = features.copy()

    # connectivity
    if 'sample_idx' not in features.columns:
        cmat = None
    else:
        vals = features['sample_idx'].values.reshape(-1, 1)
        cmat = cdist(vals, vals, metric=lambda x, y: x != y).astype(bool)

    # compute inter-feature distances
    distances = []
    for i, d in enumerate(dims):
        # vectors
        v1 = features[d].values.reshape(-1, 1)

        # distances
        dist = scipy.spatial.distance.cdist(v1, v1)

        if relative[i] is True:
            # divisor
            basis = np.repeat(v1, v1.shape[0], axis=1)
            fix = np.repeat(v1, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # divide
            dist = np.divide(dist, basis, out=np.zeros_like(
                basis), where=basis != 0)

        # check tol
        distances.append(dist / tol[i])

    # stack distances
    distances = np.dstack(distances)

    # max distance
    distances = np.max(distances, axis=-1)

    # perform clustering
    try:
        clustering = AgglomerativeClustering(n_clusters=None,
                                             linkage='complete',
                                             affinity='precomputed',
                                             distance_threshold=1,
                                             connectivity=cmat).fit(distances)
        features['cluster'] = clustering.labels_
    except:
        features['cluster'] = np.arange(len(features.index))

    return features
