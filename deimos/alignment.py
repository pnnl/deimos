import numpy as np
import scipy
from scipy import sparse
from scipy.spatial import KDTree
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

    '''

    if a is None or b is None:
        return None, None

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # Check dims
    deimos.utils.check_length([dims, tol, relative])

    # Compute inter-feature distances
    idx = []
    for i, f in enumerate(dims):
        # vectors
        v1 = a[f].values.reshape(-1, 1)
        v2 = b[f].values.reshape(-1, 1)

        # Distances
        d = scipy.spatial.distance.cdist(v1, v2)

        if relative[i] is True:
            # Divisor
            basis = np.repeat(v1, v2.shape[0], axis=1)
            fix = np.repeat(v2, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # Divide
            d = np.divide(d, basis, out=np.zeros_like(basis), where=basis != 0)

        # Check tol
        idx.append(d <= tol[i])

    # Stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1, dtype=bool)

    # Compute normalized 3d distance
    v1 = a[dims].values / tol
    v2 = b[dims].values / tol
    dist3d = scipy.spatial.distance.cdist(v1, v2, 'cityblock')
    dist3d = np.multiply(dist3d, idx)

    # Normalize to 0-1
    mx = dist3d.max()
    if mx > 0:
        dist3d = dist3d / dist3d.max()

    # Intensities
    intensity = np.repeat(a['intensity'].values.reshape(-1, 1),
                          b.shape[0], axis=1)
    intensity = np.multiply(intensity, idx)

    # Max over dims
    maxcols = np.max(intensity, axis=0, keepdims=True)

    # Zero out nonmax over dims
    intensity[intensity != maxcols] = 0

    # Break ties by distance
    intensity = intensity - dist3d

    # Max over clusters
    maxrows = np.max(intensity, axis=1, keepdims=True)

    # Where max and nonzero
    ii, jj = np.where((intensity == maxrows) & (intensity > 0))

    # Reorder
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

    '''

    if a is None or b is None:
        return None, None

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # Check dims
    deimos.utils.check_length([dims, tol, relative])

    # Compute inter-feature distances
    idx = []
    for i, f in enumerate(dims):
        # Vectors
        v1 = a[f].values.reshape(-1, 1)
        v2 = b[f].values.reshape(-1, 1)

        # Distances
        d = scipy.spatial.distance.cdist(v1, v2)

        if relative[i] is True:
            # Divisor
            basis = np.repeat(v1, v2.shape[0], axis=1)
            fix = np.repeat(v2, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # Divide
            d = np.divide(d, basis, out=np.zeros_like(basis), where=basis != 0)

        # Check tol
        idx.append(d <= tol[i])

    # Stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1, dtype=bool)

    # Per-dataset indices
    ii, jj = np.where(idx > 0)

    # Reorder
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

    # Uniqueify
    x = a[align].values
    y = b[align].values
    arr = np.vstack((x, y)).T
    arr = np.unique(arr, axis=0)

    # Check kwargs
    if 'kernel' in kwargs:
        kernel = kwargs.get('kernel')
    else:
        kernel = 'linear'

    # Construct interpolation axis
    newx = np.linspace(arr[:, 0].min(), arr[:, 0].max(), 1000)

    # Linear kernel
    if kernel == 'linear':
        reg = scipy.stats.linregress(x, y)
        newy = reg.slope * newx + reg.intercept

    # Other kernels
    else:
        # Fit
        svr = SVR(**kwargs)
        svr.fit(arr[:, 0].reshape(-1, 1), arr[:, 1])

        # Predict
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

    '''

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

    # Connectivity
    if 'sample_idx' not in features.columns:
        cmat = None
    else:
        vals = features['sample_idx'].values.reshape(-1, 1)
        cmat = cdist(vals, vals, metric=lambda x, y: x != y).astype(bool)

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
            dist = np.divide(dist, basis, out=np.zeros_like(
                basis), where=basis != 0)

        # Check tol
        distances.append(dist / tol[i])

    # Stack distances
    distances = np.dstack(distances)

    # Max distance
    distances = np.max(distances, axis=-1)

    # Perform clustering
    try:
        clustering = AgglomerativeClustering(n_clusters=None,
                                             linkage='complete',
                                             metric='precomputed',
                                             distance_threshold=1,
                                             connectivity=cmat).fit(distances)
        features['cluster'] = clustering.labels_

    # All data points are singleton clusters
    except:
        features['cluster'] = np.arange(len(features.index))

    return features


def merge_features(features,
                   dims=['mz', 'drift_time', 'retention_time'],
                   tol=[20E-6, 0.03, 0.3],
                   relative=[True, True, False]):
    '''
    Merge features within provided tolerances.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
        Input feature coordinates and intensities per sample.
    dims : str or list
        Dimensions considered in clustering.
    tol : float or list
        Tolerance in each dimension to define maximum cluster tolerance.
        distance.
    relative : bool or list
        Whether to use relative or absolute tolerances per dimension.

    Returns
    -------
    features : :obj:`~pandas.DataFrame`
        Features concatenated over samples with cluster labels.

    '''

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
    
    # Sort
    features = features.sort_values(by='intensity', ascending=False).reset_index(drop=True)

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
        sdm = tree.sparse_distance_matrix(tree, max_tol, output_type='coo_matrix')

        # Only consider forward case, exclude diagonal
        sdm = sparse.triu(sdm, k=1)

        # Filter relative distances
        if relative[i] is True:
            # Compute relative distances
            rel_dists = sdm.data / values[sdm.row] # or col?
            
            # Indices of relative distances less than tolerance
            idx = rel_dists <= tol[i]
            
            # Reconstruct sparse distance matrix
            sdm = sparse.coo_matrix((rel_dists[idx], (sdm.row[idx], sdm.col[idx])),
                                    shape=(len(values), len(values)))
        
        # Cast as binary matrix
        sdm.data = np.ones_like(sdm.data)

        # Stack distances
        if distances is None:
            distances = sdm
        else:
            distances = distances.multiply(sdm)
    
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
