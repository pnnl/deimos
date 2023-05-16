import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from ripser import ripser
from scipy import sparse
from scipy.spatial import KDTree
from sklearn.utils.sparsefuncs import _get_median

import deimos


def std(a, size):
    '''
    N-dimensional convolution of a standard deviation filter.

    Parameters
    ----------
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensity data.

    '''

    c1 = ndi.uniform_filter(a, size, mode='reflect')
    c2 = ndi.uniform_filter(np.square(a), size, mode='reflect')
    return np.abs(np.lib.scimath.sqrt(c2 - np.square(c1)))


def std_pdf(edges, a, size):
    '''
    N-dimensional convolution of a standard deviation probability density
    function filter.

    Parameters
    ----------
    edges : list of :obj:`~numpy.array`
        Edges coordinates along each grid axis.
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    list of :obj:`~numpy.array`
        Filtered edge data.

    '''

    edges = np.meshgrid(*edges, indexing='ij')
    f = ndi.uniform_filter(a, size=size, mode='constant')

    res = []
    for e in edges:
        wmu = ndi.uniform_filter(a * e, size=size, mode='constant')
        mu = wmu / f

        wvar = ndi.uniform_filter(
            a * (e - mu) ** 2, size=size, mode='constant')
        var = wvar / f
        res.append(np.sqrt(var))

    return res


def maximum(a, size):
    '''
    N-dimensional convolution of a maximum filter.

    Parameters
    ----------
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensity data.

    '''

    return ndi.maximum_filter(a, size=size, mode='constant', cval=-np.inf)


def minimum(a, size):
    '''
    N-dimensional convolution of a minimum filter.

    Parameters
    ----------
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensity data.

    '''

    return ndi.minimum_filter(a, size=size, mode='constant', cval=np.inf)


def sum(a, size):
    '''
    N-dimensional convolution of a sum filter.

    Parameters
    ----------
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensity data.

    '''

    size = deimos.utils.safelist(size)

    if len(size) == 1:
        size = size[0]
        ksize = size ** a.ndim
    else:
        ksize = np.prod(size)

    return ksize * ndi.uniform_filter(a, size, mode='constant', cval=0.0)


def mean(a, size):
    '''
    N-dimensional convolution of a mean filter.

    Parameters
    ----------
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensity data.

    '''

    return ndi.uniform_filter(a, size=size, mode='constant', cval=0.0)


def mean_pdf(edges, a, size):
    '''
    N-dimensional convolution of a mean probability density function filter.

    Parameters
    ----------
    edges : list of :obj:`~numpy.array`
        Edges coordinates along each grid axis.
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    list of :obj:`~numpy.array`
        Filtered edge data.

    '''

    edges = np.meshgrid(*edges, indexing='ij')
    f = ndi.uniform_filter(a, size=size, mode='constant')

    res = []
    for e in edges:
        w = ndi.uniform_filter(a * e, size=size, mode='constant')
        res.append(w / f)

    return res


def matched_gaussian(a, size):
    '''
    N-dimensional convolution of a matched Gaussian filter.

    Parameters
    ----------
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensity data.

    '''

    return np.square(ndi.gaussian_filter(a, size, mode='constant', cval=0.0))


def count(a, size, nonzero=False):
    '''
    N-dimensional convolution of a counting filter.

    Parameters
    ----------
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.
    nonzero : bool
        Only count nonzero values.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensity data.

    '''

    if nonzero is True:
        a = np.where(np.nan_to_num(a) > 0, 1.0, 0.0)
    else:
        a = np.where(np.isnan(a), 0.0, 1.0)

    return deimos.filters.sum(a, size)


def skew_pdf(edges, a, size):
    '''
    N-dimensional convolution of a skew probability density function filter.

    Parameters
    ----------
    edges : list of :obj:`~numpy.array`
        Edges coordinates along each grid axis.
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    list of :obj:`~numpy.array`
        Filtered edge data.

    '''

    edges = np.meshgrid(*edges, indexing='ij')
    f = ndi.uniform_filter(a, size=size, mode='constant')

    res = []
    for e in edges:
        wmu = ndi.uniform_filter(a * e, size=size, mode='constant')
        mu = wmu / f

        wvar = ndi.uniform_filter(
            a * (e - mu) ** 2, size=size, mode='constant')
        var = wvar / f
        sigma = np.sqrt(var)

        wskew = ndi.uniform_filter(
            a * ((e - mu) / sigma) ** 3, size=size, mode='constant')
        skew = wskew / f
        res.append(skew)

    return res


def kurtosis_pdf(edges, a, size):
    '''
    N-dimensional convolution of a kurtosis probability density function
    filter.

    Parameters
    ----------
    edges : list of :obj:`~numpy.array`
        Edges coordinates along each grid axis.
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    list of :obj:`~numpy.array`
        Filtered edge data.

    '''

    edges = np.meshgrid(*edges, indexing='ij')
    f = ndi.uniform_filter(a, size=size, mode='constant')

    res = []
    for e in edges:
        wmu = ndi.uniform_filter(a * e, size=size, mode='constant')
        mu = wmu / f

        wvar = ndi.uniform_filter(
            a * (e - mu) ** 2, size=size, mode='constant')
        var = wvar / f
        sigma = np.sqrt(var)

        wkurtosis = ndi.uniform_filter(
            a * ((e - mu) / sigma) ** 4, size=size, mode='constant')
        kurtosis = wkurtosis / f
        res.append(kurtosis - 3)

    return res


def embed_unique_indices(a):

    def count_tens(n):
        # Count tens
        ntens = (n - 1) // 10

        while True:
            ntens_test = (ntens + n - 1) // 10

            if ntens_test == ntens:
                return ntens
            else:
                ntens = ntens_test

    def arange_exclude_10s(n):
        # How many 10s will there be?
        ntens = count_tens(n)

        # Base array
        arr = np.arange(0, n + ntens)

        # Exclude 10s
        arr = arr[(arr == 0) | (arr % 10 != 0)][:n]

        return arr

    # Creates an array of indices, sorted by unique element
    idx_sort = np.argsort(a)
    idx_unsort = np.argsort(idx_sort)

    # Sorts records array so all unique elements are together
    sorted_a = a[idx_sort]

    # Returns the unique values, the index of the first occurrence,
    # and the count for each element
    vals, idx_start, count = np.unique(sorted_a, return_index=True, return_counts=True)

    # Splits the indices into separate arrays
    splits = np.split(idx_sort, idx_start[1:])

    # Creates unique indices for each split
    idx_unq = np.concatenate([arange_exclude_10s(len(x)) for x in splits])

    # Reorders according to input array
    idx_unq = idx_unq[idx_unsort]

    # Magnitude of each index
    exp = np.log10(idx_unq,
                   where=idx_unq > 0,
                   out=np.zeros_like(idx_unq, dtype=np.float64))
    idx_unq_mag = np.power(10, np.floor(exp) + 1)

    # Result
    return a + idx_unq / idx_unq_mag


def sparse_upper_star(idx, V):
    '''
    Sparse implementation of an upper star filtration.
    Parameters
    ----------
    idx : :obj:`~numpy.array`
        Edge indices for each dimension (MxN).
    V : :obj:`~numpy.array`
        Array of intensity data (Mx1).
    Returns
    -------
    idx : :obj:`~numpy.array`
        Index of filtered points (Mx1).
    persistence : :obj:`~numpy.array`
        Persistence of each filtered point (Mx1).
    '''

    # Invert
    V = -1 * V.copy().astype(int)

    # Embed indices
    V = embed_unique_indices(V)

    # Connectivity matrix
    cmat = KDTree(idx)
    cmat = cmat.sparse_distance_matrix(
        cmat, 1, p=np.inf, output_type='coo_matrix')
    cmat.setdiag(1)
    cmat = sparse.triu(cmat)

    # Pairwise minimums
    I, J = cmat.nonzero()
    d = np.maximum(V[I], V[J])

    # Delete connectiity matrix
    cmat_shape = cmat.shape
    del cmat

    # Sparse distance matrix
    sdm = sparse.coo_matrix((d, (I, J)), shape=cmat_shape)

    # Delete pairwise mins
    del d, I, J

    # Persistence homology
    ph = ripser(sdm, distance_matrix=True, maxdim=0)['dgms'][0]

    # Bound death values
    ph[ph[:, 1] == np.inf, 1] = np.max(V)

    # Construct tree to query against
    tree = KDTree(V.reshape((-1, 1)))

    # Get the indexes of the first nearest neighbor by birth
    _, nn = tree.query(ph[:, 0].reshape((-1, 1)), k=1, workers=-1)

    return nn, -(ph[:, 0] // 1 - ph[:, 1] // 1)


def sparse_mean_filter(idx, V, radius=[0, 1, 1]):
    '''
    Sparse implementation of a mean filter.

    Parameters
    ----------
    idx : :obj:`~numpy.array`
        Edge indices for each dimension (MxN).
    V : :obj:`~numpy.array`
        Array of intensity data (Mx1).
    radius : float or list
        Radius of the sparse filter in each dimension. Values less than
        zero indicate no connectivity in that dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensities (Mx1).

    '''

    # Copy indices
    idx = idx.copy().astype(V.dtype)

    # Scale
    for i, r in enumerate(radius):
        # Increase inter-index distance
        if r < 1:
            idx[:, i] *= 2

        # Do nothing
        elif r == 1:
            pass

        # Decrease inter-index distance
        else:
            idx[:, i] /= r

    # Connectivity matrix
    cmat = KDTree(idx)
    cmat = cmat.sparse_distance_matrix(
        cmat, 1, p=np.inf, output_type='coo_matrix')
    cmat.setdiag(1)

    # Pair indices
    I, J = cmat.nonzero()

    # Delete cmat
    cmat_shape = cmat.shape
    del cmat

    # Sum over columns
    V_sum = sparse.bsr_matrix((V[J], (I, I)),
                              shape=cmat_shape,
                              dtype=V.dtype).diagonal(0)

    # Count over columns
    V_count = sparse.bsr_matrix((np.ones_like(J), (I, I)),
                                shape=cmat_shape,
                                dtype=V.dtype).diagonal(0)

    return V_sum / V_count


def sparse_weighted_mean_filter(idx, V, w, radius=[1, 1, 1], pindex=None):
    '''
    Sparse implementation of a weighted mean filter.

    Parameters
    ----------
    idx : :obj:`~numpy.array`
        Edge indices for each dimension (MxN).
    V : :obj:`~numpy.array`
        Array of edge data (MxN).
    w : :obj:`~numpy.array`
        Array of intensity data (Mx1).
    radius : float or list
        Radius of the sparse filter in each dimension. Values less than
        one indicate no connectivity in that dimension.
    pindex : :obj:`~numpy.array`
        Index of points to evaluate the weighted mean.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered edges (MxN).

    '''

    # Copy indices
    idx = idx.copy().astype(w.dtype)

    # Scale
    for i, r in enumerate(radius):
        # Increase inter-index distance
        if r < 1:
            idx[:, i] *= 2

        # Do nothing
        elif r == 1:
            pass

        # Decrease inter-index distance
        else:
            idx[:, i] /= r

    # If not supplied, index is all points
    if pindex is None:
        pindex = np.arange(len(idx))

    # Connectivity matrix
    tree_all = KDTree(idx)
    tree_subset = KDTree(idx[pindex])
    cmat = tree_subset.sparse_distance_matrix(
        tree_all, 1, p=np.inf, output_type='coo_matrix')
    del tree_all, tree_subset

    # Pair indices
    I, J = cmat.nonzero()

    # Self
    I = np.concatenate((I, np.arange(len(pindex))))
    J = np.concatenate((J, pindex))

    # Delete connectivity matrix
    cmat_shape = cmat.shape
    del cmat

    # Sum weights over columns
    # Only need to do this once
    V_count = sparse.bsr_matrix((w[J], (I, I)),
                                shape=cmat_shape,
                                dtype=w.dtype).diagonal(0)

    # Reshape V if 1D
    if V.ndim == 1:
        V = V.reshape((-1, 1))

    # Output container
    V_out = np.empty_like(V[pindex, :], dtype=w.dtype)

    # Enumerate value columns
    for i in range(V_out.shape[1]):
        # Sum weighted values over columns
        V_sum = sparse.bsr_matrix((w[J] * V[J, i], (I, I)),
                                  shape=cmat_shape,
                                  dtype=w.dtype).diagonal(0)

        V_out[:, i] = V_sum / V_count

    # Flatten if 1D
    if V_out.shape[1] == 1:
        return V_out.flatten()

    return V_out


def sparse_median_filter(idx, V, radius=[0, 1, 1]):
    '''
    Sparse implementation of a median filter.

    Parameters
    ----------
    idx : :obj:`~numpy.array`
        Edge indices for each dimension (MxN).
    V : :obj:`~numpy.array`
        Array of intensity data (Mx1).
    radius : float or list
        Radius of the sparse filter in each dimension. Values less than
        zero indicate no connectivity in that dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensities (Mx1).

    '''

    # Copy indices
    idx = idx.copy().astype(V.dtype)

    # Scale
    for i, r in enumerate(radius):
        # Increase inter-index distance
        if r < 1:
            idx[:, i] *= 2

        # Do nothing
        elif r == 1:
            pass

        # Decrease inter-index distance
        else:
            idx[:, i] /= r

    # Connectivity matrix
    cmat = KDTree(idx)
    cmat = cmat.sparse_distance_matrix(
        cmat, 1, p=np.inf, output_type='coo_matrix')
    cmat.setdiag(1)

    # Pair indices
    I, J = cmat.nonzero()

    # Delete cmat
    cmat_shape = cmat.shape
    del cmat

    X = sparse.csc_matrix((V[I], (I, J)),
                          shape=cmat_shape,
                          dtype=V.dtype)

    indptr = X.indptr
    n_samples, n_features = X.shape
    median = np.empty(n_features, dtype=V.dtype)

    for f_ind, (start, end) in enumerate(zip(indptr[:-1], indptr[1:])):
        # Prevent modifying X in place
        data = np.copy(X.data[start:end])
        median[f_ind] = _get_median(data, 0)

    return median


def smooth(features, index=None, factors=None, dims=['mz', 'drift_time', 'retention_time'],
           radius=[0, 1, 1], iterations=1, tol=0.0):
    '''
    Smooth data by sparse mean filtration.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    index : dict
        Index of features in original data array.
    factors : dict
        Unique sorted values per dimension.
    dims : str or list
        Dimensions to perform peak detection in.
    radius : float or list
        Radius of the sparse filter in each dimension. Values less than
        zero indicate no connectivity in that dimension.
    iterations : int
        Maximum number of smoothing iterations to perform.
    tol : float
        Stopping criteria based on residual with previous iteration.
        Selecting zero will perform all specified iterations.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Smoothed feature coordinates and intensities.

    '''

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    radius = deimos.utils.safelist(radius)

    # Check dims
    deimos.utils.check_length([dims, radius])

    # Check factors and index mutually exclusive
    if (factors is not None) & (index is not None):
        raise ValueError('Specify either `index`, `factors`, or neither.')

    # Copy input
    features = features.copy()

    # Build index from features directly
    if (factors is None) & (index is None):
        index = {dim: pd.factorize(features[dim], sort=True)[0].astype(np.float32)
                 for dim in dims}

    # Build index from factors
    if (factors is not None) & (index is None):
        index = deimos.build_index(features, factors)

    # Index supplied directly
    index = np.vstack([index[dim] for dim in dims]).T

    # Values
    V = features['intensity'].values

    # Residual exit criteria
    resid = np.inf

    # Sparse mean filtration iterations
    for i in range(iterations):
        # Previous iteration
        V_prev = V.copy()
        resid_prev = resid

        # Compute smooth
        V = deimos.filters.sparse_mean_filter(index, V, radius=radius)

        # Calculate residual with previous iteration
        resid = np.sqrt(np.mean(np.square(V - V_prev)))

        # Evaluate convergence
        if i > 0:
            # Percent change in residual
            test = np.abs(resid - resid_prev) / resid_prev

            # Exit criteria
            if test <= tol:
                break

    # Overwrite values
    features['intensity'] = V

    return features
