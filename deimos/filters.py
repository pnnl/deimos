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

    # add noise to uniqueify
    V += np.random.uniform(0, 1, V.shape)

    # connectivity matrix
    cmat = KDTree(idx)
    cmat = cmat.sparse_distance_matrix(
        cmat, 1, p=np.inf, output_type='coo_matrix')
    cmat.setdiag(1)

    # pairwise minimums
    I, J = cmat.nonzero()
    d = np.minimum(V[I], V[J])

    # delete connectiity matrix
    cmat_shape = cmat.shape
    del cmat

    # sparse distance matrix
    sdm = sparse.coo_matrix((d, (I, J)), shape=cmat_shape, dtype=V.dtype)

    # delete pairwise mins
    del d, I, J

    # persistence homology
    # negative for upper star, then revert
    ph = -ripser(-sdm, distance_matrix=True, maxdim=0)["dgms"][0]

    # delete distance matrix
    del sdm

    # bound death values
    ph[ph[:, 1] == -np.inf, 1] = np.min(V)

    # construct tree to query against
    tree = KDTree(V.reshape((-1, 1)))

    # get the indexes of the first nearest neighbor by birth
    _, nn = tree.query(ph[:, 0].reshape((-1, 1)), k=1, workers=-1)

    return nn, ph[:, 0] - ph[:, 1]


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

    # copy indices
    idx = idx.copy().astype(V.dtype)

    # scale
    for i, r in enumerate(radius):
        # increase inter-index distance
        if r < 1:
            idx[:, i] *= 2

        # do nothing
        elif r == 1:
            pass

        # decrease inter-index distance
        else:
            idx[:, i] /= r

    # connectivity matrix
    cmat = KDTree(idx)
    cmat = cmat.sparse_distance_matrix(
        cmat, 1, p=np.inf, output_type='coo_matrix')
    cmat.setdiag(1)

    # pair indices
    I, J = cmat.nonzero()

    # delete cmat
    cmat_shape = cmat.shape
    del cmat

    # sum over columns
    V_sum = sparse.bsr_matrix((V[J], (I, I)),
                              shape=cmat_shape,
                              dtype=V.dtype).diagonal(0)

    # count over columns
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

    # copy indices
    idx = idx.copy().astype(w.dtype)

    # scale
    for i, r in enumerate(radius):
        # increase inter-index distance
        if r < 1:
            idx[:, i] *= 2

        # do nothing
        elif r == 1:
            pass

        # decrease inter-index distance
        else:
            idx[:, i] /= r

    if pindex is None:
        pindex = np.arange(len(idx))

    # connectivity matrix
    tree_all = KDTree(idx)
    tree_subset = KDTree(idx[pindex])
    cmat = tree_subset.sparse_distance_matrix(
        tree_all, 1, p=np.inf, output_type='coo_matrix')
    del tree_all, tree_subset

    # pair indices
    I, J = cmat.nonzero()

    # self
    I = np.concatenate((I, np.arange(len(pindex))))
    J = np.concatenate((J, pindex))

    # delete connectivity matrix
    cmat_shape = cmat.shape
    del cmat

    # sum weights over columns
    # only need to do this once
    V_count = sparse.bsr_matrix((w[J], (I, I)),
                                shape=cmat_shape,
                                dtype=w.dtype).diagonal(0)

    # reshape V if 1D
    if V.ndim == 1:
        V = V.reshape((-1, 1))

    # output container
    V_out = np.empty_like(V[pindex, :], dtype=w.dtype)

    # enumerate value columns
    for i in range(V_out.shape[1]):
        # sum weighted values over columns
        V_sum = sparse.bsr_matrix((w[J] * V[J, i], (I, I)),
                                  shape=cmat_shape,
                                  dtype=w.dtype).diagonal(0)

        V_out[:, i] = V_sum / V_count

    # flatten if 1D
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

    # copy indices
    idx = idx.copy().astype(V.dtype)

    # scale
    for i, r in enumerate(radius):
        # increase inter-index distance
        if r < 1:
            idx[:, i] *= 2

        # do nothing
        elif r == 1:
            pass

        # decrease inter-index distance
        else:
            idx[:, i] /= r

    # connectivity matrix
    cmat = KDTree(idx)
    cmat = cmat.sparse_distance_matrix(
        cmat, 1, p=np.inf, output_type='coo_matrix')
    cmat.setdiag(1)

    # pair indices
    I, J = cmat.nonzero()

    # delete cmat
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


def smooth(features, dims=['mz', 'drift_time', 'retention_time'],
           radius=[0, 1, 1]):
    '''
    Smooth data by sparse mean filtration.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    dims : str or list
        Dimensions to perform peak detection in.
    radius : float or list
        Radius of the sparse filter in each dimension. Values less than
        zero indicate no connectivity in that dimension.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Smoothed feature coordinates and intensities.

    '''

    # safely cast to list
    dims = deimos.utils.safelist(dims)
    radius = deimos.utils.safelist(radius)

    # check dims
    deimos.utils.check_length([dims, radius])

    # copy input
    features = features.copy()

    # get indices
    idx = np.vstack([pd.factorize(features[dim], sort=True)[0].astype(np.int32)
                    for dim in dims]).T

    # values
    V = features['intensity'].values

    # sparse mean filtration
    features['intensity'] = sparse_mean_filter(idx, V, radius)

    return features
