import scipy.ndimage as ndi
import numpy as np
import deimos


def stdev(a, size):
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

    c1 = ndi.filters.uniform_filter(a, size, mode='constant')
    c2 = ndi.filters.uniform_filter(np.square(a), size, mode='constant')
    return np.abs(np.lib.scimath.sqrt(c2 - np.square(c1)))


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

    return ndi.maximum_filter(a, size=size, mode='constant', cval=0)


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

    return ndi.maximum_filter(-a, size=size, mode='constant', cval=0)


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
        ksize = size ** len(a.shape)
    else:
        ksize = np.prod(size)
    return ksize * ndi.filters.uniform_filter(a, size, mode='constant')


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

    a = np.where(np.nan_to_num(a) > 0, 1.0, 0.0)
    return ndi.uniform_filter(a, size=size, mode='constant', cval=0)


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

    return np.square(ndi.gaussian_filter(a, size, mode='constant', cval=0))


def signal_to_noise_ratio(a, size):
    '''
    N-dimensional convolution of a signal-to-noise filter.

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

    c1 = ndi.filters.uniform_filter(a, size, mode='constant')
    c2 = ndi.filters.uniform_filter(np.square(a), size, mode='constant')
    std = np.abs(np.lib.scimath.sqrt(c2 - np.square(c1)))
    return np.square(np.divide(c1, std, out=np.zeros_like(c1), where=std != 0))


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


def kurtosis(edges, a, size):
    '''
    N-dimensional convolution of a standard deviation filter.

    Parameters
    ----------
    edges : list of :obj:`~numpy.array`
        Edge coordinates along each grid axis.
    a : :obj:`~numpy.array`
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    :obj:`~numpy.array`
        Filtered intensity data.

    '''

    k = []
    for i, (e, s) in enumerate(zip(edges, size)):
        ax = tuple(x for x in range(len(edges)) if x is not i)
        freq = np.sum(a, axis=ax)

        # conv
        total = s * ndi.filters.uniform_filter(freq, s, mode='constant')
        xbar = s * ndi.filters.uniform_filter(e * freq, s,
                                              mode='constant') / total
        m2 = s * ndi.filters.uniform_filter(np.square(e - xbar) * freq, s,
                                            mode='constant') / total
        m4 = s * ndi.filters.uniform_filter(np.power(e - xbar, 4) * freq, s,
                                            mode='constant') / total
        k.append(m4 / np.square(m2) - 3.0)

    k = np.meshgrid(*k, indexing='ij')
    return k
