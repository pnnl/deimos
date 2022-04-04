import deimos
import numpy as np
import scipy.ndimage as ndi


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


def snr(a, size):
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

    # get min
    mn = ndi.minimum_filter(a, size=size, mode='constant', cval=np.inf)
    
    # get median
    median = ndi.median_filter(a, size=size, mode='nearest')
    
    # noise basis
    floor = 2 * (median - mn)

    # compute snr
    return np.square(np.divide(a - mn, floor,
                               out=np.ones_like(a),
                               where=floor != 0))


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
    N-dimensional convolution of a kurtosis filter. Kurtosis
    is determined per dimension.

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

    # argument checks
    edges = deimos.utils.safelist(edges)
    size = deimos.utils.safelist(size)
    deimos.utils.check_length([edges, size, a.shape])

    # result container
    k = []

    # define edges on grid
    edges = np.meshgrid(*edges)

    # enumerate dims
    for i, (e, s) in enumerate(zip(edges, size)):
        freq = ndi.uniform_filter1d(a, size=s, axis=i)

        # frequency sum
        total = s * ndi.uniform_filter(freq, s, mode='constant', cval=0.0)

        # x-weighted sum
        xbar = s * ndi.uniform_filter(e * freq, s,
                                      mode='constant', cval=0.0) / total

        # second moment
        m2 = s * ndi.uniform_filter(np.square(e - xbar) * freq, s,
                                    mode='constant', cval=0.0) / total

        # fourth moment
        m4 = s * ndi.uniform_filter(np.power(e - xbar, 4) * freq, s,
                                    mode='constant', cval=0.0) / total

        # kurtosis calculation
        k.append(m4 / np.square(m2) - 3.0)

    return k
