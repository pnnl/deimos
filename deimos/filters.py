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
