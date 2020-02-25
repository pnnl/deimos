import scipy.ndimage as ndi
import numpy as np


def stdev_filter(a, size):
    """
    N-dimensional convolution of a standard deviation filter.

    Parameters
    ----------
    a : ndarray
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    out : ndarray
        Filtered intensity data.

    """

    c1 = ndi.filters.uniform_filter(a, size, mode='constant')
    c2 = ndi.filters.uniform_filter(np.square(a), size, mode='constant')
    return np.abs(np.lib.scimath.sqrt(c2 - np.square(c1)))


def maximum(a, size):
    """
    N-dimensional convolution of a maximum filter.

    Parameters
    ----------
    a : ndarray
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    out : ndarray
        Filtered intensity data.

    """

    return ndi.maximum_filter(a, size=size, mode='constant', cval=0)


def matched_gaussian(a, size):
    """
    N-dimensional convolution of a matched Gaussian filter.

    Parameters
    ----------
    a : ndarray
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    out : ndarray
        Filtered intensity data.

    """

    return np.square(ndi.gaussian_filter(a, size, mode='constant', cval=0))


def signal_to_noise_ratio(a, size):
    """
    N-dimensional convolution of a signal-to-noise filter.

    Parameters
    ----------
    a : ndarray
        N-dimensional array of intensity data.
    size : int or list
        Size of the convolution kernel in each dimension.

    Returns
    -------
    out : ndarray
        Filtered intensity data.

    """

    c1 = ndi.filters.uniform_filter(a, size, mode='constant')
    c2 = ndi.filters.uniform_filter(np.square(a), size, mode='constant')
    std = np.abs(np.lib.scimath.sqrt(c2 - np.square(c1)))
    return np.square(np.divide(c1, std, out=np.zeros_like(c1), where=std != 0))
