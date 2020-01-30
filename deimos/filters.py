import scipy.ndimage as ndi
import numpy as np


def stdev_filter(ndarray, size):
    c1 = ndi.filters.uniform_filter(ndarray, size, mode='constant')
    c2 = ndi.filters.uniform_filter(np.square(ndarray), size, mode='constant')
    return np.abs(np.lib.scimath.sqrt(c2 - np.square(c1)))


def non_maximum_suppression(ndarray, size):
    return ndi.maximum_filter(ndarray, size=size, mode='constant', cval=0)


def matched_gaussian(ndarray, size):
    return np.square(ndi.gaussian_filter(ndarray, size, mode='constant', cval=0))


def signal_to_noise_ratio(ndarray, size):
    c1 = ndi.filters.uniform_filter(ndarray, size, mode='constant')
    c2 = ndi.filters.uniform_filter(np.square(ndarray), size, mode='constant')
    std = np.abs(np.lib.scimath.sqrt(c2 - np.square(c1)))
    return np.square(np.divide(c1, std, out=np.zeros_like(c1), where=std != 0))
