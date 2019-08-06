import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import disk


def _gaussian_footprint(fwhm=3):
    sigma = fwhm / 2.35482004503
    struct = disk(int(4 * sigma))
    return struct


def non_maximum_suppression(ndarray, struct=_gaussian_footprint()):
    # peak pick
    H_max = ndi.maximum_filter(ndarray, footprint=struct, mode='constant')
    peaks = np.where(ndarray == H_max, ndarray, 0)

    return peaks


def matched_filter(ndarray, struct):
    return ndi.correlate(ndarray, struct, mode='constant', cval=0.0)
