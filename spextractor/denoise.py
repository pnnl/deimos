import skimage.ndimage as ndi


def median(ndarray, selem):
    return ndi.median_filter(ndarray, footprint=selem, mode='constant')


def percentile(ndarray, selem, p=0.5):
    return ndi.percentile_filter(ndarray, percentile=p, footprint=selem, mode='constant')


def gaussian(ndarray, sigma):
    return ndi.gaussian_filter(ndarray, sigma, mode='constant')
