import numpy as np
import deimos


def auto(data, features=['mz', 'drift_time', 'retention_time'],
         res=[0.002445220947265625, 0.12024688720703125, 0.03858184814453125],
         sigma=[0.025, 0.2, 0.11], truncate=4):
    """
    Helper function to perform peak detection on a single partition.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    features : str or list
        Feature dimensions to perform peak detection in
        (omitted dimensions will be collapsed and summed accross)..
    res : float or list
        Acquisition resolution in each dimension.
    sigma : float or list
        Width of a prototypical peak in each dimension.
    truncate : int
        Number of sigmas on either side of a peak considered
        during local non-maximum suppression.

    Returns
    -------
    peaks : DataFrame
        Coordinates of detected peaks and associated apex intensitites.

    """

    # safely cast to list
    features = deimos.utils.safelist(features)
    res = deimos.utils.safelist(res)
    sigma = deimos.utils.safelist(sigma)

    # check dims
    deimos.utils.check_length([features, res, sigma])

    # grid data
    edges, H = deimos.grid.data2grid(data, features=features)

    # sigma in terms of n points
    points = [s / r for s, r in zip(sigma, res)]

    # truncate * sigma size
    size = [np.ceil(truncate * x) for x in points]

    # matched filter
    corr = deimos.filters.matched_gaussian(H, points)

    # peak detection
    H_max = deimos.filters.maximum(corr, size)
    peaks = np.where(corr == H_max, H, 0)

    # convert to dataframe
    peaks = deimos.grid.grid2df(edges, peaks, features=features)

    return peaks
