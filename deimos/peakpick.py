import numpy as np
import deimos


def auto(data, features=['mz', 'drift_time', 'retention_time'],
         res=[0.002445220947265625, 0.12024688720703125, 0.03858184814453125],
         sigma=[0.004, 0.2, 0.11], truncate=4, threshold=1E3):
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
    H_max = deimos.filters.non_maximum_suppression(corr, size)
    peaks = np.where(corr == H_max, H, 0)

    # convert to dataframe
    peaks = deimos.grid.grid2df(edges, peaks, features=features)

    # threshold
    peaks = peaks.loc[peaks['intensity'] > threshold, :]

    return peaks
