import numpy as np
import deimos
import pandas as pd
import multiprocessing as mp
from functools import partial


def auto(data, features=['mz', 'drift_time', 'retention_time'],
         res=[0.002445220947265625, 0.12024688720703125, 0.03858184814453125],
         sigma=[0.004, 0.2, 0.11], truncate=4, threshold=1E3,
         split_on='mz', partitions=500, overlap=0.05, processes=mp.cpu_count()):

    # safely cast to list
    features = deimos.utils.safelist(features)
    res = deimos.utils.safelist(res)
    sigma = deimos.utils.safelist(sigma)

    # check dims
    deimos.utils.check_length([features, res, sigma])

    # unique to split on
    idx = np.unique(data[split_on].values)

    # determine partition bounds
    splits = [[x.min(), x.max()] for x in np.array_split(idx, partitions)]
    for i in range(1, len(splits)):
        splits[i][0] = splits[i - 1][1]

    # partition data
    data = [deimos.targeted.slice(data, by=split_on, low=a - overlap, high=b) for a, b in splits]

    # peak pick in parallel
    with mp.Pool(processes=processes) as p:
        peaks = p.map(partial(_run, features=features, res=res,
                              sigma=sigma, truncate=truncate, threshold=threshold),
                      data)

    # reconcile overlap
    for i in range(len(peaks)):
        a, b = splits[i]

        # first partition
        if i < 1:
            b = b - overlap / 2

        # middle partitions
        elif i < len(peaks) - 1:
            a = a + overlap / 2
            b = b - overlap / 2

        # last partition
        else:
            a = a + overlap / 2

        peaks[i] = deimos.targeted.slice(peaks[i], by=split_on, low=a, high=b)

    # combine partitions
    peaks = pd.concat(peaks).reset_index(drop=True)

    return peaks


def _run(data, features=['mz', 'drift_time', 'retention_time'],
         res=[0.002445220947265625, 0.12024688720703125, 0.03858184814453125],
         sigma=[0.004, 0.2, 0.11], truncate=4, threshold=1E3):

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
