import numpy as np
import scipy.ndimage as ndi
import spextractor as spx
import pandas as pd


def auto(data, features=['mz', 'drift_time', 'retention_time'], intensity='intensity',
         res=[0.01, 0.12, 1], sigma=[0.06, 0.3, 1], truncate=4, threshold=1000):
    # safely cast to list
    features = spx.utils.safelist(features)
    res = spx.utils.safelist(res)
    sigma = spx.utils.safelist(sigma)

    # check dims
    spx.utils.check_length([features, res, sigma])

    # grid data
    edges, H = spx.grid.data2grid(data, features=features, intensity=intensity, res=res)

    # sigma in terms of n points
    points = [int(s / r) for s, r in zip(sigma, res)]

    # matched filter
    corr = matched_filter(H, [x + 1 if x % 2 == 0 else x for x in points])

    # peak detection
    footprint = [(truncate * x) + 1 if (truncate * x) % 2 == 0 else truncate * x for x in points]
    peaks = non_maximum_suppression(corr, footprint)

    # convert to dataframe
    peaks = spx.grid.grid2df(edges, peaks, features=features)

    # threshold
    peaks = peaks.loc[peaks['intensity'] > threshold, :]

    # reconcile with original data
    peaks = reconcile(peaks, data, features=features, intensity=intensity,
                      sigma=sigma, truncate=truncate)

    # threshold
    peaks = peaks.loc[peaks['intensity'] > threshold, :].reset_index(drop=True)

    return peaks


def reconcile(peaks, data, features=['mz', 'drift_time', 'retention_time'], intensity='intensity',
              sigma=[0.06, 0.3, 1], truncate=4):
    # safely cast to list
    features = spx.utils.safelist(features)
    sigma = spx.utils.safelist(sigma)

    # check dims
    spx.utils.check_length([features, sigma])

    # build containers
    res = {k: [] for k in features}
    res[intensity] = []

    # iterate peaks
    for idx, row in peaks.iterrows():
        # targeted search
        subset = spx.targeted.find_feature(data,
                                           by=features,
                                           loc=row[features].values,
                                           tol=sigma * np.array(truncate))

        # combine by each feature
        for f in features:
            # sum
            subset_f = subset.groupby(by=f, as_index=False).agg({intensity: np.sum})
            res[f].append(subset_f.loc[subset_f[intensity].idxmax(), f])

            # get peak intensity from mz dim
            if f == 'mz':
                res[intensity].append(subset_f[intensity].max())

    return pd.DataFrame(res)


def non_maximum_suppression(ndarray, size):
    # peak pick
    H_max = ndi.maximum_filter(ndarray, size=size, mode='constant', cval=0)
    peaks = np.where(ndarray == H_max, ndarray, 0)

    return peaks


def matched_filter(ndarray, size):
    return np.square(ndi.gaussian_filter(ndarray, size, mode='constant', cval=0))


def guided(data, by=['mz', 'drift_time', 'retention_time'], intensity='intensity',
           res=[0.01, 0.12, 1], loc=[0, 0, 0], sigma=[0.06, 0.3, 1], truncate=4, threshold=1000):
    # safely cast to list
    by = spx.utils.safelist(by)
    res = spx.utils.safelist(res)
    loc = spx.utils.safelist(loc)
    sigma = spx.utils.safelist(sigma)

    # check dims
    spx.utils.check_length([by, res, loc, sigma])

    # targeted search
    subset = spx.targeted.find_feature(data,
                                       by=by,
                                       loc=loc,
                                       tol=sigma * np.array(truncate))

    # if no data found
    if subset is None:
        return subset

    # peakpick on subset
    peaks = auto(subset, features=by, intensity=intensity,
                 res=res, sigma=sigma, truncate=truncate, threshold=threshold)

    # if no peaks found
    if len(peaks.index) == 0:
        return None

    return peaks
