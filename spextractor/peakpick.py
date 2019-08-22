import numpy as np
import scipy.ndimage as ndi
import spextractor as spx
import pandas as pd


def auto(ms, res=[0.005, 0.12], sigma=[0.06, 0.3], truncate=4, threshold=1000):
    # grid data
    xe, ye, H = spx.grid.data2grid(ms['mz'], ms['drift_time'], ms['intensity'],
                                   x_res=res[0], y_res=res[1])

    points = [int(s / r) for s, r in zip(sigma, res)]

    # matched filter
    corr = matched_filter(H, [x + 1 if x % 2 == 0 else x for x in points])

    # peaks
    footprint = [(truncate * x) + 1 if (truncate * x) % 2 == 0 else truncate * x for x in points]
    peaks = non_maximum_suppression(corr, footprint)
    peaks = spx.grid.grid2df(xe, ye, peaks)
    peaks = peaks.loc[peaks['intensity'] > threshold, :]

    # reconcile with original data
    peaks = reconcile(peaks, ms, sigma=sigma, truncate=truncate)
    peaks = peaks.loc[peaks['intensity'] > threshold, :]

    return peaks


def reconcile(peaks, data, sigma=[0.06, 0.3], truncate=4):
    res = {'mz': [], 'drift_time': [], 'intensity': []}
    for idx, row in peaks.iterrows():
        mz_i = row['mz']
        dt_i = row['drift_time']
        subset = spx.targeted.find_feature(data,
                                           mz=mz_i,
                                           dt=dt_i,
                                           mz_tol=sigma[0] * truncate,
                                           dt_tol=sigma[1] * truncate)

        # sum
        subset_mz = subset.groupby(by='mz', as_index=False).agg({'intensity': np.sum})
        subset_dt = subset.groupby(by='drift_time', as_index=False).agg({'intensity': np.sum})

        # store
        res['mz'].append(subset_mz.loc[subset_mz['intensity'].idxmax(), 'mz'])
        res['drift_time'].append(subset_dt.loc[subset_dt['intensity'].idxmax(), 'drift_time'])
        res['intensity'].append(subset_mz['intensity'].max())

    return pd.DataFrame(res)


def non_maximum_suppression(ndarray, size):
    # peak pick
    H_max = ndi.maximum_filter(ndarray, size=size, mode='constant', cval=0)
    peaks = np.where(ndarray == H_max, ndarray, 0)

    return peaks


def matched_filter(ndarray, size):
    return np.square(ndi.gaussian_filter(ndarray, size, mode='constant', cval=0))


def guided(ms, mz=None, dt=None, mz_tol=6E-6, dt_tol=0.12,
           res=[0.005, 0.12], sigma=[0.06, 0.3], truncate=4, threshold=1000):
    subset = spx.targeted.find_feature(ms, mz=mz, dt=dt, mz_tol=mz_tol, dt_tol=dt_tol)

    if subset is None:
        return subset

    peaks = auto(subset, res=res, sigma=sigma, truncate=truncate, threshold=threshold)
    return peaks
