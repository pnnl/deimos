from scipy import stats
import numpy as np
import pandas as pd


def data2grid(mz, dt, intensity, mz_bins='auto', dt_bins='auto'):
    if mz_bins.lower() == 'auto':
        mz_bins = (mz.max() - mz.min()) / np.min(np.diff(np.sort(mz.unique())))
    if dt_bins.lower() == 'auto':
        dt_bins = (dt.max() - dt.min()) / np.min(np.diff(np.sort(dt.unique())))

    H, xe, ye, bn = stats.binned_statistic_2d(mz, dt, intensity,
                                              statistic='sum',
                                              bins=(mz_bins, dt_bins))
    H = np.nan_to_num(H)
    XX, YY = np.meshgrid(xe, ye, indexing='ij')

    return XX, YY, H


def grid2df(x, y, z, top=None):
    # bin centers
    x = (x[1:, 1:] + x[:-1, :-1]) / 2
    y = (y[1:, 1:] + y[:-1, :-1]) / 2

    data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
    data = pd.DataFrame(data, columns=['mz', 'drift_time', 'intensity'])
    data = data.loc[data['intensity'] > 0, :].sort_values(by='intensity', ascending=False).reset_index(drop=True)

    if top is not None:
        return data.loc[:top, :]
    else:
        return data
