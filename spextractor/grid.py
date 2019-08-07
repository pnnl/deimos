from scipy import stats
import numpy as np
import pandas as pd


def data2grid(x, y, z, x_res='auto', y_res='auto'):
    if x_res == 'auto':
        x_res = np.min(np.diff(np.sort(x.unique())))
    if y_res == 'auto':
        y_res = np.min(np.diff(np.sort(y.unique())))

    x_bins = (x.max() - x.min()) / x_res
    y_bins = (x.max() - x.min()) / y_res

    H, xe, ye, bn = stats.binned_statistic_2d(x, y, z,
                                              statistic='sum',
                                              bins=(x_bins, y_bins))
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
