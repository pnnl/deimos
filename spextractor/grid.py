from scipy import stats
import numpy as np
import pandas as pd


def data2grid(features, intensity, resolution='auto'):
    if resolution == 'auto':
        res = np.min(np.diff(np.sort(np.unique(features, axis=0), axis=0), axis=0), axis=0)
    elif len(resolution) == features.shape[-1]:
        res = np.array(resolution)
    else:
        raise ValueError('dimension mismatch between features and resolution')

    bins = (features.max(axis=0) - features.min(axis=0)) / res

    H, edges, bn = stats.binned_statistic_dd(features, intensity,
                                             statistic='sum',
                                             bins=bins)
    H = np.nan_to_num(H)
    grid = [x.flatten() for x in np.meshgrid(*edges)]
    grid = [(x[1:] + x[:-1]) / 2 for x in grid]
    grid = np.stack(grid).T

    return grid, H


def grid2df(grid, intensity, columns=['mz', 'drift_time', 'retention_time'], top=None):
    data = pd.DataFrame(grid, columns=columns)
    data['intensity'] = intensity
    data = data.loc[data['intensity'] > 0, :].sort_values(by='intensity', ascending=False).reset_index(drop=True)

    if top is not None:
        return data.loc[:top, :]
    else:
        return data
