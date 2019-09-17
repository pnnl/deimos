from scipy import stats
import numpy as np
import pandas as pd
from spx.utils import safelist, check_length


def data2grid(data, features=['mz', 'drift_time', 'retention_time'], intensity='intensity',
              res='auto'):
    # safely cast to list
    features = safelist(features)

    # auto detect res
    if res == 'auto':
        res = np.min(np.diff(np.sort(np.unique(features, axis=0), axis=0), axis=0), axis=0)
    else:
        # safely cast to list
        res = safelist(res)

        # check dims
        check_length([features, res])

    # separate features, intensity
    f = data[features].values
    i = data[intensity].values

    # calculate bin counts
    bins = (f.max(axis=0) - f.min(axis=0)) / res

    # grid
    H, edges, bn = stats.binned_statistic_dd(f, i,
                                             statistic='sum',
                                             bins=bins)

    # nans to zeros
    H = np.nan_to_num(H)

    return edges, H


def grid2df(edges, intensity, features=['mz', 'drift_time', 'retention_time'], top=None):
    # safely cast to list
    features = safelist(features)

    # edges to grid
    grid = edges2grid(edges)

    # create dataframe
    data = pd.DataFrame(grid, columns=features)
    data['intensity'] = intensity

    # threshold and sort
    data = data.loc[data['intensity'] > 0, :].sort_values(by='intensity', ascending=False).reset_index(drop=True)

    # return top hits
    if top is not None:
        return data.loc[:top, :]

    # return all hits
    return data


def edges2grid(edges):
    # mesh edges
    grid = np.meshgrid(*edges)

    # flatten
    grid = [x.flatten() for x in grid]

    # center bins
    grid = [(x[1:] + x[:-1]) / 2 for x in grid]

    # stack and transpose
    grid = np.stack(grid).T

    return grid
