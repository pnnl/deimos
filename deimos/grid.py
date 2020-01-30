import numpy as np
import pandas as pd
import deimos


def data2grid(data, features=['mz', 'drift_time', 'retention_time']):
    # safely cast to list
    features = deimos.utils.safelist(features)

    data = deimos.utils.collapse(data, keep=features, how=np.sum)

    idx = [np.unique(data.loc[:, f].values, return_inverse=True) for f in features]
    idx_i = [x[-1] for x in idx]
    idx = [x[0] for x in idx]

    grid = np.zeros([len(x) for x in idx], dtype=np.float32)
    grid[tuple(idx_i)] = data.loc[:, 'intensity'].values

    return idx, grid


def grid2df(edges, intensity, features=['mz', 'drift_time', 'retention_time']):
    # edges to grid
    axes = np.meshgrid(*edges, indexing='ij')
    grid = np.hstack([x.reshape(-1, 1) for x in axes])

    # create dataframe
    data = pd.DataFrame(grid, columns=features)
    data['intensity'] = intensity.flatten()

    # threshold and sort
    data = data.loc[data['intensity'] > 0, :].sort_values(by='intensity', ascending=False).reset_index(drop=True)

    # return all hits
    return data
