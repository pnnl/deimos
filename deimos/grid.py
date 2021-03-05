import numpy as np
import pandas as pd
import deimos


def data2grid(data, features=['mz', 'drift_time', 'retention_time']):
    """
    Converts data frame representation to a dense, N-dimensional grid.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    features : str or list
        Feature dimension(s) to create the dense grid (omitted dimensions will
        be collapsed and summed accross).

    Returns
    -------
    edges : ndarray(s)
        Edges coordinates along each grid axis.
    grid : ndarray
        Resulting N-dimensional grid.

    """

    # safely cast to list
    features = deimos.utils.safelist(features)

    if len(features) < len(deimos.utils.detect_features(data)):
        data = deimos.utils.collapse(data, keep=features, how=np.sum)

    idx = [np.unique(data.loc[:, f].values,
                     return_inverse=True) for f in features]
    idx_i = [x[-1] for x in idx]
    idx = [x[0] for x in idx]

    grid = np.full([len(x) for x in idx], np.nan, dtype=np.float32)
    grid[tuple(idx_i)] = data.loc[:, 'intensity'].values

    return idx, grid


def grid2df(edges, grid, features=['mz', 'drift_time', 'retention_time'],
            additional=None):
    """
    Converts dense grid representation to a data frame.

    Parameters
    ----------
    edges : ndarray(s)
        Edges coordinates along each grid axis.
    grid : ndarray
        N-dimensional dense grid of intensities.
    features : str or list
        Feature label(s) for each grid dimension.

    Returns
    -------
    out : DataFrame
        Data frame representation of input grid.

    """

    # column labels container
    cols = features.copy()

    # flatten grid and threshold
    grid = grid.flatten()
    idx = grid > 0
    grid = grid[idx].reshape(-1, 1)

    # edges to grid
    data = np.meshgrid(*edges, indexing='ij')
    data = [x.reshape(-1, 1)[idx] for x in data]

    # append intensity
    data.append(grid)
    cols.append('intensity')

    # append additional columns
    if additional is not None:
        for k, v in additional.items():
            data.append(v.flatten()[idx].reshape(-1, 1))
            cols.append(k)

    del additional, idx

    return pd.DataFrame(np.hstack(data), columns=cols, dtype=np.float32)
