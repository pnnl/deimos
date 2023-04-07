import numpy as np
import pandas as pd

import deimos


def data2grid(features, dims=['mz', 'drift_time', 'retention_time']):
    '''
    Converts data frame representation to a dense, N-dimensional grid.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    dims : str or list
        Dimension(s) to create the dense grid (omitted dimensions will
        be collapsed and summed accross).

    Returns
    -------
    edges : list of :obj:`~numpy.array`
        Edges coordinates along each grid axis.
    grid : :obj:`~numpy.array`
        Resulting N-dimensional grid.

    '''

    # Safely cast to list
    dims = deimos.utils.safelist(dims)

    # Collapse data if necessary
    if len(dims) < len(deimos.utils.detect_dims(features)):
        features = deimos.collapse(features, keep=dims, how=np.sum)

    # Unique indices
    idx = [np.unique(features.loc[:, d].values,
                     return_inverse=True) for d in dims]
    idx_i = [x[-1] for x in idx]
    idx = [x[0] for x in idx]

    # Populate grid
    grid = np.zeros([len(x) for x in idx], dtype=float)
    grid[tuple(idx_i)] = features.loc[:, 'intensity'].values

    return idx, grid


def grid2df(edges, grid, dims=['mz', 'drift_time', 'retention_time'],
            additional=None):
    '''
    Converts dense grid representation to a data frame.

    Parameters
    ----------
    edges : list of :obj:`~numpy.array`
        Edges coordinates along each grid axis.
    grid : :obj:`~numpy.array`
        N-dimensional dense grid of intensities.
    dims : str or list
        Label(s) for each grid dimension.
    additional : dict or None
        Additional grids to process.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Feature coordinates, intensities, and any other attributes from
        `additional`.

    '''

    # Cast to list safely
    dims = deimos.utils.safelist(dims)

    # Column labels container
    cols = dims.copy()

    # Flatten grid
    grid = grid.flatten()

    # Threshold
    idx = grid > 0

    # Reshape
    grid = grid[idx].reshape(-1, 1)

    # Edges to grid
    data = np.meshgrid(*edges, indexing='ij')
    data = [x.reshape(-1, 1)[idx] for x in data]

    # Append intensity
    data.append(grid)
    cols.append('intensity')

    # Append additional columns
    if additional is not None:
        for k, v in additional.items():
            data.append(v.flatten()[idx].reshape(-1, 1))
            cols.append(k)

    del additional, idx

    return pd.DataFrame(np.hstack(data), columns=cols, dtype=float)
