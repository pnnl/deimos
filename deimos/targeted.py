from deimos.utils import safelist, check_length
import pandas as pd
import numpy as np


def find_feature(data, by=['mz', 'drift_time', 'retention_time'],
                 loc=[0, 0, 0], tol=[0, 0, 0], return_index=False):
    """
    Given a feature coordinate and tolerances, return a subset
    of the data.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    by : str or list
        Feature(s) by which to subset the data
    loc : float or list
        Coordinate location.
    tol : float or list
        Tolerance in each dimension.
    return_index : bool
        Return indices of subset.

    Returns
    -------
    out : DataFrame
        Subset of feature coordinates and intensities.
    index : array
        If `return_index` is True, indices of slice elements.

    """

    # safely cast to list
    by = safelist(by)
    loc = safelist(loc)
    tol = safelist(tol)

    # check dims
    check_length([by, loc, tol])

    if data is None:
        if return_index is True:
            return None, None
        else:
            return None

    # store index
    rindex = data.index.values

    # extend columns
    cols = data.columns
    cidx = [cols.get_loc(x) for x in by]

    # subset by each dim
    data = data.values
    idx = np.full(data.shape[0], True, dtype=bool)
    for i, x, dx in zip(cidx, loc, tol):
        idx *= (data[:, i] <= x + dx) & (data[:, i] >= x - dx)

    data = data[idx]
    rindex = rindex[idx]

    if return_index is True:
        # data found
        if data.shape[0] > 0:
            return pd.DataFrame(data, index=rindex, columns=cols), idx

        # no data
        return None, idx
    else:
        # data found
        if data.shape[0] > 0:
            return pd.DataFrame(data, index=rindex, columns=cols)

        # no data
        return None


def slice(data, by=['mz', 'drift_time', 'retention_time'],
          low=[0, 0, 0], high=[0, 0, 0], return_index=False):
    """
    Given a feature coordinate and bounds, return a subset
    of the data.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    by : str or list
        Feature(s) by which to subset the data
    low : float or list
        Lower bound(s) in each dimension.
    high : float or list
        Upper bound(s) in each dimension.
    return_index : bool
        Return indices of subset.

    Returns
    -------
    out : DataFrame
        Subset of feature coordinates and intensities.
    index : array
        If `return_index` is True, indices of slice elements.

    """

    # safely cast to list
    by = safelist(by)
    low = safelist(low)
    high = safelist(high)

    # check dims
    check_length([by, low, high])

    if data is None:
        if return_index is True:
            return None, None
        else:
            return None

    # store index
    rindex = data.index.values

    # extend columns
    cols = data.columns
    cidx = [cols.get_loc(x) for x in by]

    # subset by each dim
    data = data.values
    idx = np.full(data.shape[0], True, dtype=bool)
    for i, lb, ub in zip(cidx, low, high):
        idx *= (data[:, i] <= ub) & (data[:, i] >= lb)

    data = data[idx]
    rindex = rindex[idx]

    if return_index is True:
        # data found
        if data.shape[0] > 0:
            return pd.DataFrame(data, index=rindex, columns=cols), idx

        # no data
        return None, idx
    else:
        # data found
        if data.shape[0] > 0:
            return pd.DataFrame(data, index=rindex, columns=cols)

        # no data
        return None
