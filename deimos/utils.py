import numpy as np
import pandas as pd


def safelist(x):
    '''
    Ensures passed object is of correct list-like format.

    Parameters
    ----------
    x : any
        Object to be cast as list.

    Returns
    -------
    list-like
        Input safely cast to list-like.

    '''

    if not isinstance(x, (list, pd.core.series.Series, np.ndarray)):
        return [x].copy()

    return x.copy()


def check_length(lists):
    '''
    Ensures collection of lists passed are of equal length.

    Parameters
    ----------
    lists : list of list
        List of lists for length evaluation.

    Raises
    ------
    ValueError
        If lists are not the same length.

    '''

    it = iter(lists)
    length = len(next(it))

    if not all(len(x) == length for x in it):
        raise ValueError('Per-dimension inputs must have same dimension.')


def detect_dims(features):
    '''
    Detects non-intensity columns in the input.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.

    Returns
    -------
    list
        List of non-intensity features in the input.

    '''

    dims = list(features.columns)
    dims.remove('intensity')
    return dims
