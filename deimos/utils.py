import pandas as pd
import numpy as np


def safelist(x):
    """
    Ensures passed object is of correct list-like format.

    Parameters
    ----------
    x : any
        Object to be cast as list.

    Returns
    -------
    out : list_like
        Input safely cast to list-like.

    """

    if not isinstance(x, (list, pd.core.series.Series, np.ndarray)):
        return [x].copy()
    return x.copy()


def check_length(lists):
    """
    Ensures collection of lists passed are of equal length. If not,
    a value error is raised.

    Parameters
    ----------
    lists : list
        List of lists for length evaluation.

    Returns
    -------
    None.

    """

    it = iter(lists)
    length = len(next(it))
    if not all(len(x) == length for x in it):
        raise ValueError('per-dimension inputs must have same dimension')


def detect_features(data):
    """
    Detects non-intensity feature columns in the input.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.

    Returns
    -------
    features : list
        List of non-intensity features in the input.

    """

    features = list(data.columns)
    features.remove('intensity')
    return features
