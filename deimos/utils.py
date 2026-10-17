from typing import Any, overload

import numpy as np
import pandas as pd


# Series and arrays keep their type; any other input is wrapped in a list
@overload
def safelist(x: pd.Series) -> pd.Series: ...  # pyright: ignore[reportOverlappingOverload]
@overload
def safelist(x: np.ndarray) -> np.ndarray: ...  # pyright: ignore[reportOverlappingOverload]
@overload
def safelist(x: Any) -> list[Any]: ...
def safelist(x: Any) -> list[Any] | pd.Series | np.ndarray:
    """
    Ensures passed object is of correct list-like format.

    Parameters
    ----------
    x : any
        Object to be cast as list.

    Returns
    -------
    list-like
        Input safely cast to list-like.

    """

    if not isinstance(x, (list, pd.Series, np.ndarray)):
        return [x].copy()

    return x.copy()


def check_length(lists):
    """
    Ensures collection of lists passed are of equal length.

    Parameters
    ----------
    lists : list of list
        List of lists for length evaluation.

    Raises
    ------
    ValueError
        If lists are not the same length.

    """

    it = iter(lists)
    length = len(next(it))

    if not all(len(x) == length for x in it):
        raise ValueError("Per-dimension inputs must have same dimension.")


def detect_dims(features):
    """
    Detects non-intensity columns in the input.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.

    Returns
    -------
    list
        List of non-intensity features in the input.

    """

    dims = list(features.columns)
    dims.remove("intensity")
    return dims
