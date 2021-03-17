from deimos.utils import safelist, check_length
import pandas as pd
import numpy as np
from functools import partial
import multiprocessing as mp


def threshold(data, by='intensity', threshold=0):
    '''
    Thresholds input :obj:`~pandas.DataFrame` using `by` keyword, greater than
    value passed to `threshold`.

    Parameters
    ----------
    data : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : str
        Variable to threshold by.
    threshold : float
        Threshold value.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Thresholded feature coordinates.

    '''

    return data.loc[data[by] > threshold, :].reset_index(drop=True)


def collapse(data, keep=['mz', 'drift_time', 'retention_time'], how=np.sum):
    '''
    Collpases input data such that only specified dimensions remain, according
    to the supplied aggregation function.

    Parameters
    ----------
    data : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    keep : str or list
        Features to keep during collapse operation.
    how : function or str
        Aggregation function for collapse operation.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Collapsed feature coordinates and aggregated
        intensities.

    '''

    return data.groupby(by=keep,
                        as_index=False,
                        sort=False).agg({'intensity': how})


def locate(data, by=['mz', 'drift_time', 'retention_time'],
           loc=[0, 0, 0], tol=[0, 0, 0], return_index=False):
    '''
    Given a feature coordinate and tolerances, return a subset
    of the data.

    Parameters
    ----------
    data : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : str or list
        Feature(s) by which to subset the data.
    loc : float or list
        Coordinate location.
    tol : float or list
        Tolerance in each dimension.
    return_index : bool
        Return boolean index of subset if True.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Subset of feature coordinates and intensities.
    :obj:`~numpy.array`
        If `return_index` is True, boolean index of subset elements,
        i.e. `data[index] = subset`.

    '''

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
    '''
    Given a feature coordinate and bounds, return a subset of the data.

    Parameters
    ----------
    data : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    by : str or list
        Feature(s) by which to subset the data
    low : float or list
        Lower bound(s) in each dimension.
    high : float or list
        Upper bound(s) in each dimension.
    return_index : bool
        Return boolean index of subset if True.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Subset of feature coordinates and intensities.
    :obj:`~numpy.array`
        If `return_index` is True, boolean index of subset elements,
        i.e. `data[index] = subset`.

    '''

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


class Partitions:
    '''
    Generator object that will lazily build and return each partition.

    Attributes
    ----------
    data : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    split_on : str
        Dimension to partition the data.
    size : int
        Target partition size.
    overlap : float
        Amount of overlap between partitions to ameliorate edge effects.

    '''

    def __init__(self, data, split_on='mz', size=1000, overlap=0.05):
        '''
        Initialize :obj:`~deimos.subset.Partitions` instance.

        Parameters
        ----------
        data : :obj:`~pandas.DataFrame`
            Input feature coordinates and intensities.
        split_on : str
            Dimension to partition the data.
        size : int
            Target partition size.
        overlap : float
            Amount of overlap between partitions to ameliorate edge effects.

        '''

        self.data = data
        self.split_on = split_on
        self.size = size
        self.overlap = overlap

        self._compute_splits()

    def _compute_splits(self):
        '''
        Determines data splits for partitioning.

        '''

        # unique to split on
        idx = np.unique(self.data[self.split_on].values)

        # number of partitions
        partitions = np.ceil(len(idx) / self.size)

        # determine partition bounds
        bounds = [[x.min(), x.max()] for x in np.array_split(idx, partitions)]
        for i in range(1, len(bounds)):
            bounds[i][0] = bounds[i - 1][1] - self.overlap

        if (self.overlap > 0) & (len(bounds) > 1):
            # functional bounds
            fbounds = []
            for i in range(len(bounds)):
                a, b = bounds[i]

                # first partition
                if i < 1:
                    b = b - self.overlap / 2

                # middle partitions
                elif i < len(bounds) - 1:
                    a = a + self.overlap / 2
                    b = b - self.overlap / 2

                # last partition
                else:
                    a = a + self.overlap / 2

                fbounds.append([a, b])
        else:
            fbounds = bounds

        self.bounds = bounds
        self.fbounds = fbounds

    def __iter__(self):
        '''
        Yields each partition.

        Yields
        ------
        :obj:`~pandas.DataFrame`
            Partition of feature coordinates and intensities.

        '''

        for a, b in self.bounds:
            yield slice(self.data, by=self.split_on, low=a, high=b)

    def map(self, func, processes=1, **kwargs):
        '''
        Maps `func` to each partition, then returns the combined result,
        accounting for overlap regions.

        Parameters
        ----------
        func : function
            Function to apply to partitions.
        processes : int
            Number of parallel processes. If less than 2, a serial mapping is
            applied.
        kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            Combined result of `func` applied to partitions.

        '''

        # serial
        if processes < 2:
            result = [func(x, **kwargs) for x in self]

        # parallel
        else:
            with mp.Pool(processes=processes) as p:
                result = list(p.imap(partial(func, **kwargs), self))

        # reconcile overlap
        result = [slice(result[i], by=self.split_on, low=a, high=b)
                  for i, (a, b) in enumerate(self.fbounds)]

        # combine partitions
        return pd.concat(result).reset_index(drop=True)

    def zipmap(self, func, b, processes=1, **kwargs):
        '''
        Maps `func` to each partition pair resulting from the zip operation of
        `self` and `b`, then returns the combined result, accounting for
        overlap regions.

        Parameters
        ----------
        func : function
            Function to apply to zipped partitions. Must accept and return two
            :obj:`~pandas.DataFrame` instances.
        b : :obj:`~pandas.DataFrame`
            Input feature coordinates and intensities.
        processes : int
            Number of parallel processes. If less than 2, a serial mapping is
            applied.
        kwargs
            Keyword arguments passed to `func`.

        Returns
        -------
        a, b : :obj:`~pandas.DataFrame`
            Result of `func` applied to paired partitions.

        '''

        # partition other dataset
        partitions = (slice(b, by=self.split_on, low=a, high=b_)
                      for a, b_ in self.bounds)

        # serial
        if processes < 2:
            result = [func(a, b_, **kwargs) for a, b_ in zip(self, partitions)]

        # parallel
        else:
            with mp.Pool(processes=processes) as p:
                result = list(p.starmap(partial(func, **kwargs),
                              zip(self, partitions)))

        result = {'a': [x[0] for x in result], 'b': [x[1] for x in result]}

        # reconcile overlap
        tmp = [slice(result['a'][i], by=self.split_on, low=a, high=b_,
                     return_index=True)
               for i, (a, b_) in enumerate(self.fbounds)]

        result['a'] = [x[0] for x in tmp]
        idx = [x[1] for x in tmp]
        result['b'] = [p.iloc[i, :] if i is not None else None for p, i in zip(result['b'], idx)]

        # combine partitions
        result['a'] = pd.concat(result['a'])
        result['b'] = pd.concat(result['b'])

        return result['a'], result['b']


def partition(data, split_on='mz', size=1000, overlap=0.05):
    '''
    Partitions data along a feature dimension.

    Parameters
    ----------
    data : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    split_on : str
        Dimension to partition the data.
    size : int
        Target partition size.
    overlap : float
        Amount of overlap between partitions to ameliorate edge effects.

    Returns
    -------
    :obj:`~deimos.subset.Partitions`
        A generator object that will lazily build and return each partition.

    '''

    return Partitions(data, split_on, size, overlap)
