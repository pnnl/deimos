import numpy as np
import pandas as pd

import deimos


def local_maxima(features, dims=['mz', 'drift_time', 'retention_time'],
                 bins=[37, 9, 37], scale_by=None, ref_res=None,
                 scale=None):
    '''
    N-dimensional non-maximum suppression peak detection method.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    dims : str or list
        Dimensions to perform peak detection in (omitted dimensions
        will be collapsed and summed accross).
    bins : float or list
        Number of bins representing approximate peak width in each dimension.
    scale_by : str
        Dimension to scale bin widths by. Only applies when data is partitioned
        by `scale_by` (see :func:`deimos.utils.partition`).
    ref_res : float
        Minimum acquisition resolution of `scale_by` dimension.
    scale : str or list
        Dimensions to scale, according to `scale_by`.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Coordinates of detected peaks and associated apex intensitites.

    '''

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    bins = deimos.utils.safelist(bins)

    # Check dims
    deimos.utils.check_length([dims, bins])

    # Scaling
    if None not in [scale_by, ref_res, scale]:
        scale = deimos.utils.safelist(scale)
        sf = np.min(np.diff(np.unique(features[scale_by]))) / ref_res

        # Enumerate dimensions
        for i, d in enumerate(dims):

            # Scale
            if d in scale:
                bins[i] *= sf

    # No scaling
    elif not any([scale_by, ref_res, scale]):
        pass

    # Improper scaling kwargs
    else:
        raise ValueError(
            '`scale_by`, `ref_res`, and `scale` must all be supplied')

    # Footprint rounded up to nearest odd
    bins = [np.ceil(x) // 2 * 2 + 1 for x in bins]
    # bins_half = [np.ceil(x / 2) // 2 * 2 + 1 for x in bins]
    # bins_half[0] = 3

    # Ggrid data
    edges, H = deimos.grid.data2grid(features, dims=dims)

    # # Mean pdf
    # additional = {dim + '_mean': x for dim, x in zip(dims,
    #                                                  deimos.filters.mean_pdf(edges, H, bins_half))}

    # # Coverage
    # additional['coverage'] = deimos.filters.count(H, bins, nonzero=True) / np.prod(bins)

    # # Smooth
    # H = deimos.filters.sum(H, [1, 3, 3])

    # Peak detection
    H = np.where(H == deimos.filters.maximum(H, bins), H, 0)

    # Convert to dataframe
    peaks = deimos.grid.grid2df(edges, H, dims=dims)

    return peaks


def persistent_homology(features, index=None, factors=None, dims=['mz', 'drift_time', 'retention_time'],
                        radius=None):
    '''
    Peak detection by persistent homology, implemented as a sparse upper star
    filtration.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    index : dict
        Index of features in original data array.
    factors : dict
        Unique sorted values per dimension.
    dims : str or list
        Dimensions to perform peak detection in.
    radius : float, list, or None
        If specified, radius of the sparse weighted mean filter in each dimension.
        Values less than one indicate no connectivity in that dimension.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Coordinates of detected peaks, associated apex intensitites, and
        persistence.

    '''

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    if radius is not None:
        radius = deimos.utils.safelist(radius)

    # Check lenghts
    if radius is not None:
        deimos.utils.check_length([dims, radius])

    # Check factors and index mutually exclusive
    if (factors is not None) & (index is not None):
        raise ValueError('Specify either `index`, `factors`, or neither.')

    # Build index from features directly
    if (factors is None) & (index is None):
        index = {dim: pd.factorize(features[dim], sort=True)[0].astype(np.float32)
                 for dim in dims}

    # Build index from factors
    if (factors is not None) & (index is None):
        index = deimos.build_index(features, factors)

    # Index built, shape appropriately
    index = np.vstack([index[dim] for dim in dims]).T

    # Values
    V = features['intensity'].values

    # Upper star filtration
    pidx, pers = deimos.filters.sparse_upper_star(index, V)
    pidx = pidx[pers > 1]
    pers = pers[pers > 1]

    # Get peaks
    peaks = features.iloc[pidx, :].reset_index(drop=True)

    # Add persistence column
    peaks['persistence'] = pers

    # Weighted mean
    if radius is not None:
        vals = deimos.filters.sparse_weighted_mean_filter(index, features[dims].values,
                                                          V, radius=radius, pindex=pidx)
        for i, dim in enumerate(dims):
            peaks[dim + '_weighted'] = vals[:, i]

    return peaks
