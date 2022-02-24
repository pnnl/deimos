import deimos
import numpy as np


def local_maxima(features, dims=['mz', 'drift_time', 'retention_time'],
                 bins=[2.7, 0.94, 3.64], scale_by=None, ref_res=None,
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
        Number of bins representing 1 sigma in each dimension.
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

    Raises
    ------
    ValueError
        If `scale_by`, `ref_res`, and `scale` are not all None or not all
        supplied.
    ValueError
        If `dims` and `bins` are not the same length.

    '''

    # safely cast to list
    dims = deimos.utils.safelist(dims)
    bins = deimos.utils.safelist(bins)

    # check dims
    deimos.utils.check_length([dims, bins])

    # scaling
    if None not in [scale_by, ref_res, scale]:
        scale = deimos.utils.safelist(scale)
        sf = np.min(np.diff(np.unique(features[scale_by]))) / ref_res

        # enumerate dimensions
        for i, d in enumerate(dims):

            # scale
            if d in scale:
                bins[i] *= sf

    # no scaling
    elif not any([scale_by, ref_res, scale]):
        pass

    # improper scaling kwargs
    else:
        raise ValueError(
            '`scale_by`, `ref_res`, and `scale` must all be supplied')

    # footprint rounded up to nearest odd
    # sigma2 = [np.ceil(x * 2) // 2 * 2 + 1 for x in bins]
    sigma4 = [np.ceil(x * 4) // 2 * 2 + 1 for x in bins]
    # sigma8 = [np.ceil(x * 8) // 2 * 2 + 1 for x in bins]

    # container
    additional = {}

    # grid data
    edges, H = deimos.grid.data2grid(features, dims=dims)

    # # data counts
    # additional['npoints_2'] = deimos.filters.count(H, sigma2)
    # additional['nonzero_2'] = deimos.filters.count(H, sigma2, nonzero=True)
    # additional['npoints_4'] = deimos.filters.count(H, sigma4)
    # additional['nonzero_4'] = deimos.filters.count(H, sigma4, nonzero=True)

    # # nan to num
    # H = np.nan_to_num(H)

    # # sum
    # additional['sum_2'] = deimos.filters.sum(H, sigma2)
    # additional['sum_4'] = deimos.filters.sum(H, sigma4)

    # # minimum
    # additional['min_4'] = deimos.filters.minimum(H, sigma4)
    # additional['min_8'] = deimos.filters.minimum(H, sigma8)

    # # kurtosis
    # for k, v in zip(dims, deimos.filters.kurtosis(edges, H, sigma4)):
    #     additional['k_{}'.format(k)] = v

    # peak detection
    H = np.where(H == deimos.filters.maximum(H, sigma4), H, 0)

    # convert to dataframe
    peaks = deimos.grid.grid2df(edges, H, dims=dims,
                                additional=additional)

    # # add bins info
    # for i, d in enumerate(dims):
    #     peaks['sigma_{}_2'.format(d)] = sigma2[i]
    #     peaks['sigma_{}_4'.format(d)] = sigma4[i]

    return peaks
