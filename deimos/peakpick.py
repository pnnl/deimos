import deimos
import numpy as np


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
    bins = [np.ceil(x) // 2 * 2 + 1 for x in bins]

    # grid data
    edges, H = deimos.grid.data2grid(features, dims=dims)
    
    # mean pdf
    H_mean_pdf = {dim + '_mean': x for dim, x in zip(dims,
                                                     deimos.filters.mean_pdf(edges, H, bins))}
    
    # smooth
    H = deimos.filters.mean(H, [1, 3, 3])

    # peak detection
    H = np.where(H == deimos.filters.maximum(H, bins), H, 0)

    # convert to dataframe
    peaks = deimos.grid.grid2df(edges, H, dims=dims,
                                additional=H_mean_pdf)

    return peaks
