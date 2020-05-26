import numpy as np
import deimos


def auto(data, features=['mz', 'drift_time', 'retention_time'],
         bins=[2.7, 0.94, 3.64], scale_by=None, ref_res=None, scale=None):
    """
    Helper function to perform peak detection on a single partition.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    features : str or list
        Feature dimensions to perform peak detection in
        (omitted dimensions will be collapsed and summed accross).
    bins : float or list
        Number of bins representing 1 sigma in each dimension.
    scale_by : str
        Dimension to scale bin widths by. Only applies when data is
        partitioned by `scale_by` (see `deimos.utils.partition`).
    ref_res : float
        Minimum acquisition resolution of `scale_by` dimension.
    scale : str or list
        Dimensions to scale, according to `scale_by`.

    Returns
    -------
    peaks : DataFrame
        Coordinates of detected peaks and associated apex intensitites.

    """

    # safely cast to list
    features = deimos.utils.safelist(features)
    bins = deimos.utils.safelist(bins)

    # check dims
    deimos.utils.check_length([features, bins])

    # scaling
    if None not in [scale_by, ref_res, scale]:
        scale = deimos.utils.safelist(scale)
        sf = np.min(np.diff(np.unique(data[scale_by]))) / ref_res

        # enumerate features
        for i, f in enumerate(features):

            # scale
            if f in scale:
                bins[i] *= sf

    # no scaling
    elif not any([scale_by, ref_res, scale]):
        pass

    # improper scaling kwargs
    else:
        raise ValueError('`scale_by`, `ref_res`, and `scale` must all be supplied')

    # bounds
    sigma2 = [round(2 * x) for x in bins]
    sigma4 = [round(4 * x) for x in bins]

    # container
    additional = {}

    # grid data
    edges, H = deimos.grid.data2grid(data, features=features)

    # data counts
    additional['npoints_2'] = deimos.filters.count(H, sigma2)
    additional['nonzero_2'] = deimos.filters.count(H, sigma2, nonzero=True)
    additional['npoints_4'] = deimos.filters.count(H, sigma4)
    additional['nonzero_4'] = deimos.filters.count(H, sigma4, nonzero=True)

    # nan to num
    H = np.nan_to_num(H)

    # sum
    additional['sum_2'] = deimos.filters.sum(H, sigma2)
    additional['sum_4'] = deimos.filters.sum(H, sigma4)

    # peak detection
    H_max = deimos.filters.maximum(H, sigma4)
    peaks = np.where(H == H_max, H, 0)

    # clean up
    del H_max, H

    # convert to dataframe
    peaks = deimos.grid.grid2df(edges, peaks, features=features,
                                additional=additional)

    # add bins info
    for i, f in enumerate(features):
        peaks['sigma_{}_2'.format(f)] = sigma2[i]
        peaks['sigma_{}_4'.format(f)] = sigma4[i]

    return peaks
