import numpy as np
import pandas as pd
import scipy

import deimos


def OrderedSet(x):
    return list({k: None for k in x})


def detect(features, dims=['mz', 'drift_time', 'retention_time'],
           tol=[0.1, 0.2, 0.3], delta=1.003355, max_isotopes=4, max_charge=1,
           max_error=50E-6):
    '''
    Perform isotope detection according to expected patterning.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    dims : str or list
        Dimensions to perform isotope detection in.
    tol : float or list
        Tolerance in each dimension to be considered a match.
    delta : float
        Expected spacing between isotopes (e.g. C_13=1.003355).
    max_isotopes : int
        Maximum number of isotopes to search for per parent feature.
    max_charge : int
        Maximum charge to search for per parent feature.
    max_error : float
        Maximum relative error between search pattern and putative isotopic
        feature.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Features grouped by isotopic pattern.

    '''

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)

    # Check dims
    deimos.utils.check_length([dims, tol])

    # Isolate mz dimension
    mz_idx = dims.index('mz')
    else_idx = [i for i, j in enumerate(dims) if i != mz_idx]

    isotopes = []
    idx = []

    # Tolerance in other dimensions
    for i in else_idx:
        arr = features[dims[i]].values.reshape((-1, 1))
        dist = scipy.spatial.distance.cdist(arr, arr)

        # Less than tolerance
        idx.append(dist <= tol[i])

    # Stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1)

    # Half matrix
    idx = np.tril(idx, k=-1)

    # Isotopic distances
    arr = features[dims[mz_idx]].values.reshape((-1, 1))
    d = scipy.spatial.distance.cdist(arr, arr)
    d = np.multiply(d, idx)

    # Enumerate putative spacings
    for charge in range(1, max_charge + 1):
        for mult in range(1, max_isotopes + 1):

            dx_i = mult * (delta / charge)
            r, c = np.where((d > dx_i - tol[mz_idx])
                            & (d < dx_i + tol[mz_idx]))
            a = features.iloc[c, :]
            b = features.iloc[r, :]
            z = charge * np.ones(len(a))
            m = mult * np.ones(len(a))
            dx_i = dx_i * np.ones(len(a))

            isotopes.append(pd.DataFrame(np.vstack((a['mz'].values,
                                                    a['intensity'].values,
                                                    z,
                                                    m,
                                                    dx_i,
                                                    b['mz'].values,
                                                    b['intensity'].values,
                                                    a.index.values,
                                                    b.index.values)).T,
                                         columns=['mz', 'intensity', 'charge',
                                                  'multiple', 'dx', 'mz_iso',
                                                  'intensity_iso', 'idx',
                                                  'idx_iso']))

    # Combine
    isotopes = pd.concat(isotopes, axis=0, ignore_index=True)

    # Stats
    isotopes['error'] = np.abs(
        (isotopes['mz_iso'] - isotopes['mz']) - isotopes['dx']) / isotopes['mz']
    isotopes['decay'] = isotopes['intensity_iso'] / isotopes['intensity']

    # Cull non-decreasing
    isotopes = isotopes.loc[isotopes['intensity']
                            > isotopes['intensity_iso'], :]

    # Cull high error
    isotopes = isotopes.loc[isotopes['error'] < max_error, :]

    # Cull children
    isotopes = isotopes.loc[~isotopes['idx'].isin(isotopes['idx_iso']), :]

    # Group by parent
    grouped = isotopes.groupby(by=['mz', 'charge', 'idx', 'intensity'],
                               as_index=False).agg(OrderedSet)
    grouped['n'] = [len(x) for x in grouped['multiple'].values]

    # grouped['n_sum'] = [sum(x) for x in grouped['multiple'].values]
    # grouped['check'] = np.abs(grouped['n'] * (grouped['n'] + 1) / 2 - grouped['n_sum'])

    return grouped.sort_values(by=['intensity', 'n'],
                               ascending=False).reset_index(drop=True)
