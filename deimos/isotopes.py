import deimos
import numpy as np
import pandas as pd
import scipy


def detect(data, features=['mz', 'drift_time', 'retention_time'],
           tol=[0.1, 0.2, 0.3], delta=1.003355, max_isotopes=4, max_charge=1):
    '''
    Perform isotope detection according to expected patterning.

    Parameters
    ----------
    features : str or list
        Feature dimensions to perform isotope detection in.
    tol : float or list
        Tolerance in each dimension to be considered a match.
    delta : float
        Expected spacing between isotopes (e.g. C13=1.003355).
    max_isotopes : int
        Maximum number of isotopes to search for per parent feature.
    max_charge : int
        Maximum charge to search for per parent feature.

    Returns
    -------
    :obj:`pandas.DataFrame`
        Features grouped by isotopic pattern.

    '''

    # safely cast to list
    features = deimos.utils.safelist(features)
    tol = deimos.utils.safelist(tol)

    # check dims
    deimos.utils.check_length([features, tol])

    # isolate mz dimension
    mz_idx = features.index('mz')
    else_idx = [i for i, j in enumerate(features) if i != mz_idx]

    isotopes = []
    idx = []
    for i in else_idx:
        arr = data[features[i]].values.reshape((-1, 1))
        d = scipy.spatial.distance.cdist(arr, arr)

        # less than tolerance
        idx.append(d <= tol[i])

    # stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1)

    # half matrix
    idx = np.tril(idx, k=-1)

    # isotopic distances
    arr = data[features[mz_idx]].values.reshape((-1, 1))
    d = scipy.spatial.distance.cdist(arr, arr)
    d = np.multiply(d, idx)

    for charge in range(1, max_charge + 1):
        for mult in range(1, max_isotopes + 1):

            dx_i = mult * (delta / charge)
            r, c = np.where((d > dx_i - tol[mz_idx])
                            & (d < dx_i + tol[mz_idx]))
            a = data.iloc[c, :]
            b = data.iloc[r, :]
            z = charge * np.ones(len(a))
            m = mult * np.ones(len(a))
            dx_i = dx_i * np.ones(len(a))

            isotopes.append(pd.DataFrame(np.vstack((a['mz'].values,
                                                    a['sum_2'].values,
                                                    z,
                                                    m,
                                                    dx_i,
                                                    b['mz'].values,
                                                    b['sum_2'].values,
                                                    a['idx'].values,
                                                    b['idx'].values)).T,
                                         columns=['mz', 'intensity', 'charge',
                                                  'multiple', 'dx', 'mz_iso',
                                                  'intensity_iso', 'idx',
                                                  'idx_iso']))

    # combine
    isotopes = pd.concat(isotopes, axis=0, ignore_index=True)

    # stats
    isotopes['error'] = 1E6 * np.abs((isotopes['mz_iso'] - isotopes['mz']) - isotopes['dx']) / isotopes['mz']
    isotopes['decay'] = isotopes['intensity_iso'] / isotopes['intensity']

    # cull non-decreasing
    isotopes = isotopes.loc[isotopes['intensity'] > isotopes['intensity_iso'], :]

    # cull high error
    isotopes = isotopes.loc[isotopes['error'] < 50, :]

    # cull children
    isotopes = isotopes.loc[~isotopes['idx'].isin(isotopes['idx_iso']), :]

    return isotopes
