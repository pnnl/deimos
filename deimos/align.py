import scipy
import numpy as np
import deimos
from sklearn.svm import SVR
import pandas as pd
import time


def match(a, b, features=['mz', 'drift_time', 'retention_time'],
          tol=[5E-6, 0.015, 0.3], relative=[True, True, False]):
    """
    Identify features in `b` within tolerance of those in `a`. Matches are
    bidirectionally one-to-one by highest intensity.

    Parameters
    ----------
    a, b : DataFrame
        Input feature coordinates and intensities. Features from a are matched
        to features in b.
    features : str or list
        Features to match against.
    tol : float or list
        Tolerance in each feature dimension to define a match.
    relative : bool or list
        Whether to use ppm or absolute values when determining m/z tolerance.

    Returns
    -------
    a, b : DataFrame
        Features matched within tolerances. E.g., a[i..n] and b[i..n] each
        represent matched features.

    """

    if a is None or b is None:
        return None, None

    # safely cast to list
    features = deimos.utils.safelist(features)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # check dims
    deimos.utils.check_length([features, tol, relative])

    # compute inter-feature distances
    idx = []
    for i, f in enumerate(features):
        # vectors
        v1 = a[f].values.reshape(-1, 1)
        v2 = b[f].values.reshape(-1, 1)

        # distances
        d = scipy.spatial.distance.cdist(v1, v2)

        if relative[i] is True:
            # divisor
            basis = np.repeat(v1, v2.shape[0], axis=1)
            fix = np.repeat(v2, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # divide
            d = np.divide(d, basis, out=np.zeros_like(basis), where=basis != 0)

        # check tol
        idx.append(d <= tol[i])

    # stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1, dtype=bool)

    # compute normalized 3d distance
    v1 = a[features].values / tol
    v2 = b[features].values / tol
    # v1 = (v1 - v1.min(axis=0)) / (v1.max(axis=0) - v1.min(axis=0))
    # v2 = (v2 - v1.min(axis=0)) / (v1.max(axis=0) - v1.min(axis=0))
    dist3d = scipy.spatial.distance.cdist(v1, v2, 'cityblock')
    dist3d = np.multiply(dist3d, idx)

    # normalize to 0-1
    mx = dist3d.max()
    if mx == 0:
        return None, None

    dist3d = dist3d / dist3d.max()

    # intensities
    intensity = np.repeat(a['intensity'].values.reshape(-1, 1),
                          b.shape[0], axis=1)
    intensity = np.multiply(intensity, idx)

    # max over features
    maxcols = np.max(intensity, axis=0, keepdims=True)

    # zero out nonmax over features
    intensity[intensity != maxcols] = 0

    # break ties by distance
    intensity = intensity - dist3d

    # max over clusters
    maxrows = np.max(intensity, axis=1, keepdims=True)

    # where max and nonzero
    ii, jj = np.where((intensity == maxrows) & (intensity > 0))

    # reorder
    a = a.iloc[ii]
    b = b.iloc[jj]

    if len(a.index) < 1 or len(b.index) < 1:
        return None, None

    return a, b


def threshold(a, b, features=['mz', 'drift_time', 'retention_time'],
              tol=[5E-6, 0.025, 0.3], relative=[True, True, False]):
    """
    Identify features in `b` within tolerance of those in `a`.
    Matches are potentially many-to-one.

    Parameters
    ----------
    a, b : DataFrame
        Input feature coordinates and intensities. Features from a are matched
        to features in b.
    features : str or list
        Features to match against.
    tol : float or list
        Tolerance in each feature dimension to define a match.
    relative : bool or list
        Whether to use ppm or absolute values when determining m/z tolerance.

    Returns
    -------
    a, b : DataFrame
        Features matched within tolerances. E.g., a[i..n] and b[i..n] each
        represent matched features.

    """

    if a is None or b is None:
        return None, None

    # safely cast to list
    features = deimos.utils.safelist(features)
    tol = deimos.utils.safelist(tol)

    # check dims
    deimos.utils.check_length([features, tol])

    # compute inter-feature distances
    idx = []
    for i, f in enumerate(features):
        # vectors
        v1 = a[f].values.reshape(-1, 1)
        v2 = b[f].values.reshape(-1, 1)

        # distances
        d = scipy.spatial.distance.cdist(v1, v2)

        if relative[i] is True:
            # divisor
            basis = np.repeat(v1, v2.shape[0], axis=1)
            fix = np.repeat(v2, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # divide
            d = np.divide(d, basis, out=np.zeros_like(basis), where=basis != 0)

        # check tol
        idx.append(d <= tol[i])

    # stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1, dtype=bool)

    # per-dataset indices
    ii, jj = np.where(idx > 0)

    # reorder
    a = a.iloc[ii]
    b = b.iloc[jj]

    if len(a.index) < 1 or len(b.index) < 1:
        return None, None

    return a, b


def fit_spline(a, b, align='retention_time', **kwargs):
    """
    Fit a support vector regressor to matched features.

    Parameters
    ----------
    a, b : DataFrame
        Matched input feature coordinates and intensities.
    align : str
        Feature to align.
    kwargs :
        Keyword arguments for scikit-learn support vector regressor
        (`sklearn.svm.SVR`).

    Returns
    -------
    interp : interpolator
        Interpolated fit of the SVR result.

    """

    # uniqueify
    x = a[align].values
    y = b[align].values
    arr = np.vstack((x, y)).T
    arr = np.unique(arr, axis=0)

    # check kwargs
    if 'kernel' in kwargs:
        kernel = kwargs.get('kernel')
    else:
        kernel = 'linear'

    newx = np.linspace(arr[:, 0].min(), arr[:, 0].max(), 1000)

    if kernel == 'linear':
        reg = scipy.stats.linregress(x, y)
        newy = reg.slope * newx + reg.intercept

    else:
        # fit
        svr = SVR(**kwargs)
        svr.fit(arr[:, 0].reshape(-1, 1), arr[:, 1])

        # predict
        newy = svr.predict(newx.reshape(-1, 1))

    return scipy.interpolate.interp1d(newx, newy,
                                      kind='linear', fill_value='extrapolate')

    # # linear edges
    # N = int(len(arr) * buffer)
    # if N > 2:
    #     # fit
    #     lin1 = SVR(kernel='linear', **kwargs)
    #     lin1.fit(arr[:N, 0].reshape(-1, 1), arr[:N, 1])
    #     lin2 = SVR(kernel='linear', **kwargs)
    #     lin2.fit(arr[-N:, 0].reshape(-1, 1), arr[-N:, 1])

    #     # predict
    #     ylin1 = lin1.predict(newx.reshape(-1, 1))
    #     ylin2 = lin2.predict(newx.reshape(-1, 1))

    #     # overwrite
    #     # newy[newx < arr[N, 0]] = ylin1[newx < arr[N, 0]]
    #     newy[newx > arr[-N, 0]] = ylin2[newx > arr[-N, 0]]

    #     # fit spline for continuity
    #     spl = scipy.interpolate.UnivariateSpline(newx, newy, s=2, k=3)
    #     newy = spl(newx)

    # return interpolator


def join(paths, features=['mz', 'drift_time', 'retention_time'],
         quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], processes=4,
         partition_kwargs={}, match_kwargs={}):
    '''
    Iteratively apply `deimos.alignment.threshold` and `deimos.alignment.match`
    across multiple datasets, generating a growing set of "clusters", similar
    to the "join aligner" method of MZmine.

    Parameters
    ----------
    paths : list
        List of dataset paths to align.
    features : str or list
        Features to align with.
    quantiles : array_like
        Quantiles of feature intensities to iteratively perform alignment.
    processes : int
        Number of parallel processes. If less than 2, a serial mapping is
        applied.
    partition_kwargs : dict
        Keyword arguments for `deimos.utils.partition`.
    match_kwargs : dict
        Keyword arugments for `deimos.alignment.threshold` and
        `deimos.alignment.match`.

    Returns
    -------
    clusters : DataFrame
        Coordinates of detected clusters, average intensitites, and number of
        datasets observed.

    '''

    def helper(samp, clusters, verbose=True):
        if len(samp.index) > 1000:
            partition_kwargs['size'] = 500
        else:
            partition_kwargs['size'] = len(samp.index)

        # partition
        c_parts = deimos.utils.partition(pd.concat((samp, clusters), axis=0),
                                         **partition_kwargs)
        partitions = deimos.utils.partition(samp, **partition_kwargs)
        partitions.bounds = c_parts.bounds
        partitions.fbounds = c_parts.fbounds

        # filter by tolerance
        samp_pass, clust_pass = partitions.zipmap(deimos.alignment.threshold,
                                                  clusters,
                                                  processes=processes,
                                                  **match_kwargs)

        # drop duplicates
        samp_pass = samp_pass[~samp_pass.index.duplicated(keep='first')]
        clust_pass = clust_pass[~clust_pass.index.duplicated(keep='first')]

        # unmatched
        unmatched = samp.loc[samp.index.difference(samp_pass.index), :]

        # one-to-one mapping
        if samp_pass.index.equals(clust_pass.index):
            samp_match = samp_pass
            clust_match = clust_pass

        # many-to-many mapping
        else:
            # repartition
            partitions = deimos.utils.partition(samp_pass, **partition_kwargs)
            partitions.bounds = c_parts.bounds
            partitions.fbounds = c_parts.fbounds
            c_parts = None

            # match
            samp_match, clust_match = partitions.zipmap(deimos.alignment.match,
                                                        clust_pass,
                                                        processes=processes,
                                                        **match_kwargs)

        # match stats
        if verbose:
            print('clusters:\t{}'.format(len(clusters.index)))
            print('sample count:\t{}'.format(len(samp.index)))
            p = len(samp_pass.index) / len(samp.index) * 100
            print('matched:\t{} ({:.1f}%)'.format(len(samp_pass.index), p))
            p = len(samp_match.index) / len(samp.index) * 100
            print('kept:\t\t{} ({:.1f}%)'.format(len(samp_match.index), p))
            p = len(unmatched.index) / len(samp.index) * 100
            print('unmatched:\t{} ({:.1f}%)'.format(len(unmatched.index), p))

        return samp_match, clust_match, unmatched

    # column indices
    colnames = features + ['intensity']

    # iterate quantiles
    for k in range(1, len(quantiles)):
        high = quantiles[-k]
        low = quantiles[-k - 1]
        print('quantile range: ({}, {}]'.format(low, high))

        # iterate datasets
        for i in range(len(paths)):
            start = time.time()
            print(i, paths[i])

            # load
            samp = deimos.utils.load_hdf(paths[i])
            samp['intensity'] = samp['sum_2']

            # filter
            samp = samp.loc[(samp['intensity'] <= samp['intensity'].quantile(high))
                            & (samp['intensity'] > samp['intensity'].quantile(low)), :]
            samp = samp[colnames]

            # no clusters initialized
            if (k == 1) and (i == 0):
                clusters = samp

            # match
            samp_match, clust_match, unmatched = helper(samp, clusters)

            # initialize unique clusters
            if (k == 1) and (i == 0):
                clusters = samp_match.reset_index(drop=True)
                clusters['n'] = 1
                clusters['quantile'] = low

            # update clusters
            else:
                # unique matched clusters
                idx = clust_match.index

                # increment intensity
                clusters.loc[idx, 'intensity'] += samp_match.loc[:, 'intensity'].values
                clusters.loc[idx, 'n'] += 1

                # update cluster centers
                ab = clusters.loc[idx, 'intensity'].values
                b = samp_match.loc[:, 'intensity'].values
                a = ab - b

                for f in features:
                    clusters.loc[idx, f] = (a * clusters.loc[idx, f].values
                                            + b * samp_match.loc[:, f].values) / ab

                # uniqueify unmatched
                if len(unmatched.index) > 0:
                    unmatched, _, _ = helper(unmatched, unmatched,
                                             verbose=False)
                    unmatched['n'] = 1
                    unmatched['quantile'] = low

                # new cluster counts
                p = len(unmatched.index) / len(samp.index) * 100
                print('new:\t\t{} ({:.1f}%)'.format(len(unmatched.index), p))

                # combine
                clusters = pd.concat((clusters, unmatched), axis=0,
                                     ignore_index=True)

            print('time:\t\t{:.2f}\n'.format(time.time() - start))

    return clusters
