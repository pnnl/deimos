import deimos
import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVR
import time


def match(a, b, dims=['mz', 'drift_time', 'retention_time'],
          tol=[5E-6, 0.015, 0.3], relative=[True, True, False]):
    '''
    Identify features in `b` within tolerance of those in `a`. Matches are
    bidirectionally one-to-one by highest intensity.

    Parameters
    ----------
    a, b : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities. Features from `a` are
        matched to features in `b`.
    dims : str or list
        Dimensions considered in matching.
    tol : float or list
        Tolerance in each dimension to define a match.
    relative : bool or list
        Whether to use relative or absolute tolerances per dimension.

    Returns
    -------
    a, b : :obj:`~pandas.DataFrame`
        Features matched within tolerances. E.g., `a`[i..n] and `b`[i..n] each
        represent matched features.

    Raises
    ------
    ValueError
        If `dims`, `tol`, and `relative` are not the same length.

    '''

    if a is None or b is None:
        return None, None

    # safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # check dims
    deimos.utils.check_length([dims, tol, relative])

    # compute inter-feature distances
    idx = []
    for i, f in enumerate(dims):
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
    v1 = a[dims].values / tol
    v2 = b[dims].values / tol
    # v1 = (v1 - v1.min(axis=0)) / (v1.max(axis=0) - v1.min(axis=0))
    # v2 = (v2 - v1.min(axis=0)) / (v1.max(axis=0) - v1.min(axis=0))
    dist3d = scipy.spatial.distance.cdist(v1, v2, 'cityblock')
    dist3d = np.multiply(dist3d, idx)

    # normalize to 0-1
    mx = dist3d.max()
    if mx > 0:
        dist3d = dist3d / dist3d.max()

    # intensities
    intensity = np.repeat(a['intensity'].values.reshape(-1, 1),
                          b.shape[0], axis=1)
    intensity = np.multiply(intensity, idx)

    # max over dims
    maxcols = np.max(intensity, axis=0, keepdims=True)

    # zero out nonmax over dims
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


def tolerance(a, b, dims=['mz', 'drift_time', 'retention_time'],
              tol=[5E-6, 0.025, 0.3], relative=[True, True, False]):
    '''
    Identify features in `b` within tolerance of those in `a`. Matches are
    potentially many-to-one.

    Parameters
    ----------
    a, b : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities. Features from `a` are
        matched to features in `b`.
    dims : str or list
        Dimensions considered in matching.
    tol : float or list
        Tolerance in each dimension to define a match.
    relative : bool or list
        Whether to use relative or absolute tolerances per dimension.

    Returns
    -------
    a, b : :obj:`~pandas.DataFrame`
        Features matched within tolerances. E.g., `a`[i..n] and `b`[i..n] each
        represent matched features.

    Raises
    ------
    ValueError
        If `dims`, `tol`, and `relative` are not the same length.

    '''

    if a is None or b is None:
        return None, None

    # safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # check dims
    deimos.utils.check_length([dims, tol, relative])

    # compute inter-feature distances
    idx = []
    for i, f in enumerate(dims):
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
    '''
    Fit a support vector regressor to matched features.

    Parameters
    ----------
    a, b : :obj:`~pandas.DataFrame`
        Matched input feature coordinates and intensities.
    align : str
        Dimension to align.
    kwargs
        Keyword arguments for support vector regressor
        (:class:`sklearn.svm.SVR`).

    Returns
    -------
    :obj:`~scipy.interpolate.interp1d`
        Interpolated fit of the SVR result.

    '''

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


def _agglomerative_clustering(features,
                              dims=['mz', 'drift_time', 'retention_time'],
                              tol=[20E-6, 0.03, 0.3],
                              relative=[True, True, False]):
    '''
    Cluster features within provided linkage tolerances. Recursively merges
    the pair of clusters that minimally increases a given linkage distance.
    See :class:`sklearn.cluster.AgglomerativeClustering`.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
        Input feature coordinates and intensities per sample.
    dims : str or list
        Dimensions considered in clustering.
    tol : float or list
        Tolerance in each feature dimension to define maximum cluster linkage
        distance.
    relative : bool or list
        Whether to use relative or absolute tolerances per dimension.

    Returns
    -------
    features : :obj:`~pandas.DataFrame`
        Features concatenated over samples with cluster labels.
    clustering : :obj:`~sklearn.cluster.AgglomerativeClustering`
        Result of the agglomerative clustering operation.

    Raises
    ------
    ValueError
        If `dims`, `tol`, and `relative` are not the same length.

    '''

    # safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)
    relative = deimos.utils.safelist(relative)

    # check dims
    deimos.utils.check_length([dims, tol, relative])

    # connectivity
    if 'sample_idx' not in features.columns:
        cmat = None
    else:
        vals = features['sample_idx'].values.reshape(-1, 1)
        cmat = cdist(vals, vals, metric=lambda x, y: x != y).astype(bool)

    # compute inter-feature distances
    distances = []
    for i, d in enumerate(dims):
        # vectors
        v1 = features[d].values.reshape(-1, 1)

        # distances
        dist = scipy.spatial.distance.cdist(v1, v1)

        if relative[i] is True:
            # divisor
            basis = np.repeat(v1, v1.shape[0], axis=1)
            fix = np.repeat(v1, v1.shape[0], axis=1).T
            basis = np.where(basis == 0, fix, basis)

            # divide
            dist = np.divide(dist, basis, out=np.zeros_like(
                basis), where=basis != 0)

        # check tol
        distances.append(dist / tol[i])

    # stack distances
    distances = np.dstack(distances)

    # max distance
    distances = np.max(distances, axis=-1)

    # perform clustering
    clustering = AgglomerativeClustering(n_clusters=None,
                                         linkage='single',
                                         affinity='precomputed',
                                         distance_threshold=1,
                                         connectivity=cmat).fit(distances)
    features['cluster'] = clustering.labels_
    return features


def agglomerative_clustering(features,
                             dims=['mz', 'drift_time', 'retention_time'],
                             tol=[20E-6, 0.03, 0.3],
                             relative=[True, True, False],
                             meta=None,
                             aggregate=None,
                             min_cluster_size=None,
                             processes=1,
                             partition_kwargs={}):
    '''
    Cluster features within provided linkage tolerances. Recursively merges
    the pair of clusters that minimally increases a given linkage distance.
    See :class:`sklearn.cluster.AgglomerativeClustering`.

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame` or :obj:`~dask.dataframe.DataFrame`
        Input feature coordinates and intensities per sample.
    dims : str or list
        Dimensions considered in clustering.
    tol : float or list
        Tolerance in each feature dimension to define maximum cluster linkage
        distance.
    relative : bool or list
        Whether to use relative or absolute tolerances per dimension.
    meta : list of str
        Meta data columns.
    aggregate : list of str
        Sequential list of meta data columns to aggregate over. Each must be
        present in `meta`, and len(`meta`) > len(`aggregate`).
    min_cluster_size : list of int
        Filter clusters by minimum size at corresponding aggregation level.
        Must be same length as aggregate.
    processes : int
        Number of partitions to process in parallel.
    partition_kwargs : dict
        Arguments passed to :func:`~deimos.subset.multi_sample_partition`.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Features concatenated over samples with cluster labels.

    Raises
    ------
    ValueError
        If `dims`, `tol`, and `relative` are not the same length.
    ValueError
        If len(`meta`) <= len(`aggregate`).
    KeyError
        If `aggregate` entry not in `meta`.
    ValueError
        If `aggregate` and `min_cluster_size` are not the same length.

    '''
    
    def apply_func(group):
        # partition data
        partitions = deimos.subset.multi_sample_partition(group, **partition_kwargs)

        # map agglomerative clustering routine
        res = partitions.map(deimos.alignment._agglomerative_clustering,
                             dims=dims,
                             tol=tol,
                             relative=relative,
                             processes=processes)

        # drop intra-dataset duplicates
        res = res.sort_values(by='intensity', ascending=False).drop_duplicates(subset=['cluster',
                                                                                       'partition_idx',
                                                                                       'sample_idx']).reset_index(drop=True)
        
        # unique cluster indices
        res['cluster'] = res.groupby(by=['partition_idx', 'cluster']).ngroup().reset_index(drop=True)
        
        # cluster counts
        res['n'] = res.groupby(by='cluster')['partition_idx'].transform('size').reset_index(drop=True)
        
        return res.drop(columns='partition_idx')
        
    res = features.copy()
    
    if None not in [meta, aggregate, min_cluster_size]:
        meta = deimos.utils.safelist(meta)
        aggregate = deimos.utils.safelist(aggregate)
        min_cluster_size = deimos.utils.safelist(min_cluster_size)
        
        deimos.utils.check_length([aggregate, min_cluster_size])
        
        if len(meta) <= len(aggregate):
            raise ValueError('`meta` list must be longer than `aggregate` list.')
        if not all([x in meta for x in aggregate]):
            raise KeyError('Entries in `aggregate` must be present in `meta`.')
        
        meta_object = [(c, 'object') for c in res.columns] + [('cluster', 'int'), ('n', 'int')]

        for agg_over, min_size in zip(aggregate, min_cluster_size):
            meta.remove(agg_over)
            
            res = res.groupby(by=meta).apply(apply_func, meta=meta_object).reset_index(drop=True)
            res = res.query('n >= {}'.format(min_size)).drop(columns='n').reset_index(drop=True)
    
    else:
        res = apply_func(res).drop(columns='n')

    return res.compute()


def join(paths, dims=['mz', 'drift_time', 'retention_time'],
         quantiles=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], processes=4,
         partition_kwargs={}, match_kwargs={}):
    '''
    Iteratively apply :func:`deimos.alignment.tolerance` and
    :func:`deimos.alignment.match` across multiple datasets, generating a
    growing set of "clusters", similar to the "join align" approach in MZmine.

    Parameters
    ----------
    paths : list
        List of dataset paths to align.
    dims : str or list
        Dimensions considered in alignment.
    quantiles : :obj:`numpy.array`
        Quantiles of feature intensities to iteratively perform alignment.
    processes : int
        Number of parallel processes. If less than 2, a serial mapping is
        applied.
    partition_kwargs : dict
        Keyword arguments for :func:`deimos.subset.partition`.
    match_kwargs : dict
        Keyword arugments for :func:`deimos.alignment.tolerance` and
        :func:`deimos.alignment.match`.

    Returns
    -------
    :obj:`~pandas.DataFrame`
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
        samp_pass, clust_pass = partitions.zipmap(deimos.alignment.tolerance,
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

    # safely cast to list
    dims = deimos.utils.safelist(dims)

    # column indices
    colnames = dims + ['intensity']

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
                clusters.loc[idx, 'intensity'] += samp_match.loc[:,
                                                                 'intensity'].values
                clusters.loc[idx, 'n'] += 1

                # update cluster centers
                ab = clusters.loc[idx, 'intensity'].values
                b = samp_match.loc[:, 'intensity'].values
                a = ab - b

                for d in dims:
                    clusters.loc[idx, d] = (a * clusters.loc[idx, d].values
                                            + b * samp_match.loc[:, d].values) / ab

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
