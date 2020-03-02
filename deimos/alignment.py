import scipy
import numpy as np
import deimos
import pandas as pd
import statsmodels


def match_features(a, b, features=['mz', 'drift_time', 'retention_time'],
                   ignore=None, tol=[10E-6, 0.2, 0.11]):
    """
    Match features by their proximity to the closest feature in another dataset.

    Parameters
    ----------
    a, b : DataFrame
        Input feature coordinates and intensities. Features from a are
        matched to features in b.
    features : list
        Features to match against.
    tol : float or list
        Tolerance in each feature dimension to define a match.
    ignore : str or list
        Ignore during distance calculation, e.g. for highly misaligned
        dimensions. Does not affect tolerance filter. Ambiguous matches
        in the ignored dimensions are dropped.

    Returns
    -------
    a, b : DataFrame
        Matched feature coordinates and intensities. E.g., a[i..n] and
        b[i..n] each represent matched features.

    """

    # safely cast to list
    features = deimos.utils.safelist(features)
    tol = deimos.utils.safelist(tol)

    # check dims
    deimos.utils.check_length([features, tol])

    # mask ignored
    if ignore is not None:
        ignore = deimos.utils.safelist(ignore)
        mask_idx = [features.index(x) for x in ignore]
        features_masked = [j for i, j in enumerate(features) if i not in mask_idx]
        tol_masked = [j for i, j in enumerate(tol) if i not in mask_idx]
    else:
        features_masked = features
        tol_masked = tol

    # groupby ops
    if ignore is not None:
        # placeholder for group counts
        a['count'] = 1
        b['count'] = 1

        # aggregator
        agg = {k: np.mean for k in ignore}
        agg['intensity'] = np.sum
        agg['count'] = np.sum

        # group
        a = a.groupby(by=features_masked, as_index=False, sort=False).agg(agg)
        b = b.groupby(by=features_masked, as_index=False, sort=False).agg(agg)

        # filter by counts
        a = a.loc[a['count'] == 1, :].drop('count', axis=1)
        b = b.loc[b['count'] == 1, :].drop('count', axis=1)

    # compute normalized 3d distance
    v1 = a[features_masked].values / np.array(tol_masked)
    v2 = b[features_masked].values / np.array(tol_masked)
    dist3d = scipy.spatial.distance.cdist(v1, v2)

    # compute inter-feature distances
    distances = []
    for f in features:
        v1 = a[f].values.reshape(-1, 1)
        v2 = b[f].values.reshape(-1, 1)

        d = scipy.spatial.distance.cdist(v1, v2)

        if f == 'mz':
            d = np.divide(d, scipy.spatial.distance.cdist(v1, v2, min))

        distances.append(d)

    # stack
    distances = np.dstack(distances)

    # compute indices where each condition true
    idx = []
    for i in range(len(features)):
        idx.append(distances[:, :, i] <= tol[i])

    # stack truth arrays
    idx = np.prod(np.dstack(idx), axis=-1, dtype=bool)

    # per-dataset indices
    mn = np.amin(dist3d, axis=1, where=idx, initial=np.inf, keepdims=True)
    ii, jj = np.where(np.repeat(mn, dist3d.shape[1], axis=1) == dist3d)

    # reorder
    a = a.iloc[ii].reset_index(drop=True)
    b = b.iloc[jj].reset_index(drop=True)

    return a, b


def lowess(a, b, align='retention_time', tol=[10E-6, 0.2],
           frac=0.2, it=10, s=2):
    """
    Match features by their proximity to the closest feature in another dataset.

    Parameters
    ----------
    a, b : DataFrame
        Matched input feature coordinates and intensities.
    align : str
        Feature to align.
    frac : float
        Between 0 and 1. The fraction of the data used for the LOWESS fit.
    it : int
        The number of residual-based reweightings to perform during LOWESS fit.
    s : int
        Positive smoothing factor used to choose the number of knots in the
        univariate spline approximation.

    Returns
    -------
    spl, spl_inv : UnivariateSpline
        Forward and inverse spline approximations of the LOWESS
        fit.

    """

    # uniqueify
    x = a[align].values
    y = b[align].values
    arr = np.vstack((x, y)).T
    arr = np.unique(arr, axis=0)

    # fit forward lowess
    lw = statsmodels.nonparametric.smoothers_lowess.lowess(arr[:, 1],
                                                           arr[:, 0],
                                                           frac=frac,
                                                           it=it)

    # unique x
    _, idx = np.unique(lw[:, 0], return_index=True)

    # spline
    spl = scipy.interpolate.UnivariateSpline(lw[idx, 0], lw[idx, 1], s=2)

    # fit reverse lowess
    lw_inv = statsmodels.nonparametric.smoothers_lowess.lowess(arr[:, 0],
                                                               arr[:, 1],
                                                               frac=frac,
                                                               it=it)

    # unique x
    _, idx = np.unique(lw_inv[:, 0], return_index=True)

    # spline
    spl_inv = scipy.interpolate.UnivariateSpline(lw_inv[idx, 0], lw_inv[idx, 1], s=s)

    return spl, spl_inv


def internal_standards(data, masses, tol=0.02):
    """
    Detect internal standards (by mass only) in the dataset.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    masses : array_like
        Expected masses of the internal standards.
    tol : float
        Tolerance in mass to define a match.

    Returns
    -------
    out : DataFrame
        Feature coordinates and intensities that matched to
        provided internal standards by mass.

    """

    out = []
    features = deimos.utils.detect_features(data)

    for m in masses:
        tmp = deimos.targeted.find_feature(data, by='mz', loc=m, tol=tol)
        if tmp is not None:
            out.append(tmp)

    out = pd.concat(out).reset_index(drop=True)
    out = deimos.utils.collapse(out, keep=features)
    return out
