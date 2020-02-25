import scipy
import numpy as np
from collections import OrderedDict
import deimos
import pandas as pd


class Aligner(OrderedDict):
    """
    Stores regression results as an OrderedDict for later application
    through the apply() method.

    """

    def apply(self, data, inverse=False):
        """
        Applies stored regression results to a dataset.

        Parameters
        ----------
        data : DataFrame
            Input feature coordinates and intensities to be aligned.
        inverse : bool
            Signals whether to perform the inverse alignment
            operation (i.e., if the stored alignment is a aligned to b,
            perform b aligned to a).

        Returns
        -------
        out : DataFrame
            Aligned feature coordinates and intensities.

        """

        for k, v in self.items():
            if inverse:
                data[k] = (data[k] - v.intercept) / v.slope
            else:
                data[k] = v.slope * data[k] + v.intercept

        return data

    def __repr__(self):
        """
        Print representation.

        Parameters
        ----------
        None.

        Returns
        -------
        s : string
            String representation of self.

        """

        s = ''
        for k, v in self.items():
            s += k + ':\n'
            s += '\tslope:\t\t{}\n'.format(v.slope)
            s += '\tintercept:\t{}\n'.format(v.intercept)
            s += '\tr-value:\t{}\n'.format(v.rvalue)
            s += '\tp-value:\t{}\n'.format(v.pvalue)
            s += '\tstderr:\t\t{}\n'.format(v.stderr)

        return s[:-2]


def proximity_screen(data, features=['mz', 'drift_time', 'retention_time'],
                     tol=[10E-6, 0.2, 0.11]):
    """
    Filter features by their proximity to the next closest feature.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    features : list
        Features to filter against.
    tol : float or list
        Tolerance in each feature dimension to define proximity.

    Returns
    -------
    out : DataFrame
        Filtered feature coordinates and intensities.

    """

    # safely cast to list
    features = deimos.utils.safelist(features)
    tol = deimos.utils.safelist(tol)

    # check dims
    deimos.utils.check_length([features, tol])

    # compute intra-feature distances
    idx = []
    for i, f in enumerate(features):
        arr = data[f].values.reshape((-1, 1))
        d = scipy.spatial.distance.cdist(arr, arr)

        if f == 'mz':
            d = np.divide(d, scipy.spatial.distance.cdist(arr, arr, min))

        # overwrite identity
        d[d == 0] = np.inf

        # take minimum
        d = np.min(d, axis=0)

        # greater than tolerance
        idx.append(d > tol[i])

    # stack truth arrays
    idx = np.nonzero(np.prod(np.vstack(idx), axis=0))[0]

    return data.iloc[idx, :].reset_index(drop=True)


def match_features(a, b, features=['mz', 'drift_time', 'retention_time'],
                   tol=[10E-6, 0.2, 0.11]):
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

    # compute normalized 3d distance
    v1 = a[features].values / np.array(tol)
    v2 = b[features].values / np.array(tol)
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


def fit(a, b, features=['mz', 'drift_time', 'retention_time']):
    """
    Given a set of matched features, fit a linear regression defining
    the alignment.

    Parameters
    ----------
    a, b : DataFrame
        Input feature coordinates and intensities. Features from a are
        matched to features in b.
    features : list
        Features to match against.

    Returns
    -------
    result : Aligner
        Aligner object containing regression results.

    """

    # safely cast to list
    features = deimos.utils.safelist(features)

    # perform regressions
    result = Aligner()
    for f in features:
        result[f] = scipy.stats.linregress(a[f].values, b[f].values)

    return result


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
