import scipy
import numpy as np
from collections import OrderedDict
import deimos


class Aligner(OrderedDict):
    def apply(self, data, inverse=False):
        for k, v in self.items():
            if inverse:
                data[k] = (data[k] - v.intercept) / v.slope
            else:
                data[k] = v.slope * data[k] + v.intercept

        return data

    def __repr__(self):
        s = ''
        for k, v in self.items():
            s += k + ':\n'
            s += '\tslope:\t\t{}\n'.format(v.slope)
            s += '\tintercept:\t{}\n'.format(v.intercept)
            s += '\tr-value:\t{}\n'.format(v.rvalue)
            s += '\tp-value:\t{}\n'.format(v.pvalue)
            s += '\tstderr:\t\t{}\n'.format(v.stderr)

        return s[:-2]


def proximity_screen(data, features=['mz', 'drift_time', 'retention_time'], tol=[10E-6, 0.2, 0.11]):
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


def match_features(a, b, features=['mz', 'drift_time', 'retention_time'], tol=[10E-6, 0.2, 0.11]):
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
    # perform regressions
    res = Aligner()
    for f in features:
        res[f] = scipy.stats.linregress(a[f].values, b[f].values)

    return res
