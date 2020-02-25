import h5py
import pandas as pd
import numpy as np
import pymzml
from collections import OrderedDict, defaultdict


def read_mzml(path, accession={'drift_time': 'MS:1002476',
                               'retention_time': 'MS:1000016'}):
    """
    Read in an mzML file, parsing for accession values, to yield
    a long-format data frame.

    Parameters
    ----------
    path : str
        Path to input mzML file.
    accession : dict
        Key-value pairs signaling which features to parse for
        in the mzML file.

    Returns
    -------
    out : DataFrame
        Parsed feature coordinates and intensities.

    """

    # open file
    data = pymzml.run.Reader(path)

    # ordered dict
    accession = OrderedDict(accession)

    # result container
    res = defaultdict(list)

    # parse
    for spec in data:
        # init
        arr = np.empty((spec.mz.shape[0], len(accession) + 2))
        cols = ['mz', 'intensity']

        # fill
        arr[:, 0] = spec.mz
        arr[:, 1] = spec.i

        for i, (k, v) in enumerate(accession.items()):
            cols.append(k)
            arr[:, 2 + i] = spec.get(v)

        res['ms{}'.format(spec.ms_level)].append(arr)

    for level in res.keys():
        res[level] = pd.DataFrame(np.concatenate(res[level], axis=0), columns=cols)

    return res


def save_hdf(path, data, dtype={}, compression_level=5):
    """
    Saves data frame to HDF5 container.

    Parameters
    ----------
    path : str
        Path to output file.
    data : dict
        Dictionary of data frames to be saved. Dictionary keys
        will be saved as 'groups' (e.g., MS level) and
        data frame columns will be saved as 'datasets'
        in the HDF5 container.
    dtype : dict
        Specifies what data type to save each column, provided as column:dtype
        pairs. Defaults to 32-bit float if unspecified.
    compression_level : int
        A value from 0-9 signaling the number of compression operations to apply.
        Higher values result in greater compression at the expense of computational
        overhead.

    Returns
    -------
    None.

    """

    with h5py.File(path, 'w') as f:
        for level in data.keys():
            f.create_group(level)
            for c in data[level].columns:
                if c not in dtype.keys():
                    dtype[c] = np.float32

                f[level].create_dataset(c, data=data[level][c].values,
                                        dtype=dtype[c],
                                        compression="gzip",
                                        compression_opts=compression_level)


def load_hdf(path, level='ms1'):
    """
    Loads data frame from HDF5 container.

    Parameters
    ----------
    path : str
        Path to input HDF5 file.
    level : str
        Access this level (group) of the HDF5 container.
        E.g., 'ms1' or 'ms2' for MS levels 1 or 2,
        respectively.

    Returns
    -------
    out : DataFrame
        Feature coordinates and intensities for the
        specified level.

    """

    with h5py.File(path, 'r') as f:
        g = f[level]
        return pd.DataFrame({k: g[k] for k in list(g.keys())})


def safelist(x):
    """
    Ensures passed object is of correct list-like format.

    Parameters
    ----------
    x : any
        Object to be cast as list.

    Returns
    -------
    out : list_like
        Input safely cast to list-like.

    """

    if not isinstance(x, (list, pd.core.series.Series, np.ndarray)):
        return [x]
    return x


def check_length(lists):
    """
    Ensures collection of lists passed are of equal length. If not,
    a value error is raised.

    Parameters
    ----------
    lists : list
        List of lists for length evaluation.

    Returns
    -------
    None.

    """

    it = iter(lists)
    length = len(next(it))
    if not all(len(x) == length for x in it):
        raise ValueError('per-dimension inputs must have same dimension')


def collapse(data, keep=['mz', 'drift_time', 'retention_time'], how=np.sum):
    """
    Collpases input data such that only specified dimensions remain
    according to the supplied aggregation function.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    keep : str or list
        Features to keep during collapse operation.
    how : func
        Aggregation function for collapse operation.

    Returns
    -------
    out : DataFrame
        Collapsed feature coordinates and aggregated
        intensities.

    """

    return data.groupby(by=keep, as_index=False, sort=False).agg({'intensity': how})


def threshold(data, threshold=1000):
    """
    Thresholds input by intensity.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.
    threshold : float
        Intensity threshold value.

    Returns
    -------
    out : DataFrame
        Thresholded feature coordinates and intensities.

    """

    return data.loc[data['intensity'] >= threshold, :].reset_index(drop=True)


def detect_features(data):
    """
    Detects non-intensity feature columns in the input.

    Parameters
    ----------
    data : DataFrame
        Input feature coordinates and intensities.

    Returns
    -------
    features : list
        List of non-intensity features in the input.

    """

    features = list(data.columns)
    features.remove('intensity')
    return features


def save_mgf(path, data, charge='1+'):
    """
    Saves data to MGF format.

    Parameters
    ----------
    path : str
        Path to output file.
    data : DataFrame
        Precursor m/z and intensities paired to
        MS2 spectra.

    Returns
    -------
    None.

    """

    template = ('BEGIN IONS\n'
                'PEPMASS={} {}\n'
                'CHARGE={}\n'
                'TITLE=Spectrum {}\n'
                '{}\n'
                'END IONS\n')

    with open(path, 'w') as f:
        for i, row in data.iterrows():
            precursor_mz = row['mz']
            precursor_int = row['intensity']
            ms2 = row['ms2']

            # check for ms2 spectra
            if ms2 is not np.nan:
                f.write(template.format(precursor_mz, precursor_int, charge, i, ms2.replace(';', '\n')))
