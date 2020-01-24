import h5py
import pandas as pd
import numpy as np
import pymzml
from os.path import *
from collections import OrderedDict, defaultdict


def read_mzml(path, accession={'drift_time': 'MS:1002476',
                               'retention_time': 'MS:1000016'}):
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

        # drop
        arr = arr[np.where(arr[:, 1] > 0)]

        for i, (k, v) in enumerate(accession.items()):
            cols.append(k)
            arr[:, 2 + i] = spec.get(v)

        res['ms{}'.format(spec.ms_level)].append(arr)

    for level in res.keys():
        res[level] = pd.DataFrame(np.concatenate(res[level], axis=0), columns=cols)

    return res


def save_hdf(path, data, dtype={}, compression_level=5):
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
    with h5py.File(path, 'r') as f:
        g = f[level]
        return pd.DataFrame({k: g[k] for k in list(g.keys())})


def safelist(x):
    if not isinstance(x, (list, pd.core.series.Series, np.ndarray)):
        return [x]
    return x


def check_length(lists):
    it = iter(lists)
    length = len(next(it))
    if not all(len(x) == length for x in it):
        raise ValueError('per-dimension inputs must have same dimension')


def collapse(data, keep=['mz', 'drift_time', 'retention_time'], how=np.sum):
    return data.groupby(by=keep, as_index=False, sort=False).agg({'intensity': how})


def save_mgf(data, path, charge='1+'):
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
