import h5py
import pandas as pd
import numpy as np
from pyteomics import mzml
import gzip
from os.path import *
import multiprocessing as mp


dtypes = {'mz': np.float32,
          'intensity': np.uint32,
          'retention_time': np.float32,
          'drift_time': np.float32,
          'ms_level': np.uint8}


def read_mzml(path):
    # check for zip
    ext = splitext(path)[-1].lower()
    if ext == '.gz':
        zipped = True
        f = gzip.open(path, 'rb')
    else:
        zipped = False
        f = path

    # mzml file handle
    data = mzml.read(f)

    # parse in parallel
    with mp.Pool(mp.cpu_count()) as p:
        df = pd.concat([x for x in p.imap_unordered(_parse, data, chunksize=1000)], ignore_index=True)

    # close zip file
    if zipped:
        f.close()

    # replace with nan
    df = df.replace(-1, np.nan)

    # drop missing axes
    df = df.dropna(axis=1, how='all')

    return df


def _parse(d):
    # drift time
    try:
        dt = d['scanList']['scan'][0]['ion mobility drift time']
    except:
        dt = -1

    # retention time
    try:
        rt = d['scanList']['scan'][0]['scan start time']
    except:
        rt = -1

    # precursor info
    try:
        precursor = d['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
        pre_mz = precursor['selected ion m/z']
        pre_int = precursor['peak intensity']
    except:
        pre_mz = -1
        pre_int = -1

    df = pd.DataFrame(data={'mz': d['m/z array'], 'intensity': d['intensity array']})

    df['ms_level'] = d['ms level']
    df['drift_time'] = dt
    df['retention_time'] = rt
    df['mz_precursor'] = pre_mz
    df['intensity_precursor'] = pre_int

    # filter zero intensity out
    df = df.loc[df['intensity'] > 0, :]

    # return df.values
    return df


def dtype(k):
    if k in dtypes.keys():
        return dtypes[k]
    else:
        return np.float32


def save_hdf(df, path, compression_level=5):
    idx = np.array(df['ms_level'] == 1)
    with h5py.File(path, 'w') as f:
        for level in ['ms1', 'ms2']:
            f.create_group(level)

        for c in df.columns:
            if c == 'ms_level':
                pass
            else:
                f['ms1'].create_dataset(c, data=df.loc[idx, c].values,
                                        dtype=dtype(c),
                                        compression="gzip",
                                        compression_opts=compression_level)
                f['ms2'].create_dataset(c, data=df.loc[~idx, c].values,
                                        dtype=dtype(c),
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
