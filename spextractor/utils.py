import h5py
import pandas as pd
import numpy as np


dtypes = {'mz': np.float32,
          'intensity': np.uint32,
          'retention_time': np.float32,
          'drift_time': np.float32,
          'ms_level': np.uint8}


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
