import h5py
import pandas as pd
import numpy as np
import warnings


dtype_dict = {'drift_time': np.float32,
              'mz_ms1': np.float32,
              'intensity_ms1': np.uint8,
              'mz_ms2': np.float32,
              'intensity_ms2': np.uint8,
              'mz': np.float32,
              'intensity': np.uint8,
              'ms_level': np.uint8}


def types(keys):
    return {k: dtype_dict[k] for k in keys}


def save_hdf(df, path):
    with h5py.File(path, 'w') as f:
        for c in df.columns:
            if c in dtype_dict.keys():
                f.create_dataset(c, data=df[c].values.astype(dtype_dict[c]), compression_opts=9, compression="gzip")
            else:
                f.create_dataset(c, data=df[c].values, compression_opts=9, compression="gzip")


def load_hdf(path, group=False):
    with h5py.File(path, 'r') as f:
        if group is False:
            return pd.DataFrame({k: np.array(f[k]) for k in list(f.keys())}).astype(types(f.keys()))
        elif group is True:
            return _group(pd.DataFrame({k: np.array(f[k]) for k in list(f.keys())}).astype(types(f.keys())))
        else:
            raise KeyError(group)


def _group(df):
    # check all keys present
    for key in ['drift_time', 'mz_ms1', 'intensity_ms1', 'mz_ms2', 'intensity_ms2']:
        if key not in df.columns:
            warnings.warn("This function requires a pandas data frame with columns \
                          ['drift_time', 'mz_ms1', 'intensity_ms1', 'mz_ms2', 'intensity_ms2']")
            raise KeyError(key)

    # group
    g = df.groupby(by=['drift_time',
                       'mz_ms1',
                       'intensity_ms1']).agg({'mz_ms2': lambda x: list(x),
                                              'intensity_ms2': lambda x: list(x)}).reset_index()

    # sort
    return g.sort_values(by=['intensity_ms1', 'drift_time', 'mz_ms1'], ascending=[False, True, True])
