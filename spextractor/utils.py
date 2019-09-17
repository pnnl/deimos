import h5py
import pandas as pd
import numpy as np


def save_hdf(df, path, compression_level=5):
    with h5py.File(path, 'w') as f:
        for c in df.columns:
            f.create_dataset(c, data=df[c].values,
                             compression="gzip",
                             compression_opts=compression_level)


def load_hdf(path):
    with h5py.File(path, 'r') as f:
        return pd.DataFrame({k: np.array(f[k]) for k in list(f.keys())})


def safelist(x):
    if not isinstance(x, list):
        return list(x)
    return x


def check_length(lists):
    it = iter(lists)
    length = len(next(it))
    if not all(len(x) == length for x in it):
        raise ValueError('per-dimension inputs must have same dimension')
