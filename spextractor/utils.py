import h5py
import pandas as pd
import numpy as np


def save_hdf(df, path):
    with h5py.File(path, 'w') as f:
        for c in df.columns:
            f.create_dataset(c, data=df[c].values, compression_opts=9, compression="gzip")


def load_hdf(path):
    with h5py.File(path, 'r') as f:
        return pd.DataFrame({k: np.array(f[k]) for k in list(f.keys())})
