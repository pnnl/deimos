import h5py
import pandas as pd
import numpy as np
import spextractor as spx


def save_hdf(df, path):
    with h5py.File(path, 'w') as f:
        for c in df.columns:
            f.create_dataset(c, data=df[c].values, compression_opts=9, compression="gzip")


def load_hdf(path, group=False):
    with h5py.File(path, 'r') as f:
        if group is False:
            return pd.DataFrame({k: np.array(f[k]) for k in list(f.keys())})
        elif group is True:
            return spx.process._group(pd.DataFrame({k: np.array(f[k]) for k in list(f.keys())}))
        else:
            raise KeyError(group)
