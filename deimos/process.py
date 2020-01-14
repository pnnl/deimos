from pyteomics import mzml
import gzip
from os.path import *
import pandas as pd
import multiprocessing as mp
import deimos
import numpy as np


def mzml2hdf(path, output):
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

    # save
    deimos.utils.save_hdf(df, output)


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


# class HDFConcat:
#     def __init__(self, path):
#         self.path = path
#         self.store = pd.HDFStore(self.path)

#     def append(self, df):
#         self.store.append('data', df, data_columns=True)

#     def get(self):
#         df = self.store.select('data')
#         self.store.close()
#         os.remove(self.path)
#         return df