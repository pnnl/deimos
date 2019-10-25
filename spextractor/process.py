from pyteomics import mzml
import gzip
from os.path import *
import pandas as pd
import multiprocessing as mp
import spextractor as spx
import time
import numpy as np
import dask.dataframe as dd


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
        # df = pd.DataFrame(np.concatenate([x for x in p.imap_unordered(_parse, data, chunksize=100)]),
        #                   columns=['retention_time', 'drift_time', 'mz', 'ms_level', 'intensity'])
        df = pd.concat([x for x in p.imap_unordered(_parse, data, chunksize=100)], ignore_index=True)

    # close zip file
    if zipped:
        f.close()

    # groupby
    df = dd.from_pandas(df, npartitions=1000)
    df = df.groupby(by=['retention_time', 'drift_time', 'mz', 'ms_level']).sum().reset_index().compute()

    # drop missing axes
    df = df.dropna(axis=1, how='all')

    # save
    spx.utils.save_hdf(df, output)


def _parse(d):
    try:
        dt = [x['ion mobility drift time'] for x in d['scanList']['scan']][0]
    except:
        dt = np.nan

    try:
        rt = [x['scan start time'] for x in d['scanList']['scan']][0]
    except:
        rt = np.nan

    df = pd.DataFrame(data={'mz': d['m/z array'], 'intensity': d['intensity array']})

    df['ms_level'] = d['ms level']
    df['drift_time'] = dt
    df['retention_time'] = rt

    # group
    df = df.groupby(by=['retention_time', 'drift_time', 'mz', 'ms_level'],
                    sort=False).sum().reset_index()

    # filter zero intensity out
    df = df.loc[df['intensity'] > 0, :]

    # return df.values
    return df


class HDFConcat:
    def __init__(self, path):
        self.path = path
        self.store = pd.HDFStore(self.path)

    def append(self, df):
        self.store.append('data', df, data_columns=True)

    def get(self):
        df = self.store.select('data')
        self.store.close()
        os.remove(self.path)
        return df
