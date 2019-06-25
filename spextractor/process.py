from pyteomics import mzml
import gzip
from os.path import *
import pandas as pd
import multiprocessing as mp
import spextractor as spx


def _parse(d):
    dt = [x['ion mobility drift time'] for x in d['scanList']['scan']]
    if len(dt) < 2:
        dt = dt[0]
    return [d['ms level'], dt, d['m/z array'], d['intensity array']]


def mzML(path, output):
    # check for zip
    if splitext(path)[-1].lower() == 'mzml':
        f = path

        # process mzml
        data = [x for x in mzml.read(f)]
    else:
        f = gzip.open(path, 'rb')

        # process mzml
        data = [x for x in mzml.read(f)]

        # close file
        f.close()

    with mp.Pool(mp.cpu_count()) as p:
        parsed = p.map(_parse, data)

    # generate dataframe
    df = pd.DataFrame(parsed, columns=['ms_level', 'drift_time', 'mz', 'intensity'])

    # explode m/z
    a = df.set_index(['ms_level', 'drift_time'])['mz'].apply(pd.Series).stack()
    a = a.reset_index()
    a.columns = ['ms_level', 'drift_time', 'sample', 'mz']

    # explode intensity
    b = df.set_index(['ms_level', 'drift_time'])['intensity'].apply(pd.Series).stack()
    b = b.reset_index()
    b.columns = ['ms_level', 'drift_time', 'sample', 'intensity']

    # combine
    a['intensity'] = b['intensity'].values
    a.drop('sample', axis=1, inplace=True)

    # filter zero intensity out
    a = a.loc[a['intensity'] > 0, :]

    # group
    a = a.groupby(by=['drift_time', 'mz', 'ms_level'], sort=False).sum().reset_index()

    # save
    spx.utils.save_hdf(a, output)


def merge(path, output, stdev=[4, 3]):
    # read input
    df = spx.utils.load_hdf(path)

    # separate ms levels
    ms1 = df.loc[df['ms_level'] == 1, :].drop('ms_level', axis=1).reset_index(drop=True)
    ms2 = df.loc[df['ms_level'] == 2, :].drop('ms_level', axis=1).reset_index(drop=True)

    # top ms
    if stdev[0] > 0:
        ms1 = ms1.loc[ms1['intensity'] > ms1['intensity'].mean() + stdev[0] * ms1['intensity'].std(), :].reset_index(drop=True)
    if stdev[1] > 0:
        ms2 = ms2.loc[ms2['intensity'] > ms2['intensity'].mean() + stdev[1] * ms2['intensity'].std(), :].reset_index(drop=True)

    features = ms1.merge(ms2, on='drift_time', suffixes=['_ms1', '_ms2']).sort_values(by=['intensity_ms1', 'drift_time', 'mz_ms1', 'mz_ms2'], ascending=[False, True, True, True])

    # save
    spx.utils.save_hdf(features, output)
