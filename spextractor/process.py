from pyteomics import mzml
import gzip
from os.path import *
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial


def _parse(d):
    dt = [x['ion mobility drift time'] for x in d['scanList']['scan']]
    if len(dt) < 2:
        dt = dt[0]
    return [d['ms level'], dt, d['m/z array'], d['intensity array']]


def mzML(path, output):
    # check for zip
    if splitext(path)[-1].lower() == 'mzml':
        f = path
        close = False
    else:
        f = gzip.open(path, 'rb')
        close = True

    # process mzml
    data = [x for x in mzml.read(f)]

    # close gzip file
    if close:
        f.close()

    # parse contents
    # parsed = []
    # for d in data:
    #     dt = [x['ion mobility drift time'] for x in d['scanList']['scan']]
    #     if len(dt) < 2:
    #             dt = dt[0]
    #     parsed.append([d['ms level'], dt, d['m/z array'], d['intensity array']])

    with mp.Pool(mp.cpu_count()) as p:
        parsed = p.map(_parse, data)

    # generate dataframe
    df = pd.DataFrame(parsed, columns=['ms level', 'drift time', 'm/z', 'intensity'])

    # explode m/z
    a = df.set_index(['ms level', 'drift time'])['m/z'].apply(pd.Series).stack()
    a = a.reset_index()
    a.columns = ['ms level', 'drift time', 'sample', 'm/z']

    # explode intensity
    b = df.set_index(['ms level', 'drift time'])['intensity'].apply(pd.Series).stack()
    b = b.reset_index()
    b.columns = ['ms level', 'drift time', 'sample', 'intensity']

    # combine
    a['intensity'] = b['intensity'].values
    a.drop('sample', axis=1, inplace=True)

    # clean variables
    df = None
    b = None

    # filter zero intensity out
    a = a.loc[a['intensity'] > 0, :]

    # group
    a = a.groupby(by=['drift time', 'm/z']).sum().reset_index()

    # save
    a.to_hdf(output, key='msms', mode='w', complevel=9)


def _feature_mapper(row, ms2):
    index, row = row
    feature = {}
    feature['drift time'] = row['drift time']
    feature['m/z'] = row['m/z']
    feature['intensity'] = row['intensity']
    feature['fragments'] = ms2.loc[ms2['drift time'] == row['drift time'], ['m/z', 'intensity']]
    return feature


def align(path, output, stdev=[3, 0]):
    # read input
    df = pd.read_hdf(path, 'msms')

    # separate ms levels
    ms1 = df.loc[df['ms level'] == 1, :].sort_values(by='intensity', ascending=False).reset_index()
    ms2 = df.loc[df['ms level'] == 2, :].sort_values(by=['drift time', 'm/z', 'intensity'], ascending=[True, True, False]).reset_index()

    # top ms
    if stdev[0] > 0:
        ms1 = ms1.loc[ms1['intensity'] > ms1['intensity'].mean() + stdev[0] * ms1['intensity'].std(), :].reset_index()
    if stdev[1] > 0:
        ms2 = ms2.loc[ms2['intensity'] > ms2['intensity'].mean() + stdev[1] * ms2['intensity'].std(), :].reset_index()

    with mp.Pool(mp.cpu_count()) as p:
        features = p.map(partial(_feature_mapper, ms2=ms2), ms1.iterrows())

    np.save(output, features)
