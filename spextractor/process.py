from pyteomics import mzml
import gzip
from os.path import *
import pandas as pd
import multiprocessing as mp
from scipy import stats
import numpy as np
import spextractor as spx


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


def merge(path, output, thresh=0, grid=True, xbins='auto', ybins='auto'):
    # read input
    df = spx.utils.load_hdf(path)

    # separate ms levels
    ms1 = df.loc[df['ms_level'] == 1, :].drop('ms_level', axis=1).reset_index(drop=True)
    ms2 = df.loc[df['ms_level'] == 2, :].drop('ms_level', axis=1).reset_index(drop=True)

    # grid reduction
    if grid is True:
        loginf = {'ms1': len(ms1.index), 'ms2': len(ms2.index)}
        ms1 = _grid_reduce(ms1, xbins=xbins, ybins=ybins)
        print('ms1 grid compression ratio:\t%.2f' % (loginf['ms1'] / len(ms1.index)))
        ms2 = _grid_reduce(ms2, xbins=xbins, ybins=ybins)
        print('ms2 grid compression ratio:\t%.2f' % (loginf['ms2'] / len(ms2.index)))

    # threshold
    ms1 = ms1.loc[ms1['intensity'] > thresh, :]
    ms2 = ms2.loc[ms2['intensity'] > thresh, :]

    # merge ms levels
    features = _chunkmerge(ms1, ms2, chunksize=1000)

    # save
    spx.utils.save_hdf(features, output)


def _parse(d):
    dt = [x['ion mobility drift time'] for x in d['scanList']['scan']]
    if len(dt) < 2:
        dt = dt[0]
    return [d['ms level'], dt, d['m/z array'], d['intensity array']]


def _grid_reduce(df, x='mz', y='drift_time', z='intensity', xbins='auto', ybins='auto'):
    if xbins.lower() == 'auto':
        xbins = (df[x].max() - df[x].min()) / np.mean(np.diff(np.sort(df[x].unique())))
    if ybins.lower() == 'auto':
        ybins = (df[y].max() - df[y].min()) / np.mean(np.diff(np.sort(df[y].unique())))

    H, xe, ye, bn = stats.binned_statistic_2d(df[x], df[y], df[z],
                                              statistic='sum',
                                              bins=(xbins, ybins))
    H = np.nan_to_num(H)
    XX, YY = np.meshgrid(xe, ye, indexing='ij')

    # bin centers
    XX = (XX[1:, 1:] + XX[:-1, :-1]) / 2
    YY = (YY[1:, 1:] + YY[:-1, :-1]) / 2

    # construct data frame
    res = np.hstack((XX.reshape(-1, 1), YY.reshape(-1, 1), H.reshape(-1, 1)))
    res = pd.DataFrame(res, columns=['mz', 'drift_time', 'intensity']).astype(spx.utils.types(['mz', 'drift_time', 'intensity']))
    res = res.loc[res['intensity'] > 0, :]

    return res


def _chunkmerge(df1, df2, chunksize=1000):
    res = []
    chunks = int(np.ceil(len(df2.index) / chunksize))
    print('merging...')
    for i in range(0, chunks):
        print('\t%.2f%%' % (i / chunks))
        res.append(df1.merge(df2.loc[i * chunksize:(i + 1) * chunksize, :], on='drift_time', how='inner', suffixes=['_ms1', '_ms2']))

    print('concatenating...')
    return pd.concat(res, axis=0, ignore_index=True)


def _merge2(df1, df2):
    def f(row, df):
        res = df.loc[df['drift_time'] == row['drift_time'], ['mz', 'intensity']]
        row['mz_ms2'] = res['mz'].values.flatten().astype(np.float32)
        row['intensity_ms2'] = res['intensity'].values.flatten().astype(np.int8)
        return row

    return df1.apply(f, args=(df2), axis=1)
