from pyteomics import mzml
import gzip
from os.path import *
import pandas as pd
import multiprocessing as mp
import spextractor as spx


def mzml2hdf(path, output):
    print('reading')
    # check for zip
    if splitext(path)[-1].lower() == '.mzml':
        f = path

        # process mzml
        data = mzml.read(f)
    else:
        f = gzip.open(path, 'rb')

        # process mzml
        data = mzml.read(f)

        # close file
        f.close()

    # parse
    print('parsing with {} threads'.format(mp.cpu_count()))
    with mp.Pool(mp.cpu_count()) as p:
        parsed = p.map(_parse, data)

    # generate dataframe
    print('concatenating dataframes')
    df = pd.concat(parsed)

    # group
    df = df.groupby(by=['drift_time', 'mz', 'ms_level'], sort=False).sum().reset_index()

    # save
    print('saving')
    spx.utils.save_hdf(df, output)

    print('done')


def _parse(d):
    dt = [x['ion mobility drift time'] for x in d['scanList']['scan']]
    if len(dt) < 2:
        dt = dt[0]

    df = pd.DataFrame([d['ms level'], dt, d['m/z array'], d['intensity array']],
                      columns=['ms_level', 'drift_time', 'mz', 'intensity'])

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

    # group
    a = a.groupby(by=['drift_time', 'mz', 'ms_level'], sort=False).sum().reset_index()

    # filter zero intensity out
    a = a.loc[a['intensity'] > 0, :]

    return a
