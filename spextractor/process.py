from pyteomics import mzml
import gzip
from os.path import *
import pandas as pd
import multiprocessing as mp
import spextractor as spx


def mzml2hdf(path, output):
    print('reading')
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

    # parse
    print('parsing with {} threads'.format(mp.cpu_count()))
    with mp.Pool(mp.cpu_count()) as p:
        parsed = p.map(_parse, data)

    # generate dataframe
    print('casting to dataframe')
    df = pd.DataFrame(parsed, columns=['ms_level', 'drift_time', 'mz', 'intensity'])

    # explode m/z
    print('exploding mz')
    a = df.set_index(['ms_level', 'drift_time'])['mz'].apply(pd.Series).stack()
    a = a.reset_index()
    a.columns = ['ms_level', 'drift_time', 'sample', 'mz']

    # explode intensity
    print('exploding intensity')
    b = df.set_index(['ms_level', 'drift_time'])['intensity'].apply(pd.Series).stack()
    b = b.reset_index()
    b.columns = ['ms_level', 'drift_time', 'sample', 'intensity']

    # combine
    print('merging mz and intensity')
    a['intensity'] = b['intensity'].values
    a.drop('sample', axis=1, inplace=True)

    # filter zero intensity out
    print('filtering zeros')
    a = a.loc[a['intensity'] > 0, :]

    # group
    print('grouping')
    a = a.groupby(by=['drift_time', 'mz', 'ms_level'], sort=False).sum().reset_index()

    # save
    print('saving')
    spx.utils.save_hdf(a, output)

    print('done')


def _parse(d):
    dt = [x['ion mobility drift time'] for x in d['scanList']['scan']]
    if len(dt) < 2:
        dt = dt[0]
    return [d['ms level'], dt, d['m/z array'], d['intensity array']]
