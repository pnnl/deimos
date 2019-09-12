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
        parsed = [x for x in p.imap(_parse, data)]

    # generate dataframe
    print('concatenating dataframes')
    df = pd.concat(parsed, ignore_index=True)

    # group
    print('grouping')
    df = df.groupby(by=['drift_time', 'mz', 'ms_level'], sort=False).sum().reset_index()

    # save
    print('saving')
    spx.utils.save_hdf(df, output)

    print('done')


def _parse(d):
    dt = [x['ion mobility drift time'] for x in d['scanList']['scan']]
    if len(dt) < 2:
        dt = dt[0]

    df = pd.DataFrame(data={'mz': d['m/z array'], 'intensity': d['intensity array']})

    df['ms_level'] = d['ms level']
    df['drift_time'] = dt

    # group
    df = df.groupby(by=['drift_time', 'mz', 'ms_level'], sort=False).sum().reset_index()

    # filter zero intensity out
    df = df.loc[df['intensity'] > 0, :]

    return df
