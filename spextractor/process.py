from pyteomics import mzml
import gzip
from os.path import *
import pandas as pd
import multiprocessing as mp
import spextractor as spx


def mzml2hdf(path, output):
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
    with mp.Pool(mp.cpu_count()) as p:
        parsed = [x for x in p.imap(_parse, data)]

    # generate dataframe
    df = pd.concat(parsed, ignore_index=True)

    # group
    df = df.groupby(by=['retention_time', 'drift_time', 'mz', 'ms_level'],
                    sort=False).sum().reset_index()

    # save
    spx.utils.save_hdf(df, output)


def _parse(d):
    try:
        dt = [x['ion mobility drift time'] for x in d['scanList']['scan']][0]
    except:
        rt = np.nan

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

    return df
