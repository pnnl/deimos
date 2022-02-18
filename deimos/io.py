from collections import OrderedDict
import dask.dataframe as dd
import h5py
import numpy as np
import os
import pandas as pd
import pymzml
import warnings


def read_mzml(path, accession={'drift_time': 'MS:1002476',
                               'retention_time': 'MS:1000016'}):
    '''
    Read in an mzML file, parsing for accession values, to yield a long-format
    :obj:`~pandas.DataFrame`.

    Parameters
    ----------
    path : str
        Path to input mzML file.
    accession : dict
        Key-value pairs signaling which features to parse for in the mzML file.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Parsed feature coordinates and intensities.

    '''

    # open file
    data = pymzml.run.Reader(path)

    # ordered dict
    accession = OrderedDict(accession)

    # define columns with known integer values
    integer_cols = ['frame', 'scan', 'intensity']

    # column name container
    cols = {}

    # result container
    res = {}

    # enumerate spectra
    for i, spec in enumerate(data):
        # number of rows
        n = spec.mz.shape[0]
        
        # no measurements
        if n == 0:
            continue
            
        # dimension check
        if len(spec.mz) != len(spec.i):
            warnings.warn("m/z and intensity array dimension mismatch")
            continue
            
        # scan/frame info
        id_dict = spec.id_dict
            
        # check for precursor
        precursor_info = {}
        has_precursor = False
        if spec.selected_precursors:
            has_precursor = True
            precursor_info = {'precursor_mz': spec.selected_precursors[0].get('mz', None)}
        
        # get ms level
        level = 'ms{}'.format(spec.ms_level)
        
        # append to result container
        if level not in res:
            res[level] = []

        # columns
        cols[level] = list(id_dict.keys()) \
                    + list(accession.keys()) \
                    + ['mz', 'intensity'] \
                    + list(precursor_info.keys())
        m = len(cols[level])
        
        # array init
        arr = np.empty((n, m), dtype=float)
        inx = 0
                    
        # populate scan/frame info
        for k, v in id_dict.items():
            arr[:, inx] = v
            inx += 1

        # populate accession fields
        for k, v in accession.items():
            arr[:, inx] = spec.get(v)
            inx += 1
            
        # populate m/z
        arr[:, inx] = spec.mz
        inx += 1
            
        # populate intensity
        arr[:, inx] = spec.i
        inx += 1

        # populate precursor information
        if has_precursor:
            for k, v in precursor_info.items():
                arr[:, inx] = v
                inx += 1

        # append array
        res[level].append(arr)

    # concatenate
    to_int = [x for x in cols[level] if x in integer_cols]
    for level in res.keys():
        res[level] = pd.DataFrame(np.concatenate(res[level], axis=0), columns=cols[level])
        res[level][to_int] = res[level][to_int].astype(int)
    
    return res


def save_hdf(path, data, key='ms1', compression_level=5, **kwargs):
    '''
    Saves dictionary of :obj:`~pandas.DataFrame`s to HDF5 container.

    Parameters
    ----------
    path : str
        Path to output file.
    data : :obj:`~pandas.DataFrame`
        :obj:`~pandas.DataFrame to be saved.
    key : str
        Save to this level (group) of the HDF5 container. E.g., "ms1" or "ms2"
        for MS levels 1 or 2, respectively.
    compression_level : int
        A value from 0-9 signaling the number of compression operations to
        apply. Higher values result in greater compression at the expense of
        computational overhead.
    kwargs
        Keyword arguments exposed by :meth:`~pandas.DataFrame.to_hdf`.

    '''

    data.to_hdf(path, key, format='table', complib='blosc',
                complevel=compression_level, **kwargs)


def load_hdf(path, key='ms1', columns=None):
    '''
    Loads data frame from HDF5 container.

    Parameters
    ----------
    path : str
        Path to input HDF5 file.
    key : str
        Access this level (group) of the HDF5 container. E.g., "ms1" or "ms2"
        for MS levels 1 or 2, respectively.
    columns : list
        A list of columns names to return.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Feature coordinates and intensities for the specified level.

    '''

    return pd.read_hdf(path, key=key, columns=columns)


def load_hdf_multi(paths, key='ms1', columns=None, chunksize=1E7, meta=None):
    '''
    Loads data frame from HDF5 container using Dask.

    Parameters
    ----------
    paths : list of str
        Paths to input HDF5 files.
    key : str
        Access this level (group) of the HDF5 container. E.g., "ms1" or "ms2"
        for MS levels 1 or 2, respectively.
    columns : list
        A list of columns names to return.
    chunksize : int
        Dask partition chunksize.
    meta : dict
        Dictionary of meta data per path.

    Returns
    -------
    :obj:`~dask.dataframe.DataFrame`
        Feature coordinates and intensities for the specified level.

    '''

    df = [dd.read_hdf(x, key=key, chunksize=int(chunksize), columns=columns) for x in paths]

    # label each sample
    for i in range(len(paths)):
        df[i]['sample_idx'] = i  # force unique label in toy case
        df[i]['sample_id'] = os.path.splitext(os.path.basename(paths[i]))[0]

        if meta is not None:
            for k, v in meta.items():
                df[i][k] = v[i]

    # concat results
    return dd.concat(df, axis=0)


def _save_hdf(path, data, dtype={}, compression_level=5):
    '''
    Deprecated version. Saves dictionary of :obj:`~pandas.DataFrame`s to HDF5 container.

    Parameters
    ----------
    path : str
        Path to output file.
    data : dict of :obj:`~pandas.DataFrame`
        Dictionary of :obj:`~pandas.DataFrame`s to be saved. Dictionary keys
        are saved as "groups" (e.g., MS level) and data frame columns are saved
        as "datasets" in the HDF5 container.
    dtype : dict
        Specifies what data type to save each column, provided as column:dtype
        pairs. Defaults to 32-bit float if unspecified.
    compression_level : int
        A value from 0-9 signaling the number of compression operations to
        apply. Higher values result in greater compression at the expense of
        computational overhead.

    '''

    with h5py.File(path, 'w') as f:
        for level in data.keys():
            f.create_group(level)
            for c in data[level].columns:
                if c not in dtype.keys():
                    dtype[c] = float

                f[level].create_dataset(c, data=data[level][c].values,
                                        dtype=dtype[c],
                                        compression="gzip",
                                        compression_opts=compression_level)


def _load_hdf(path, level='ms1'):
    '''
    Deprecated version. Loads data frame from HDF5 container.

    Parameters
    ----------
    path : str
        Path to input HDF5 file.
    level : str
        Access this level (group) of the HDF5 container. E.g., "ms1" or "ms2"
        for MS levels 1 or 2, respectively.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Feature coordinates and intensities for the specified level.

    '''

    with h5py.File(path, 'r') as f:
        g = f[level]
        return pd.DataFrame({k: g[k] for k in list(g.keys())})


def save_mgf(path, features, charge='1+'):
    '''
    Saves data to MGF format.

    Parameters
    ----------
    path : str
        Path to output file.
    features : :obj:`~pandas.DataFrame`
        Precursor m/z and intensities paired to MS2 spectra.

    '''

    template = ('BEGIN IONS\n'
                'PEPMASS={} {}\n'
                'CHARGE={}\n'
                'TITLE=Spectrum {}\n'
                '{}\n'
                'END IONS\n')

    with open(path, 'w') as f:
        for i, row in features.iterrows():
            precursor_mz = row['mz']
            precursor_int = row['intensity']
            ms2 = row['ms2']

            # check for ms2 spectra
            if ms2 is not np.nan:
                f.write(template.format(precursor_mz,
                                        precursor_int,
                                        charge,
                                        i,
                                        ms2.replace(';', '\n')))
