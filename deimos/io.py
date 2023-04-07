import os
import warnings
from collections import OrderedDict, defaultdict

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import pymzml

import deimos


def load(path, key='ms1', columns=None, chunksize=1E7, meta=None, accession={}, dtype=np.float32):
    '''
    Loads data from HDF5 or mzML file.

    Parameters
    ----------
    path : str or list of str
        Path to input file (or files if HDF5).
    key : str
        Access this level (group) of the HDF5 container. E.g., "ms1" or "ms2"
        for MS levels 1 or 2, respectively. HDF5 format only.
    columns : list
        A list of columns names to return. HDF5 format only.
    chunksize : int
        Dask partition chunksize. HDF5 format only. Unused when loading single
        file.
    meta : dict
        Dictionary of meta data per path. HDF5 format only. Unused when loading
        single file.
    accession : dict
        Key-value pairs signaling which features to parse for in the mzML file.
        mzML format only. See :func:`~deimos.io.get_accessions` to obtain
        available values.
    dtype : data type
        Data type to encode values. mzML format only.

    Returns
    -------
    :obj:`~pandas.DataFrame` or :obj:`dict` of :obj:`~pandas.DataFrame`
        Feature coordinates and intensities for the specified level.
        Pandas is used when loading a single file, Dask for multiple files.
        Loading an mzML file returns a dictionary with keys per MS level.

    '''

    # Check number of inputs
    paths = deimos.utils.safelist(path)

    # Ensure extensions match
    exts = [os.path.splitext(x)[-1].lower() for x in paths]
    if not all(x == exts[0] for x in exts):
        raise ValueError('All inputs must have same filetype extension.')

    # Get the extension
    ext = exts[0]

    # Multi loader
    if len(paths) > 1:
        # Hdf5
        if ext in ['.h5', '.hdf']:
            return deimos.io.load_hdf_multi(paths,
                                            key=key,
                                            columns=columns,
                                            chunksize=chunksize,
                                            meta=meta)

        # Other
        raise ValueError(
            'Only HDF5 currently supported for multi-file loading.')

    # Single loader
    # Hdf5
    if ext in ['.h5', '.hdf']:
        return deimos.io.load_hdf_single(path, key=key, columns=columns)

    # Mzml
    if ext in ['.gz', '.mzml']:
        return deimos.io.load_mzml(path, accession=accession, dtype=dtype)

    # Other
    raise ValueError('Only HDF5 and mzML currently supported.')


def build_factors(data, dims='detect'):
    '''
    Determine sorted unique elements (factors) for each dimension in data.

    Parameters
    ----------
    data : :obj:`~pandas.DataFrame`
        Feature coordinates and intensities.
    dims : str or list
        Dimensions to determine factors for. Attempts to autodetect by
        default.

    Returns
    -------
    :obj:`dict` of :obj:`~numpy.array`
        Unique sorted values per dimension.

    '''

    # Autodetect
    if dims == 'detect':
        dims = deimos.utils.detect_dims(data)

    # Safely cast to list
    dims = deimos.utils.safelist(dims)

    # Construct per-dimension factor arrays
    return {dim: pd.factorize(data[dim], sort=True)[1].astype(np.float32) for dim in dims}


def build_index(data, factors):
    '''
    Construct data index from precomputed factors.

    Parameters
    ----------
    data : :obj:`~pandas.DataFrame`
        Feature coordinates and intensities.
    factors : dict
        Per-dimension arrays of unique values.

    Returns
    -------
    :obj:`dict` of :obj:`~numpy.array`
        Index per dimension.

    '''

    return {dim: np.searchsorted(factors[dim], data[dim]).astype(np.float32) for dim in factors}


def save(path, data, key='ms1', **kwargs):
    '''
    Saves :obj:`~pandas.DataFrame` to HDF5 or MGF container.

    Parameters
    ----------
    path : str
        Path to output file.
    data : :obj:`~pandas.DataFrame`
        Feature coordinates and intensities to be saved. Precursor m/z and
        intensities should be paired to MS2 spectra for MGF format.
    key : str
        Save to this level (group) of the HDF5 container. E.g., "ms1" or "ms2"
        for MS levels 1 or 2, respectively. HDF5 format only.
    kwargs
        Keyword arguments exposed by :meth:`~pandas.DataFrame.to_hdf`
        or :func:`~deimos.io.save_mgf`.

    '''

    # Path extension
    ext = os.path.splitext(path)[-1].lower()

    # Hdf5
    if ext in ['.h5', '.hdf']:
        return deimos.io.save_hdf(path, data, key=key, **kwargs)

    # Mzml
    if ext in ['.mgf']:
        return deimos.io.save_mgf(path, data, **kwargs)

    # Other
    raise ValueError('Only HDF5 and MGF currently supported.')


def get_accessions(path):
    '''
    Determines accession fields available in the mzML file.

    Parameters
    ----------
    path : str
        Path to mzML file.

    Returns
    -------
    :obj:`dict` of str
        Dictionary of accession fields.

    '''

    # Open file
    data = pymzml.run.Reader(path)

    # Iterate single spec instance
    for spec in data:
        spec._read_accessions()
        break

    # Return accessions
    return spec.accessions


def load_mzml(path, accession={}, dtype=np.float32):
    '''
    Loads in an mzML file, parsing for accession values, to yield a
    :obj:`~pandas.DataFrame`.

    Parameters
    ----------
    path : str
        Path to input mzML file.
    accession : dict
        Key-value pairs signaling which features to parse for in the mzML file.
        See :func:`~deimos.io.get_accessions` to obtain available values. Scan,
        frame, m/z, and intensity are parsed by default.
    dtype : data type
        Data type to encode values.

    Returns
    -------
    :obj:`dict` of :obj:`~pandas.DataFrame`
        Dictionary containing parsed feature coordinates and intensities, indexed
        by keys per MS level.

    '''

    # Open file
    data = pymzml.run.Reader(path)

    # Ordered dict
    accession = OrderedDict(accession)

    # Result container
    res = {}

    # Row count container
    counter = {}

    # Column name container
    cols = {}

    # First pass: get nrows
    N = defaultdict(lambda: 0)
    for i, spec in enumerate(data):
        # Get ms level
        level = 'ms{}'.format(spec.ms_level)

        # Number of rows
        N[level] += spec.mz.shape[0]

    # Second pass: parse
    for i, spec in enumerate(data):
        # Number of rows
        n = spec.mz.shape[0]

        # No measurements
        if n == 0:
            continue

        # Dimension check
        if len(spec.mz) != len(spec.i):
            warnings.warn("m/z and intensity array dimension mismatch")
            continue

        # Scan/frame info
        id_dict = spec.id_dict

        # Check for precursor
        precursor_info = {}
        has_precursor = False
        if spec.selected_precursors:
            has_precursor = True
            precursor_info = {
                'precursor_mz': spec.selected_precursors[0].get('mz', None)}

        # Get ms level
        level = 'ms{}'.format(spec.ms_level)

        # Columns
        cols[level] = list(id_dict.keys()) \
            + list(accession.keys()) \
            + ['mz', 'intensity'] \
            + list(precursor_info.keys())
        m = len(cols[level])

        # Subarray init
        arr = np.empty((n, m), dtype=dtype)
        inx = 0

        # Populate scan/frame info
        for k, v in id_dict.items():
            arr[:, inx] = v
            inx += 1

        # Populate accession fields
        for k, v in accession.items():
            arr[:, inx] = spec.get(v)
            inx += 1

        # Populate m/z
        arr[:, inx] = spec.mz
        inx += 1

        # Populate intensity
        arr[:, inx] = spec.i
        inx += 1

        # Populate precursor information
        if has_precursor:
            for k, v in precursor_info.items():
                arr[:, inx] = v
                inx += 1

        # Initialize output container
        if level not in res:
            res[level] = np.empty((N[level], m), dtype=dtype)
            counter[level] = 0

        # Insert subarray
        res[level][counter[level]:counter[level] + n, :] = arr
        counter[level] += n

    # Construct data frames
    for level in res.keys():
        res[level] = pd.DataFrame(res[level], columns=cols[level])

    return res


def save_hdf(path, data, key='ms1', complevel=5, **kwargs):
    '''
    Saves :obj:`~pandas.DataFrame` to HDF5 container.

    Parameters
    ----------
    path : str
        Path to output file.
    data : :obj:`~pandas.DataFrame`
        Feature coordinates and intensities to be saved.
    key : str
        Save to this level (group) of the HDF5 container. E.g., "ms1" or "ms2"
        for MS levels 1 or 2, respectively.
    kwargs
        Keyword arguments exposed by :meth:`~pandas.DataFrame.to_hdf`.

    '''

    data.to_hdf(path, key, format='table', complib='blosc',
                complevel=complevel, **kwargs)


def load_hdf(path, key='ms1', columns=None, chunksize=1E7, meta=None):
    '''
    Loads data frame from HDF5 container(s). 

    Parameters
    ----------
    path : str or list of str
        Path to input HDF5 file or files.
    key : str
        Access this level (group) of the HDF5 container. E.g., "ms1" or "ms2"
        for MS levels 1 or 2, respectively.
    columns : list
        A list of columns names to return.
    chunksize : int
        Dask partition chunksize. Unused when loading single file.
    meta : dict
        Dictionary of meta data per path. Unused when loading single file.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Feature coordinates and intensities for the specified level.
        Pandas is used when loading a single file, Dask for multiple files.

    '''

    # Check number of inputs
    paths = deimos.utils.safelist(path)

    # Ensure extensions match
    exts = [os.path.splitext(x)[-1].lower() for x in paths]
    if not all(x == exts[0] for x in exts):
        raise ValueError('All inputs must have same filetype extension.')

    # Multi loader
    if len(paths) > 1:
        return deimos.io.load_hdf_multi(paths,
                                        key=key,
                                        columns=columns,
                                        chunksize=chunksize,
                                        meta=meta)

    # Single loader
    return deimos.io.load_hdf_single(paths,
                                     key=key,
                                     columns=columns)


def load_hdf_single(path, key='ms1', columns=None):
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
    Loads data frame from HDF5 containers using Dask. Appends column to indicate
    source filenames.

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

    # Load as dask
    df = [dd.read_hdf(x, key=key, chunksize=int(
        chunksize), columns=columns) for x in paths]

    # Label each sample
    for i in range(len(paths)):
        df[i]['sample_idx'] = i  # force unique label in toy case
        df[i]['sample_id'] = os.path.splitext(os.path.basename(paths[i]))[0]

        if meta is not None:
            for k, v in meta.items():
                df[i][k] = v[i]

    # Concatenate results
    return dd.concat(df, axis=0)


def _save_hdf(path, data, dtype={}, compression_level=5):
    '''
    Deprecated version. Saves dictionary of :obj:`~pandas.DataFrame` to HDF5 container.

    Parameters
    ----------
    path : str
        Path to output file.
    data : dict of :obj:`~pandas.DataFrame`
        Dictionary of feature coordinates and intensities to be saved. Dictionary
        keys are saved as "groups" (e.g., MS level) and data frame columns are saved
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

            # Check for ms2 spectra
            if ms2 is not np.nan:
                f.write(template.format(precursor_mz,
                                        precursor_int,
                                        charge,
                                        i,
                                        ms2.replace(';', '\n')))
