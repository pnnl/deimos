import os
import warnings
from collections import OrderedDict, defaultdict

import dask.dataframe as dd
import h5py
import hdf5plugin
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
        mzML and MZA format only. See :func:`~deimos.io.get_accessions` to obtain
        available values.
    dtype : data type
        Data type to encode values. mzML format only.

    Returns
    -------
    :obj:`~pandas.DataFrame` or :obj:`dict` of :obj:`~pandas.DataFrame`
        Feature coordinates and intensities for the specified level.
        Pandas is used when loading a single file, Dask for multiple files.
        Loading an mzML or MZA file returns a dictionary with keys per MS level.

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

    # MZA
    if ext in ['.mza']:
        return deimos.io.load_mza(path, accession=accession)

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

    # MGF
    if ext in ['.mgf']:
        return deimos.io.save_mgf(path, data, **kwargs)

    # MGF
    if ext in ['.msp']:
        return deimos.io.save_msp(path, data, **kwargs)

    # CSV
    if ext in ['.csv']:
        return data.to_csv(path, sep=',', index=False, **kwargs)

    # TSV
    if ext in ['.tsv', '.txt']:
        return data.to_csv(path, sep='\t', index=False, **kwargs)

    # Other
    raise ValueError(
        'Only HDF5, MGF, MSP, TSV, and CSV formats currently supported.')


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


def load_mza(path, accession={}):
    '''
    Loads in an MZA file, parsing for accession values, to yield a
    :obj:`~pandas.DataFrame`.

    Parameters
    ----------
    path : str
        Path to input mzML file.
    accession : dict
        Key-value pairs signaling which features to parse for in the mzML file.
        See :func:`~deimos.io.get_accessions` to obtain available values. Scan,
        frame, m/z, and intensity are parsed by default.

    Returns
    -------
    :obj:`dict` of :obj:`~pandas.DataFrame`
        Dictionary containing parsed feature coordinates and intensities, indexed
        by keys per MS level.

    '''

    # Invert dict
    accession_inv = {v: k for k, v in accession.items()}

    # Ensure complete metadata
    metadata_fields = list(accession.values())
    metadata_fields += ['Scan', 'MSLevel']
    metadata_fields = set(metadata_fields)

    # Intitialize result container
    container = defaultdict(list)

    # Open mza file
    with h5py.File(path, 'r') as mza:

        # Access metadata
        metadata = pd.DataFrame(
            {x: mza["Metadata"][x][()] for x in metadata_fields})

        # Enumerate scans
        for i, row in metadata.iterrows():

            scan = int(row['Scan'])
            ms_level = int(row['MSLevel'])

            # m/z values
            mzs = mza['Arrays_mz/{}'.format(scan)][:]

            # Intensity values
            ints = mza['Arrays_intensity/{}'.format(scan)][:]

            # Initialize scan dataframe
            df = pd.DataFrame({'mz': mzs, 'intensity': ints})

            # Populate other metadata fields
            for field in metadata_fields:
                if field not in ['Scan', 'MSLevel']:
                    df[field] = row[field]

            # Append to appropriate MS level
            container['ms{}'.format(ms_level)].append(df)

    # Concatenate scan dataframes
    container = {k: pd.concat(v, ignore_index=True)
                 for k, v in container.items()}

    # Drop all-zero columns
    container = {k: v.loc[:, (v != 0).any(axis=0)]
                 for k, v in container.items()}

    # Rename columns
    container = {k: v.rename(columns=accession_inv)
                 for k, v in container.items()}

    return container


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


def save_mgf(path, features,
             groupby='index_ms1',
             precursor_mz='mz_ms1',
             fragment_mz='mz_ms2',
             fragment_intensity='intensity_ms2',
             precursor_metadata=None,
             sample_metadata=None):
    '''
    Saves data to MGF format.

    Parameters
    ----------
    path : str
        Path to output file.
    features : :obj:`~pandas.DataFrame`
        Precursor m/z and intensities paired to MS2 spectra.
    groupby : str or list of str
        Column(s) to group fragments by.
    precursor_mz : str
        Column containing precursor m/z values.
    fragment_mz : str
        Column containing fragment m/z values.
    fragment_intensity : str
        Column containing fragment intensity values.
    precursor_metadata : dict
        Precursor metadata key:value pairs of {MGF entry name}:{column name}.
    sample_metadata : dict
        Sample metadata key:value pairs of {MGF entry name}:{value}. 

    '''

    # Initialize default fields
    metadata = ['TITLE', 'PEPMASS', 'PEPINTENSITY', 'CHARGE',
                'PRECURSORTYPE', 'INSTRUMENTTYPE',
                'INSTRUMENT', 'IONMODE', 'COLLISIONENERGY',
                'SMILES', 'INCHI', 'INCHIKEY', 'FORMULA',
                'RETENTIONTIME', 'DRIFTTIME', 'CCS']
    metadata = OrderedDict([(x, None) for x in metadata])

    # Initialize precursor metadata dict
    if precursor_metadata is None:
        precursor_metadata = {}

    # Initialize sample metadata dict
    if sample_metadata is None:
        sample_metadata = {}

    # Add required field
    precursor_metadata['PEPMASS'] = precursor_mz

    # Update defaults
    metadata.update(precursor_metadata)
    metadata.update(sample_metadata)

    # Build template
    template = 'BEGIN IONS\n'
    columns = []
    for k in metadata:
        # Key from precursor metadata
        if k in precursor_metadata:
            template += k + '={}\n'
            columns.append(metadata[k])

        # Key from sample metadata
        elif k in sample_metadata:
            template += k + '={}\n'.format(metadata[k])

        # Key was not specified
        else:
            pass

    # Append MS2 template information
    template += ('{}\n'
                 'END IONS\n\n')

    # Open file object
    with open(path, 'w') as f:
        # Enumerate groups
        for name, grp in features.groupby(by=groupby):

            # Format MS2 string
            ms2_str = '\n'.join('{}\t{}'.format(a, b) for a, b in zip(grp[fragment_mz].values,
                                                                      grp[fragment_intensity].values))

            # Precursor metadata values
            values = list(grp[columns].values[0])

            # Add MS2 info
            values += [ms2_str]

            # Write to template
            f.write(template.format(*values))


def save_msp(path, features,
             groupby='index_ms1',
             precursor_mz='mz_ms1',
             fragment_mz='mz_ms2',
             fragment_intensity='intensity_ms2',
             precursor_metadata=None,
             sample_metadata=None):
    '''
    Saves data to MSP format.

    Parameters
    ----------
    path : str
        Path to output file.
    features : :obj:`~pandas.DataFrame`
        Precursor m/z and intensities paired to MS2 spectra.
    groupby : str or list of str
        Column(s) to group fragments by.
    precursor_mz : str
        Column containing precursor m/z values.
    fragment_mz : str
        Column containing fragment m/z values.
    fragment_intensity : str
        Column containing fragment intensity values.
    precursor_metadata : dict
        Precursor metadata key:value pairs of {MSP entry name}:{column name}.
    sample_metadata : dict
        Sample metadata key:value pairs of {MSP entry name}:{value}. 

    '''

    # Initialize default fields
    metadata = ['NAME', 'PRECURSORMZ', 'PRECURSORINTENSITY',
                'PRECURSORTYPE', 'INSTRUMENTTYPE',
                'INSTRUMENT', 'IONMODE', 'COLLISIONENERGY',
                'SMILES', 'INCHI', 'INCHIKEY', 'FORMULA',
                'RETENTIONTIME', 'DRIFTTIME', 'CCS']
    metadata = OrderedDict([(x, None) for x in metadata])

    # Initialize precursor metadata dict
    if precursor_metadata is None:
        precursor_metadata = {}

    # Initialize sample metadata dict
    if sample_metadata is None:
        sample_metadata = {}

    # Add required field
    precursor_metadata['PRECURSORMZ'] = precursor_mz

    # Update defaults
    metadata.update(precursor_metadata)
    metadata.update(sample_metadata)

    # Build template
    template = ''
    columns = []
    for k in metadata:
        # Key from precursor metadata
        if k in precursor_metadata:
            template += k + ': {}\n'
            columns.append(metadata[k])

        # Key from sample metadata
        elif k in sample_metadata:
            template += k + ': {}\n'.format(metadata[k])

        # Key was not specified
        else:
            pass

    # Append MS2 template information
    template += ('Num Peaks: {}\n'
                 '{}\n\n')

    # Open file object
    with open(path, 'w') as f:
        # Enumerate groups
        for name, grp in features.groupby(by=groupby):
            # Number of MS2
            n = len(grp.index)

            # Format MS2 string
            ms2_str = '\n'.join('{}\t{}'.format(a, b) for a, b in zip(grp[fragment_mz].values,
                                                                      grp[fragment_intensity].values))

            # Precursor metadata values
            values = list(grp[columns].values[0])

            # Add MS2 info
            values += [n, ms2_str]

            # Write to template
            f.write(template.format(*values))
