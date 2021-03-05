import pymzml
import h5py
from collections import OrderedDict, defaultdict
import warnings
import pandas as pd
import numpy as np


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

    # result container
    res = defaultdict(list)

    # precursor rename
    pdict = {'mz': 'precursor_mz',
             'i': 'precursor_intensity',
             'charge': 'precursor_charge'}

    # parse
    for spec in data:
        # dimension check
        if len(spec.mz) != len(spec.i):
            warnings.warn("m/z and intensity array dimension mismatch")
            continue

        # init
        cols = ['mz', 'intensity']

        # check for precursor
        if spec.selected_precursors:
            arr = np.empty((spec.mz.shape[0],
                            len(accession)
                            + len(spec.selected_precursors[0]) + 2),
                           dtype=np.float32)

        # no precursor
        else:
            arr = np.empty((spec.mz.shape[0], len(accession) + 2),
                           dtype=np.float32)

        # fill
        arr[:, 0] = spec.mz
        arr[:, 1] = spec.i

        # populate accession fields
        for i, (k, v) in enumerate(accession.items()):
            cols.append(k)
            arr[:, 2 + i] = spec.get(v)

        # populate precursor information
        if spec.selected_precursors:
            for i, (k, v) in enumerate(spec.selected_precursors[0].items()):
                cols.append(pdict[k])
                arr[:, 2 + len(accession) + i] = v

        # append dataframe
        res['ms{}'.format(spec.ms_level)].append(pd.DataFrame(arr,
                                                              columns=cols))

    # concatenate dataframes
    for level in res.keys():
        res[level] = pd.concat(res[level], axis=0, ignore_index=True)

    return res


def save_hdf(path, data, dtype={}, compression_level=5):
    '''
    Saves dictionary of :obj:`~pandas.DataFrame`s to HDF5 container.

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
                    dtype[c] = np.float32

                f[level].create_dataset(c, data=data[level][c].values,
                                        dtype=dtype[c],
                                        compression="gzip",
                                        compression_opts=compression_level)


def load_hdf(path, level='ms1'):
    '''
    Loads data frame from HDF5 container.

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


def save_mgf(path, data, charge='1+'):
    '''
    Saves data to MGF format.

    Parameters
    ----------
    path : str
        Path to output file.
    data : :obj:`~pandas.DataFrame`
        Precursor m/z and intensities paired to MS2 spectra.

    '''

    template = ('BEGIN IONS\n'
                'PEPMASS={} {}\n'
                'CHARGE={}\n'
                'TITLE=Spectrum {}\n'
                '{}\n'
                'END IONS\n')

    with open(path, 'w') as f:
        for i, row in data.iterrows():
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
