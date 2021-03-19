import pytest
import deimos
import os
import pandas as pd
from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load_hdf(localfile('resources/example_data.h5'),
                           level='ms1')


@pytest.fixture()
def ms2():
    return deimos.load_hdf(localfile('resources/example_data.h5'),
                           level='ms2')


def test_read_mzml():
    raise NotImplementedError


@pytest.mark.parametrize('dtype,compression_level',
                         [({}, 5),
                          ({'intensity': int}, 4)])
def test_save_hdf(ms1, ms2, dtype, compression_level):
    deimos.save_hdf(localfile('resources/test_save.h5'),
                    {'ms1': ms1, 'ms2': ms2},
                    compression_level=compression_level)

    os.remove(localfile('resources/test_save.h5'))


def test_load_hdf(ms1, ms2):
    assert type(ms1) is pd.DataFrame
    assert type(ms2) is pd.DataFrame

    for col in ['mz', 'drift_time', 'retention_time']:
        assert col in ms1.columns
        assert col in ms2.columns

    assert len(ms1.index) == 1958605
    assert len(ms2.index) == 1991829


def test_save_mgf():
    raise NotImplementedError
