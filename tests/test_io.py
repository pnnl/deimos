import os

import deimos
import pandas as pd
import pytest

from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load(localfile('resources/example_data.h5'),
                       key='ms1')


@pytest.fixture()
def ms2():
    return deimos.load(localfile('resources/example_data.h5'),
                       key='ms2')


def test_load_mzml():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


def test_save(ms1, ms2):
    deimos.save(localfile('resources/test_save.h5'),
                ms1, key='ms1')
    deimos.save(localfile('resources/test_save.h5'),
                ms2, key='ms2')

    os.remove(localfile('resources/test_save.h5'))


def test_load(ms1, ms2):
    assert type(ms1) is pd.DataFrame
    assert type(ms2) is pd.DataFrame

    for col in ['mz', 'drift_time', 'retention_time']:
        assert col in ms1.columns
        assert col in ms2.columns

    assert len(ms1.index) == 1958605
    assert len(ms2.index) == 1991829


def test_load_hdf_multi():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


def test_save_mgf():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


def test_build_factors():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


def test_build_index():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


def test_get_accessions():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
