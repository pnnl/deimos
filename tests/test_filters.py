import pytest
import deimos
from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load_hdf(localfile('resources/example_data.h5'),
                           level='ms1')


@pytest.fixture()
def ms2():
    return deimos.load_hdf(localfile('resources/example_data.h5'),
                           level='ms2')


def test_stdev():
    raise NotImplementedError


def test_maximum():
    raise NotImplementedError


def test_minimum():
    raise NotImplementedError


def test_sum():
    raise NotImplementedError


def test_mean():
    raise NotImplementedError


def test_matched_gaussian():
    raise NotImplementedError


def test_signal_to_noise_ratio():
    raise NotImplementedError


def test_count():
    raise NotImplementedError


def test_kurtosis():
    raise NotImplementedError
