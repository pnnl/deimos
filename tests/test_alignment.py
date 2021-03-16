import pytest
import deimos
from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load_hdf(localfile('resources/example_data.h5')
                           level='ms1')


@pytest.fixture()
def ms2():
    return deimos.load_hdf(localfile('resources/example_data.h5')
                           level='ms2')


def test_match():
    raise NotImplementedError


def test_tolerance():
    raise NotImplementedError


def test_fit_spline():
    raise NotImplementedError


def test_join():
    raise NotImplementedError
