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


def test_data2grid():
    raise NotImplementedError


def test_grid2df():
    raise NotImplementedError
