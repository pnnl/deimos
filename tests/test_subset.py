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


def test_threshold():
    raise NotImplementedError


def test_collapse():
    raise NotImplementedError


def test_locate():
    raise NotImplementedError


def test_slice():
    raise NotImplementedError


class TestPartitions:

    def test_init(self):
        raise NotImplementedError

    def test__compute_splits(self):
        raise NotImplementedError

    def test_iter(self):
        raise NotImplementedError

    def test_map(self):
        raise NotImplementedError

    def test_zipmap(self):
        raise NotImplementedError


def test_partition():
    raise NotImplementedError
