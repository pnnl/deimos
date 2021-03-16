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


class TestArrivalTimeCalibration:

    def test_init(self):
        raise NotImplementedError

    def test__check(self):
        raise NotImplementedError

    def test_calibrate(self):
        raise NotImplementedError

    def test_arrival2ccs(self):
        raise NotImplementedError

    def test_ccs2arrival(self):
        raise NotImplementedError
