import deimos
import numpy as np
import pytest


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5), 5, [1, 1, 1.414, 1.414, 2])])
def test_stdev(array, size, expected):
    result = deimos.filters.stdev(array, size)
    assert (np.abs(result - expected) <= 1E-3).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5), 5, [2, 3, 4, 4, 4])])
def test_maximum(array, size, expected):
    result = deimos.filters.maximum(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5), 5, [0, 0, 0, 0, 0])])
def test_minimum(array, size, expected):
    result = deimos.filters.minimum(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5), 5, [0, 5, 10, 10, 5])])
def test_sum(array, size, expected):
    result = deimos.filters.sum(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5), 5, [0.4, 0.6, 0.8, 0.8, 0.6])])
def test_mean(array, size, expected):
    result = deimos.filters.mean(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5), 5, [0, 0, 0, 0, 0])])
def test_matched_gaussian(array, size, expected):
    result = deimos.filters.matched_gaussian(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5), 5, [0, 1, 2, 2, 0.25])])
def test_signal_to_noise_ratio(array, size, expected):
    result = deimos.filters.signal_to_noise_ratio(array, size)
    assert (np.abs(result - expected) <= 1E-3).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5), 5, [3, 4, 5, 4, 3])])
def test_count(array, size, expected):
    result = deimos.filters.count(array, size)
    assert (result == expected).all()


def test_kurtosis():
    raise NotImplementedError
