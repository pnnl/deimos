import deimos
import numpy as np
import pytest
from scipy import signal


@pytest.fixture()
def g3d():
    return 10E4 * np.einsum('i,j,k',
                            signal.windows.gaussian(70, 3),
                            signal.windows.gaussian(70, 3),
                            signal.windows.gaussian(70, 3))


@pytest.fixture()
def g1d():
    return 10E4 * signal.windows.gaussian(70, 3)


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [0.74833148, 1.16619038, 1.41421356, 1.16619038, 0.74833148])])
def test_std(array, size, expected):
    result = deimos.filters.std(array, size)
    assert (np.abs(result - expected) <= 1E-3).all()


def test_std_pdf():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [2, 3, 4, 4, 4])])
def test_maximum(array, size, expected):
    result = deimos.filters.maximum(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [0, 0, 0, 1, 2])])
def test_minimum(array, size, expected):
    result = deimos.filters.minimum(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [3., 6., 10., 10., 9.])])
def test_sum(array, size, expected):
    result = deimos.filters.sum(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [0.6, 1.2, 2., 2., 1.8])])
def test_mean(array, size, expected):
    result = deimos.filters.mean(array, size)
    assert (result == expected).all()


def test_mean_pdf():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [0.43195254, 0.52385218, 0.5883779, 0.61199644, 0.58946036])])
def test_matched_gaussian(array, size, expected):
    result = deimos.filters.matched_gaussian(array, size)
    assert (np.abs(result - expected) <= 1E-3).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [3, 4, 5, 4, 3])])
def test_count(array, size, expected):
    result = deimos.filters.count(array, size)
    assert (result == expected).all()


def test_skew_pdf():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
    

def test_kurtosis_pdf():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
    

def test_sparse_upper_star():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


def test_sparse_mean_filter():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
    

def test_sparse_weighted_mean_filter():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
    

def test_sparse_median_filter():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
    

def test_smooth():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
