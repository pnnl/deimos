import deimos
import numpy as np
import pytest
from scipy import signal


@pytest.fixture()
def gaussian_3d():
    return 10E4 * np.einsum('i,j,k',
                            signal.windows.gaussian(70, 3),
                            signal.windows.gaussian(70, 3),
                            signal.windows.gaussian(70, 3))


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [0.8, 1.16619038, 1.41421356, 1.41421356, 1.6])])
def test_stdev(array, size, expected):
    result = deimos.filters.stdev(array, size)
    assert (np.abs(result - expected) <= 1E-3).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [2, 3, 4, 4, 4])])
def test_maximum(array, size, expected):
    result = deimos.filters.maximum(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [0, 0, 0, 0, 0])])
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


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [0.43195254, 0.52385218, 0.5883779 , 0.61199644, 0.58946036])])
def test_matched_gaussian(array, size, expected):
    result = deimos.filters.matched_gaussian(array, size)
    assert (np.abs(result - expected) <= 1E-3).all()


# @pytest.mark.parametrize('array,size,expected',
#                          [(np.arange(5.), 5, [0.5625, 1.05882353, 2., 2., 1.265625])])
# def test_signal_to_noise_ratio(array, size, expected):
#     result = deimos.filters.signal_to_noise_ratio(array, size)
#     assert (np.abs(result - expected) <= 1E-3).all()


@pytest.mark.parametrize('array,size,expected',
                         [(np.arange(5.), 5, [3, 4, 5, 4, 3])])
def test_count(array, size, expected):
    result = deimos.filters.count(array, size)
    assert (result == expected).all()


@pytest.mark.parametrize('size,expected',
                         [([69, 69, 69], 0)])
def test_kurtosis(gaussian_3d, size, expected):
    edges = [np.arange(gaussian_3d.shape[x]) for x in range(gaussian_3d.ndim)]
    result = deimos.filters.kurtosis(edges, gaussian_3d, size)

    for dim in result:
        # compare center of array to expectation
        # (because evaluation will be of a Gaussian at center)
        assert np.abs(np.take(dim, dim.size // 2) - expected) < 1E-3
