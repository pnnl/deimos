import pytest
import deimos
import numpy as np
from pandas.core.series import Series
from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load_hdf(localfile('resources/example_data.h5'),
                           level='ms1')


@pytest.mark.parametrize('x,expected',
                         [('a', ['a']),
                          (['a', 'b', 'c'], ['a', 'b', 'c']),
                          (1, [1]),
                          ([1, 2, 3], [1, 2, 3])])
def test_safelist(x, expected):
    # list
    assert deimos.utils.safelist(x) == expected

    # array
    assert np.all(deimos.utils.safelist(np.array(x)) == np.array(expected))

    # series
    assert (deimos.utils.safelist(Series(x)) == Series(expected)).all()


@pytest.mark.parametrize('lists',
                         [([['a', 'b', 'c'], ['a', 'b', 'c']]),
                          ([np.arange(5), np.arange(5)])])
def test_check_length(lists):
    deimos.utils.check_length(lists)


@pytest.mark.parametrize('lists',
                         [([['a', 'b', 'c'], ['a', 'b']]),
                          ([np.arange(5), np.arange(4)])])
def test_check_length_fail(lists):
    with pytest.raises(ValueError):
        deimos.utils.check_length(lists)


@pytest.mark.parametrize('contains',
                         [(['mz', 'drift_time', 'retention_time'])])
def test_detect_features(ms1, contains):
    features = deimos.utils.detect_features(ms1)
    for f in contains:
        assert f in features

    assert 'intensity' not in features
