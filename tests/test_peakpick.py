import deimos
import pandas as pd
import pytest

from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load(localfile('resources/example_data.h5'),
                       key='ms1')


@pytest.mark.parametrize('dims,bins,scale_by,ref_res,scale',
                         [(['mz', 'drift_time', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           None,
                           None,
                           None),
                          (['mz', 'drift_time', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           'mz',
                           0.002445221,
                           'drift_time'),
                          (['mz', 'drift_time'],
                           [2.7, 0.94],
                           'mz',
                           0.002445221,
                           'drift_time'),
                          ('retention_time',
                           3.64,
                           None,
                           None,
                           None)])
def test_local_maxima(ms1, dims, bins, scale_by, ref_res, scale):
    # make smaller for testing
    subset = deimos.slice(ms1, by='mz', low=200, high=300)

    peaks = deimos.peakpick.local_maxima(subset, dims=dims,
                                         bins=bins, scale_by=scale_by,
                                         ref_res=ref_res, scale=scale)

    assert type(peaks) is pd.DataFrame

    for d in deimos.utils.safelist(dims) + ['intensity']:
        assert d in peaks.columns


@pytest.mark.parametrize('dims,bins,scale_by,ref_res,scale',
                         [(['mz', 'drift_time', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           'mz',
                           0.002445221,
                           None),
                          (['mz', 'drift_time', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           'mz',
                           None,
                           'drift_time'),
                          (['mz', 'drift_time', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           None,
                           0.002445221,
                           'drift_time'),
                          (['mz', 'drift_time', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           'mz',
                           None,
                           None),
                          (['mz', 'drift_time', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           None,
                           0.002445221,
                           None),
                          (['mz', 'drift_time', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           None,
                           None,
                           'drift_time'),
                          (['mz', 'retention_time'],
                           [2.7, 0.94, 3.64],
                           'mz',
                           0.002445221,
                           None),
                          (['mz', 'drift_time', 'retention_time'],
                           2.7,
                           'mz',
                           0.002445221,
                           None)])
def test_local_maxima_fail(ms1, dims, bins, scale_by, ref_res, scale):
    with pytest.raises(ValueError):
        deimos.peakpick.local_maxima(ms1, dims=dims,
                                     bins=bins, scale_by=scale_by,
                                     ref_res=ref_res, scale=scale)
        

def test_persistent_homology():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
