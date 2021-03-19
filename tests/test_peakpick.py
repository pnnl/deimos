import pytest
import deimos
import pandas as pd
from tests import localfile


@pytest.fixture()
def ms1():
    return deimos.load_hdf(localfile('resources/example_data.h5'),
                           level='ms1')


@pytest.mark.parametrize('features,bins,scale_by,ref_res,scale',
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
def test_non_max_suppression(ms1, features, bins, scale_by, ref_res, scale):
    # make smaller for testing
    subset = deimos.slice(ms1, by='mz', low=200, high=300)

    peaks = deimos.peakpick.non_max_suppression(subset, features=features,
                                                bins=bins, scale_by=scale_by,
                                                ref_res=ref_res, scale=scale)

    assert type(peaks) is pd.DataFrame

    for f in deimos.utils.safelist(features) + ['intensity']:
        assert f in peaks.columns


@pytest.mark.parametrize('features,bins,scale_by,ref_res,scale',
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
def test_non_max_suppression_fail(ms1, features, bins, scale_by, ref_res, scale):
    with pytest.raises(ValueError):
        deimos.peakpick.non_max_suppression(ms1, features=features,
                                            bins=bins, scale_by=scale_by,
                                            ref_res=ref_res, scale=scale)
