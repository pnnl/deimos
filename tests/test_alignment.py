import deimos
import pytest

from tests import localfile


@pytest.fixture()
def ms1_peaks():
    ms1 = deimos.load(localfile('resources/isotope_example_data.h5'),
                      key='ms1')
    peaks = deimos.peakpick.persistent_homology(ms1,
                                                dims=['mz',
                                                      'drift_time',
                                                      'retention_time'],
                                                radius=[2, 10, 0])
    return deimos.threshold(peaks, threshold=1E3)


@pytest.fixture()
def ms2_peaks():
    ms2 = deimos.load(localfile('resources/isotope_example_data.h5'),
                      key='ms2')
    peaks = deimos.peakpick.persistent_homology(ms2,
                                                dims=['mz',
                                                      'drift_time',
                                                      'retention_time'],
                                                radius=[2, 10, 0])
    return deimos.threshold(peaks, threshold=1E3)


def test_match(ms1_peaks):
    a, b = deimos.alignment.match(ms1_peaks, ms1_peaks,
                                  dims=['mz',
                                        'drift_time',
                                        'retention_time'],
                                  tol=[5E-6, 0.015, 0.3],
                                  relative=[True, True, False])

    # identity match
    assert a.equals(b)

    # one-to-one constraint
    assert len(a.index) <= len(ms1_peaks.index)
    assert len(b.index) <= len(ms1_peaks.index)

    # tolerance check
    assert all(abs(a['mz'] - b['mz']) / a['mz'] <= 5E-6)
    assert all(abs(a['drift_time'] - b['drift_time']) /
               a['drift_time'] <= 0.015)
    assert all(abs(a['retention_time'] - b['retention_time']) <= 0.3)

    # one-to-one check
    assert len(a.index) == len(a.drop_duplicates(subset=['mz',
                                                         'drift_time',
                                                         'retention_time']).index)
    assert len(b.index) == len(b.drop_duplicates(subset=['mz',
                                                         'drift_time',
                                                         'retention_time']).index)


@pytest.mark.parametrize('a_none,b_none',
                         [(True, True),
                          (True, False),
                          (False, True)])
def test_match_pass_none(ms1_peaks, a_none, b_none):
    if a_none:
        a = None
    else:
        a = ms1_peaks

    if b_none:
        b = None
    else:
        b = ms1_peaks

    a_, b_ = deimos.alignment.match(a, b,
                                    dims=['mz',
                                          'drift_time',
                                          'retention_time'],
                                    tol=[5E-6, 0.015, 0.3],
                                    relative=[True, True, False])

    assert a_ is None
    assert b_ is None


def test_match_return_none(ms1_peaks):
    min_mz = ms1_peaks['mz'].min()
    mid_mz = ms1_peaks['mz'].median()
    max_mz = ms1_peaks['mz'].max()

    a = deimos.slice(ms1_peaks, by='mz', low=min_mz, high=mid_mz - 1)
    b = deimos.slice(ms1_peaks, by='mz', low=mid_mz + 1, high=max_mz)

    a_, b_ = deimos.alignment.match(a, b,
                                    dims=['mz',
                                          'drift_time',
                                          'retention_time'],
                                    tol=[5E-6, 0.015, 0.3],
                                    relative=[True, True, False])

    assert a_ is None
    assert b_ is None


def test_tolerance(ms1_peaks):
    a, b = deimos.alignment.tolerance(ms1_peaks, ms1_peaks,
                                      dims=['mz',
                                            'drift_time',
                                            'retention_time'],
                                      tol=[5E-6, 0.015, 0.3],
                                      relative=[True, True, False])

    assert all(abs(a['mz'] - b['mz']) / a['mz'] <= 5E-6)
    assert all(abs(a['drift_time'] - b['drift_time']) /
               a['drift_time'] <= 0.015)
    assert all(abs(a['retention_time'] - b['retention_time']) <= 0.3)


@pytest.mark.parametrize('a_none,b_none',
                         [(True, True),
                          (True, False),
                          (False, True)])
def test_tolerance_pass_none(ms1_peaks, a_none, b_none):
    if a_none:
        a = None
    else:
        a = ms1_peaks

    if b_none:
        b = None
    else:
        b = ms1_peaks

    a_, b_ = deimos.alignment.tolerance(a, b,
                                        dims=['mz',
                                              'drift_time',
                                              'retention_time'],
                                        tol=[5E-6, 0.015, 0.3],
                                        relative=[True, True, False])

    assert a_ is None
    assert b_ is None


def test_tolerance_return_none(ms1_peaks):
    min_mz = ms1_peaks['mz'].min()
    mid_mz = ms1_peaks['mz'].median()
    max_mz = ms1_peaks['mz'].max()

    a = deimos.slice(ms1_peaks, by='mz', low=min_mz, high=mid_mz - 1)
    b = deimos.slice(ms1_peaks, by='mz', low=mid_mz + 1, high=max_mz)

    a_, b_ = deimos.alignment.tolerance(a, b,
                                        dims=['mz',
                                              'drift_time',
                                              'retention_time'],
                                        tol=[5E-6, 0.015, 0.3],
                                        relative=[True, True, False])

    assert a_ is None
    assert b_ is None


def test_fit_spline():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError


def test_agglomerative_clustering():
    with pytest.raises(NotImplementedError):
        raise NotImplementedError
