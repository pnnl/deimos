import deimos
import pytest

from tests import localfile


@pytest.fixture()
def ms1_peaks():
    ms1 = deimos.load(localfile('resources/isotope_example_data.h5'),
                      key='ms1')
    return deimos.peakpick.persistent_homology(ms1,
                                               dims=['mz',
                                                     'drift_time',
                                                     'retention_time'],
                                        radius=[2, 10, 0])


# need to test more configurations
def test_detect(ms1_peaks):
    isotopes = deimos.isotopes.detect(ms1_peaks,
                                      dims=['mz',
                                            'drift_time',
                                            'retention_time'],
                                      tol=[0.1, 0.2, 0.3],
                                      delta=1.003355,
                                      max_isotopes=5,
                                      max_charge=1,
                                      max_error=50E-6)

    # grab the most intense isotopic pattern
    isotopes = isotopes.sort_values(by='intensity', ascending=False)
    isotopes = isotopes.iloc[0, :]

    assert abs(isotopes['mz'] - 387.024353) <= 1E-3
    assert isotopes['n'] == 4
    assert all([x <= 50E-6 for x in isotopes['error']])
    assert isotopes['dx'] == [1.003355, 2.00671, 3.010065, 4.01342]
