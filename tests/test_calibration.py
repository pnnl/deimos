import deimos
import numpy as np
import pytest


@pytest.fixture()
def ccs_cal():
    return deimos.calibration.CCSCalibration()


@pytest.fixture()
def pos():
    return {'mz': [118.086255, 322.048121, 622.028960, 922.009798, 1221.990636, 1521.971475],
            'ta': [13.72, 18.65, 25.20, 30.44, 35.36, 39.83],
            'ccs': [121.3, 153.7, 203, 243.6, 282.2, 317],
            'q': [1, 1, 1, 1, 1, 1]}


@pytest.fixture()
def neg():
    return {'mz': [301.998139, 601.978977, 1033.988109, 1333.968947, 1633.949786],
            'ta': [17.04, 22.53, 32.13, 36.15, 40.70],
            'ccs': [140, 180.8, 255.3, 284.8, 319],
            'q': [1, 1, 1, 1, 1]}


class TestCCSCalibration:

    def test_init(self, ccs_cal):
        for attr, expected in zip(['buffer_mass', 'beta', 'tfix', 'fit'],
                                  [None, None, None, dict]):
            assert hasattr(ccs_cal, attr)

            tmp = getattr(ccs_cal, attr)

            if expected is None:
                assert tmp is expected

            else:
                assert type(tmp) is expected

                for k in ['r', 'p', 'se']:
                    assert k in tmp.keys()
                    assert tmp[k] is None

    @pytest.mark.parametrize('beta,tfix',
                             [(1, 0)])
    def test__check(self, ccs_cal, beta, tfix):
        ccs_cal.beta = beta
        ccs_cal.tfix = tfix
        ccs_cal._check()

    @pytest.mark.parametrize('beta,tfix',
                             [(1, None),
                              (None, 0),
                              (None, None)])
    def test__check_fail(self, ccs_cal, beta, tfix):
        ccs_cal.beta = beta
        ccs_cal.tfix = tfix

        with pytest.raises(ValueError):
            ccs_cal._check()

    @pytest.mark.parametrize('calc,beta,tfix,beta_exp,tfix_exp',
                             [(False, 1, 0, 1, 0),
                              (True, 1, 0, 0.12722, -0.11387),
                              (True, None, None, 0.12722, -0.11387)])
    def test_calibrate(self, ccs_cal, pos, calc, beta, tfix, beta_exp, tfix_exp):
        if calc is True:
            ccs_cal.calibrate(beta=beta, tfix=tfix, **pos)
            for k in ['r', 'p', 'se']:
                assert ccs_cal.fit[k] is not None
        else:
            ccs_cal.calibrate(beta=beta, tfix=tfix)

        assert abs(ccs_cal.beta - beta_exp) <= 1E-3
        assert abs(ccs_cal.tfix - tfix_exp) <= 1E-3

    def test_arrival2ccs(self, ccs_cal, pos, neg):
        for data in [pos, neg]:
            ccs_cal.calibrate(**data)
            ccs = ccs_cal.arrival2ccs(data['mz'], data['ta'], q=data['q'])

            error = np.abs(ccs - data['ccs']) / data['ccs']

            assert (error <= 0.005).all()

    def test_ccs2arrival(self, ccs_cal, pos, neg):
        for data in [pos, neg]:
            ccs_cal.calibrate(**data)
            ta = ccs_cal.ccs2arrival(data['mz'], data['ccs'], q=data['q'])

            error = np.abs(ta - data['ta']) / data['ta']

            assert (error <= 0.005).all()


@pytest.mark.parametrize('calc,beta,tfix,beta_exp,tfix_exp',
                         [(False, 1, 0, 1, 0),
                          (True, 1, 0, 0.12722, -0.11387),
                          (True, None, None, 0.12722, -0.11387)])
def test_calibrate_ccs(pos, calc, beta, tfix, beta_exp, tfix_exp):
    if calc is True:
        ccs_cal = deimos.calibration.calibrate_ccs(beta=beta, tfix=tfix, **pos)
        for k in ['r', 'p', 'se']:
            assert ccs_cal.fit[k] is not None
    else:
        ccs_cal = deimos.calibration.calibrate_ccs(beta=beta, tfix=tfix)

    assert type(ccs_cal) is deimos.calibration.CCSCalibration
    assert abs(ccs_cal.beta - beta_exp) <= 1E-3
    assert abs(ccs_cal.tfix - tfix_exp) <= 1E-3
