import deimos
import numpy as np
import pytest


@pytest.fixture()
def ccs_cal():
    return deimos.calibration.CCSCalibration()


@pytest.fixture()
def pos():
    return deimos.calibration.AGILENT_CCS_REFERENCE['pos']


@pytest.fixture()
def neg():
    return deimos.calibration.AGILENT_CCS_REFERENCE['neg']


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

        with pytest.raises(RuntimeError):
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
