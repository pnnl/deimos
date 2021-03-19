import pytest
import deimos
import numpy as np
from tests import localfile


@pytest.fixture()
def atc():
    return deimos.calibration.ArrivalTimeCalibration()


@pytest.fixture()
def pos():
    return {'mz': [118.086255, 322.048121, 622.028960, 922.009798,
                   1221.990636, 1521.971475],
            'ta': [13.72, 18.65, 25.20, 30.44, 35.36, 39.83],
            'ccs': [120.8, 152.8, 201.6, 241.8, 279.9, 314.4],
            'q': [1, 1, 1, 1, 1, 1]}


@pytest.fixture()
def neg():
    return {'mz': [301.998139, 601.978977, 1033.988109,
                   1333.968947, 1633.949786],
            'ta': [17.04, 22.53, 32.13, 36.15, 40.70],
            'ccs': [139.8, 179.9, 254.2, 283.6, 317.7],
            'q': [1, 1, 1, 1, 1]}


class TestArrivalTimeCalibration:

    def test_init(self, atc):
        for attr, expected in zip(['buffer_mass', 'beta', 'tfix', 'fit'],
                                  [None, None, None, dict]):
            assert hasattr(atc, attr)

            tmp = getattr(atc, attr)

            if expected is None:
                assert tmp is expected

            else:
                assert type(tmp) is expected

                for k in ['r', 'p', 'se']:
                    assert k in tmp.keys()
                    assert tmp[k] is None

    @pytest.mark.parametrize('beta,tfix',
                             [(1, 0)])
    def test__check(self, atc, beta, tfix):
        atc.beta = beta
        atc.tfix = tfix
        atc._check()

    @pytest.mark.parametrize('beta,tfix',
                             [(1, None),
                              (None, 0),
                              (None, None)])
    def test__check_fail(self, atc, beta, tfix):
        atc.beta = beta
        atc.tfix = tfix

        with pytest.raises(ValueError):
            atc._check()

    @pytest.mark.parametrize('calc,beta,tfix,beta_exp,tfix_exp',
                             [(False, 1, 0, 1, 0),
                              (True, 1, 0, 0.12856, -0.20273),
                              (True, None, None, 0.12856, -0.20273)])
    def test_calibrate(self, atc, pos, calc, beta, tfix, beta_exp, tfix_exp):
        if calc is True:
            atc.calibrate(beta=beta, tfix=tfix, **pos)
            for k in ['r', 'p', 'se']:
                assert atc.fit[k] is not None
        else:
            atc.calibrate(beta=beta, tfix=tfix)

        assert abs(atc.beta - beta_exp) <= 1E-3
        assert abs(atc.tfix - tfix_exp) <= 1E-3

    def test_arrival2ccs(self, atc, pos, neg):
        for data in [pos, neg]:
            atc.calibrate(**data)
            ccs = atc.arrival2ccs(data['mz'], data['ta'], q=data['q'])

            error = np.abs(ccs - data['ccs']) / data['ccs']

            assert (error <= 0.005).all()

    def test_ccs2arrival(self, atc, pos, neg):
        for data in [pos, neg]:
            atc.calibrate(**data)
            ta = atc.ccs2arrival(data['mz'], data['ccs'], q=data['q'])

            error = np.abs(ta - data['ta']) / data['ta']

            assert (error <= 0.005).all()


@pytest.mark.parametrize('calc,beta,tfix,beta_exp,tfix_exp',
                         [(False, 1, 0, 1, 0),
                          (True, 1, 0, 0.12856, -0.20273),
                          (True, None, None, 0.12856, -0.20273)])
def test_calibrate_ccs(pos, calc, beta, tfix, beta_exp, tfix_exp):
    if calc is True:
        atc = deimos.calibration.calibrate_ccs(beta=beta, tfix=tfix, **pos)
        for k in ['r', 'p', 'se']:
            assert atc.fit[k] is not None
    else:
        atc = deimos.calibration.calibrate_ccs(beta=beta, tfix=tfix)

    assert type(atc) is deimos.calibration.ArrivalTimeCalibration
    assert abs(atc.beta - beta_exp) <= 1E-3
    assert abs(atc.tfix - tfix_exp) <= 1E-3
