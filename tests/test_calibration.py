import deimos
import numpy as np
import pytest


@pytest.fixture()
def mass_cal():
    return deimos.calibration.MassCalibration()


@pytest.fixture()
def ccs_cal():
    return deimos.calibration.CCSCalibration()


@pytest.fixture()
def pos():
    return {
        "mz": [
            118.086255,
            322.048121,
            622.028960,
            922.009798,
            1221.990636,
            1521.971475,
        ],
        "ta": [13.72, 18.65, 25.20, 30.44, 35.36, 39.83],
        "ccs": [121.30, 153.73, 202.96, 243.64, 282.20, 316.96],
        "q": [1, 1, 1, 1, 1, 1],
    }


@pytest.fixture()
def neg():
    return {
        "mz": [301.998139, 601.978977, 1033.988109, 1333.968947, 1633.949786],
        "ta": [17.04, 22.53, 32.13, 36.15, 40.70],
        "ccs": [140.04, 180.77, 255.34, 284.76, 319.03],
        "q": [1, 1, 1, 1, 1],
    }


@pytest.fixture()
def twims_data():
    """TWIMS-like data that would benefit from power function calibration"""
    return {
        "mz": [242.2846, 298.3472, 354.4099, 410.4726, 466.5356],
        "ta": [1.55012, 4.56077, 7.76386, 11.11285, 14.59229],
        "ccs": [165.8, 190.61, 213.84, 236.4, 258.55],
        "q": [1, 1, 1, 1, 1],
    }

class TestMassCalibration:
    def test_init(self, ccs_cal):
        for attr, expected in zip(["beta", "tfix", "fit"], [None, None, dict]):
            assert hasattr(ccs_cal, attr)

            tmp = getattr(ccs_cal, attr)

            if expected is None:
                assert tmp is expected

            else:
                assert type(tmp) is expected

                for k in ["r", "p", "se"]:
                    assert k in tmp.keys()
                    assert tmp[k] is None

    @pytest.mark.parametrize("beta,tfix", [(1, 0)])
    def test__check(self, mass_cal, beta, tfix):
        mass_cal.beta = beta
        mass_cal.tfix = tfix
        mass_cal._check()

    @pytest.mark.parametrize("beta,tfix", [(1, None), (None, 0), (None, None)])
    def test__check_fail(self, mass_cal, beta, tfix):
        mass_cal.beta = beta
        mass_cal.tfix = tfix

        with pytest.raises(RuntimeError):
            mass_cal._check()

    @pytest.mark.parametrize(
        "calc,beta,tfix,beta_exp,tfix_exp",
        [(False, 1, 0, 1, 0), (True, 1, 0, 1, 0), (True, None, None, 1, 0)],
    )
    def test_calibrate(self, mass_cal, pos, calc, beta, tfix, beta_exp, tfix_exp):
        if calc is True:
            mass_cal.calibrate(
                beta=beta, tfix=tfix, mz=pos["mz"], reference_mz=pos["mz"]
            )
            for k in ["r", "p", "se"]:
                assert mass_cal.fit[k] is not None
        else:
            mass_cal.calibrate(beta=beta, tfix=tfix)

        assert abs(mass_cal.beta - beta_exp) <= 1e-3
        assert abs(mass_cal.tfix - tfix_exp) <= 1e-3

    def test_apply(self, mass_cal, pos, neg):
        for data in [pos, neg]:
            mass_cal.calibrate(mz=data["mz"], reference_mz=data["mz"])
            mz = mass_cal.apply(data["mz"])

            error = np.abs(mz - data["mz"]) / data["mz"]

            assert (error <= 0.005).all()


@pytest.mark.parametrize(
    "calc,beta,tfix,beta_exp,tfix_exp",
    [
        (False, 1, 0, 1, 0),
        (True, 1, 0, 41.596354, 4.898403),
        (True, None, None, 41.596354, 4.898403),
    ],
)
def test_calibrate_ccs_standalone(pos, calc, beta, tfix, beta_exp, tfix_exp):
    if calc is True:
        mass_cal = deimos.calibration.calibrate_ccs(beta=beta, tfix=tfix, **pos)
        for k in ["r", "p", "se"]:
            assert mass_cal.fit[k] is not None
    else:
        mass_cal = deimos.calibration.calibrate_ccs(beta=beta, tfix=tfix)

    assert type(mass_cal) is deimos.calibration.CCSCalibration
    assert abs(mass_cal.beta - beta_exp) <= 1e-3
    assert abs(mass_cal.tfix - tfix_exp) <= 1e-3


class TestCCSCalibration:
    def test_init(self, ccs_cal):
        for attr, expected in zip(
            ["buffer_mass", "beta", "tfix", "fit", "power"], [None, None, None, dict, False]
        ):
            assert hasattr(ccs_cal, attr)

            tmp = getattr(ccs_cal, attr)

            if expected is None:
                assert tmp is expected
            elif expected is False:
                assert tmp is False
            else:
                assert type(tmp) is expected

                for k in ["r", "p", "se"]:
                    assert k in tmp.keys()
                    assert tmp[k] is None

    @pytest.mark.parametrize("beta,tfix", [(1, 0)])
    def test__check(self, ccs_cal, beta, tfix):
        ccs_cal.beta = beta
        ccs_cal.tfix = tfix
        ccs_cal._check()

    @pytest.mark.parametrize("beta,tfix", [(1, None), (None, 0), (None, None)])
    def test__check_fail(self, ccs_cal, beta, tfix):
        ccs_cal.beta = beta
        ccs_cal.tfix = tfix

        with pytest.raises(RuntimeError):
            ccs_cal._check()

    @pytest.mark.parametrize(
        "calc,beta,tfix,beta_exp,tfix_exp",
        [
            (False, 1, 0, 1, 0),
            (True, 1, 0, 41.596354, 4.898403),
            (True, None, None, 41.596354, 4.898403),
        ],
    )
    def test_calibrate(self, ccs_cal, pos, calc, beta, tfix, beta_exp, tfix_exp):
        if calc is True:
            ccs_cal.calibrate(beta=beta, tfix=tfix, **pos)
            for k in ["r", "p", "se"]:
                assert ccs_cal.fit[k] is not None
        else:
            ccs_cal.calibrate(beta=beta, tfix=tfix)

        assert abs(ccs_cal.beta - beta_exp) <= 1e-3
        assert abs(ccs_cal.tfix - tfix_exp) <= 1e-3

    def test_arrival2ccs(self, ccs_cal, pos, neg):
        for data in [pos, neg]:
            ccs_cal.calibrate(**data)
            ccs = ccs_cal.arrival2ccs(data["mz"], data["ta"], q=data["q"])

            error = np.abs(ccs - data["ccs"]) / data["ccs"]

            assert (error <= 0.005).all()

    def test_ccs2arrival(self, ccs_cal, pos, neg):
        for data in [pos, neg]:
            ccs_cal.calibrate(**data)
            ta = ccs_cal.ccs2arrival(data["mz"], data["ccs"], q=data["q"])

            error = np.abs(ta - data["ta"]) / data["ta"]

            assert (error <= 0.005).all()

    def test_calibrate_power(self, ccs_cal, twims_data):
        """Test power function calibration for TWIMS data"""
        ccs_cal.calibrate(power=True, **twims_data)
        
        # Check that power mode is enabled
        assert ccs_cal.power is True
        
        # Check that all required attributes exist
        assert hasattr(ccs_cal, 'a')
        assert ccs_cal.a is not None
        assert ccs_cal.beta is not None
        assert ccs_cal.tfix is not None
        
        # Test forward conversion (arrival time -> CCS)
        ccs = ccs_cal.arrival2ccs(twims_data["mz"], twims_data["ta"], q=twims_data["q"])
        error = np.abs(ccs - twims_data["ccs"]) / twims_data["ccs"]
        assert (error <= 0.01).all()  # Allow slightly higher tolerance for power function
        
        # Test reverse conversion (CCS -> arrival time)
        ta = ccs_cal.ccs2arrival(twims_data["mz"], twims_data["ccs"], q=twims_data["q"])
        error = np.abs(ta - twims_data["ta"]) / twims_data["ta"]
        assert (error <= 0.01).all()


@pytest.mark.parametrize(
    "calc,beta,tfix,beta_exp,tfix_exp",
    [
        (False, 1, 0, 1, 0),
        (True, 1, 0, 41.596354, 4.898403),
        (True, None, None, 41.596354, 4.898403),
    ],
)
def test_calibrate_ccs_function(pos, calc, beta, tfix, beta_exp, tfix_exp):
    if calc is True:
        ccs_cal = deimos.calibration.calibrate_ccs(beta=beta, tfix=tfix, **pos)
        for k in ["r", "p", "se"]:
            assert ccs_cal.fit[k] is not None
    else:
        ccs_cal = deimos.calibration.calibrate_ccs(beta=beta, tfix=tfix)

    assert type(ccs_cal) is deimos.calibration.CCSCalibration
    assert abs(ccs_cal.beta - beta_exp) <= 1e-3
    assert abs(ccs_cal.tfix - tfix_exp) <= 1e-3


def test_calibrate_ccs_function_power(twims_data):
    """Test power function calibration via calibrate_ccs function"""
    ccs_cal = deimos.calibration.calibrate_ccs(power=True, **twims_data)
    
    assert type(ccs_cal) is deimos.calibration.CCSCalibration
    assert ccs_cal.power is True
    assert ccs_cal.a is not None
    assert ccs_cal.beta is not None
    assert ccs_cal.tfix is not None
    
    # Verify calibration accuracy
    ccs = ccs_cal.arrival2ccs(twims_data["mz"], twims_data["ta"], q=twims_data["q"])
    error = np.abs(ccs - twims_data["ccs"]) / twims_data["ccs"]
    assert (error <= 0.01).all()
