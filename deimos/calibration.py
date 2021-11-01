import deimos
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress


class CCSCalibration:
    '''
    Performs calibration and stores result to enable convenient application.

    Attributes
    ----------
    buffer_mass : float
        Mass of the buffer gas used in ion mobility experiment.
    beta : float
        Slope of calibration curve.
    tfix : float
        Intercept of calibration curve.
    fit : dict of float
        Fit parameters of calibration curve.

    '''

    def __init__(self):
        '''
        Initializes :obj:`~deimos.calibration.CCSCalibration` object.

        '''

        # initialize variables
        self.buffer_mass = None
        self.beta = None
        self.tfix = None
        self.fit = {'r': None, 'p': None, 'se': None}

    def _check(self):
        '''
        Helper method to check for calibration parameters.

        Raises
        ------
        ValueError
            If `self.tfix` or `self.beta` are None.

        '''

        if (self.beta is None) or (self.tfix is None):
            raise ValueError('Must perform calibration to yield beta and '
                             'tfix.')

    def calibrate(self, mz=None, ta=None, ccs=None, q=None,
                  beta=None, tfix=None, buffer_mass=28.013):
        '''
        Performs calibration if `mz`, `ta`, `ccs`, and `q` arrays are provided,
        otherwise calibration parameters `beta` and `tfix` must be supplied
        directly.

        Parameters
        ----------
        mz : :obj:`~numpy.array`
            Calibration mass-to-charge ratios.
        ta : :obj:`~numpy.array`
            Calibration arrival times.
        ccs : :obj:`~numpy.array`
            Calibration collision cross sections.
        q : :obj:`~numpy.array`
            Calibration nominal charges.
        beta : float
            Provide calibration parameter "beta" (slope) directly.
        tfix : float
            Provide calibration parameter "tfix" (intercept) directly.
        buffer_mass : float
            Mass of the buffer gas.

        Raises
        ------
        ValueError
            If appropriate keyword arguments are not supplied (either `mz`,
            `ta`, `ccs`, and `q`; or `beta` and `tfix`).

        '''

        # buffer mass
        self.buffer_mass = buffer_mass

        # calibrant arrays supplied
        if (mz is not None) and (ta is not None) and (ccs is not None) \
           and (q is not None):
            self.mz = np.array(mz)
            self.ta = np.array(ta)
            self.ccs = np.array(ccs)
            self.q = np.array(q)

            # linear regression
            beta, tfix, r, p, se = linregress(np.sqrt(self.mz / (self.mz + self.buffer_mass)) * self.ccs / self.q,
                                              self.ta)

            # store params
            self.beta = beta
            self.tfix = tfix
            self.fit['r'] = r
            self.fit['p'] = p
            self.fit['se'] = se
            return

        # beta and tfix supplied
        if (beta is not None) and (tfix is not None):
            # store params
            self.beta = beta
            self.tfix = tfix
            return

        raise ValueError('Must supply arrays for calibration or calibration '
                         'parameters.')

    def arrival2ccs(self, mz, ta, q=1):
        '''
        Calculates collision cross section (CCS) from arrival time, m/z, and
        nominal charge, according to calibration parameters.

        Parameters
        ----------
        mz : float or list of float
            Feature mass-to-charge ratio.
        ta : float or list of float
            Feature arrival time (ms).
        q : int or list of int
            Feature nominal charge.

        Returns
        -------
        :obj:`~numpy.array`
            Feature collision cross section (A^2).

        '''

        self._check()

        mz = np.array(mz)
        ta = np.array(ta)
        q = np.array(q)

        return q / self.beta * (ta - self.tfix) / np.sqrt(mz / (mz + self.buffer_mass))

    def ccs2arrival(self, mz, ccs, q=1):
        '''
        Calculates arrival time from collsion cross section (CCS), m/z, and
        nominal charge, according to calibration parameters.

        Parameters
        ----------
        mz : float or list of float
            Feature mass-to-charge ratio.
        ccs : float or list of float
            Feature collision cross section (A^2).
        q : int or list of int
            Feature nominal charge.

        Returns
        -------
        :obj:`~numpy.array`
            Feature arrival time (ms).

        '''

        self._check()

        mz = np.array(mz)
        ccs = np.array(ccs)
        q = np.array(q)

        return self.beta / q * np.sqrt(mz / (mz + self.buffer_mass)) * ccs + self.tfix


def calibrate_ccs(mz=None, ta=None, ccs=None, q=None, beta=None, tfix=None, buffer_mass=28.013):
    '''
    Convenience function for :class:`~deimos.calibration.CCSCalibration`.
    Performs calibration if `mz`, `ta`, `ccs`, and `q` arrays are provided,
    otherwise calibration parameters `beta` and `tfix` must be supplied
    directly.

    Parameters
    ----------
    mz : :obj:`~numpy.array`
        Calibration mass-to-charge ratios.
    ta : :obj:`~numpy.array`
        Calibration arrival times.
    ccs : :obj:`~numpy.array`
        Calibration collision cross sections.
    q : :obj:`~numpy.array`
        Calibration nominal charges.
    beta : float
        Provide calibration parameter "beta" (slope) directly.
    tfix : float
        Provide calibration parameter "tfix" (intercept) directly.
    buffer_mass : float
        Mass of the buffer gas.

    Returns
    -------
    :obj:`~deimos.calibration.ArrivalTimeCalibration`
        Instance of calibrated `~deimos.calibration.ArrivalTimeCalibration`
        object.

    Raises
    ------
    ValueError
        If appropriate keyword arguments are not supplied (either `mz`,
        `ta`, `ccs`, and `q`; or `beta` and `tfix`).

    '''

    ccsc = CCSCalibration()
    ccsc.calibrate(mz=mz, ta=ta, ccs=ccs, q=q, beta=beta, tfix=tfix,
                   buffer_mass=buffer_mass)

    return ccsc


class MS2DriftCalibration:

    def __init__(self):
        '''
        Initializes :obj:`~deimos.calibration.ms2DriftCalibration` object.

        '''

        # initialize variables
        self.mode = None
        self.mz = None
        self.df = None
        self.b = None
        self.calibrants = pd.DataFrame()

    def add_calibrant_pairs(self, ms1_dt=None, ms2_dt=None, voltage=None):
        if (ms1_dt is not None) and (ms2_dt is not None) and (voltage is not None):
            deimos.utils.check_length([ms1_dt, ms2_dt])
            df = pd.DataFrame({'ms1': ms1_dt, 'ms2': ms2_dt, 'voltage': np.full_like(
                ms1_dt, voltage, dtype=np.double), 'bias': np.ones_like(ms1_dt)})
            self.calibrants = self.calibrants.append(df)
        else:
            raise('Must specify ms1_dt, ms2_dt, and voltage')
        return

    def regress(self):
        '''
        Does 3D regression for drift times, voltage, and bias stored in memory.
        '''

        df = self.calibrants.copy()
        x = df['ms2'].values.reshape(-1, 1)
        y = df['ms1'].values.reshape(-1, 1)
        z = df['voltage'].values.reshape(-1, 1)
        bias = df['bias'].values.reshape(-1, 1)
        X = np.hstack((x, z, bias))
        b = np.linalg.pinv(X).dot(y)
        self.b = b
        return

    def shift(self, dt, voltage):
        '''
        Provided a drift time and voltage at which drift time should be shifted by.

        Parameters
        ----------
        dt : np.float
        voltage : np.double

        Returns
        -------
        shifted_dt : np.float
        '''
        shifted_dt = np.array([dt, voltage, 1]).dot(self.b)
        return shifted_dt


def generate_ms2_drift_calibration(ms1_dt=None, ms2_dt=None, voltages=None):
    '''
    Parameters
    ----------
    ms1_dt : list of np.array()
    ms2_dt : list of np.array()
    calibration_voltages : np.array() of np.double or np.int
    input_dt : np.array() of np.float or np.double
    input_voltage : np.array() of np.double or np.int

    Returns
    -------
    output_dt : np.array() of np.float or np.double
    '''
    ms2dc = MS2DriftCalibration()
    for ms1_sub, ms2_sub, volt_sub in zip(ms1_dt, ms2_dt, voltages):
        ms2dc.add_calibrant_pairs(ms1_dt=ms1_sub, ms2_dt=ms2_sub, voltage=volt_sub)
    ms2dc.regress()
    return ms2dc


def calibrate_drift(ms2dc, input_dt=None, input_voltage=None):
    output_dt = np.array()
    for dt in input_dt:
        output_dt = np.concatenate(output_dt, ms2dc.shift(dt, input_voltage))
    return output_dt


class TuneMixCalibrants:
    def __init__(self, features):
        '''
        Initializes :obj:`~deimos.calibration.MS2DriftCalibration` object.

        '''

        # initialize variables
        if isinstance(features, list):
            self.features = [feat.copy() for feat in features]
        else:
            self.features = features.copy()
        self.volts = None
        self.mode = None
        self.mz = None
        self.ccs = None
        self.q = None
        self.bm = None
        self.mzt = None
        self.dtt = None
        self.dt = None

    def set_mode(self, mode):
        '''
        Parameters
        ----------
        mode : string
            String of mode of ionization; must be specified as:
                negative, neg, -, positive, pos, or +
        '''
        if mode.lower() in ['positive', 'pos', '+']:
            self.mode = 'positive'
            return
        elif mode.lower() in ['negative', 'neg', '-']:
            self.mode = 'negative'
            return

    def set_mz(self, mz):
        '''
        Parameters
        ----------
        mz : list
            List of float mz values to use in calibration of drift time
        '''
        if mz is None:
            if self.mode == 'positive':
                self.mz = np.array([118.086255, 322.048121, 622.028960,
                                    922.009798, 1221.990636, 1521.971475])
            elif self.mode == 'negative':
                self.mz = np.array([301.998139, 601.978977,
                                    1033.988109, 1333.968947, 1633.949786])
        elif mz is not None:
            self.mz = np.array(mz)
        return

    def set_ccs(self, ccs):
        '''
        Parameters
        ----------
        ccs : list
            List of float ccs values to use in calibration of drift time
        '''
        if ccs is None:
            if self.mode == 'positive':
                self.ccs = np.array([120.8, 152.8, 201.6, 241.8, 279.9, 314.4])
            elif self.mode == 'negative':
                self.ccs = np.array([139.8, 179.9, 254.2, 283.6, 317.7])
        elif ccs is not None:
            self.ccs = np.array(ccs)
        return

    def set_q(self, q):
        '''
        Parameters
        ----------
        q : list
            List of int q values to use in calibration of drift time
        '''
        if q is None:
            if self.mode == 'positive':
                self.q = np.array([1, 1, 1, 1, 1, 1])
            elif self.mode == 'negative':
                self.q = np.array([1, 1, 1, 1, 1])
        else:
            self.q = np.array(q)
        return

    def set_buffer_mass(self, buffer_mass):
        self.bm = buffer_mass
        return

    def set_mz_tol(self, mz_tol):
        self.mzt = mz_tol
        return

    def set_dt_tol(self, dt_tol):
        self.dtt = dt_tol
        return

    def set_volts(self, voltages):
        self.volts = voltages
        return

    def set_input(self, mode=None, voltages=None, mz=None, ccs=None, q=None, buffer_mass=28.013, mz_tol=200E-6, dt_tol=0.04):
        '''
        Parameters
        ----------
        mz : :obj:`~numpy.array`
            Calibration mass-to-charge ratios.
        ccs : :obj:`~numpy.array`
            Calibration collision cross sections.
        q : :obj:`~numpy.array`
            Calibration nominal charges.
        buffer_mass : float
            Mass of the buffer gas.
        mz_tol : float
            Tolerance in ppm to isolate tune ion.
        dt_tol : float
            Fractional tolerance to define drift time window bounds.
        '''
        self.set_mode(mode)
        self.set_buffer_mass(buffer_mass)
        self.set_mz_tol(mz_tol)
        self.set_dt_tol(dt_tol)
        self.set_volts(voltages)
        if self.mode is not None:
            if (mz is not None) and (ccs is not None) and (q is not None):
                # check lengths
                deimos.utils.check_length([mz, ccs, q])
            self.set_mz(mz)
            self.set_ccs(ccs)
            self.set_q(q)
            return
        if (mz is not None) and (ccs is not None) and (q is not None):
            # check lengths
            deimos.utils.check_length([mz, ccs, q])
            self.set_mz(mz)
            self.set_ccs(ccs)
            self.set_q(q)
        elif mz is not None:
            self.set_mz(mz)
        else:
            raise("Must specify ionization mode ('positive' or 'negative') or supply mz (np.array() or list) to look for, at a minimum.")
        return

    def iterate_ions_ccs(self):
        features = self.features.copy()
        # iterate tune ions
        dt_list = []
        for mz_i in self.mz:
            # slice ms1
            subset = deimos.slice(features, by='mz',
                                  low=mz_i - 0.1 * self.mzt,
                                  high=mz_i + mz_i * 0.9 * self.mzt)

            # extract dt info
            dt_profile = deimos.collapse(subset, keep='drift_time')
            dt_i = dt_profile.sort_values(by='intensity', ascending=False)['drift_time'].values[0]
            dt_profile = deimos.locate(dt_profile, by='drift_time', loc=dt_i,
                                       tol=self.dtt * dt_i).sort_values(by='drift_time')

            # interpolate spline
            x = dt_profile['drift_time'].values
            y = dt_profile['intensity'].values

            spl = interp1d(x, y, kind='quadratic')
            newx = np.arange(x.min(), x.max(), 0.001)
            newy = spl(newx)
            dt_j = newx[np.argmax(newy)]

            dt_list.append(dt_j)
        self.dt = np.array(dt_list)
        return

    def iterate_ions_drift(self):
        features = self.features.copy()
        self.features = []
        for feat_df in features:
            feat_df = feat_df.copy()
            for mz_i in self.mz:
                subset = feat_df[feat_df['mz_ms1'].between(mz_i-2, mz_i+2)]
                subset2 = subset[subset['mz_ms2'].between(mz_i-2, mz_i+2)]
                feat_df = feat_df.append(subset2)
            self.features.append(feat_df)
        return

    def calibrate_ccs(self):
        return deimos.calibration.calibrate_ccs(mz=self.mz, ta=self.dt, ccs=self.ccs, q=self.q, buffer_mass=self.bm)

    def generate_ms2_drift_calibration(self):
        return deimos.calibration.generate_ms2_drift_calibration(ms1_dt=[np.array(feats['drift_time_ms1']) for feats in self.features], ms2_dt=[np.array(feats['drift_time_ms2']) for feats in self.features], voltages=self.volts)


def tunemix_calibrate_ccs(features, mode=None, **kwargs):
    '''
    Parameters
    ----------
    features : pd.DataFrame

    '''
    tmc = TuneMixCalibrants(features)
    tmc.set_input(mode=mode, **kwargs)
    tmc.iterate_ions_ccs()
    ccsc = tmc.calibrate_ccs()
    return ccsc


def tunemix_calibrate_ms2_drift(features, voltages, mode=None, **kwargs):
    '''
    Parameters
    ----------
    features : list of pd.DataFrames
    voltages : list of np.double or np.int

    '''
    tmc = TuneMixCalibrants(features)
    tmc.set_input(voltages=voltages, mode=mode, **kwargs)
    tmc.iterate_ions_drift()
    ms2dc = tmc.generate_ms2_drift_calibration()
    return ms2dc
