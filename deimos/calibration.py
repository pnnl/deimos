import deimos
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress


class ArrivalTimeCalibration:
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
        Initializes :obj:`~deimos.calibration.ArrivalTimeCalibration` object.

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


def calibrate_ccs(mz=None, ta=None, ccs=None, q=None,
                  beta=None, tfix=None, buffer_mass=28.013):
    '''
    Convenience function for :class:`~deimos.calibration.ArrivalTimeCalibration`.
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

    atc = ArrivalTimeCalibration()
    atc.calibrate(mz=mz, ta=ta, ccs=ccs, q=q, beta=beta, tfix=tfix,
                  buffer_mass=buffer_mass)

    return atc


def tunemix(features,
            mz=[112.985587, 301.998139, 601.978977, 1033.988109, 1333.968947, 1633.949786],
            ccs=[108.4, 139.8, 179.9, 254.2, 283.6, 317.7],
            q=[1, 1, 1, 1, 1, 1], buffer_mass=28.013, mz_tol=200E-6, dt_tol=0.04):
    '''
    Provided tune mix data with known calibration ions (i.e. known m/z, CCS, and nominal charge),
    determine the arrival time for each to define a CCS calibration.
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
    Returns
    -------
    :obj:`~deimos.calibration.ArrivalTimeCalibration`
        Instance of calibrated `~deimos.calibration.ArrivalTimeCalibration`
        object.
    '''

    # cast to numpy array
    mz = np.array(mz)
    ccs = np.array(ccs)
    q = np.array(q)

    # check lengths
    deimos.utils.check_length([mz, ccs, q])

    # iterate tune ions
    ta = []
    for mz_i, ccs_i, q_i in zip(mz, ccs, q):
        # slice ms1
        subset = deimos.slice(features, by='mz',
                              low=mz_i - 0.1 * mz_tol,
                              high=mz_i + mz_i * 0.9 * mz_tol)

        # extract dt info
        dt_profile = deimos.collapse(subset, keep='drift_time')
        dt_i = dt_profile.sort_values(by='intensity', ascending=False)['drift_time'].values[0]
        dt_profile = deimos.locate(dt_profile, by='drift_time', loc=dt_i,
                                   tol=dt_tol * dt_i).sort_values(by='drift_time')

        # interpolate spline
        x = dt_profile['drift_time'].values
        y = dt_profile['intensity'].values

        spl = interp1d(x, y, kind='quadratic')
        newx = np.arange(x.min(), x.max(), 0.001)
        newy = spl(newx)
        dt_j = newx[np.argmax(newy)]

        ta.append(dt_j)

    # calibrate
    ta = np.array(ta)
    return deimos.calibration.calibrate_ccs(mz=mz, ta=ta, ccs=ccs, q=q, buffer_mass=buffer_mass)


class ms2DriftCalibration:

    def __init__(self):
        '''
        Initializes :obj:`~deimos.calibration.ms2DriftCalibration` object.

        '''

        # initialize variables
        self.mode = None
        self.mz = None
        self.df = None
        self.b = None

    def set_mode(self, mode=None):
        '''
        Parameters
        ----------
        mode : string
            String of mode of ionization; must be specified as:
                negative, neg, -, positive, pos, or +
        '''
        self.mode = mode
        return

    def set_mz(self, mz=None):
        '''
        Parameters
        ----------
        mz : list
            List of float mz values to use in calibration of drift time
        '''
        if mz is None:
            if self.mode.lower() in ['positive', 'pos', '+']:
                self.mz = np.array([118.086255, 322.048121, 622.028960,
                                    922.009798, 1221.990636, 1521.971475])
            elif self.mode.lower() in ['negative', 'neg', '-']:
                self.mz = np.array([112.985587, 301.998139, 601.978977,
                                    1033.988109, 1333.968947, 1633.949786])
        else:
            self.mz = np.array(mz)
        return

    def subset_mz(self, decon_data, orbitrap=False):
        '''
        Provided data with known calibration ions (i.e. known m/z),
        subset the deconvolution output by those ions.

        Parameters
        ----------


        Returns
        -------

        '''
        decon_data['bias'] = 1

        meta_df = pd.DataFrame(columns=decon_data.columns)
        for mz_i in self.mz:
            subset = decon_data[decon_data['mz_ms1'].between(mz_i-2, mz_i+2)]
            # if orbitrap == True:
            # TODO compare to orbitrap here
            # subset['OrbitrapScore'] = cosine_score()
            subset2 = subset[subset['mz_ms2'].between(mz_i-2, mz_i+2)]
            meta_df = meta_df.append(subset2)
        self.df = meta_df
        return

    def regress_drift(self):
        '''

        Parameters
        ----------

        Returns
        -------


        Raises
        ------


        '''

        df = self.df.copy()
        x = df['drift_time_ms2'].values
        y = df['drift_time_ms1'].values
        z = df['voltage'].values
        bias = df['bias'].values
        X = np.hstack(x, z, bias)
        b = np.linalg.pinv(X).dot(y)
        self.b = b
        return


def tunemix_calibrate(decon_data, mode, **kwargs):
    '''
    Parameters
    ----------
    decon_data : pd.DataFrame
        output from `deimos.calibration.stitch_deconvolutions`
    '''
    mdc = ms2DriftCalibration()
    mdc.set_mode(mode)
    mdc.set_mz()
    mdc.subset_mz(decon_data, **kwargs)
    mdc.regress_drift()
    return mdc


def add_voltage_to_deconvolution(decon_data, voltage):
    '''
    Parameters
    ----------
    decon_data : pd.DataFrame
        deimos.deconvolution.deconvolve_ms2 output
    voltage : int
        int value of voltage
    '''
    decon_data['voltage'] = voltage
    return decon_data


def stitch_deconvolutions(list_of_decons):
    '''
    Parameters
    ----------
    list_of_decons : list of pd.DataFrame
        list of `deimos.calibration.add_voltage_to_deconvolution` output
    '''
    return pd.concat(list_of_decons)


def shift_drift(ms2, mdc):
    '''
    Provided an ms2 or ms2_peaks object, update the drift_time column with the calibration object.

    Parameters
    ----------
    ms2 : pd.DataFrame
        (1) deimos.load_hdf(key='ms2') output or
        (2) deimos.peakpick.local_maxima() output or
        (3) deimos.threshold() output

    Returns
    -------
    ms2 : pd.DataFrame
        same as input with modified `drift_time` column
    '''
    ms2_df = ms2.copy()
    ms2['drift_time'] = ms2['drift_time'].apply(lambda dt: [dt, voltage].dot(tmc.b))
    return ms2
