from scipy.stats import linregress
import numpy as np


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
            mz = np.array(mz)
            ta = np.array(ta)
            ccs = np.array(ccs)
            q = np.array(q)

            # linear regression
            beta, tfix, r, p, se = linregress(np.sqrt(mz / (mz + self.buffer_mass)) * ccs / q, ta)

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
        mz : float
            Feature mass-to-charge ratio.
        ta : float
            Feature arrival time (ms).
        q : int
            Feature nominal charge.

        Returns
        -------
        float
            Feature collision cross section (A^2).

        '''

        self._check()

        return q / self.beta * (ta - self.tfix) / np.sqrt(mz / (mz + self.buffer_mass))

    def ccs2arrival(self, mz, ccs, q=1):
        '''
        Calculates arrival time from collsion cross section (CCS), m/z, and
        nominal charge, according to calibration parameters.

        Parameters
        ----------
        mz : float
            Feature mass-to-charge ratio.
        ccs : float
            Feature collision cross section (A^2).
        q : int
            Feature nominal charge.

        Returns
        -------
        float
            Feature arrival time (ms).

        '''

        self._check()
        return self.beta / q * np.sqrt(mz / (mz + self.buffer_mass)) * ccs + self.tfix
