from scipy.stats import linregress
import numpy as np


class ArrivalTimeCalibration:
    def __init__(self):
        """
        Initialization method.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # initialize variables
        self.buffer_mass = None
        self.beta = None
        self.tfix = None
        self.fit = {'r': None, 'p': None, 'se': None}

    def _check(self):
        """
        Helper method to check for calibration parameters

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        if (self.beta is None) or (self.tfix is None):
            raise ValueError('Must perform calibration to yield beta and tfix.')

    def calibrate(self, mass=None, ta=None, ccs=None, q=None,
                  beta=None, tfix=None, buffer_mass=28.013):
        """
        Performs calibration if 'mass', 'ta', 'ccs', and 'q' arrays are provided,
        otherwise calibration parameters 'beta' and 'tfix' must be supplied directly.

        Parameters
        ----------
        mass : array, optional
            Calibration masses.
        ta : array, optional
            Calibration arrival times.
        ccs : array, optional
            Calibration collision cross sections.
        q : array, optional
            Calibration nominal charges.
        beta : float, optional
            Provide calibration parameter 'beta' directly.
        tfix : float, optional
            Provide calibration parameter 'tfix' directly

        Returns
        -------
        None.
        """
        # buffer mass
        self.buffer_mass = buffer_mass

        # calibrant arrays supplied
        if (mass is not None) and (ta is not None) and (ccs is not None) and (q is not None):
            # linear regression
            beta, tfix, r, p, se = linregress(np.sqrt(mass / (mass + self.buffer_mass)) * ccs / q, ta)

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

        raise ValueError('Must supply arrays for calibration or calibration parameters.')

    def arrival2ccs(self, mass, ta, q=1):
        """
        Calculates collision cross section (CCS) from arrival time, mass,
        and nominal charge, according to calibration parameters.

        Parameters
        ----------
        mass : float
            Feature mass (m/z).
        ta : float
            Feature arrival time (ms).
        q : int, optional (default=1)
            Feature nominal charge.

        Returns
        -------
        ccs : float
            Feature collision cross section (A^2).
        """
        self._check()
        return q / self.beta * (ta - self.tfix) / np.sqrt(mass / (mass + self.buffer_mass))

    def ccs2arrival(self, mass, ccs, q=1):
        """
        Calculates arrival time from collsion cross section (CCS), mass,
        and nominal charge, according to calibration parameters.

        Parameters
        ----------
        mass : float
            Feature mass (m/z).
        ccs : float
            Feature collision cross section (A^2).
        q : int, optional (default=1)
            Feature nominal charge.

        Returns
        -------
        ta : float
            Feature arrival time (ms).
        """
        self._check()
        return self.beta / q * np.sqrt(mass / (mass + self.buffer_mass)) * ccs + self.tfix
