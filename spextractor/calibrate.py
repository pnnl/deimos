from scipy.stats import linregress
import numpy as np
import spextractor as spx
import matplotlib.pyplot as plt


def tunemix(path, mz, ccs, q, mz_tol=0.1, threshold=1E3, verbosity=0):
    """
    Calibrate using Agilent TuneMix.

    Parameters
    ----------
    path : str
        Path to TuneMix file stored as HDF.
    mz : array
        Calibration masses (m/z).
    ccs : array
        Calibration collision cross sections (A^2).
    q : array
        Calibration nomincal charges.
    mz_tol : float, optional (default=0.01)
        Tolerance in mass (m/z).
    verbosity : int, optional (default=0).
        0 - Silent operation.
        1 - Prints found peak information.
        2 - Additionally plots drift time peaks.

    Returns
    -------
    measurements : dict
        Experimental calibration points.
    calibration :  ArrivalTimeCalibration instance
        Calibration result.
    """
    data = spx.utils.load_hdf(path)
    measurements = {'mz': [],
                    'ccs': [],
                    'q': [],
                    'ta': []}
    for mz_i, ccs_i, q_i in zip(mz, ccs, q):
        # find peak by mz
        peaks = spx.peakpick.guided(data,
                                    mz=mz_i,
                                    mz_tol=mz_tol,
                                    threshold=threshold)

        if peaks is not None:
            peak = peaks.loc[0, :]
            mz_exp = peak['mz']
            ta_exp = peak['drift_time']
            measurements['mz'].append(mz_exp)
            measurements['ccs'].append(ccs_i)
            measurements['q'].append(q_i)
            measurements['ta'].append(ta_exp)

            if verbosity > 0:
                error = abs(mz_i - mz_exp) / mz_i * 1E6
                print('reference mass:\t\t', mz_i)
                print('experimental mass:\t', mz_exp)
                print('mass error (ppm):\t', error)
                print('experimental ta:\t', ta_exp)
                print()
            if verbosity > 1:
                spx.plot.fill_between(peak['drift_time'], peak['intensity'])
                plt.show()
        else:
            if verbosity > 0:
                print('reference mass:\t\t', mz_i)
                print('peak not found.\n')

    # calibration
    c = ArrivalTimeCalibration()
    c.calibrate(**measurements)

    return measurements, c


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

    def calibrate(self, mz=None, ta=None, ccs=None, q=None,
                  beta=None, tfix=None, buffer_mass=28.013):
        """
        Performs calibration if 'mz', 'ta', 'ccs', and 'q' arrays are provided,
        otherwise calibration parameters 'beta' and 'tfix' must be supplied directly.

        Parameters
        ----------
        mz : array, optional
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
        if (mz is not None) and (ta is not None) and (ccs is not None) and (q is not None):
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

        raise ValueError('Must supply arrays for calibration or calibration parameters.')

    def arrival2ccs(self, mz, ta, q=1):
        """
        Calculates collision cross section (CCS) from arrival time, mz,
        and nominal charge, according to calibration parameters.

        Parameters
        ----------
        mz : float
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
        return q / self.beta * (ta - self.tfix) / np.sqrt(mz / (mz + self.buffer_mass))

    def ccs2arrival(self, mz, ccs, q=1):
        """
        Calculates arrival time from collsion cross section (CCS), mz,
        and nominal charge, according to calibration parameters.

        Parameters
        ----------
        mz : float
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
        return self.beta / q * np.sqrt(mz / (mz + self.buffer_mass)) * ccs + self.tfix
