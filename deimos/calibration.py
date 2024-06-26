import warnings

import numpy as np
import tabula
from scipy.interpolate import interp1d
from scipy.stats import linregress

import deimos

AGILENT_CCS_REFERENCE = {
    "pos": {
        "mz": [
            118.086255,
            322.048121,
            622.028960,
            922.009798,
            1221.990636,
            1521.971475,
        ],
        "ccs": [121.30, 153.73, 202.96, 243.64, 282.20, 316.96],
        "q": [1, 1, 1, 1, 1, 1],
    },
    "neg": {
        "mz": [
            # 112.985587,
            301.998139,
            601.978977,
            1033.988109,
            1333.968947,
            1633.949786,
        ],
        "ccs": [
            # 108.23,
            140.04, 180.77, 255.34, 284.76, 319.03],
        "q": [
            # 1, 
            1, 1, 1, 1, 1],
    },
}


class CCSCalibration:
    """
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

    """

    def __init__(self):
        """
        Initializes :obj:`~deimos.calibration.CCSCalibration` object.

        """

        # Initialize variables
        self.buffer_mass = None
        self.beta = None
        self.tfix = None
        self.fit = {"r": None, "p": None, "se": None}

    def _check(self):
        """
        Helper method to check for calibration parameters.

        """

        if (self.beta is None) or (self.tfix is None):
            raise ValueError("Must perform calibration to yield beta and " "tfix.")

    def calibrate(
        self,
        mz=None,
        ta=None,
        ccs=None,
        q=None,
        beta=None,
        tfix=None,
        buffer_mass=28.013,
        power=False,
    ):
        """
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
        power : bool
            Indicate whether to use linearize power function for calibration,
            i.e. in traveling wave ion moblility spectrometry.

        """

        # Buffer mass
        self.buffer_mass = buffer_mass

        # Power function indicator
        self.power = power

        # Calibrant arrays supplied
        if (
            (mz is not None)
            and (ta is not None)
            and (ccs is not None)
            and (q is not None)
        ):
            self.mz = np.array(mz)
            self.ta = np.array(ta)
            self.ccs = np.array(ccs)
            self.q = np.array(q)

            # Derived variables
            self.gamma = (
                np.sqrt(self.mz * self.q / (self.mz * self.q + self.buffer_mass))
                / self.q
            )
            self.reduced_ccs = self.ccs * self.gamma

            # Linear regression
            if self.power:
                beta, tfix, r, p, se = linregress(
                    np.log(self.reduced_ccs), np.log(self.ta)
                )
            else:
                beta, tfix, r, p, se = linregress(self.reduced_ccs, self.ta)

            # Store params
            self.beta = beta
            self.tfix = tfix
            self.fit["r"] = r
            self.fit["p"] = p
            self.fit["se"] = se
            return

        # Beta and tfix supplied
        if (beta is not None) and (tfix is not None):
            # store params
            self.beta = beta
            self.tfix = tfix
            return

        raise ValueError(
            "Must supply arrays for calibration or calibration " "parameters."
        )

    def arrival2ccs(self, mz, ta, q=1):
        """
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

        """

        # Check for required attributes
        self._check()

        # Cast to numpy array
        mz = np.array(mz)
        ta = np.array(ta)
        q = np.array(q)

        # Derived variables
        gamma = np.sqrt(mz * q / (mz * q + self.buffer_mass)) / q

        # Power model
        if self.power:
            return np.exp((np.log(ta) - self.tfix) / self.beta) / gamma

        # Linear model
        return (ta - self.tfix) / (self.beta * gamma)

    def ccs2arrival(self, mz, ccs, q=1):
        """
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

        """

        # Check for required attributes
        self._check()

        # Cast to numpy array
        mz = np.array(mz)
        ccs = np.array(ccs)
        q = np.array(q)

        # Derived variables
        gamma = np.sqrt(mz * q / (mz * q + self.buffer_mass)) / q

        # Power model
        if self.power:
            return np.exp(self.beta * np.log(gamma * ccs) + self.tfix)

        # Linear model
        else:
            return self.beta * gamma * ccs + self.tfix


def calibrate_ccs(
    mz=None,
    ta=None,
    ccs=None,
    q=None,
    beta=None,
    tfix=None,
    buffer_mass=28.013,
    power=False,
):
    """
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
    power : bool
        Indicate whether to use linearize power function for calibration,
        i.e. in traveling wave ion moblility spectrometry.

    Returns
    -------
    :obj:`~deimos.calibration.CCSCalibration`
        Instance of calibrated `~deimos.calibration.CCSCalibration`
        object.

    """

    # Initialize calibration instance
    ccs_cal = CCSCalibration()

    # Perform calibration
    ccs_cal.calibrate(
        mz=mz,
        ta=ta,
        ccs=ccs,
        q=q,
        beta=beta,
        tfix=tfix,
        buffer_mass=buffer_mass,
        power=power,
    )

    return ccs_cal


def tunemix(
    features,
    mz=[112.985587, 301.998139, 601.978977, 1033.988109, 1333.968947, 1633.949786],
    ccs=[108.23, 140.04, 180.77, 255.34, 284.76, 319.03],
    q=[1, 1, 1, 1, 1, 1],
    buffer_mass=28.013,
    mz_tol=200e-6,
    dt_tol=0.04,
    power=False,
):
    """
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
    power : bool
        Indicate whether to use linearize power function for calibration,
        i.e. in traveling wave ion moblility spectrometry.

    Returns
    -------
    :obj:`~deimos.calibration.CCSCalibration`
        Instance of calibrated `~deimos.calibration.CCSCalibration`
        object.

    """

    # Cast to numpy array
    mz = np.array(mz)
    ccs = np.array(ccs)
    q = np.array(q)

    # Check lengths
    deimos.utils.check_length([mz, ccs, q])

    # Iterate tune ions
    ta = []
    for mz_i, ccs_i, q_i in zip(mz, ccs, q):
        # Slice ms1
        subset = deimos.slice(
            features, by="mz", low=mz_i - 0.1 * mz_tol, high=mz_i + mz_i * 0.9 * mz_tol
        )

        # Extract dt info
        dt_profile = deimos.collapse(subset, keep="drift_time")
        dt_i = dt_profile.sort_values(by="intensity", ascending=False)[
            "drift_time"
        ].values[0]
        dt_profile = deimos.locate(
            dt_profile, by="drift_time", loc=dt_i, tol=dt_tol * dt_i
        ).sort_values(by="drift_time")

        # X and Y arrays
        x = dt_profile["drift_time"].values
        y = dt_profile["intensity"].values

        # Interpolate spline
        spl = interp1d(x, y, kind="quadratic")

        # Higher resolution x-axis
        newx = np.arange(x.min(), x.max(), 0.001)

        # Evaluate
        newy = spl(newx)

        # Take argmax
        dt_j = newx[np.argmax(newy)]
        ta.append(dt_j)

    # Calibrate
    ta = np.array(ta)
    return deimos.calibration.calibrate_ccs(
        mz=mz, ta=ta, ccs=ccs, q=q, buffer_mass=buffer_mass, power=power
    )


def parse_agilent_calibration_pdf(path, mode="pos", tol=15e-3):
    """
    Reads calibration information from Agilent PDF.

    Parameters
    ----------
    path : str
        Path to PDF file.
    mode : str
        Ionization mode.
    tol : float
        m/z tolerance to identify calibrant ions.

    Returns
    -------
    tof_cal : :obj:`~pandas.DataFrame`
        Calibration data for the time-of-flight instrument.
    im_cal : :obj:`~pandas.DataFrame`
        Calibration data for the ion mobility instrument.

    """

    def map_mz_to_ccs(mz, mode="pos", tol=15e-3):
        masses = np.array(AGILENT_CCS_REFERENCE[mode]["mz"])
        ccss = np.array(AGILENT_CCS_REFERENCE[mode]["ccs"])

        dists = np.abs(mz - masses)

        if np.min(dists) > tol:
            return np.nan

        return ccss[np.argmin(dists)]

    if "pos" in mode.lower():
        mode = "pos"
    elif "neg" in mode.lower():
        mode = "neg"

    # Read PDF tables
    with warnings.catch_warnings(action="ignore"):
        dfs = tabula.read_pdf(path, pages="all")

    # IM calibration
    im_cal = dfs[0].loc[2:].reset_index(drop=True)
    im_cal.columns = [
        "theoretical",
        "actual",
        "tof abundance",
        "tof resolution",
        "corrected residuals",
        "im drift time (ms)",
        "im abundance",
        "im resolution",
    ]

    # Replace commas
    for col in ["tof abundance", "tof resolution", "im abundance"]:
        im_cal[col] = im_cal[col].str.replace(",", "").astype(float)

    # Cast as float
    for col in [
        "theoretical",
        "actual",
        "corrected residuals",
        "im drift time (ms)",
        "im resolution",
    ]:
        im_cal[col] = im_cal[col].astype(float)

    # TOF calibration
    tmp = dfs[1].loc[2:].reset_index(drop=True)
    tof_cal = tmp["TOF Mass Calibration Data"].str.split(" ", n=2, expand=True)
    tof_cal.columns = ["theoretical", "actual", "time"]

    # Rename botched col names
    for src_col, target_col in zip(
        tmp.columns[1:],
        [
            "abundance",
            "calibration abundance",
            "resolution",
            "primary residuals",
            "corrected residuals",
        ],
    ):
        tof_cal[target_col] = tmp[src_col].str.replace(",", "").astype(float)

    # Cast as float
    for col in ["theoretical", "actual", "time"]:
        tof_cal[col] = tof_cal[col].astype(float)

    # Add ionization mode column
    im_cal["mode"] = mode
    tof_cal["mode"] = mode

    # Add CCS values to IM calibration
    im_cal["ccs"] = [
        map_mz_to_ccs(x, mode=mode, tol=tol) for x in im_cal["theoretical"].values
    ]

    # Drop missing
    tof_cal = tof_cal.dropna(axis=0).reset_index(drop=True)
    im_cal = im_cal.dropna(axis=0).reset_index(drop=True)

    return tof_cal, im_cal


def calibrate_ccs_agilent_pdf(path, mode="pos", tol=15e-3):
    """
    Reads calibration information from Agilent PDF and performs CCS calibration.

    Parameters
    ----------
    path : str
        Path to PDF file.
    mode : str
        Ionization mode.
    tol : float
        m/z tolerance to identify calibrant ions.

    Returns
    -------
    :obj:`~deimos.calibration.CCSCalibration`
        Instance of calibrated `~deimos.calibration.CCSCalibration`
        object.

    """

    _, im_cal = parse_agilent_calibration_pdf(path, mode=mode, tol=tol)

    return calibrate_ccs(
        mz=im_cal["actual"].values,
        ta=im_cal["im drift time (ms)"].values,
        ccs=im_cal["ccs"].values,
        q=np.ones(len(im_cal.index)),
        buffer_mass=28.013,
        power=False,
    )
