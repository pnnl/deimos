import warnings

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import linregress

import deimos


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
                  beta=None, tfix=None, buffer_mass=28.013,
                  power=False):
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
        power : bool
            Indicate whether to use linearize power function for calibration,
            i.e. in traveling wave ion moblility spectrometry.

        Raises
        ------
        ValueError
            If appropriate keyword arguments are not supplied (either `mz`,
            `ta`, `ccs`, and `q`; or `beta` and `tfix`).

        '''

        # buffer mass
        self.buffer_mass = buffer_mass

        # power function indicator
        self.power = power

        # calibrant arrays supplied
        if (mz is not None) and (ta is not None) and (ccs is not None) \
           and (q is not None):
            self.mz = np.array(mz)
            self.ta = np.array(ta)
            self.ccs = np.array(ccs)
            self.q = np.array(q)

            # derived variables
            self.gamma = np.sqrt(
                self.mz * self.q / (self.mz * self.q + self.buffer_mass)) / self.q
            self.reduced_ccs = self.ccs * self.gamma

            # linear regression
            if self.power:
                beta, tfix, r, p, se = linregress(np.log(self.reduced_ccs),
                                                  np.log(self.ta))
            else:
                beta, tfix, r, p, se = linregress(self.reduced_ccs,
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

        # derived variables
        gamma = np.sqrt(mz * q / (mz * q + self.buffer_mass)) / q

        if self.power:
            return np.exp((np.log(ta) - self.tfix) / self.beta) / gamma

        return (ta - self.tfix) / (self.beta * gamma)

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

        # derived variables
        gamma = np.sqrt(mz * q / (mz * q + self.buffer_mass)) / q

        if self.power:
            return np.exp(self.beta * np.log(gamma * ccs) + self.tfix)
        else:
            return self.beta * gamma * ccs + self.tfix


def calibrate_ccs(mz=None, ta=None, ccs=None, q=None,
                  beta=None, tfix=None, buffer_mass=28.013, power=False):
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
    power : bool
        Indicate whether to use linearize power function for calibration,
        i.e. in traveling wave ion moblility spectrometry.

    Returns
    -------
    :obj:`~deimos.calibration.CCSCalibration`
        Instance of calibrated `~deimos.calibration.CCSCalibration`
        object.

    Raises
    ------
    ValueError
        If appropriate keyword arguments are not supplied (either `mz`,
        `ta`, `ccs`, and `q`; or `beta` and `tfix`).

    '''

    ccs_cal = CCSCalibration()
    ccs_cal.calibrate(mz=mz, ta=ta, ccs=ccs, q=q, beta=beta, tfix=tfix,
                      buffer_mass=buffer_mass, power=power)

    return ccs_cal


def tunemix(features,
            mz=[112.985587, 301.998139, 601.978977,
                1033.988109, 1333.968947, 1633.949786],
            ccs=[108.4, 139.8, 179.9, 254.2, 283.6, 317.7],
            q=[1, 1, 1, 1, 1, 1], buffer_mass=28.013, mz_tol=200E-6, dt_tol=0.04,
            power=False):
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
    power : bool
        Indicate whether to use linearize power function for calibration,
        i.e. in traveling wave ion moblility spectrometry.

    Returns
    -------
    :obj:`~deimos.calibration.CCSCalibration`
        Instance of calibrated `~deimos.calibration.CCSCalibration`
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
        dt_i = dt_profile.sort_values(by='intensity', ascending=False)[
            'drift_time'].values[0]
        dt_profile = deimos.locate(
            dt_profile, by='drift_time', loc=dt_i, tol=dt_tol * dt_i).sort_values(by='drift_time')

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
    return deimos.calibration.calibrate_ccs(mz=mz, ta=ta, ccs=ccs, q=q, buffer_mass=buffer_mass,
                                            power=power)


def cosine(a, b):
    '''
    Cosine distance (1 - similarity) between two arrays.

    Parameters
    ----------
    a, b : :obj:`~numpy.array`
        N-dimensional arrays.

    Returns
    -------
    float
        Cosine distance.

    '''

    a_ = a.flatten()
    b_ = b.flatten()
    return 1 - np.dot(a_, b_) / np.sqrt(a_.dot(a_) * b_.dot(b_))


def objective(x, src, target, order, loss):
    '''
    Objective function for contrained (i.e. independent per dimension) affine
    transformation.

    Parameters
    ----------
    x : :obj:`~numpy.array`
        Array of affine tranformation parameters (constrained).
    src : :obj:`~numpy.array`
        Array to be transformed.
    target : :obj:`~numpy.array`
        Array against which the affine tranformation is evaluated.
    order : int
        Order of interpolation used during the affine transformation.
    loss : func
        Loss function that takes two arrays as input, returning a measure
        of distance (e.g. cosine).

    Returns
    -------
    float
        Objective loss between transformed `src` and `target`.

    '''

    # identity transform
    A = np.eye(src.ndim, src.ndim + 1)

    # fill diagonal (scale)
    np.fill_diagonal(A, x[:src.ndim])

    # fill last column (offset)
    A[:, -1] = x[src.ndim:]

    # transform
    transformed = ndi.affine_transform(src, A,
                                       output_shape=target.shape,
                                       prefilter=False, order=order)

    # suppress warnings
    # (because division error common)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # compute loss
        return loss(target, transformed)


def rescale(a, src=[0, 1], target=[0, 1]):
    '''
    Rescale array values based on ranges in source and target.

    Parameters
    ----------
    a : :obj:`~numpy.array`
        N-dimensional array.
    src : list of float
        Range of source values.
    target : list of float
        Range of target values.

    Returns
    -------
    :obj:`~numpy.array`
        Scaled N-dimensional array.

    Raises
    ------
    ValueError
        If `src` and `target` are not length 2.

    '''

    # check ranges
    if not all(len(x) == 2 for x in [src, target]):
        raise ValueError('Source and target ranges must be length 2.')

    # scale
    return (target[1] - target[0]) * (a - src[0]) / (src[1] - src[0]) + target[0]


class MS2OffsetCalibration:
    '''
    Performs MS2 offset calibration and stores result to enable convenient application.

    Attributes
    ----------
    bounds : dict
        Bounds of each dimension for each MS level.
    res : :obj:`~scipy.optimize.OptimizeResult`
        The optimization result. See `~scipy.optimize.OptimizeResult`.
    A : :obj:`~numpy.array`
        Affine transformation matrix.

    '''

    def __init__(self, ms1, ms2, dims=['drift_time', 'retention_time']):
        '''
        Initializes :obj:`~deimos.calibration.MS2OffsetCalibration` object.

        Parameters
        ----------
        ms1, ms2 : :obj:`~pandas.DataFrame`
            Input feature coordinates and intensities for MS1 and MS2,
            respectively.
        dims : str or list
            Dimension(s) to perform the affine transform in (omitted dimensions 
            will be collapsed and summed across).

        '''

        # placeholder
        self.res = None

        # store dims
        self.dims = deimos.utils.safelist(dims)
        self.ndim = len(self.dims)
        self.shape = (self.ndim, self.ndim + 1)

        # get dimension bounds
        self.bounds = {}
        self.bounds['ms1'] = [[ms1[dim].min(), ms1[dim].max()]
                              for dim in self.dims]
        self.bounds['ms2'] = [[ms2[dim].min(), ms2[dim].max()]
                              for dim in self.dims]

        # shared grid space
        _, self.ms1 = deimos.grid.data2grid(deimos.collapse(ms1, keep=self.dims),
                                            dims=self.dims)
        _, self.ms2 = deimos.grid.data2grid(deimos.collapse(ms2, keep=self.dims),
                                            dims=self.dims)

    def calibrate(self, normalize=None, order=1, loss=cosine, method='BFGS', X0=None):
        '''
        Perform the affine transform to align MS1 and MS2.

        Parameters
        ----------
        normalize : func
            Function to perform normalization of input data (certain loss
            functions are sensitive to unnormalized data).
        order : int
            Order of interpolation used during the affine transformation.
        loss : func
            Loss function that takes two arrays as input, returning a measure
            of distance (e.g. cosine).
        method : str
            Optimization solver to use (recommend 'BFGS' or 'Powell'). See
            `method` argument of `scipy.optimize.minimize`.
        X0 : :obj:`~numpy.array`
            Initial solution guess. Array of real elements of size (n,),
            where n is the number of independent variables. Defaults to
            the identity transformation if None.

        Returns
        -------
        :obj: `~scipy.optimize.OptimizeResult`
            The optimization result. See `~scipy.optimize.OptimizeResult`.

        '''

        # normalize if function provided
        if normalize is not None:
            self.ms1 = normalize(self.ms1)
            self.ms2 = normalize(self.ms2)

        # set X0 to identity if not provided
        if X0 is None:
            X0 = np.eye(self.ndim, self.ndim + 1)

        # get diagonal
        diag = list(np.diagonal(X0))

        # get last column
        offset = list(X0[:, -1])

        # constrained X0
        X0 = diag + offset

        # run optimization
        self.res = minimize(objective,
                            X0,
                            args=(self.ms2,
                                  self.ms1,
                                  order,
                                  loss),
                            method=method)

        # construct affine matrix
        A_grid = np.eye(self.ndim, self.ndim + 1)

        # fill diagonal (scale)
        np.fill_diagonal(A_grid, self.res.x[:self.ndim])

        # fill last column (offset)
        A_grid[:, -1] = self.res.x[self.ndim:]

        # create "full" matrix
        bottom_row = np.zeros((1, self.ndim + 1))
        bottom_row[0, -1] = 1
        A_grid = np.append(A_grid, bottom_row, axis=0)

        # map to measurement coordinate system
        A = A_grid.copy()
        for i, dim in enumerate(self.dims):
            A[i, -1] = rescale(A[i, -1],
                               src=[0, self.ms1.shape[i]],
                               target=self.bounds['ms1'][i])

        # invert
        self.A = np.linalg.inv(A)

        return self.res

    def apply(self, ms2):
        '''
        Applies calibration to correct MS2 offset according to fit
        calibration parameters.

        Parameters
        ----------
        ms2 : :obj:`~pandas.DataFrame`
            Input feature coordinates and intensities to transform.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            Transformed feature coordinates and intensities.

        '''

        # copy input df
        ms2_ = ms2.copy()

        # relevant columns as array
        arr = ms2_[self.dims].values

        # add offset dim to array
        arr = np.hstack([arr,
                         np.ones((len(ms2.index), 1))])

        # transform to ms1 index coords
        arr = np.dot(arr, self.A.T)[:, :-1]

        # overwrite columns in ms2
        ms2_[self.dims] = arr

        return ms2_


def calibrate_ms2(ms1, ms2, dims=['drift_time', 'retention_time'],
                  normalize=None, order=1, loss=cosine, method='BFGS',
                  X0=None):
    '''
    Convenience function for :class:`~deimos.calibration.MS2OffsetCalibration`.

    Parameters
    ----------
    ms1, ms2 : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities for MS1 and MS2,
        respectively.
    dims : str or list
        Dimension(s) to perform the affine transform in (omitted dimensions 
        will be collapsed and summed across).
    normalize : func
        Function to perform normalization of input data (certain loss
        functions are sensitive to unnormalized data).
    order : int
        Order of interpolation used during the affine transformation.
    loss : func
        Loss function that takes two arrays as input, returning a measure
        of distance (e.g. cosine).
    method : str
        Optimization solver to use (recommend 'BFGS' or 'Powell'). See
        `method` argument of `scipy.optimize.minimize`.
    X0 : :obj:`~numpy.array`
        Initial solution guess. Array of real elements of size (n,),
        where n is the number of independent variables. Defaults to
        the identity transformation if None.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Transformed feature coordinates and intensities.

    '''

    # initialize
    ms2_cal = MS2OffsetCalibration(ms1, ms2, dims=dims)

    # calibrate
    ms2_cal.calibrate(normalize=normalize, order=order, loss=loss,
                      method=method, X0=X0)

    # apply
    return ms2_cal.apply(ms2)
