import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.spatial import KDTree

import deimos


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


def get_1D_profiles(features, dims=['mz', 'drift_time', 'retention_time']):
    '''
    Extract 1D profile for each of the indicated dimension(s).

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    dims : str or list
        Dimensions considered in generating 1D profile(s).

    Returns
    -------
    :obj:`dict` of :obj:`~scipy.interpolate.UnivariateSpline`
        Dictionary indexed by dimension containing univariate
        splines for each 1D profile.

    '''

    # Safely cast to list
    dims = deimos.utils.safelist(dims)

    profiles = {}
    for dim in dims:
        # Collapse to 1D profile
        profile = deimos.collapse(features, keep=dim).sort_values(
            by=dim, ignore_index=True)

        # Interpolate spline
        x = profile[dim].values
        y = profile['intensity'].values

        # Fit univariate spline
        try:
            uspline = UnivariateSpline(x, y, s=0, ext=1)
        except:
            def uspline(x): return np.zeros_like(x)

        profiles[dim] = uspline

    return profiles


def offset_correction_model(dt_ms2, mz_ms2, mz_ms1, ce=0,
                            params=[1.02067031, -0.02062323,  0.00176694]):
    # Cast params as array
    params = np.array(params).reshape(-1, 1)
    
    # Convert collision energy to array
    ce = np.ones_like(dt_ms2) * np.log(ce)
    
    # Create constant vector
    const = np.ones_like(dt_ms2)
    
    # Sqrt
    mu_ms1 = np.sqrt(mz_ms1)
    mu_ms2 = np.sqrt(mz_ms2)
    
    # Ratio
    mu_ratio = mu_ms2 / mu_ms1
    
    # Create dependent array
    x = np.stack((const, mu_ratio, ce), axis=1)
    
    # Predict
    y = np.dot(x, params).flatten() * dt_ms2
    
    return y


class MS2Deconvolution:
    '''
    Performs MS2 deconvolution by correlating non-m/z separation dimension
    profiles and scoring the agreement between precursor and fragment.

    '''

    def __init__(self, ms1_features, ms1_data, ms2_features, ms2_data):
        '''
        Initializes :obj:`~deimos.calibration.ArrivalTimeCalibration` object.

        Parameters
        ----------
        ms1_features : :obj:`~pandas.DataFrame`
            MS1 peak locations and intensities.
        ms1_data : :obj:`~pandas.DataFrame`
            Complete MS1 data.
        ms2_features : :obj:`~pandas.DataFrame`
            MS2 peak locations and intensities.
        ms2_data : :obj:`~pandas.DataFrame`
            Complete MS1 data.

        '''

        self.ms1_features = ms1_features
        self.ms1_data = ms1_data
        self.ms2_features = ms2_features
        self.ms2_data = ms2_data

    def construct_putative_pairs(self, dims=['drift_time', 'retention_time'],
                                 low=[-0.12, -0.1], high=[1.4, 0.1], ce=None,
                                 model=offset_correction_model,
                                 require_ms1_greater_than_ms2=True,
                                 error_tolerance=0.12):
        '''
        Determine each possible MS1:MS2 pair within specified tolerances. If
        considering drift time, apply a model to correct for drift time 
        offset such that MS2 can be downselected by respective error in drift
        time (i.e. a poor model correction suggests an incorrect MS1:MS2 pair).

        Parameters
        ----------
        dims : str or list
            Dimension(s) by which to match MS1:MS2.
        low : float or list
            Lower bound(s) in each dimension.
        high : float or list
            Upper bound(s) in each dimension.
        ce : float
            Collision energy of MS2 collection.
        model : function
            Model used to correct for drift time offset between MS1 precursor
            and MS2 fragment. If omitted, a default model is used.
        require_ms1_greater_than_ms2 : bool
            Signal whether precursor intensity must be greater than fragment
            intensity for putative assignments.
        error_tolerance : float
            Acceptable difference between precursor and fragment drift times
            following correction by offset model.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            All MS1:MS2 pairings and associated agreement scores per requested dimension.

        '''

        # Safely cast to list
        dims = deimos.utils.safelist(dims)
        low = deimos.utils.safelist(low)
        high = deimos.utils.safelist(high)

        # Check dims
        deimos.utils.check_length([dims, low, high])

        # Check collision energy is supplied
        if ce is None:
            raise ValueError("Collision energy must be specified.")

        # Match vectors
        v_ms1 = self.ms1_features[dims].values
        v_ms2 = self.ms2_features[dims].values

        # Normalize dims for query
        for i, dim in enumerate(dims):
            # Cast range as radius
            tol = (high[i] - low[i]) / 2

            # Offset to center search
            v_ms2[:, i] = v_ms2[:, i] + tol + low[i]

            # Normalize
            v_ms1[:, i] = v_ms1[:, i] / tol
            v_ms2[:, i] = v_ms2[:, i] / tol

        # Create k-d trees
        ms1_tree = KDTree(v_ms1)
        ms2_tree = KDTree(v_ms2)

        # Query
        sdm = ms1_tree.sparse_distance_matrix(
            ms2_tree, 1, p=np.inf, output_type='coo_matrix')

        # Pairs within tolerance
        ms1_matches = self.ms1_features.loc[sdm.row, :].reset_index()
        ms2_matches = self.ms2_features.loc[sdm.col, :].reset_index()

        # Correct drift time
        if 'drift_time' in dims:
            ms2_matches['drift_time_raw'] = ms2_matches['drift_time'].values
            ms2_matches['drift_time'] = model(ms2_matches['drift_time'].values,
                                              ms2_matches['mz'].values,
                                              ms1_matches['mz'].values, ce=ce)

        # Rename columns
        ms1_matches.columns = [x + '_ms1' for x in ms1_matches.columns]
        ms2_matches.columns = [x + '_ms2' for x in ms2_matches.columns]

        # Construct MS1:MS2 data frame
        decon_pairs = pd.concat((ms1_matches,
                                 ms2_matches), axis=1)

        # Compute offset correction error
        if 'drift_time' in dims:
            decon_pairs['drift_time_error'] = np.abs(
                decon_pairs['drift_time_ms1'] - decon_pairs['drift_time_ms2'])

        # MS1 intensity greater than MS2 intensity
        if require_ms1_greater_than_ms2 is True:
            decon_pairs = decon_pairs.loc[decon_pairs['intensity_ms1']
                                          >= decon_pairs['intensity_ms2'], :]

        # Error tolerance
        if 'drift_time' in dims:
            decon_pairs = decon_pairs.loc[decon_pairs['drift_time_error']
                                          <= error_tolerance, :]

        # Remove empty groups
        decon_pairs = decon_pairs.groupby(
            by='index_ms1', as_index=False).filter(lambda x: len(x) > 0)

        # Sort by index
        decon_pairs = decon_pairs.sort_values(
            by=['index_ms1', 'index_ms2']).reset_index(drop=True)

        self.decon_pairs = decon_pairs

        return self.decon_pairs

    def configure_profile_extraction(self, dims=['mz', 'drift_time', 'retention_time'],
                                     low=[-100E-6, -0.05, -0.3], high=[400E-6, 0.05, 0.3],
                                     relative=[True, True, False]):
        '''
        Configure parameters to generate extracted ion subsets of the data.

        Parameters
        ----------
        dims : str or list
            Dimension(s) by which to subset the data.
        low : float or list
            Lower tolerance(s) in each dimension.
        high : float or list
            Upper tolerance(s) in each dimension.
        relative : bool or list
            Whether to use relative or absolute tolerance per dimension.

        '''

        def abstract_fxn(features, data, dims=None, low=None, high=None, relative=None):
            '''
            Function abstraction to bake in asymmetrical tolerances at runtime.

            '''

            res = []
            for i, row in features.iterrows():
                subset = deimos.locate_asym(data, by=dims, loc=row[dims].values,
                                            low=low, high=high, relative=relative)
                res.append(subset)

            return res

        # Safely cast to list
        dims = deimos.utils.safelist(dims)
        low = deimos.utils.safelist(low)
        high = deimos.utils.safelist(high)
        relative = deimos.utils.safelist(relative)

        # Check dims
        deimos.utils.check_length([dims, low, high, relative])

        # Recast as dictionaries indexed by dimension
        self.profile_low = {k: v for k, v in zip(dims, low)}
        self.profile_high = {k: v for k, v in zip(dims, high)}
        self.profile_relative = {k: v for k, v in zip(dims, relative)}

        # Construct pre-configured profile extraction function
        self.profiler = lambda x, y: abstract_fxn(
            x, y, dims=dims, low=low, high=high, relative=relative)

    def apply(self, dims=['drift_time', 'retention_time'], resolution=[0.01, 0.01]):
        '''
        Perform deconvolution according to mathed features and their
        extracted profiles.

        Parameters
        ----------
        dims : : str or list
            Dimensions(s) for which to calculate MS1:MS2 correspondence
            by 1D profile agreement (i.e. non-m/z).
        resolution : float or list
            Resolution applied to per-dimension profile interpolations.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            All MS1:MS2 pairings and associated agreement scores.

        '''
        # Safely cast to list
        dims = deimos.utils.safelist(dims)
        resolution = deimos.utils.safelist(resolution)

        # Check dims
        deimos.utils.check_length([dims, resolution])

        # Ensure m/z not in dims
        if 'mz' in dims:
            raise ValueError('MS1:MS2 similarity does not consider m/z.')

        # Recast as dictionaries indexed by dimension
        self.profile_resolution = {k: v for k, v in zip(dims, resolution)}

        # Extracted ions
        ms1_xis = self.profiler(self.ms1_features, self.ms1_data)
        ms2_xis = self.profiler(self.ms2_features, self.ms2_data)

        # Container for profile similarity scores
        scores = {dim: [] for dim in dims}

        # Enumerate MS1 featres
        for name, grp in self.decon_pairs.groupby(by=['index_ms1'], as_index=False):
            # MS1 feature index
            idx_i = int(name[0])

            # Extracted ion
            ms1_xi = ms1_xis[idx_i]

            # Build MS1 profile
            ms1_profiles = get_1D_profiles(ms1_xi, dims=dims)

            # Shared interpolation axis
            newx = {}
            for dim in dims:
                # Determine upper and lower bounds
                if self.profile_relative[dim] is True:
                    lb = min(grp[dim + "_ms1"].min(), grp[dim +
                             "_ms2"].min()) * (1 + self.profile_low[dim])
                    ub = max(grp[dim + "_ms1"].max(), grp[dim +
                             "_ms2"].max()) * (1 + self.profile_high[dim])
                else:
                    lb = min(grp[dim + "_ms1"].min(), grp[dim +
                             "_ms2"].min()) + self.profile_low[dim]
                    ub = max(grp[dim + "_ms1"].max(), grp[dim +
                             "_ms2"].max()) + self.profile_high[dim]

                # Determine shared x-axis
                newx[dim] = np.arange(lb, ub, self.profile_resolution[dim])

            # Evaluate MS1 profile
            ms1_profiles = {dim: ms1_profiles[dim](newx[dim]) for dim in dims}

            # Enumerate MS2 features
            for j, row in grp.reset_index().iterrows():
                # MS2 feature index
                idx_j = int(row['index_ms2'])

                # Extracted ion
                ms2_xi = ms2_xis[idx_j].copy()

                if 'drift_time' in dims:
                    # Determine offset
                    offset = row['drift_time_ms2'] - row['drift_time_raw_ms2']

                    # Apply ofset
                    ms2_xi['drift_time'] += offset

                # Build MS2 profile
                ms2_profiles = get_1D_profiles(ms2_xi, dims=dims)

                # Evaluate MS2 profile
                ms2_profiles = {dim: ms2_profiles[dim](
                    newx[dim]) for dim in dims}

                # Compute similarity
                for dim in dims:
                    scores[dim].append(
                        1 - cosine(ms1_profiles[dim], ms2_profiles[dim]))

        # Append score columns
        for dim in dims:
            self.decon_pairs[dim + '_score'] = np.array(scores[dim])

        return self.decon_pairs
