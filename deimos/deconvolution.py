import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist

import deimos


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

    # safely cast to list
    dims = deimos.utils.safelist(dims)

    profiles = {}
    for dim in dims:
        # collapse to 1D profile
        profile = deimos.collapse(features, keep=dim).sort_values(
            by=dim, ignore_index=True)

        # interpolate spline
        x = profile[dim].values
        y = profile['intensity'].values

        # fit univariate spline
        try:
            uspline = UnivariateSpline(x, y, s=0, ext=3)
        except:
            def uspline(x): return np.zeros_like(x)

        profiles[dim] = uspline

    return profiles


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

        self.ms1_features['ms_level'] = 1
        self.ms2_features['ms_level'] = 2

    def cluster(self, dims=['drift_time', 'retention_time'],
                tol=[0.1, 0.3], relative=[True, False]):
        '''
        Performs clustering in deconvolution dimensions in MS1 and MS2
        simultaneously.

        Parameters
        ----------
        dims : : str or list
            Dimensions(s) by which to cluster the data (i.e. non-m/z).
        tol : float or list
            Tolerance in each dimension to define maximum cluster linkage
            distance.
        relative : bool or list
            Whether to use relative or absolute tolerances per dimension.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            Features concatenated over MS levels with cluster labels.

        '''

        clusters = deimos.alignment.agglomerative_clustering(pd.concat((self.ms1_features,
                                                                        self.ms2_features),
                                                                       ignore_index=True,
                                                                       axis=0),
                                                             dims=dims,
                                                             tol=tol,
                                                             relative=relative)
        self.clusters = clusters
        return self.clusters

    def configure_profile_extraction(self, dims=['mz', 'drift_time', 'retention_time'],
                                     low=[-100E-6, -0.05, -0.3], high=[400E-6, 0.05, 0.3],
                                     relative=[True, True, False], resolution=[0.01, 0.01, 0.01]):
        '''
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
        resolution : float or list
            Resolution applied to per-dimension profile interpolations.

        '''

        def abstract_fxn(features, data, dims=None, low=None, high=None, relative=None):
            '''
            Function abstraction to bake in asymmetrical tolerances at runtime.

            '''

            res = []
            for i, row in features.iterrows():
                subset = deimos.locate_asym(data, by=dims, loc=row[dims].values,
                                            low=low, high=high, relative=relative)
                profiles = get_1D_profiles(subset, dims=dims)
                res.append(profiles)

            return res

        # safely cast to list
        dims = deimos.utils.safelist(dims)
        low = deimos.utils.safelist(low)
        high = deimos.utils.safelist(high)
        relative = deimos.utils.safelist(relative)
        resolution = deimos.utils.safelist(resolution)

        # check dims
        deimos.utils.check_length([dims, low, high, relative, resolution])

        # recast as dictionaries indexed by dimension
        self.profile_low = {k: v for k, v in zip(dims, low)}
        self.profile_high = {k: v for k, v in zip(dims, high)}
        self.profile_relative = {k: v for k, v in zip(dims, relative)}
        self.profile_resolution = {k: v for k, v in zip(dims, resolution)}

        # construct pre-configured profile extraction function
        self.profiler = lambda x, y: abstract_fxn(
            x, y, dims=dims, low=low, high=high, relative=relative)

    def apply(self, dims=['drift_time', 'retention_time']):
        '''
        Perform deconvolution according to clustered features and their
        extracted profiles.

        Parameters
        ----------
        dims : : str or list
            Dimensions(s) for which to calculate MS1:MS2 correspondence
            by 1D profile agreement (i.e. non-m/z).

        Returns
        -------
        :obj:`~pandas.DataFrame`
            All MS1:MS2 pairings and associated agreement scores.

        '''

        # safely cast to list
        dims = deimos.utils.safelist(dims)

        # initialize result container
        decon = []

        # enumerate clusters
        for name, grp in self.clusters.groupby('cluster'):
            # group by ms level
            ms1_peaks_subset = grp.loc[grp['ms_level'] == 1, :].drop(
                columns=['ms_level', 'cluster'])
            ms2_peaks_subset = grp.loc[grp['ms_level'] == 2, :].drop(
                columns=['ms_level', 'cluster'])

            # sort by intensity
            ms1_peaks_subset = ms1_peaks_subset.sort_values(
                by='intensity', ascending=False)
            ms2_peaks_subset = ms2_peaks_subset.sort_values(
                by='intensity', ascending=False)

            # filter duplicate masses
            ms1_peaks_subset = ms1_peaks_subset.drop_duplicates(
                subset='mz').reset_index(drop=True)
            ms2_peaks_subset = ms2_peaks_subset.drop_duplicates(
                subset='mz').reset_index(drop=True)

            if (len(ms1_peaks_subset.index) > 0) & (len(ms2_peaks_subset.index) > 0):
                # extract 1d profiles
                ms1_profiles = self.profiler(ms1_peaks_subset, self.ms1_data)
                ms2_profiles = self.profiler(ms2_peaks_subset, self.ms2_data)

                # determine possible MS1:MS2 pairings
                combos = np.array(np.meshgrid(
                    ms1_peaks_subset.index, ms2_peaks_subset.index)).T.reshape(-1, 2)

                # rename columns
                ms1_peaks_subset.columns = [
                    x + '_ms1' for x in ms1_peaks_subset.columns]
                ms2_peaks_subset.columns = [
                    x + '_ms2' for x in ms2_peaks_subset.columns]

                # construct MS1:MS2 data frame
                res = pd.concat((ms1_peaks_subset.loc[combos[:, 0], :].reset_index(drop=True),
                                 ms2_peaks_subset.loc[combos[:, 1], :].reset_index(drop=True)), axis=1)

                # score MS1:MS2 assignments per dimension
                for dim in dims:
                    if self.profile_relative[dim] is True:
                        lb = grp[dim].min() * (1 + self.profile_low[dim])
                        ub = grp[dim].max() * (1 + self.profile_high[dim])
                    else:
                        lb = grp[dim].min() + self.profile_low[dim]
                        ub = grp[dim].max() + self.profile_high[dim]

                    newx = np.arange(lb, ub, self.profile_resolution[dim])

                    v_ms1 = np.vstack([x[dim](newx) for x in ms1_profiles])
                    v_ms2 = np.vstack([x[dim](newx) for x in ms2_profiles])

                    # similarity matrix

                    H = 1 - cdist(v_ms1, v_ms2, metric='cosine')

                    # add column
                    res[dim + '_score'] = H.reshape(-1, 1)

                # append to container
                decon.append(res)

        # combine and return
        return pd.concat(decon, ignore_index=True)


def deconvolve_ms2(ms1_features, ms1_data, ms2_features, ms2_data,
                   cluster_kwargs, profile_kwargs, apply_kwargs):
    '''
    Convenience function to perform all necessary deconvolution steps.

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
    cluster_kwargs : :obj:`dict`
        Dictionary of keyword arguments for clustering
        (see :meth:`~deimos.deconvolution.MS2Deconvolution.cluster`).
    profile_kwargs : :obj:`dict`
        Dictionary of keyword arguments for profile extraction
        (see :meth:`~deimos.deconvolution.MS2Deconvolution.configure_profile_extraction`).
    apply_kwargs : :obj:`dict`
        Dictionary of keyword arguments for applying deconvolution
        (see :meth:`~deimos.deconvolution.MS2Deconvolution.apply`).

    Returns
    -------
    :obj:`~pandas.DataFrame`
        All MS1:MS2 pairings and associated agreement scores per requested dimension.

    '''

    # init
    decon = MS2Deconvolution(ms1_features, ms1_data, ms2_features, ms2_data)

    # cluster
    decon.cluster(**cluster_kwargs)

    # configure profiling
    decon.configure_profile_extraction(**profile_kwargs)

    # decon
    return decon.apply(**apply_kwargs)
