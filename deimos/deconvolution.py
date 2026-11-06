from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.spatial import KDTree

import deimos


def cosine(a, b, axis=None):
    """
    Cosine distance (1 - similarity) between two arrays.

    By default `a` and `b` are flattened and compared as one vector each.
    With `axis`, the arrays are broadcast against each other and compared
    along that axis, so a 2D array of profiles against one 1D profile gives
    one distance per row.

    Parameters
    ----------
    a, b : :obj:`~numpy.array`
        N-dimensional arrays. With `axis`, they must broadcast to a common
        shape.
    axis : int, optional
        Axis along which to compare. Default is None, which flattens `a`
        and `b`.

    Returns
    -------
    float or :obj:`~numpy.array`
        Cosine distance, or with `axis`, one distance per position of the
        other axes. A distance is NaN where a compared vector is all zero.

    """

    if axis is None:
        a_ = a.flatten()
        b_ = b.flatten()

        return 1 - np.dot(a_, b_) / np.sqrt(a_.dot(a_) * b_.dot(b_))

    # Compare along the axis after broadcasting the arrays together
    a_, b_ = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    num = (a_ * b_).sum(axis=axis)
    denom = np.sqrt((a_ * a_).sum(axis=axis) * (b_ * b_).sum(axis=axis))

    return 1 - num / denom


def pearson(a, b, axis=None):
    """
    Pearson correlation distance (1 - correlation) between two arrays.

    Each array is centered on its mean before comparison, so a constant offset,
    such as a baseline under a profile, does not change the distance. The
    distance is undefined (NaN) when either array is constant.

    By default `a` and `b` are flattened and compared as one vector each.
    With `axis`, each array is centered along that axis, and the arrays are
    compared along it as :func:`cosine` does.

    Parameters
    ----------
    a, b : :obj:`~numpy.array`
        N-dimensional arrays. With `axis`, they must broadcast to a common
        shape.
    axis : int, optional
        Axis along which to center and compare. Default is None, which
        flattens `a` and `b`.

    Returns
    -------
    float or :obj:`~numpy.array`
        Pearson correlation distance, or with `axis`, one distance per
        position of the other axes.

    """

    if axis is None:
        return cosine(a - a.mean(), b - b.mean())

    a_ = np.asarray(a, dtype=float)
    b_ = np.asarray(b, dtype=float)

    return cosine(
        a_ - a_.mean(axis=axis, keepdims=True),
        b_ - b_.mean(axis=axis, keepdims=True),
        axis=axis,
    )


def get_1D_profiles(features, dims=None):
    """
    Extract 1D profile for each of the indicated dimension(s).

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    dims : str or list, optional
        Dimensions considered in generating 1D profile(s).
        Default is ``["mz", "drift_time", "retention_time"]``.

    Returns
    -------
    :obj:`dict` of :obj:`~scipy.interpolate.UnivariateSpline`
        Dictionary indexed by dimension containing univariate
        splines for each 1D profile.

    """

    # Set mutable argument defaults
    if dims is None:
        dims = ["mz", "drift_time", "retention_time"]

    # Safely cast to list
    dims = deimos.utils.safelist(dims)

    profiles = {}
    for dim in dims:
        # Collapse to 1D profile
        profile = deimos.collapse(features, keep=dim).sort_values(
            by=dim, ignore_index=True
        )

        # Interpolate spline
        x = profile[dim].values
        y = profile["intensity"].values

        # Fit univariate spline
        try:
            uspline = UnivariateSpline(x, y, s=0, ext=1)
        except (IndexError, ValueError):

            def uspline(x):
                return np.zeros_like(x)

        profiles[dim] = uspline

    return profiles


def offset_correction_model(dt_ms2, mz_ms2, mz_ms1, ce, params=None):
    """
    Correct MS2 drift times for their offset from the MS1 precursor.

    Each corrected drift time is ``factor * dt_ms2``, where ``factor`` is
    ``p0 + p1 * sqrt(mz_ms2 / mz_ms1) + p2 * ln(ce)``.

    Parameters
    ----------
    dt_ms2 : :obj:`~numpy.array`
        MS2 drift times.
    mz_ms2 : :obj:`~numpy.array`
        MS2 m/z values, in the same order as `dt_ms2`.
    mz_ms1 : :obj:`~numpy.array`
        MS1 precursor m/z values, in the same order as `dt_ms2`.
    ce : float
        Collision energy of MS2 collection. Must be positive.
    params : list, optional
        Model coefficients ``[p0, p1, p2]``.
        Default is ``[1.02067031, -0.02062323, 0.00176694]``.

    Returns
    -------
    :obj:`~numpy.array`
        Corrected MS2 drift times.

    Raises
    ------
    ValueError
        If `ce` is zero or negative.

    """

    # Check collision energy is positive, as the model uses its logarithm
    if np.any(np.less_equal(ce, 0)):
        raise ValueError("Collision energy must be positive.")

    # Set mutable argument defaults
    if params is None:
        params = [1.02067031, -0.02062323, 0.00176694]

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
    """
    Performs MS2 deconvolution by correlating non-m/z separation dimension
    profiles and scoring the agreement between precursor and fragment.

    """

    def __init__(self, ms1_features, ms1_data, ms2_features, ms2_data):
        """
        Initializes :obj:`~deimos.deconvolution.MS2Deconvolution` object.

        Parameters
        ----------
        ms1_features : :obj:`~pandas.DataFrame`
            MS1 peak locations and intensities.
        ms1_data : :obj:`~pandas.DataFrame`
            Complete MS1 data.
        ms2_features : :obj:`~pandas.DataFrame`
            MS2 peak locations and intensities.
        ms2_data : :obj:`~pandas.DataFrame`
            Complete MS2 data.

        """

        self.ms1_features = ms1_features
        self.ms1_data = ms1_data
        self.ms2_features = ms2_features
        self.ms2_data = ms2_data

    def construct_putative_pairs(
        self,
        dims=None,
        low=None,
        high=None,
        ce=None,
        model=offset_correction_model,
        require_ms1_greater_than_ms2=True,
        error_tolerance=0.12,
    ):
        """
        Determine each possible MS1:MS2 pair within specified tolerances. If
        considering drift time, apply a model to correct for drift time
        offset such that MS2 can be downselected by respective error in drift
        time (i.e. a poor model correction suggests an incorrect MS1:MS2 pair).

        Parameters
        ----------
        dims : str or list, optional
            Dimension(s) by which to match MS1:MS2.
            Default is ``["drift_time", "retention_time"]``.
        low : float or list, optional
            Lower bound(s) in each dimension.
            Default is ``[-0.12, -0.1]``.
        high : float or list, optional
            Upper bound(s) in each dimension.
            Default is ``[1.4, 0.1]``.
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
            ``index_ms1`` and ``index_ms2`` contain the index labels of the features,
            which must be unique.

        """

        # Set mutable argument defaults
        if dims is None:
            dims = ["drift_time", "retention_time"]
        if low is None:
            low = [-0.12, -0.1]
        if high is None:
            high = [1.4, 0.1]

        # Safely cast to list
        dims = deimos.utils.safelist(dims)
        low = deimos.utils.safelist(low)
        high = deimos.utils.safelist(high)

        # Check dims
        deimos.utils.check_length([dims, low, high])

        # Check collision energy is supplied
        if ce is None:
            raise ValueError("Collision energy must be specified.")

        # Check index labels are unique, as they identify features in the output
        if not (
            self.ms1_features.index.is_unique and self.ms2_features.index.is_unique
        ):
            raise ValueError("Feature index labels must be unique.")

        # Copy match vectors, which are normalized in place
        v_ms1 = self.ms1_features[dims].to_numpy(copy=True)
        v_ms2 = self.ms2_features[dims].to_numpy(copy=True)

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
            ms2_tree, 1, p=np.inf, output_type="coo_matrix"
        )

        # Pairs within tolerance, by position in each table
        ms1_matches = (
            self.ms1_features.iloc[sdm.row, :].rename_axis(index="index").reset_index()
        )
        ms2_matches = (
            self.ms2_features.iloc[sdm.col, :].rename_axis(index="index").reset_index()
        )

        # Correct drift time
        if "drift_time" in dims:
            ms2_matches["drift_time_raw"] = ms2_matches["drift_time"].values
            ms2_matches["drift_time"] = model(
                ms2_matches["drift_time"].values,
                ms2_matches["mz"].values,
                ms1_matches["mz"].values,
                ce=ce,
            )

        # Rename columns
        ms1_matches.columns = [x + "_ms1" for x in ms1_matches.columns]
        ms2_matches.columns = [x + "_ms2" for x in ms2_matches.columns]

        # Construct MS1:MS2 data frame
        decon_pairs = pd.concat((ms1_matches, ms2_matches), axis=1)

        # Compute offset correction error
        if "drift_time" in dims:
            decon_pairs["drift_time_error"] = np.abs(
                decon_pairs["drift_time_ms1"] - decon_pairs["drift_time_ms2"]
            )

        # MS1 intensity greater than MS2 intensity
        if require_ms1_greater_than_ms2 is True:
            decon_pairs = decon_pairs.loc[
                decon_pairs["intensity_ms1"] >= decon_pairs["intensity_ms2"], :
            ]

        # Error tolerance
        if "drift_time" in dims:
            decon_pairs = decon_pairs.loc[
                decon_pairs["drift_time_error"] <= error_tolerance, :
            ]

        # Remove empty groups
        decon_pairs = cast(
            pd.DataFrame,
            decon_pairs.groupby(by="index_ms1", as_index=False).filter(
                lambda x: len(x) > 0
            ),
        )

        # Sort by index
        decon_pairs = decon_pairs.sort_values(
            by=["index_ms1", "index_ms2"]
        ).reset_index(drop=True)

        self.decon_pairs = decon_pairs

        return self.decon_pairs

    def configure_profile_extraction(
        self,
        dims=None,
        low=None,
        high=None,
        relative=None,
    ):
        """
        Configure parameters to generate extracted ion subsets of the data.

        Parameters
        ----------
        dims : str or list, optional
            Dimension(s) by which to subset the data.
            Default is ``["mz", "drift_time", "retention_time"]``.
        low : float or list, optional
            Lower tolerance(s) in each dimension.
            Default is ``[-100e-6, -0.05, -0.3]``.
        high : float or list, optional
            Upper tolerance(s) in each dimension.
            Default is ``[400e-6, 0.05, 0.3]``.
        relative : bool or list, optional
            Whether to use relative or absolute tolerance per dimension.
            Default is ``[True, True, False]``.

        """

        # Set mutable argument defaults
        if dims is None:
            dims = ["mz", "drift_time", "retention_time"]
        if low is None:
            low = [-100e-6, -0.05, -0.3]
        if high is None:
            high = [400e-6, 0.05, 0.3]
        if relative is None:
            relative = [True, True, False]

        def abstract_fxn(features, data, dims, low, high, relative):
            """
            Function abstraction to bake in asymmetrical tolerances at runtime.

            """

            # Locate every feature in one batch, with coordinates in the common
            # dtype of all columns of `features`
            cidx = [features.columns.get_loc(x) for x in dims]
            loc: np.ndarray = features.to_numpy()[:, cidx]
            return deimos.locate_asym(
                data,
                by=dims,
                loc=loc,
                low=low,
                high=high,
                relative=relative,
            )

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
            x, y, dims=dims, low=low, high=high, relative=relative
        )

    def apply(self, dims=None, resolution=None, similarity="cosine"):
        """
        Perform deconvolution according to matched features and their
        extracted profiles.

        Parameters
        ----------
        dims : str or list, optional
            Dimension(s) for which to calculate MS1:MS2 correspondence
            by 1D profile agreement (i.e. non-m/z).
            Default is ``["drift_time", "retention_time"]``.
        resolution : float or list, optional
            Resolution applied to per-dimension profile interpolations.
            Default is ``[0.01, 0.01]``.
        similarity : str
            Profile agreement score, either "cosine" (default) or "pearson".
            Pearson centers each profile first, so a flat background or a
            baseline under a co-eluting ion does not inflate the score.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            All MS1:MS2 pairings and associated agreement scores. A score is NaN when
            either profile is all zero, such as for a feature with no data in its
            profile window.

        """

        # Set mutable argument defaults
        if dims is None:
            dims = ["drift_time", "retention_time"]
        if resolution is None:
            resolution = [0.01, 0.01]
        # Safely cast to list
        dims = deimos.utils.safelist(dims)
        resolution = deimos.utils.safelist(resolution)

        # Check dims
        deimos.utils.check_length([dims, resolution])

        # Ensure m/z not in dims
        if "mz" in dims:
            raise ValueError("MS1:MS2 similarity does not consider m/z.")

        # Select profile distance
        distances = {"cosine": cosine, "pearson": pearson}
        if similarity not in distances:
            raise ValueError(
                f"Unknown similarity {similarity!r}; use 'cosine' or 'pearson'."
            )
        distance = distances[similarity]

        # Recast as dictionaries indexed by dimension
        self.profile_resolution = {k: v for k, v in zip(dims, resolution)}

        # Extracted ions, with no rows for a feature without data in its window
        ms1_xis = self.profiler(self.ms1_features, self.ms1_data)
        ms2_xis = self.profiler(self.ms2_features, self.ms2_data)
        ms1_xis = [self.ms1_data.iloc[:0] if x is None else x for x in ms1_xis]
        ms2_xis = [self.ms2_data.iloc[:0] if x is None else x for x in ms2_xis]

        # Container for profile similarity scores
        scores = {dim: [] for dim in dims}

        # Enumerate MS1 features
        for name, grp in self.decon_pairs.groupby(by=["index_ms1"], as_index=False):
            # MS1 feature position; grouping by a list yields tuple names
            idx_i = cast(
                int,
                self.ms1_features.index.get_loc(cast("tuple[Any, ...]", name)[0]),
            )

            # Extracted ion
            ms1_xi = ms1_xis[idx_i]

            # Build MS1 profile
            ms1_profiles = get_1D_profiles(ms1_xi, dims=dims)

            # Shared interpolation axis
            newx = {}
            for dim in dims:
                # Determine upper and lower bounds
                if self.profile_relative[dim] is True:
                    lb = min(grp[dim + "_ms1"].min(), grp[dim + "_ms2"].min()) * (
                        1 + self.profile_low[dim]
                    )
                    ub = max(grp[dim + "_ms1"].max(), grp[dim + "_ms2"].max()) * (
                        1 + self.profile_high[dim]
                    )
                else:
                    lb = (
                        min(grp[dim + "_ms1"].min(), grp[dim + "_ms2"].min())
                        + self.profile_low[dim]
                    )
                    ub = (
                        max(grp[dim + "_ms1"].max(), grp[dim + "_ms2"].max())
                        + self.profile_high[dim]
                    )

                # Determine shared x-axis
                newx[dim] = np.arange(lb, ub, self.profile_resolution[dim])

            # Evaluate MS1 profile
            ms1_profiles = {dim: ms1_profiles[dim](newx[dim]) for dim in dims}

            # Enumerate MS2 features
            for j, row in grp.reset_index().iterrows():
                # MS2 feature position
                idx_j = cast(int, self.ms2_features.index.get_loc(row["index_ms2"]))

                # Extracted ion
                ms2_xi = ms2_xis[idx_j].copy()

                if "drift_time" in dims:
                    # Determine offset
                    offset = row["drift_time_ms2"] - row["drift_time_raw_ms2"]

                    # Apply offset
                    ms2_xi["drift_time"] += offset

                # Build MS2 profile
                ms2_profiles = get_1D_profiles(ms2_xi, dims=dims)

                # Evaluate MS2 profile
                ms2_profiles = {dim: ms2_profiles[dim](newx[dim]) for dim in dims}

                # Compute similarity
                for dim in dims:
                    scores[dim].append(
                        1 - distance(ms1_profiles[dim], ms2_profiles[dim])
                    )

        # Append score columns
        for dim in dims:
            self.decon_pairs[dim + "_score"] = np.array(scores[dim])

        return self.decon_pairs
