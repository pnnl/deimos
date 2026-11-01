import numpy as np
import pandas as pd
import pytest

import deimos


@pytest.mark.parametrize(
    "b,expected",
    [
        (np.array([10.0, 11.0, 12.0, 11.0, 10.0]), 0.0),
        (np.array([0.0, 3.0, 6.0, 3.0, 0.0]), 0.0),
        (np.array([0.0, -1.0, -2.0, -1.0, 0.0]), 2.0),
        (np.array([1.0, 0.0, 1.0, 0.0, 1.0]), 1 + 0.4 / np.sqrt(3.36)),
    ],
    ids=["baseline-offset", "scaled", "inverted", "unrelated"],
)
def test_pearson(b, expected):
    a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    assert deimos.deconvolution.pearson(a, b) == pytest.approx(expected)


def test_get_1D_profiles():
    # Rows at the same retention time sum in the profile
    features = pd.DataFrame(
        {
            "retention_time": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "intensity": [1.0, 2.0, 5.0, 7.0, 2.0, 1.0],
        }
    )

    profiles = deimos.deconvolution.get_1D_profiles(features, dims="retention_time")

    assert list(profiles) == ["retention_time"]
    assert profiles["retention_time"](
        np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ) == pytest.approx([3.0, 5.0, 7.0, 2.0, 1.0])


def test_get_1D_profiles_default_dims():
    features = pd.DataFrame(
        {
            "mz": [100.0, 200.0, 300.0, 400.0],
            "drift_time": [10.0, 20.0, 30.0, 40.0],
            "retention_time": [1.0, 2.0, 3.0, 4.0],
            "intensity": [1.0, 4.0, 2.0, 3.0],
        }
    )

    profiles = deimos.deconvolution.get_1D_profiles(features)

    assert sorted(profiles) == ["drift_time", "mz", "retention_time"]
    for dim, profile in profiles.items():
        assert profile(features[dim].values) == pytest.approx([1.0, 4.0, 2.0, 3.0])


def test_get_1D_profiles_is_zero_outside_the_data():
    features = pd.DataFrame(
        {"retention_time": [1.0, 2.0, 3.0, 4.0], "intensity": [1.0, 4.0, 2.0, 3.0]}
    )

    profile = deimos.deconvolution.get_1D_profiles(features, dims="retention_time")[
        "retention_time"
    ]

    assert profile(np.array([0.5, 4.5])) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize("n_points", [0, 1, 3])
def test_get_1D_profiles_is_zero_with_too_few_points_for_a_spline(n_points):
    x = np.arange(n_points, dtype=float)
    features = pd.DataFrame({"retention_time": x, "intensity": x + 1})

    profile = deimos.deconvolution.get_1D_profiles(features, dims="retention_time")[
        "retention_time"
    ]

    assert profile(np.array([0.0, 1.0])) == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    "ce,params,expected",
    [
        # Factors of 1 + 2 * 0.5 + 3 and 1 + 2 * 1 + 3
        (np.e, [1.0, 2.0, 3.0], [50.0, 120.0]),
        # Default factors of 1.02067031 - 0.02062323 * 0.5 and 1.02067031 - 0.02062323
        (1.0, None, [10.10358695, 20.0009416]),
    ],
    ids=["explicit-params", "default-params"],
)
def test_offset_correction_model(ce, params, expected):
    dt_ms2 = np.array([10.0, 20.0])
    mz_ms2 = np.array([100.0, 400.0])
    mz_ms1 = np.array([400.0, 400.0])

    corrected = deimos.deconvolution.offset_correction_model(
        dt_ms2, mz_ms2, mz_ms1, ce=ce, params=params
    )

    assert corrected == pytest.approx(expected)


@pytest.mark.parametrize(
    "b,expected",
    [
        (np.array([1.0, 1.0, 0.0, 0.0]), 0.0),
        (np.array([3.0, 3.0, 0.0, 0.0]), 0.0),
        (np.array([0.0, 0.0, 1.0, 1.0]), 1.0),
        (np.array([-1.0, -1.0, 0.0, 0.0]), 2.0),
        (np.array([1.0, 0.0, 0.0, 0.0]), 1 - 1 / np.sqrt(2)),
    ],
    ids=["identical", "scaled", "orthogonal", "opposite", "partial"],
)
def test_cosine(b, expected):
    a = np.array([1.0, 1.0, 0.0, 0.0])

    assert deimos.deconvolution.cosine(a, b) == pytest.approx(expected)


def test_cosine_flattens_arrays():
    a = np.array([[1.0, 1.0], [0.0, 0.0]])
    b = np.array([[1.0, 0.0], [0.0, 0.0]])

    assert deimos.deconvolution.cosine(a, b) == pytest.approx(1 - 1 / np.sqrt(2))


def test_cosine_is_nan_for_a_zero_array():
    with pytest.warns(RuntimeWarning):
        distance = deimos.deconvolution.cosine(np.zeros(3), np.ones(3))

    assert np.isnan(distance)


# Precursor profile on the integer axis from 5 to 14, with a squared norm of 19
PRECURSOR_PROFILE = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])

# Fragment profile with a cosine of 3 / sqrt(19) against the precursor profile
PARTIAL_PROFILE = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def _profile_data(mz, intensity, dim="retention_time", start=5.0):
    """Build one ion's data along `dim`, on an integer axis beginning at `start`."""
    return pd.DataFrame(
        {
            "mz": mz,
            dim: start + np.arange(len(intensity), dtype=float),
            "intensity": intensity,
        }
    )


def _pair_features(dim="retention_time"):
    """Build MS1 features at m/z 500 and 800 and MS2 features at m/z 100 and 200."""
    ms1_features = pd.DataFrame(
        {"mz": [500.0, 800.0], dim: [10.0, 10.0], "intensity": [3.0, 3.0]}
    )
    ms2_features = pd.DataFrame(
        {"mz": [100.0, 200.0], dim: [10.0, 10.0], "intensity": [6.0, 3.0]}
    )
    return ms1_features, ms2_features


def _configured_decon(ms1_features, ms1_data, ms2_features, ms2_data, dim, pairs):
    """Build a deconvolution with the given pairs, windowed 5 on either side in `dim`."""
    decon = deimos.deconvolution.MS2Deconvolution(
        ms1_features, ms1_data, ms2_features, ms2_data
    )
    decon.decon_pairs = pd.DataFrame(pairs)
    decon.configure_profile_extraction(
        dims=["mz", dim],
        low=[-100e-6, -5.0],
        high=[400e-6, 5.0],
        relative=[True, False],
    )
    return decon


class TestMS2Deconvolution:
    def test_init(self):
        ms1_features, ms1_data = pd.DataFrame(), pd.DataFrame()
        ms2_features, ms2_data = pd.DataFrame(), pd.DataFrame()

        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, ms1_data, ms2_features, ms2_data
        )

        assert decon.ms1_features is ms1_features
        assert decon.ms1_data is ms1_data
        assert decon.ms2_features is ms2_features
        assert decon.ms2_data is ms2_data

    def test_construct_putative_pairs(self):
        # Offsets of MS1 minus MS2: 0.5, 1.5, 2.5, -0.5, and -1.5 from the first MS1
        # feature, and far from the second
        ms1_features = pd.DataFrame(
            {
                "mz": [500.0, 800.0],
                "retention_time": [10.0, 50.0],
                "intensity": [100.0, 100.0],
            }
        )
        ms2_features = pd.DataFrame(
            {
                "mz": [100.0, 110.0, 120.0, 130.0, 140.0],
                "retention_time": [9.5, 8.5, 7.5, 10.5, 11.5],
                "intensity": [10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        pairs = decon.construct_putative_pairs(
            dims="retention_time", low=-1.0, high=2.0, ce=20
        )

        assert pairs["index_ms1"].tolist() == [0, 0, 0]
        assert pairs["index_ms2"].tolist() == [0, 1, 3]
        assert pairs["mz_ms1"].tolist() == [500.0, 500.0, 500.0]
        assert pairs["mz_ms2"].tolist() == [100.0, 110.0, 130.0]
        assert decon.decon_pairs is pairs

    def test_construct_putative_pairs_requires_collision_energy(self):
        decon = deimos.deconvolution.MS2Deconvolution(None, None, None, None)

        with pytest.raises(ValueError, match="Collision energy"):
            decon.construct_putative_pairs()

    def test_construct_putative_pairs_checks_lengths(self):
        decon = deimos.deconvolution.MS2Deconvolution(None, None, None, None)

        with pytest.raises(ValueError, match="same dimension"):
            decon.construct_putative_pairs(
                dims=["retention_time"], low=[-1.0, 0.0], high=[1.0], ce=20
            )

    @pytest.mark.parametrize(
        "require,expected",
        [(True, [0, 1]), (False, [0, 1, 2])],
        ids=["required", "not-required"],
    )
    def test_construct_putative_pairs_ms1_greater_than_ms2(self, require, expected):
        # MS2 features below, equal to, and above the MS1 intensity
        ms1_features = pd.DataFrame(
            {"mz": [500.0], "retention_time": [10.0], "intensity": [100.0]}
        )
        ms2_features = pd.DataFrame(
            {
                "mz": [100.0, 110.0, 120.0],
                "retention_time": [10.0, 10.0, 10.0],
                "intensity": [50.0, 100.0, 150.0],
            }
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        pairs = decon.construct_putative_pairs(
            dims="retention_time",
            low=-1.0,
            high=1.0,
            ce=20,
            require_ms1_greater_than_ms2=require,
        )

        assert pairs["index_ms2"].tolist() == expected

    def test_construct_putative_pairs_corrects_drift_time(self):
        # Default model factor of 1.02067031 - 0.02062323 * 0.5, at a collision energy
        # with a logarithm of 0
        ms1_features = pd.DataFrame(
            {"mz": [400.0], "drift_time": [20.2], "intensity": [100.0]}
        )
        ms2_features = pd.DataFrame(
            {"mz": [100.0], "drift_time": [20.0], "intensity": [10.0]}
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        pairs = decon.construct_putative_pairs(
            dims="drift_time", low=-1.0, high=1.0, ce=1.0
        )

        assert pairs["drift_time_raw_ms2"].tolist() == [20.0]
        assert pairs["drift_time_ms2"].tolist() == pytest.approx([20.2071739])
        assert pairs["drift_time_error"].tolist() == pytest.approx([0.0071739])

    def test_construct_putative_pairs_calls_model(self):
        # Correction of 25 uses the MS2 drift time, both m/z values, and collision energy
        ms1_features = pd.DataFrame(
            {"mz": [400.0], "drift_time": [25.0], "intensity": [100.0]}
        )
        ms2_features = pd.DataFrame(
            {"mz": [100.0], "drift_time": [20.0], "intensity": [10.0]}
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        pairs = decon.construct_putative_pairs(
            dims="drift_time",
            low=0.0,
            high=10.0,
            ce=2.0,
            model=lambda dt, mz_ms2, mz_ms1, ce: dt + (mz_ms1 - mz_ms2) / 100 + ce,
        )

        assert pairs["drift_time_ms2"].tolist() == pytest.approx([25.0])
        assert pairs["drift_time_error"].tolist() == pytest.approx([0.0])

    def test_construct_putative_pairs_applies_error_tolerance(self):
        # Errors of 0.05, 0.5 above, and 0.5 below after a model that leaves drift time
        # unchanged
        ms1_features = pd.DataFrame(
            {"mz": [500.0], "drift_time": [20.0], "intensity": [100.0]}
        )
        ms2_features = pd.DataFrame(
            {
                "mz": [100.0, 110.0, 120.0],
                "drift_time": [20.05, 20.5, 19.5],
                "intensity": [10.0, 10.0, 10.0],
            }
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        pairs = decon.construct_putative_pairs(
            dims="drift_time",
            low=-1.0,
            high=1.0,
            ce=20,
            model=lambda dt, mz_ms2, mz_ms1, ce: dt,
            error_tolerance=0.12,
        )

        assert pairs["index_ms2"].tolist() == [0]

    def test_construct_putative_pairs_sorts_pairs(self):
        # Enough features in shuffled order that the k-d tree returns pairs out of order
        n_features = 50
        ms1_features = pd.DataFrame(
            {"mz": [500.0], "retention_time": [10.0], "intensity": [100.0]}
        )
        ms2_features = pd.DataFrame(
            {
                "mz": np.full(n_features, 100.0),
                "retention_time": np.random.default_rng(0).permutation(
                    np.linspace(9.5, 10.5, n_features)
                ),
                "intensity": np.full(n_features, 10.0),
            }
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        pairs = decon.construct_putative_pairs(
            dims="retention_time", low=-1.0, high=1.0, ce=20
        )

        assert pairs["index_ms2"].tolist() == list(range(n_features))

    def test_construct_putative_pairs_returns_no_rows_without_a_match(self):
        ms1_features = pd.DataFrame(
            {"mz": [500.0], "retention_time": [10.0], "intensity": [100.0]}
        )
        ms2_features = pd.DataFrame(
            {"mz": [100.0], "retention_time": [99.0], "intensity": [10.0]}
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        pairs = decon.construct_putative_pairs(
            dims="retention_time", low=-1.0, high=1.0, ce=20
        )

        assert len(pairs) == 0
        assert {"index_ms1", "index_ms2", "mz_ms1", "mz_ms2"} <= set(pairs.columns)

    def test_construct_putative_pairs_leaves_features_unchanged(self):
        ms1_features = pd.DataFrame(
            {"mz": [500.0], "retention_time": [10.0], "intensity": [100.0]}
        )
        ms2_features = pd.DataFrame(
            {"mz": [100.0], "retention_time": [10.0], "intensity": [10.0]}
        )
        ms1_before, ms2_before = ms1_features.copy(), ms2_features.copy()
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        decon.construct_putative_pairs(dims="retention_time", low=-1.0, high=2.0, ce=20)

        pd.testing.assert_frame_equal(ms1_features, ms1_before)
        pd.testing.assert_frame_equal(ms2_features, ms2_before)

    @pytest.mark.parametrize(
        "ms1_index,ms2_index,expected_ms1,expected_ms2",
        [
            (pd.Index([1, 0]), pd.Index([0, 1]), [0, 1], [1, 0]),
            (pd.Index(["b", "a"]), pd.Index(["x", "y"]), ["a", "b"], ["y", "x"]),
            (pd.Index([7, 9], name="peak"), pd.Index([3, 5]), [7, 9], [3, 5]),
        ],
        ids=["shuffled", "strings", "named"],
    )
    def test_construct_putative_pairs_reports_index_labels(
        self, ms1_index, ms2_index, expected_ms1, expected_ms2
    ):
        # The first features pair at retention time 10 and the second at 50, wherever
        # their labels sort
        ms1_features = pd.DataFrame(
            {
                "mz": [500.0, 800.0],
                "retention_time": [10.0, 50.0],
                "intensity": [100.0, 100.0],
            },
            index=ms1_index,
        )
        ms2_features = pd.DataFrame(
            {
                "mz": [100.0, 200.0],
                "retention_time": [10.0, 50.0],
                "intensity": [10.0, 10.0],
            },
            index=ms2_index,
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, None, ms2_features, None
        )

        pairs = decon.construct_putative_pairs(
            dims="retention_time", low=-1.0, high=1.0, ce=20
        )

        assert pairs["index_ms1"].tolist() == expected_ms1
        assert pairs["index_ms2"].tolist() == expected_ms2
        assert (pairs["retention_time_ms1"] == pairs["retention_time_ms2"]).all()

    @pytest.mark.parametrize(
        "ms1_index,ms2_index",
        [([0, 0], [0, 1]), ([0, 1], [0, 0])],
        ids=["ms1", "ms2"],
    )
    def test_construct_putative_pairs_requires_unique_labels(
        self, ms1_index, ms2_index
    ):
        features = {
            "mz": [500.0, 800.0],
            "retention_time": [10.0, 50.0],
            "intensity": [100.0, 100.0],
        }
        decon = deimos.deconvolution.MS2Deconvolution(
            pd.DataFrame(features, index=ms1_index),
            None,
            pd.DataFrame(features, index=ms2_index),
            None,
        )

        with pytest.raises(ValueError, match="unique"):
            decon.construct_putative_pairs(
                dims="retention_time", low=-1.0, high=1.0, ce=20
            )

    def test_configure_profile_extraction(self):
        # Peaks with a float64 column beside float32 coordinates, as peak picking
        # returns them
        features = pd.DataFrame(
            {
                "mz": np.array([100.0, 200.0, 300.0], dtype=np.float32),
                "drift_time": np.array([10.0, 20.0, 30.0], dtype=np.float32),
                "intensity": np.array([5.0, 6.0, 7.0], dtype=np.float32),
                "persistence": [1.0, 2.0, 3.0],
            }
        )
        # Float32 99.99 is below the first peak's lower bound in float64, the
        # common dtype of the peaks, but on it in float32
        data = pd.DataFrame(
            {
                "mz": np.array(
                    [100.0, 100.01, 100.05, 200.0, 200.0, 99.99], dtype=np.float32
                ),
                "drift_time": np.array(
                    [10.0, 10.2, 10.0, 21.0, 20.5, 10.0], dtype=np.float32
                ),
                "intensity": np.ones(6, dtype=np.float32),
            }
        )
        decon = deimos.deconvolution.MS2Deconvolution(features, data, features, data)

        decon.configure_profile_extraction(
            dims=["mz", "drift_time"],
            low=[-100e-6, -0.3],
            high=[400e-6, 0.6],
            relative=[True, False],
        )
        profiles = decon.profiler(features, data)

        assert [None if x is None else x.index.tolist() for x in profiles] == [
            [0, 1],
            [4],
            None,
        ]

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, 109 / np.sqrt(19 * 1199)),
            ({"similarity": "cosine"}, 109 / np.sqrt(19 * 1199)),
            ({"similarity": "pearson"}, 1.0),
        ],
        ids=["default", "cosine", "pearson"],
    )
    def test_apply_similarity(self, kwargs, expected):
        # A fragment profile that follows the precursor on a baseline of 10
        retention_time = np.arange(5.0, 15.0)
        precursor = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        ms1_data = pd.DataFrame(
            {"mz": 500.0, "retention_time": retention_time, "intensity": precursor}
        )
        ms2_data = pd.DataFrame(
            {
                "mz": 100.0,
                "retention_time": retention_time,
                "intensity": precursor + 10,
            }
        )
        ms1_features = pd.DataFrame(
            {"mz": [500.0], "retention_time": [10.0], "intensity": [3.0]}
        )
        ms2_features = pd.DataFrame(
            {"mz": [100.0], "retention_time": [10.0], "intensity": [13.0]}
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, ms1_data, ms2_features, ms2_data
        )
        decon.decon_pairs = pd.DataFrame(
            {
                "index_ms1": [0],
                "index_ms2": [0],
                "retention_time_ms1": [10.0],
                "retention_time_ms2": [10.0],
            }
        )
        decon.configure_profile_extraction(
            dims=["mz", "retention_time"],
            low=[-100e-6, -5.0],
            high=[400e-6, 5.0],
            relative=[True, False],
        )

        result = decon.apply(dims="retention_time", resolution=1.0, **kwargs)

        assert result["retention_time_score"].tolist() == pytest.approx([expected])

    def test_apply_unknown_similarity(self):
        decon = deimos.deconvolution.MS2Deconvolution(None, None, None, None)

        with pytest.raises(ValueError, match="similarity"):
            decon.apply(dims="drift_time", resolution=0.01, similarity="spearman")

    def test_apply(self):
        # Fragment 0 is a scaled copy of precursor 0, and fragment 1 is a copy of
        # precursor 1, which overlaps precursor 0 at one point
        ms1_features, ms2_features = _pair_features()
        ms1_data = pd.concat(
            [
                _profile_data(500.0, PRECURSOR_PROFILE),
                _profile_data(800.0, PARTIAL_PROFILE),
            ]
        )
        ms2_data = pd.concat(
            [
                _profile_data(100.0, 2 * PRECURSOR_PROFILE),
                _profile_data(200.0, PARTIAL_PROFILE),
            ]
        )
        decon = _configured_decon(
            ms1_features,
            ms1_data,
            ms2_features,
            ms2_data,
            "retention_time",
            {
                "index_ms1": [0, 0, 1],
                "index_ms2": [0, 1, 1],
                "retention_time_ms1": [10.0, 10.0, 10.0],
                "retention_time_ms2": [10.0, 10.0, 10.0],
            },
        )

        result = decon.apply(dims="retention_time", resolution=1.0)

        assert result is decon.decon_pairs
        assert result["retention_time_score"].tolist() == pytest.approx(
            [1.0, 3 / np.sqrt(19), 1.0]
        )

    def test_apply_shifts_fragment_by_the_drift_time_correction(self):
        # A fragment one unit early matches the precursor only after its correction
        ms1_features, ms2_features = _pair_features("drift_time")
        ms2_features["drift_time"] = 9.0
        decon = _configured_decon(
            ms1_features,
            _profile_data(500.0, PRECURSOR_PROFILE, "drift_time"),
            ms2_features,
            _profile_data(100.0, PRECURSOR_PROFILE, "drift_time", start=4.0),
            "drift_time",
            {
                "index_ms1": [0],
                "index_ms2": [0],
                "drift_time_ms1": [10.0],
                "drift_time_ms2": [10.0],
                "drift_time_raw_ms2": [9.0],
            },
        )

        result = decon.apply(dims="drift_time", resolution=1.0)

        assert result["drift_time_score"].tolist() == pytest.approx([1.0])

    @pytest.mark.parametrize(
        "low,high,relative",
        [(-5.0, 5.0, False), (-0.5, 0.5, True)],
        ids=["absolute", "relative"],
    )
    def test_apply_profile_window(self, low, high, relative):
        # The score depends on the points in a window from 5 up to 15
        ms1_features, ms2_features = _pair_features()
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features,
            _profile_data(500.0, PRECURSOR_PROFILE),
            ms2_features,
            _profile_data(100.0, PARTIAL_PROFILE),
        )
        decon.decon_pairs = pd.DataFrame(
            {
                "index_ms1": [0],
                "index_ms2": [0],
                "retention_time_ms1": [10.0],
                "retention_time_ms2": [10.0],
            }
        )
        decon.configure_profile_extraction(
            dims=["mz", "retention_time"],
            low=[-100e-6, low],
            high=[400e-6, high],
            relative=[True, relative],
        )

        result = decon.apply(dims="retention_time", resolution=1.0)

        assert result["retention_time_score"].tolist() == pytest.approx(
            [3 / np.sqrt(19)]
        )

    def test_apply_default_dims(self):
        # Scaled copies agree at every point of the default 0.01 resolution
        ms1_data = _profile_data(500.0, PRECURSOR_PROFILE).assign(
            drift_time=lambda x: x["retention_time"]
        )
        ms2_data = _profile_data(100.0, 2 * PRECURSOR_PROFILE).assign(
            drift_time=lambda x: x["retention_time"]
        )
        ms1_features = pd.DataFrame(
            {
                "mz": [500.0],
                "drift_time": [10.0],
                "retention_time": [10.0],
                "intensity": [3.0],
            }
        )
        ms2_features = pd.DataFrame(
            {
                "mz": [100.0],
                "drift_time": [10.0],
                "retention_time": [10.0],
                "intensity": [6.0],
            }
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, ms1_data, ms2_features, ms2_data
        )
        decon.decon_pairs = pd.DataFrame(
            {
                "index_ms1": [0],
                "index_ms2": [0],
                "drift_time_ms1": [10.0],
                "drift_time_ms2": [10.0],
                "drift_time_raw_ms2": [10.0],
                "retention_time_ms1": [10.0],
                "retention_time_ms2": [10.0],
            }
        )
        decon.configure_profile_extraction(
            dims=["mz", "drift_time", "retention_time"],
            low=[-100e-6, -5.0, -5.0],
            high=[400e-6, 5.0, 5.0],
            relative=[True, False, False],
        )

        result = decon.apply()

        assert result["drift_time_score"].tolist() == pytest.approx([1.0])
        assert result["retention_time_score"].tolist() == pytest.approx([1.0])

    def test_apply_returns_no_rows_without_pairs(self):
        ms1_features, ms2_features = _pair_features()
        ms1_data = pd.concat(
            [
                _profile_data(500.0, PRECURSOR_PROFILE),
                _profile_data(800.0, PARTIAL_PROFILE),
            ]
        )
        ms2_data = pd.concat(
            [
                _profile_data(100.0, PRECURSOR_PROFILE),
                _profile_data(200.0, PARTIAL_PROFILE),
            ]
        )
        decon = _configured_decon(
            ms1_features,
            ms1_data,
            ms2_features,
            ms2_data,
            "retention_time",
            {
                "index_ms1": pd.Series(dtype=int),
                "index_ms2": pd.Series(dtype=int),
                "retention_time_ms1": pd.Series(dtype=float),
                "retention_time_ms2": pd.Series(dtype=float),
            },
        )

        result = decon.apply(dims="retention_time", resolution=1.0)

        assert len(result) == 0
        assert "retention_time_score" in result.columns

    def test_apply_scores_constructed_pairs(self):
        # A model adds the one unit that the fragment arrives early
        ms1_features, ms2_features = _pair_features("drift_time")
        ms2_features["drift_time"] = 9.0
        ms1_data = pd.concat(
            [
                _profile_data(500.0, PRECURSOR_PROFILE, "drift_time"),
                _profile_data(800.0, PARTIAL_PROFILE, "drift_time"),
            ]
        )
        ms2_data = pd.concat(
            [
                _profile_data(100.0, 2 * PRECURSOR_PROFILE, "drift_time", start=4.0),
                _profile_data(200.0, PARTIAL_PROFILE, "drift_time", start=4.0),
            ]
        )
        decon = deimos.deconvolution.MS2Deconvolution(
            ms1_features, ms1_data, ms2_features, ms2_data
        )
        decon.construct_putative_pairs(
            dims="drift_time",
            low=-2.0,
            high=2.0,
            ce=20,
            model=lambda dt, mz_ms2, mz_ms1, ce: dt + 1,
            require_ms1_greater_than_ms2=False,
        )
        decon.configure_profile_extraction(
            dims=["mz", "drift_time"],
            low=[-100e-6, -5.0],
            high=[400e-6, 5.0],
            relative=[True, False],
        )

        result = decon.apply(dims="drift_time", resolution=1.0)

        assert result["index_ms1"].tolist() == [0, 0, 1, 1]
        assert result["index_ms2"].tolist() == [0, 1, 0, 1]
        assert result["drift_time_score"].tolist() == pytest.approx(
            [1.0, 3 / np.sqrt(19), 3 / np.sqrt(19), 1.0]
        )

    def test_apply_rejects_mz(self):
        decon = deimos.deconvolution.MS2Deconvolution(None, None, None, None)

        with pytest.raises(ValueError, match="m/z"):
            decon.apply(dims="mz", resolution=0.01)

    def test_apply_checks_lengths(self):
        decon = deimos.deconvolution.MS2Deconvolution(None, None, None, None)

        with pytest.raises(ValueError, match="same dimension"):
            decon.apply(dims=["drift_time", "retention_time"], resolution=[0.01])

    @pytest.mark.parametrize(
        "ms1_labels,ms2_labels,expected",
        [
            ([1, 0], [0, 1], [3 / np.sqrt(19), 1.0, 1.0, 3 / np.sqrt(19)]),
            ([0, 1], [1, 0], [3 / np.sqrt(19), 1.0, 1.0, 3 / np.sqrt(19)]),
            (["b", "a"], ["y", "x"], [1.0, 3 / np.sqrt(19), 3 / np.sqrt(19), 1.0]),
        ],
        ids=["ms1-shuffled", "ms2-shuffled", "strings"],
    )
    def test_apply_finds_features_by_index_label(
        self, ms1_labels, ms2_labels, expected
    ):
        # The first feature in each table is the precursor profile or its scaled copy,
        # and pairs are in sorted label order
        ms1_features, ms2_features = _pair_features()
        ms1_features.index = pd.Index(ms1_labels)
        ms2_features.index = pd.Index(ms2_labels)
        ms1_data = pd.concat(
            [
                _profile_data(500.0, PRECURSOR_PROFILE),
                _profile_data(800.0, PARTIAL_PROFILE),
            ]
        )
        ms2_data = pd.concat(
            [
                _profile_data(100.0, 2 * PRECURSOR_PROFILE),
                _profile_data(200.0, PARTIAL_PROFILE),
            ]
        )
        decon = _configured_decon(
            ms1_features,
            ms1_data,
            ms2_features,
            ms2_data,
            "retention_time",
            {
                "index_ms1": sorted(ms1_labels * 2),
                "index_ms2": sorted(ms2_labels) * 2,
                "retention_time_ms1": [10.0] * 4,
                "retention_time_ms2": [10.0] * 4,
            },
        )

        result = decon.apply(dims="retention_time", resolution=1.0)

        assert result["retention_time_score"].tolist() == pytest.approx(expected)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("missing", ["ms1", "ms2"])
    def test_apply_scores_nan_for_a_feature_without_data(self, missing):
        # The second feature of one table has no data, and the first pair still scores
        ms1_features, ms2_features = _pair_features()
        ms1_data = _profile_data(500.0, PRECURSOR_PROFILE)
        ms2_data = _profile_data(100.0, 2 * PRECURSOR_PROFILE)
        pairs = {
            "index_ms1": [0, 1] if missing == "ms1" else [0, 0],
            "index_ms2": [0, 1] if missing == "ms2" else [0, 0],
            "retention_time_ms1": [10.0, 10.0],
            "retention_time_ms2": [10.0, 10.0],
        }
        decon = _configured_decon(
            ms1_features, ms1_data, ms2_features, ms2_data, "retention_time", pairs
        )

        scores = decon.apply(dims="retention_time", resolution=1.0)[
            "retention_time_score"
        ].to_numpy()

        assert scores[0] == pytest.approx(1.0)
        assert np.isnan(scores[1])
