import tracemalloc

import numpy as np
import pandas as pd
import pytest

import deimos
from tests import localfile


@pytest.fixture()
def ms1_peaks():
    ms1 = deimos.load(localfile("resources/isotope_example_data.h5"), key="ms1")
    return deimos.peakpick.persistent_homology(
        ms1, dims=["mz", "drift_time", "retention_time"], radius=[2, 10, 0]
    )


# need to test more configurations
def test_detect(ms1_peaks):
    isotopes = deimos.isotopes.detect(
        ms1_peaks,
        dims=["mz", "drift_time", "retention_time"],
        tol=[0.1, 0.2, 0.3],
        delta=1.003355,
        max_isotopes=5,
        max_charge=1,
        max_error=50e-6,
    )

    # grab the most intense isotopic pattern
    isotopes = isotopes.sort_values(by="intensity", ascending=False)
    isotopes = isotopes.iloc[0, :]

    assert abs(isotopes["mz"] - 387.024353) <= 1e-3
    assert isotopes["n"] == 4
    assert all(x <= 50e-6 for x in isotopes["error"])
    assert sorted(set(isotopes["dx"])) == [1.003355, 2.00671, 3.010065, 4.01342]


def test_detect_finds_largest_spacing_at_tolerance_edge():
    """Test that the candidate search reaches the largest expected spacing.

    The child sits 0.099 above the M+4 position, inside the 0.1 m/z tolerance,
    so it must be found even though it is the furthest pair the search allows.
    """
    features = pd.DataFrame(
        {
            "mz": [300.0, 300.0 + 4 * 1.003355 + 0.099],
            "intensity": [1000.0, 100.0],
        }
    )

    isotopes = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], max_isotopes=4, max_error=1e-3
    )

    assert len(isotopes) == 1
    assert isotopes.iloc[0]["multiple"] == [4.0]


def test_detect_finds_low_mz_child_at_window_edge():
    """Test that a child exactly at the window edge is found at low m/z.

    This pair lies within the tolerance after rounding, and without its padding
    the window would end just short of the parent.
    """
    features = pd.DataFrame({"mz": [7.95, 9.053355], "intensity": [1000.0, 100.0]})

    isotopes = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], max_isotopes=4, max_error=0.1
    )

    assert len(isotopes) == 1
    assert isotopes.iloc[0]["multiple"] == [1.0]


def test_detect_excludes_child_inside_padding_but_outside_tolerance():
    """Test that a pair within the window padding but beyond the tolerance is excluded.

    The pair lies 0.1000005 from M+1, inside the window's 1e-6 padding and outside
    the 0.1 tolerance.
    """
    features = pd.DataFrame({"mz": [300.0, 301.1033555], "intensity": [1000.0, 100.0]})

    isotopes = deimos.isotopes.detect(features, dims=["mz"], tol=[0.1], max_error=1e-3)

    assert isotopes.empty


@pytest.mark.parametrize(
    ("max_error", "expected"),
    [(0.004, 0), (0.0041, 1)],
    ids=["at-max-error", "below-max-error"],
)
def test_detect_max_error_excludes_pair_at_the_limit(max_error, expected):
    """Test that a pair's relative error must lie below `max_error`.

    The pair lies 0.5 from the 2.0 spacing, with a child at 125.0, so its relative
    error is 0.004.
    """
    features = pd.DataFrame({"mz": [122.5, 125.0], "intensity": [1000.0, 100.0]})

    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz"],
        tol=[0.6],
        delta=2.0,
        max_isotopes=1,
        max_error=max_error,
    )

    assert len(isotopes) == expected


@pytest.mark.parametrize(
    ("require_lower_intensity", "expected"),
    [(True, 0), (False, 1)],
    ids=["required", "not-required"],
)
def test_detect_require_lower_intensity(require_lower_intensity, expected):
    """Test that a child as intense as its parent is reported only on request."""
    features = pd.DataFrame({"mz": [300.0, 301.003355], "intensity": [1000.0, 1000.0]})

    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz"],
        tol=[0.1],
        require_lower_intensity=require_lower_intensity,
    )

    assert len(isotopes) == expected


def test_detect_candidates_respect_other_dimensions():
    """Test that a pair at the right spacing but outside the retention time
    tolerance is not reported."""
    features = pd.DataFrame(
        {
            "mz": [300.0, 301.003355, 400.0, 401.003355],
            "retention_time": [5.0, 5.2, 5.0, 5.4],
            "intensity": [1000.0, 500.0, 1000.0, 500.0],
        }
    )

    isotopes = deimos.isotopes.detect(
        features, dims=["mz", "retention_time"], tol=[0.1, 0.3]
    )

    assert isotopes["idx"].tolist() == [0]
    assert isotopes.iloc[0]["idx_iso"] == [1]


def test_detect_many_candidates_in_one_window():
    """Test detection when the m/z window holds millions of candidate pairs.

    3,000 features lie within 4.1 m/z of each other, so a 10 m/z tolerance covers
    about 4.5 million pairs around M+1, more than one block of candidates. Each
    parent shares its retention time only with its own M+1 isotope, so exactly
    those 1,500 pairs are reported.
    """
    k = np.arange(1500)
    features = pd.DataFrame(
        {
            "mz": np.concatenate([300.0 + 0.002 * k, 301.003355 + 0.002 * k]),
            "retention_time": np.concatenate([10.0 * k, 10.0 * k]),
            "intensity": np.concatenate([np.full(1500, 1000.0), np.full(1500, 500.0)]),
        }
    )

    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz", "retention_time"],
        tol=[10.0, 0.3],
        max_isotopes=1,
        max_error=1.0,
    )

    pairs = sorted(
        (idx, iso[0])
        for idx, iso in zip(isotopes["idx"], isotopes["idx_iso"], strict=True)
    )
    assert pairs == [(i, i + 1500) for i in range(1500)]


def test_detect_memory_does_not_grow_with_all_pairs():
    """Test that detect does not allocate per pair of features.

    An n x n distance matrix for 20,000 features needs over 3 GB; the
    candidate search should stay far below that.
    """
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "mz": rng.uniform(100, 1100, 20000),
            "retention_time": rng.uniform(0, 30, 20000),
            "intensity": rng.uniform(1e3, 1e5, 20000),
        }
    )

    tracemalloc.start()
    deimos.isotopes.detect(features, dims=["mz", "retention_time"], tol=[0.1, 0.3])
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert peak < 200 * 2**20


def test_detect_memory_does_not_grow_with_pairs_between_spacings():
    """Test that detect does not allocate for pairs far from every spacing.

    With 200 features per m/z, 20,000 features have about 32 million pairs within
    the largest spacing of SPACINGS, which would need over 2 GB as a matrix of
    pairs by 8 expected differences. Only about 1% of the m/z range around 150 lies
    within the default 50 ppm error of an expected difference.
    """
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "mz": rng.uniform(100, 200, 20000),
            "intensity": rng.uniform(1e3, 1e5, 20000),
        }
    )

    tracemalloc.start()
    deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], delta=deimos.isotopes.SPACINGS
    )
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert peak < 200 * 2**20


@pytest.mark.parametrize(
    ("child_mz", "dtype"),
    [(310.0, object), (301.1, float)],
    ids=["no-pair-within-tolerance", "pair-beyond-max-error"],
)
def test_detect_empty_result_columns(child_mz, dtype):
    """Test the empty frame for a pair outside tolerance and outside max_error.

    At 301.1, the pair lies 0.0966 from M+1, within the 0.1 tolerance and about
    320 ppm beyond the default 50 ppm error, so the search around M+1 passes over it.
    """
    features = pd.DataFrame({"mz": [300.0, child_mz], "intensity": [1000.0, 100.0]})

    isotopes = deimos.isotopes.detect(features, dims=["mz"], tol=[0.1])

    assert isotopes.empty
    assert set(isotopes.columns) >= {"mz", "idx", "dx", "idx_iso", "n"}
    assert isotopes["dx"].dtype == dtype


def test_detect_finds_pair_whose_float32_mz_difference_rounds_to_the_spacing():
    """Test that the search reaches a pair matched only after rounding in float32.

    The difference rounds to exactly the spacing in float32, so its error is zero,
    although its exact difference lies 0.03 mDa from the spacing, beyond a `max_error`
    of 1e-9 of the child's m/z (0.0009 mDa).
    """
    features = pd.DataFrame(
        {"mz": np.array([300.1, 900.3], dtype=np.float32), "intensity": [1000.0, 100.0]}
    )

    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz"],
        tol=[0.1],
        delta=600.199951171875,
        max_isotopes=1,
        max_error=1e-9,
    )

    row = isotopes.iloc[0]
    assert row["idx_iso"] == [1]
    assert row["error"] == [0.0]


def test_detect_accepts_integer_mz():
    """Test that integer m/z values are searched."""
    features = pd.DataFrame({"mz": [300, 301], "intensity": [1000.0, 100.0]})

    isotopes = deimos.isotopes.detect(features, dims=["mz"], tol=[0.1], delta=1.0)

    row = isotopes.iloc[0]
    assert row["idx_iso"] == [1]
    assert row["error"] == [0.0]


def test_detect_negative_child_mz_passes_max_error():
    """Test that a negative child m/z is limited by the tolerance only.

    The pair lies 0.0466 from M+1, and dividing by the negative child m/z gives a
    negative relative error, which passes the default 50 ppm.
    """
    features = pd.DataFrame({"mz": [-5.0, -3.95], "intensity": [1000.0, 100.0]})

    isotopes = deimos.isotopes.detect(features, dims=["mz"], tol=[0.1])

    assert isotopes.iloc[0]["idx_iso"] == [1]


def test_detect_reports_each_expected_difference_a_pair_matches():
    """Test that one pair is reported at each expected difference within tolerance.

    A 1.5 m/z difference lies 0.5 from the first and second multiples of 1.0,
    within the 0.6 tolerance, and 0.5 / 301.5 is below `max_error`.
    """
    features = pd.DataFrame({"mz": [300.0, 301.5], "intensity": [1000.0, 100.0]})

    isotopes = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.6], delta=1.0, max_isotopes=2, max_error=0.01
    )

    row = isotopes.iloc[0]
    assert row["multiple"] == [1.0, 2.0]
    assert row["dx"] == [1.0, 2.0]


def test_detect_pairs_only_lower_mz_parents_with_wide_tolerance():
    """Test that a tolerance wider than the spacing forms no reversed or self pairs.

    The window around 0.1 below each child spans 0.6 in each direction, reaching the
    child itself and the feature above it, which must not be treated as parents.
    """
    features = pd.DataFrame({"mz": [300.0, 300.2], "intensity": [1000.0, 100.0]})

    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz"],
        tol=[0.6],
        delta=0.1,
        max_isotopes=1,
        max_error=1.0,
        require_lower_intensity=False,
        return_all_patterns=True,
    )

    assert isotopes["idx"].tolist() == [0]
    assert isotopes.iloc[0]["idx_iso"] == [1]


def test_detect_return_all_patterns():
    """Test that return_all_patterns=True preserves overlapping isotope series.

    Simulates a scenario where:
      - idx 487: low-intensity noise peak at 557.975 Da
      - idx 490: monoisotopic M peak at 558.312 Da (high intensity)
      - idx 493: M+1 peak at 558.644 Da (high intensity)

    With the default return_all_patterns=False, idx 490 is both a child of 487
    and a parent of 493, so the 490→493 pattern is silently dropped.
    With return_all_patterns=True both patterns are returned so downstream
    filtering can discard the spurious 487→490 match.
    """
    import pandas as pd

    features = pd.DataFrame(
        {
            "mz": [557.975, 558.312, 558.644],
            "drift_time": [285.84, 285.97, 285.93],
            "intensity": [1870, 831232, 892192],
        },
        # pandas 2 annotates index with a protocol that list fails to match
        index=[487, 490, 493],  # pyright: ignore[reportArgumentType]
    )

    # Default behaviour: child filter applied – only 487→490 survives because
    # 490 is marked as a child of 487 and therefore removed as a parent.
    isotopes_default = deimos.isotopes.detect(
        features,
        dims=["mz", "drift_time"],
        tol=[0.02, 1.0],
        delta=1.003355,
        max_isotopes=1,
        max_charge=3,
        require_lower_intensity=False,
    )
    default_parents = set(isotopes_default["idx"].tolist())
    assert 490 not in default_parents, (
        "idx 490 should be filtered out as a child with default settings"
    )

    # return_all_patterns=True: child filter skipped – both patterns returned.
    isotopes_all = deimos.isotopes.detect(
        features,
        dims=["mz", "drift_time"],
        tol=[0.02, 1.0],
        delta=1.003355,
        max_isotopes=1,
        max_charge=3,
        require_lower_intensity=False,
        return_all_patterns=True,
    )
    all_parents = set(isotopes_all["idx"].tolist())
    assert 487 in all_parents, "idx 487→490 pattern should be present"
    assert 490 in all_parents, (
        "idx 490→493 pattern should be present when return_all_patterns=True"
    )

    # The 490→493 entry should have idx_iso containing 493
    row_490 = isotopes_all[isotopes_all["idx"] == 490].iloc[0]
    assert 493 in row_490["idx_iso"], "idx_iso for parent 490 should include child 493"


def test_detect_mz_only(ms1_peaks):
    """Test isotope detection with only m/z dimension (single spectrum mode)."""
    # Select features from a single retention time to simulate a single spectrum
    # Use the retention time with the most features
    rt_value = ms1_peaks.groupby("retention_time").size().idxmax()
    single_spectrum = ms1_peaks[ms1_peaks["retention_time"] == rt_value].copy()

    # Keep only mz and intensity columns (simulate a simple spectrum)
    single_spectrum = single_spectrum[["mz", "intensity"]].reset_index(drop=True)

    # Verify we have enough data
    assert len(single_spectrum) > 50  # Should have plenty of features

    # Run isotope detection with only mz dimension
    isotopes = deimos.isotopes.detect(
        single_spectrum,
        dims=["mz"],
        tol=[0.1],  # Only m/z tolerance needed
        delta=1.003355,
        max_isotopes=5,
        max_charge=1,
        max_error=50e-6,
    )

    # Should find some isotope patterns
    assert len(isotopes) > 0

    # Check that all patterns have valid m/z values
    assert all(isotopes["mz"] > 0)

    # Check that n (number of isotopes) is reasonable
    assert all(isotopes["n"] >= 1)
    assert all(isotopes["n"] <= 5)

    # Verify error values are within tolerance
    for errors in isotopes["error"]:
        assert all(e <= 50e-6 for e in errors)


def test_detect_spacings_find_chlorine_isotope():
    """Test that SPACINGS finds a 37Cl isotope that 13C alone misses.

    At m/z 152 the 37Cl M+2 peak lies 9.6 mDa (63 ppm) from the 13C2 spacing,
    beyond the default 50 ppm error, and 0.23 mDa from the second spacing.
    """
    features = pd.DataFrame(
        {
            "mz": [150.0, 151.0035, 151.9971],
            "intensity": [1000.0, 100.0, 320.0],
        }
    )

    carbon_isotopes = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], max_isotopes=2
    )
    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz"],
        tol=[0.1],
        delta=deimos.isotopes.SPACINGS,
        max_isotopes=2,
    )

    assert carbon_isotopes.iloc[0]["idx_iso"] == [1]
    row = isotopes.iloc[0]
    assert row["idx_iso"] == [1, 2]
    assert row["dx"] == [1.003355, 1.996874]


def test_detect_lists_isotopes_by_child_mz_for_any_spacing_order():
    """Test that list columns follow child m/z, not the order of the spacings."""
    features = pd.DataFrame(
        {
            "mz": [150.0, 151.0035, 151.9971],
            "intensity": [1000.0, 100.0, 320.0],
        }
    )

    isotopes = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], delta=[1.996874, 1.003355], max_isotopes=1
    )

    row = isotopes.iloc[0]
    assert row["idx_iso"] == [1, 2]
    assert row["dx"] == [1.003355, 1.996874]


@pytest.mark.parametrize(
    ("child_mz", "dx", "multiple", "error"),
    [
        (302.0, 1.996874, 1.0, 0.003126 / 302.0),
        (302.006, 2.00671, 2.0, 0.00071 / 302.006),
    ],
    ids=["second-spacing-closer", "13c-closer"],
)
def test_detect_spacings_assign_closest_spacing(child_mz, dx, multiple, error):
    """Test that a pair matching several spacings is assigned the closest one.

    A 2.0 m/z difference lies 3.13 mDa from the 1.996874 spacing and 6.71 mDa
    from 13C2 (2.00671); a 2.006 difference lies 9.13 and 0.71 mDa from them.
    Each difference is within the default 50 ppm error of both spacings.
    """
    features = pd.DataFrame({"mz": [300.0, child_mz], "intensity": [1000.0, 900.0]})

    isotopes = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], delta=deimos.isotopes.SPACINGS
    )

    row = isotopes.iloc[0]
    assert row["dx"] == [dx]
    assert row["multiple"] == [multiple]
    assert row["error"] == pytest.approx([error], abs=1e-12)


@pytest.mark.parametrize(
    ("max_isotopes", "expected"),
    [(1, 0), (2, 1)],
    ids=["first-multiple", "second-multiple"],
)
def test_detect_spacings_search_multiples_of_each_spacing(max_isotopes, expected):
    """Test that max_isotopes bounds the multiple of each spacing searched.

    The child is the 37Cl2 peak, 3.99410 above its parent, which is twice the
    second spacing in SPACINGS, so it is found only when the search reaches the
    second multiple.
    """
    features = pd.DataFrame({"mz": [150.0, 153.9941], "intensity": [1000.0, 640.0]})

    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz"],
        tol=[0.1],
        delta=deimos.isotopes.SPACINGS,
        max_isotopes=max_isotopes,
    )

    assert len(isotopes) == expected


def test_detect_float_spacing_matches_one_element_sequence():
    """Test that a float spacing gives the same result as a sequence of it.

    The children lie at one and two multiples of 1.996874, so the multiples
    are 1 and 2.
    """
    features = pd.DataFrame(
        {"mz": [150.0, 151.996874, 153.993748], "intensity": [1000.0, 320.0, 100.0]}
    )

    single = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], delta=1.996874, max_isotopes=2
    )
    sequence = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], delta=[1.996874], max_isotopes=2
    )

    assert single.iloc[0]["multiple"] == [1.0, 2.0]
    pd.testing.assert_frame_equal(single, sequence)


@pytest.mark.parametrize(
    "delta",
    [-1.003355, [1.003355, 0.0], []],
    ids=["negative-float", "zero-in-sequence", "empty-sequence"],
)
def test_detect_rejects_spacings_that_are_not_positive(delta):
    """Test that a spacing of zero or less, or no spacing, is rejected."""
    features = pd.DataFrame({"mz": [150.0, 151.0], "intensity": [1000.0, 100.0]})

    with pytest.raises(ValueError, match="positive"):
        deimos.isotopes.detect(features, dims=["mz"], tol=[0.1], delta=delta)


def test_detect_unsorted_input():
    """Test that detect() handles unsorted (descending m/z) input correctly."""
    # Sorted ascending reference
    features_sorted = pd.DataFrame(
        {
            "mz": [557.975, 558.312, 558.644],
            "drift_time": [285.84, 285.97, 285.93],
            "intensity": [1870, 831232, 792192],
        }
    )

    # Same features but in reverse m/z order
    features_unsorted = features_sorted.iloc[::-1].reset_index(drop=True)

    kwargs = {
        "dims": ["mz", "drift_time"],
        "tol": [0.02, 1.0],
        "max_charge": 3,
        "max_isotopes": 1,
        "max_error": 50e-6,
        "require_lower_intensity": False,
    }

    result_sorted = deimos.isotopes.detect(features_sorted, **kwargs)
    result_unsorted = deimos.isotopes.detect(features_unsorted, **kwargs)

    # Both orderings must yield the same number of isotopic patterns
    assert len(result_sorted) == len(result_unsorted)
    # Patterns found should be consistent (non-empty)
    assert len(result_sorted) > 0


def test_detect_preserves_input_indices():
    """Test that output idx values match the original input DataFrame indices.

    This is critical when features have non-sequential indices (e.g., after filtering
    or when passing a subset of data). The function internally sorts by m/z, but must
    return idx values that correspond to the original input indices, not the sorted order.
    """
    # Create features with non-sequential indices to verify they're preserved
    features = pd.DataFrame(
        {
            "mz": [387.024, 388.027, 389.031, 390.034],  # Isotopic pattern
            "drift_time": [285.5, 285.6, 285.5, 285.6],
            "retention_time": [12.3, 12.3, 12.3, 12.3],
            "intensity": [100000, 80000, 50000, 30000],
        },
        # Non-sequential original indices; pandas 2 annotates index with a
        # protocol that list fails to match
        index=[100, 200, 300, 400],  # pyright: ignore[reportArgumentType]
    )

    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz", "drift_time", "retention_time"],
        tol=[0.1, 0.3, 0.5],
        delta=1.003355,
        max_isotopes=5,
        max_charge=1,
        max_error=50e-6,
    )

    # Should find the isotopic pattern starting at idx 100
    assert len(isotopes) > 0

    # The parent feature should have idx=100 (original index of first feature)
    parent_row = isotopes.iloc[0]
    assert parent_row["idx"] == 100, f"Expected parent idx=100, got {parent_row['idx']}"

    # The isotope indices should be 200, 300, 400 (original indices of isotopes)
    expected_isotope_indices = [200, 300, 400]
    actual_isotope_indices = parent_row["idx_iso"]

    # Check that we found at least some of the isotopes
    assert len(actual_isotope_indices) > 0

    # Check that all returned isotope indices are from the original input indices
    for iso_idx in actual_isotope_indices:
        assert iso_idx in expected_isotope_indices, (
            f"Isotope idx {iso_idx} not in expected original indices {expected_isotope_indices}"
        )

    # Verify that none of the indices are the sorted positions (0, 1, 2, 3)
    all_indices = [parent_row["idx"]] + parent_row["idx_iso"]
    for idx in all_indices:
        assert idx >= 100, (
            f"Index {idx} appears to be a sorted position, not original index"
        )


def test_detect_list_columns_keep_repeated_values():
    """Test that equal values in different isotopes are all kept.

    The M+1 and M+2 isotopes share an intensity, so the intensity and decay
    lists repeat a value. Each list must still hold one entry per isotope.
    """
    features = pd.DataFrame(
        {
            "mz": [300.0, 301.003355, 302.00671],
            "intensity": [1000.0, 500.0, 500.0],
        }
    )

    isotopes = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], max_isotopes=2, max_charge=1
    )

    row = isotopes.iloc[0]
    assert len(isotopes) == 1
    assert row["n"] == 2
    assert row["idx_iso"] == [1, 2]
    assert row["multiple"] == [1.0, 2.0]
    assert row["intensity_iso"] == [500.0, 500.0]
    assert row["decay"] == [0.5, 0.5]


def test_detect_lists_every_candidate_per_isotope():
    """Test that each isotope position reports every candidate within tolerance.

    Two features fall within tolerance of the parent's M+1 position, 0.855 and
    0.145 mDa from the expected spacing. Both are listed, every list column stays
    aligned with ``idx_iso``, and ``n`` counts the two positions.
    """
    features = pd.DataFrame(
        {
            "mz": [300.0, 301.0025, 301.0035, 302.00671],
            "intensity": [1000.0, 450.0, 400.0, 100.0],
        }
    )

    isotopes = deimos.isotopes.detect(
        features, dims=["mz"], tol=[0.1], max_isotopes=2, max_charge=1
    )

    row = isotopes.iloc[0]
    assert len(isotopes) == 1
    assert row["n"] == 2
    assert row["multiple"] == [1.0, 1.0, 2.0]
    assert row["dx"] == [1.003355, 1.003355, 2.00671]
    assert row["idx_iso"] == [1, 2, 3]
    assert row["mz_iso"] == [301.0025, 301.0035, 302.00671]
    assert row["intensity_iso"] == [450.0, 400.0, 100.0]
    assert row["decay"] == [0.45, 0.4, 0.1]
    assert row["error"] == pytest.approx(
        [0.000855 / 301.0025, 0.000145 / 301.0035, 0.0], abs=1e-12
    )


def test_detect_n_counts_expected_differences_across_spacings():
    """Test that ``n`` counts expected differences, not candidates or multiples.

    One candidate lies at 13C and two at the second spacing in SPACINGS, so the row
    has three candidates of one multiple at two expected differences.
    """
    features = pd.DataFrame(
        {
            "mz": [150.0, 151.0035, 151.9968, 151.9971],
            "intensity": [1000.0, 100.0, 320.0, 300.0],
        }
    )

    isotopes = deimos.isotopes.detect(
        features,
        dims=["mz"],
        tol=[0.1],
        delta=deimos.isotopes.SPACINGS,
        max_isotopes=1,
    )

    row = isotopes.iloc[0]
    assert row["idx_iso"] == [1, 2, 3]
    assert row["dx"] == [1.003355, 1.996874, 1.996874]
    assert row["multiple"] == [1.0, 1.0, 1.0]
    assert row["n"] == 2
