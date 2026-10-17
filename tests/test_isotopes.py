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
    assert isotopes["dx"] == [1.003355, 2.00671, 3.010065, 4.01342]


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

    3,000 features lie within 4.1 m/z of each other, so the search covers
    about 4.5 million pairs. Each parent shares its retention time only with
    its own M+1 isotope, so exactly those 1,500 pairs are reported.
    """
    import numpy as np

    k = np.arange(1500)
    features = pd.DataFrame(
        {
            "mz": np.concatenate([300.0 + 0.002 * k, 301.003355 + 0.002 * k]),
            "retention_time": np.concatenate([10.0 * k, 10.0 * k]),
            "intensity": np.concatenate([np.full(1500, 1000.0), np.full(1500, 500.0)]),
        }
    )

    isotopes = deimos.isotopes.detect(
        features, dims=["mz", "retention_time"], tol=[0.1, 0.3], max_isotopes=4
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
    import tracemalloc

    import numpy as np

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
