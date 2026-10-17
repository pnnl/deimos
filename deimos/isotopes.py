from typing import cast

import numpy as np
import pandas as pd

import deimos


def OrderedSet(x):
    return list({k: None for k in x})


def detect(
    features,
    dims=None,
    tol=None,
    delta=1.003355,
    max_isotopes=4,
    max_charge=1,
    max_error=50e-6,
    require_lower_intensity=True,
    return_all_patterns=False,
) -> pd.DataFrame:
    """
    Perform isotope detection according to expected patterning using optimized
    vectorized operations.

    This function identifies isotopic patterns in mass spectrometry data by searching
    for features that differ by expected isotopic mass differences (typically ~1.003 Da
    for C13). The algorithm:

    1. Finds candidate pairs by binary search in m/z order, up to the largest expected
       spacing, and keeps those within tolerance in the non-m/z dimensions
    2. Computes m/z distances for candidate pairs only
    3. Vectorized search across all charge states and isotope multiples simultaneously
    4. Filters results by intensity decay (parent > isotope) and mass accuracy
    5. Groups isotopes by their parent feature

    The function supports both multi-dimensional data (LC-IMS-MS) and single-dimension
    spectra. For single spectra, use dims=['mz'] and tol=[mz_tolerance].

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities. Must contain 'mz' and 'intensity' columns.
    dims : str or list, default=["mz", "drift_time", "retention_time"]
        Dimensions to perform isotope detection in. Must include 'mz'.
    tol : float or list, default=[0.1, 0.2, 0.3]
        Tolerance in each dimension to be considered a match. Length must match dims.
        For m/z, this is the absolute tolerance for the isotopic mass difference.
    delta : float, default=1.003355
        Expected spacing between isotopes in Da (e.g., C13 = 1.003355).
    max_isotopes : int, default=4
        Maximum number of isotopes to search for per parent feature.
    max_charge : int, default=1
        Maximum charge state to consider (isotopic spacing = delta / charge).
    max_error : float, default=50e-6
        Maximum relative error between expected and observed isotopic mass difference.
    require_lower_intensity : bool, default=True
        If True, filter isotope candidates to only include those with lower intensity
        than their parent feature. This enforces the expected decay pattern where isotope
        peaks are less abundant than the monoisotopic peak. For 13C isotopes, this is
        typically valid up to ~1400 m/z (natural abundance ~1.1%). However, for heavier
        elements with higher natural abundance (e.g., Fe-54/56 at ~5.8%/91.7%) or enriched
        isotope labeling experiments, set this to False to detect patterns where isotopes
        may have equal or greater intensity than the nominal parent peak.
    return_all_patterns : bool, default=False
        If False (default), features that appear as isotope children of another feature
        are removed from the results so only the top-level parent of each chain is
        reported. If True, this filter is skipped and all detected parent–isotope pairs
        are returned, including cases where a feature is simultaneously a child in one
        pair and a parent in another (overlapping isotope series). This is useful for
        large molecules where a low-intensity noise peak may incorrectly match as a
        parent, allowing downstream filtering (e.g., by intensity ratio) to remove
        invalid patterns.

    Returns
    -------
    :obj:`~pandas.DataFrame`
        Features grouped by isotopic pattern, sorted by intensity and number of isotopes.
        Each row represents one parent feature with columns:

        - mz: Parent m/z
        - intensity: Parent intensity
        - charge: Charge state
        - idx: Parent feature index
        - multiple: List of isotope multiples found (e.g., [1, 2, 3] for M+1, M+2, M+3)
        - dx: List of expected mass differences
        - mz_iso: List of isotope m/z values
        - intensity_iso: List of isotope intensities
        - idx_iso: List of isotope feature indices
        - error: List of relative errors
        - decay: List of intensity ratios (isotope/parent)
        - n: Number of isotopes detected

    Examples
    --------
    Multi-dimensional data (LC-IMS-MS):

    >>> isotopes = deimos.isotopes.detect(
    ...     ms1_peaks,
    ...     dims=["mz", "drift_time", "retention_time"],
    ...     tol=[0.1, 0.2, 0.3],
    ...     max_isotopes=5
    ... )

    Single spectrum (m/z only):

    >>> isotopes = deimos.isotopes.detect(
    ...     spectrum,
    ...     dims=["mz"],
    ...     tol=[0.1],
    ...     max_isotopes=5
    ... )

    """

    # Set mutable argument defaults
    if dims is None:
        dims = ["mz", "drift_time", "retention_time"]
    if tol is None:
        tol = [0.1, 0.2, 0.3]

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)

    # Check dims
    deimos.utils.check_length([dims, tol])

    # Isolate mz dimension
    mz_idx = dims.index("mz")
    else_idx = [i for i, j in enumerate(dims) if i != mz_idx]

    # Preserve original indices before sorting
    # Store original index as a column so we can return it in results
    features = features.copy()
    features["original_idx"] = features.index

    # Sort by m/z to ensure lower indices correspond to lower m/z values.
    # This is required for the lower-triangular candidate mask logic to correctly
    # assign parent (lower m/z) and child (higher m/z) features.
    features = features.sort_values("mz").reset_index(drop=True)

    n_features = len(features)

    # Step 1: Find candidate pairs without building an n x n mask
    # No expected spacing exceeds max_isotopes * |delta|, so each child's candidate
    # parents sit in an m/z window before it
    mz_sorted = features["mz"].values.astype(float)

    # Widen the window by 1e-6 so float rounding at its edge cannot drop a pair that
    # the exact tolerance test in step 3 accepts
    max_span = max_isotopes * np.max(np.abs(delta)) + tol[mz_idx] + 1e-6
    first = np.searchsorted(mz_sorted, mz_sorted - max_span, side="left")
    counts = np.maximum(np.arange(n_features) - first, 0)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    # Enumerate pairs child by child in blocks of about 4 million to bound memory,
    # keeping those within tolerance for drift time, retention time, etc.
    others = [(features[dims[i]].values.astype(float), tol[i]) for i in else_idx]
    rows, cols = [], []
    start = 0
    while start < n_features:
        stop = np.searchsorted(offsets, offsets[start] + 2**22, side="right") - 1
        stop = max(stop, start + 1)
        row = np.repeat(np.arange(start, stop), counts[start:stop])
        col = first[row] + np.arange(len(row)) - (offsets[row] - offsets[start])
        keep = np.ones(len(row), dtype=bool)
        for values, dim_tol in others:
            keep &= np.abs(values[row] - values[col]) <= dim_tol
        rows.append(row[keep])
        cols.append(col[keep])
        start = stop

    # Rows ascend, and parents ascend within each row; this sets the output list order
    row_idx = np.concatenate(rows) if rows else np.empty(0, dtype=int)
    col_idx = np.concatenate(cols) if cols else np.empty(0, dtype=int)

    if len(row_idx) == 0:
        # No candidates found - return empty grouped DataFrame with correct structure
        # pandas 2 annotates columns with a protocol that list fails to match
        return cast(
            pd.DataFrame,
            pd.DataFrame(
                columns=[  # pyright: ignore[reportArgumentType]
                    "mz",
                    "charge",
                    "idx",
                    "intensity",
                    "multiple",
                    "dx",
                    "mz_iso",
                    "intensity_iso",
                    "idx_iso",
                    "error",
                    "decay",
                    "n",
                ]
            )
            .groupby(by=["mz", "charge", "idx", "intensity"], as_index=False)
            .agg(OrderedSet),
        )

    # Step 2: Compute m/z distances for candidate pairs only
    # This avoids computing a full n x n distance matrix for all features
    mz_values = features["mz"].values

    # Extract m/z differences for our candidate pairs
    # This represents |mz[row] - mz[col]|, always positive
    mz_diffs = np.abs(mz_sorted[row_idx] - mz_sorted[col_idx])

    # Feature assignments:
    # - Features at column index are assigned as parent (lower triangle: col < row)
    # - Features at row index are assigned as isotopic child
    parent_idx = col_idx
    child_idx = row_idx

    # Step 3: Vectorized search for isotopic patterns
    # Pre-compute all expected mass differences for all charge/multiple combinations
    # This allows us to check all possibilities in a single vectorized operation
    charges = np.arange(1, max_charge + 1)
    multiples = np.arange(1, max_isotopes + 1)
    expected_deltas = np.outer(multiples, delta / charges).flatten()  # All combinations

    # Create charge and multiple arrays matching expected_deltas
    # Order: [(m1,c1), (m1,c2), ..., (m1,cN), (m2,c1), ..., (mM,cN)]
    charge_mult_pairs = np.array([(m, c) for m in multiples for c in charges])

    # Vectorized search: for each candidate pair, check all expected deltas
    # Shape: (n_candidates, n_delta_combinations)
    mz_diff_matrix = mz_diffs[:, np.newaxis]
    expected_delta_matrix = expected_deltas[np.newaxis, :]

    # Find matches within m/z tolerance
    matches = np.abs(mz_diff_matrix - expected_delta_matrix) <= tol[mz_idx]

    # Get indices of matches
    candidate_indices, delta_indices = np.where(matches)

    if len(candidate_indices) == 0:
        # No isotope matches found - return empty grouped DataFrame with correct structure
        # pandas 2 annotates columns with a protocol that list fails to match
        return cast(
            pd.DataFrame,
            pd.DataFrame(
                columns=[  # pyright: ignore[reportArgumentType]
                    "mz",
                    "charge",
                    "idx",
                    "intensity",
                    "multiple",
                    "dx",
                    "mz_iso",
                    "intensity_iso",
                    "idx_iso",
                    "error",
                    "decay",
                    "n",
                ]
            )
            .groupby(by=["mz", "charge", "idx", "intensity"], as_index=False)
            .agg(OrderedSet),
        )

    # Step 4: Build results DataFrame
    # Extract matched parent and child features
    matched_parents = parent_idx[candidate_indices]
    matched_children = child_idx[candidate_indices]
    matched_charges = charge_mult_pairs[delta_indices, 1]
    matched_multiples = charge_mult_pairs[delta_indices, 0]
    matched_deltas = expected_deltas[delta_indices]

    # Create DataFrame directly from arrays (more efficient than concatenating)
    isotopes = pd.DataFrame(
        {
            "mz": mz_values[matched_parents],
            "intensity": features["intensity"].values[matched_parents],
            "charge": matched_charges.astype(float),
            "multiple": matched_multiples.astype(float),
            "dx": matched_deltas,
            "mz_iso": mz_values[matched_children],
            "intensity_iso": features["intensity"].values[matched_children],
            "idx": features["original_idx"].values[matched_parents],
            "idx_iso": features["original_idx"].values[matched_children],
        }
    )

    # Step 5: Compute quality metrics
    # Relative error between observed and expected mass difference
    isotopes["error"] = (
        np.abs((isotopes["mz_iso"] - isotopes["mz"]) - isotopes["dx"])
        / isotopes["mz_iso"]
    )
    # Intensity ratio (should decay for true isotopes)
    isotopes["decay"] = isotopes["intensity_iso"] / isotopes["intensity"]

    # Step 6: Apply filters
    # Conditionally filter by intensity decay pattern
    if require_lower_intensity:
        # Remove non-decreasing intensity (parent must be more intense than isotope)
        isotopes = isotopes.loc[isotopes["intensity"] > isotopes["intensity_iso"], :]

    # Remove high mass error matches
    isotopes = isotopes.loc[isotopes["error"] < max_error, :]

    # Remove children (features that are themselves isotopes of other features)
    # This ensures we only report parent features
    if not return_all_patterns:
        isotopes = isotopes.loc[~isotopes["idx"].isin(isotopes["idx_iso"]), :]

    # Step 7: Group isotopes by parent feature
    # OrderedSet preserves insertion order for the list columns
    grouped = cast(
        pd.DataFrame,
        isotopes.groupby(by=["mz", "charge", "idx", "intensity"], as_index=False).agg(
            OrderedSet
        ),
    )
    # Count number of isotopes per parent
    grouped["n"] = [len(x) for x in grouped["multiple"].values]

    if grouped.empty:
        return grouped

    # Sort by intensity (most intense first) and number of isotopes (most isotopes first)
    grouped = grouped.sort_values(by=["intensity", "n"], ascending=False).reset_index(
        drop=True
    )
    return grouped
