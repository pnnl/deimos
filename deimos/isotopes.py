from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np
import pandas as pd

import deimos

# Candidate pairs that detect checks at once, about 4 million, to bound memory
_BLOCK_PAIRS = 2**22

# Margin added to each m/z candidate window in detect, so rounding drops no match
_WINDOW_PADDING = 1e-6

# Spacings (Da) from NIST atomic masses: 13C, and the midpoint of 34S, 37Cl, and 81Br,
# which lie within 2.2 mDa of each other
SPACINGS = (1.003355, 1.996874)


def _candidate_pairs(
    mz: np.ndarray,
    expected: np.number,
    half_width: np.ndarray,
    mz_tol: float,
    others: list[tuple[np.ndarray, float]],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield blocks of child and parent positions near one expected m/z difference.

    `mz` must ascend. Each child's window spans `half_width` around `expected` below
    its m/z, and yielded pairs lie within `mz_tol` of `expected` and within tolerance
    in each of `others`, a list of values and their tolerances.
    """

    # Locate each child's window, limited to parents before the child; a negative
    # tolerance can leave a child with none
    n_features = len(mz)
    low = np.searchsorted(mz, mz - expected - half_width, side="left")
    high = np.searchsorted(mz, mz - expected + half_width, side="right")
    counts = np.maximum(np.minimum(high, np.arange(n_features)) - low, 0)
    offsets = np.concatenate(([0], np.cumsum(counts)))

    # Check candidates in blocks of whole children to bound memory
    start = 0
    while start < n_features:
        # End the block at the last child within `_BLOCK_PAIRS` pairs, taking at
        # least one child
        stop = np.searchsorted(offsets, offsets[start] + _BLOCK_PAIRS, side="right") - 1
        stop = max(stop, start + 1)

        # Expand windows to candidate pairs, pairing each child with every parent
        # in its window
        row = np.repeat(np.arange(start, stop), counts[start:stop])
        col = low[row] + np.arange(len(row)) - (offsets[row] - offsets[start])

        # Retain pairs within tolerance for m/z, drift time, retention time, etc.
        keep = np.abs(mz[row] - mz[col] - expected) <= mz_tol
        for values, dim_tol in others:
            keep &= np.abs(values[row] - values[col]) <= dim_tol
        yield row[keep], col[keep]
        start = stop


def detect(
    features,
    dims=None,
    tol=None,
    delta: float | Sequence[float] = 1.003355,
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

    1. Finds candidate pairs by binary search in m/z order around each expected mass
       difference, across all charge states and isotope multiples, and retains those
       within tolerance in every dimension
    2. Filters pairs by mass accuracy and intensity decay (parent > isotope)
    3. Groups isotopes by their parent feature

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
    delta : float or sequence of float, default=1.003355
        Expected spacing between isotopes in Da (e.g., C13 = 1.003355). Spacings
        must be positive. With a sequence, such as :data:`SPACINGS`, multiples of
        each spacing are searched, and a pair that matches several is assigned the
        one with the smallest error, shown in ``dx``. Isotopes within a few mDa of a
        spacing also match it and differ only in abundance: 15N, 29Si, and 33S match
        13C, and 34S, 37Cl, 81Br, and 30Si match the second spacing in
        :data:`SPACINGS`. A peak with mixed heavy isotopes (37Cl plus two 13C at
        M+4) matches the nearest multiple of one spacing.
    max_isotopes : int, default=4
        Largest multiple of each spacing to search for per parent feature.
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
        Each row represents one parent feature and lists every candidate that passes
        the filters, so one expected mass difference can repeat when several features
        match it. List columns are aligned, with one entry per candidate. Columns:

        - mz: Parent m/z
        - intensity: Parent intensity
        - charge: Charge state
        - idx: Parent feature index
        - multiple: List of multiples of the matched spacing (e.g., [1, 2, 3] for
          13C M+1, M+2, M+3)
        - dx: List of expected mass differences matched by each candidate
        - mz_iso: List of isotope m/z values
        - intensity_iso: List of isotope intensities
        - idx_iso: List of isotope feature indices
        - error: List of relative errors, as fractions (6.7e-06 is 6.7 ppm)
        - decay: List of intensity ratios (isotope/parent)
        - n: Number of distinct expected mass differences matched, which is below the
          list length when several candidates match one difference

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

    # Check that the isotope spacings are positive
    spacings = np.asarray(delta, dtype=float).ravel()
    if spacings.size == 0 or (spacings <= 0).any():
        raise ValueError("Isotope spacings must be one or more positive values.")

    # Preserve original indices before sorting
    # Store original index as a column so we can return it in results
    features = features.copy()
    features["original_idx"] = features.index

    # Sort by m/z to ensure lower indices correspond to lower m/z values.
    # This is required for the candidate search to correctly assign parent
    # (lower m/z) and child (higher m/z) features.
    features = features.sort_values("mz").reset_index(drop=True)

    n_features = len(features)

    # Step 1: Pre-compute all expected mass differences for all charge/multiple
    # combinations, so only the m/z windows around them are searched
    charges = np.arange(1, max_charge + 1)
    multiples = np.arange(1, max_isotopes + 1)
    expected_deltas = np.outer(multiples, np.divide.outer(spacings, charges)).flatten()

    # Create multiple and charge arrays matching expected_deltas
    # Order: [(m1,s1,c1), ..., (m1,s1,cN), (m1,s2,c1), ..., (mM,sS,cN)]
    charge_mult_pairs = np.array(
        [(m, c) for m in multiples for _ in spacings for c in charges]
    )

    # Step 2: Size the m/z window around each expected difference
    # Get m/z in float64, so differences of float32 values are exact
    mz_values = features["mz"].values
    mz_sorted = mz_values.astype(float)

    # Widen the window by the padding so rounding drops no match
    tol_width = np.full(n_features, tol[mz_idx] + _WINDOW_PADDING)

    # Narrow the window to the largest difference that can pass max_error, padded for
    # rounding in float32 m/z; a child at or below zero m/z has no such limit
    error_width = (max_error + np.finfo(np.float32).eps) * mz_sorted
    half_width = np.where(mz_sorted > 0, np.minimum(tol_width, error_width), tol_width)

    # Get the other dimensions in float64, with their tolerances
    others = [(features[dims[i]].values.astype(float), tol[i]) for i in else_idx]

    # Step 3: Search each window, applying the per-pair filters before building the
    # results to bound memory
    intensity_values = features["intensity"].values
    rows, cols, deltas = [], [], []
    for delta_idx, expected in enumerate(expected_deltas):
        for row, col in _candidate_pairs(
            mz_sorted, expected, half_width, tol[mz_idx], others
        ):
            # Remove high mass error matches, using the relative error between
            # observed and expected mass difference
            with np.errstate(divide="ignore", invalid="ignore"):
                error = (
                    np.abs((mz_values[row] - mz_values[col]) - expected)
                    / mz_values[row]
                )
            keep = error < max_error

            # Remove non-decreasing intensity (parent must be more intense than isotope)
            if require_lower_intensity:
                keep &= intensity_values[col] > intensity_values[row]

            rows.append(row[keep])
            cols.append(col[keep])
            deltas.append(np.full(np.count_nonzero(keep), delta_idx))

    # Return an empty frame when no pair lies within tolerance of an expected difference
    if not any(len(row) > 0 for row in rows) and not any(
        len(row) > 0
        for expected in expected_deltas
        for row, _ in _candidate_pairs(
            mz_sorted, expected, tol_width, tol[mz_idx], others
        )
    ):
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
            .agg(list),
        )

    # Order matches by child, then parent, then expected difference; this determines
    # the output list order
    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    delta_indices = np.concatenate(deltas)
    order = np.lexsort((delta_indices, col_idx, row_idx))
    row_idx, col_idx, delta_indices = (
        row_idx[order],
        col_idx[order],
        delta_indices[order],
    )

    # Step 4: Build results DataFrame
    # Features at column index are parents, and features at row index are their
    # isotopic children
    matched_parents = col_idx
    matched_children = row_idx
    matched_charges = charge_mult_pairs[delta_indices, 1]
    matched_multiples = charge_mult_pairs[delta_indices, 0]
    matched_deltas = expected_deltas[delta_indices]

    # Create DataFrame directly from arrays (more efficient than concatenating)
    isotopes = pd.DataFrame(
        {
            "mz": mz_values[matched_parents],
            "intensity": intensity_values[matched_parents],
            "charge": matched_charges.astype(float),
            "multiple": matched_multiples.astype(float),
            "dx": matched_deltas,
            "mz_iso": mz_values[matched_children],
            "intensity_iso": intensity_values[matched_children],
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

    # Assign each pair the spacing with the smallest error when several match it
    if len(spacings) > 1:
        isotopes = (
            isotopes.sort_values("error", kind="stable")
            .drop_duplicates(subset=["idx", "idx_iso", "charge"])
            .sort_index()
        )

    # Step 6: Remove children (features that are themselves isotopes of other features)
    # This ensures we only report parent features
    if not return_all_patterns:
        isotopes = isotopes.loc[~isotopes["idx"].isin(isotopes["idx_iso"]), :]

    # Step 7: Group isotopes by parent feature
    # List one entry per candidate so the list columns stay aligned
    grouped = cast(
        pd.DataFrame,
        isotopes.groupby(by=["mz", "charge", "idx", "intensity"], as_index=False).agg(
            list
        ),
    )
    # Count distinct expected differences per parent, since candidates can share one
    grouped["n"] = [len(set(x)) for x in grouped["dx"].values]

    if grouped.empty:
        return grouped

    # Sort by intensity (most intense first) and number of isotopes (most isotopes first)
    grouped = grouped.sort_values(by=["intensity", "n"], ascending=False).reset_index(
        drop=True
    )
    return grouped
