import numpy as np
import pandas as pd
import scipy

import deimos


def OrderedSet(x):
    return list({k: None for k in x})


def detect(
    features,
    dims=["mz", "drift_time", "retention_time"],
    tol=[0.1, 0.2, 0.3],
    delta=1.003355,
    max_isotopes=4,
    max_charge=1,
    max_error=50e-6,
):
    """
    Perform isotope detection according to expected patterning using optimized
    vectorized operations.
    
    This function identifies isotopic patterns in mass spectrometry data by searching
    for features that differ by expected isotopic mass differences (typically ~1.003 Da
    for C13). The algorithm:
    
    1. Pre-filters candidate pairs that are within tolerance in non-m/z dimensions
       (e.g., drift time, retention time) to reduce search space
    2. Computes m/z distances for candidate pairs only
    3. Vectorized search across all charge states and isotope multiples simultaneously
    4. Filters results by intensity decay (parent > isotope) and mass accuracy
    5. Groups isotopes by their parent feature

    Parameters
    ----------
    features : :obj:`~pandas.DataFrame`
        Input feature coordinates and intensities.
    dims : str or list, default=["mz", "drift_time", "retention_time"]
        Dimensions to perform isotope detection in. Must include 'mz'.
    tol : float or list, default=[0.1, 0.2, 0.3]
        Tolerance in each dimension to be considered a match. For m/z, this is the
        absolute tolerance for the isotopic mass difference.
    delta : float, default=1.003355
        Expected spacing between isotopes in Da (e.g., C13 = 1.003355).
    max_isotopes : int, default=4
        Maximum number of isotopes to search for per parent feature.
    max_charge : int, default=1
        Maximum charge state to consider (isotopic spacing = delta / charge).
    max_error : float, default=50e-6
        Maximum relative error between expected and observed isotopic mass difference.

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

    """

    # Safely cast to list
    dims = deimos.utils.safelist(dims)
    tol = deimos.utils.safelist(tol)

    # Check dims
    deimos.utils.check_length([dims, tol])

    # Isolate mz dimension
    mz_idx = dims.index("mz")
    else_idx = [i for i, j in enumerate(dims) if i != mz_idx]

    n_features = len(features)
    
    # Step 1: Build candidate mask for non-m/z dimensions
    # Only consider pairs within tolerance for drift time, retention time, etc.
    # This dramatically reduces the search space for the m/z comparison
    candidate_mask = np.ones((n_features, n_features), dtype=bool)
    
    for i in else_idx:
        arr = features[dims[i]].values.reshape((-1, 1))
        dist = scipy.spatial.distance.cdist(arr, arr)
        candidate_mask &= (dist <= tol[i])
    
    # Apply lower triangular mask to avoid duplicate comparisons (i < j)
    # This ensures each pair is only evaluated once
    candidate_mask = np.tril(candidate_mask, k=-1)
    
    # Get candidate pairs (row, col indices where mask is True)
    # With tril, row > col (row index > column index), meaning we're in lower triangle
    row_idx, col_idx = np.where(candidate_mask)
    
    if len(row_idx) == 0:
        # No candidates found - return empty grouped DataFrame with correct structure
        return pd.DataFrame(columns=[
            "mz", "charge", "idx", "intensity", "multiple", "dx",
            "mz_iso", "intensity_iso", "idx_iso", "error", "decay", "n"
        ]).groupby(by=["mz", "charge", "idx", "intensity"], as_index=False).agg(OrderedSet)
    
    # Step 2: Compute m/z distances for candidate pairs only
    # This avoids computing a full n x n distance matrix for all features
    mz_values = features["mz"].values
    
    # Build m/z distance matrix for candidates
    # The distance matrix is computed, then element-wise multiplied by the candidate mask
    mz_arr = mz_values.reshape((-1, 1))
    mz_dist_full = scipy.spatial.distance.cdist(mz_arr, mz_arr)
    
    # Apply candidate mask to get distances for valid pairs only
    mz_dist_masked = np.multiply(mz_dist_full, candidate_mask)
    
    # Extract m/z differences for our candidate pairs
    # This represents |mz[row] - mz[col]|, always positive
    mz_diffs = mz_dist_masked[row_idx, col_idx]
    
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
        return pd.DataFrame(columns=[
            "mz", "charge", "idx", "intensity", "multiple", "dx",
            "mz_iso", "intensity_iso", "idx_iso", "error", "decay", "n"
        ]).groupby(by=["mz", "charge", "idx", "intensity"], as_index=False).agg(OrderedSet)
    
    # Step 4: Build results DataFrame
    # Extract matched parent and child features
    matched_parents = parent_idx[candidate_indices]
    matched_children = child_idx[candidate_indices]
    matched_charges = charge_mult_pairs[delta_indices, 1]
    matched_multiples = charge_mult_pairs[delta_indices, 0]
    matched_deltas = expected_deltas[delta_indices]
    
    # Create DataFrame directly from arrays (more efficient than concatenating)
    isotopes = pd.DataFrame({
        "mz": mz_values[matched_parents],
        "intensity": features["intensity"].values[matched_parents],
        "charge": matched_charges.astype(float),
        "multiple": matched_multiples.astype(float),
        "dx": matched_deltas,
        "mz_iso": mz_values[matched_children],
        "intensity_iso": features["intensity"].values[matched_children],
        "idx": features.index.values[matched_parents],
        "idx_iso": features.index.values[matched_children],
    })

    # Step 5: Compute quality metrics
    # Relative error between observed and expected mass difference
    isotopes["error"] = (
        np.abs((isotopes["mz_iso"] - isotopes["mz"]) - isotopes["dx"]) / isotopes["mz_iso"]
    )
    # Intensity ratio (should decay for true isotopes)
    isotopes["decay"] = isotopes["intensity_iso"] / isotopes["intensity"]

    # Step 6: Apply filters
    # Remove non-decreasing intensity (parent must be more intense than isotope)
    isotopes = isotopes.loc[isotopes["intensity"] > isotopes["intensity_iso"], :]

    # Remove high mass error matches
    isotopes = isotopes.loc[isotopes["error"] < max_error, :]

    # Remove children (features that are themselves isotopes of other features)
    # This ensures we only report parent features
    isotopes = isotopes.loc[~isotopes["idx"].isin(isotopes["idx_iso"]), :]

    # Step 7: Group isotopes by parent feature
    # OrderedSet preserves insertion order for the list columns
    grouped = isotopes.groupby(
        by=["mz", "charge", "idx", "intensity"], as_index=False
    ).agg(OrderedSet)
    # Count number of isotopes per parent
    grouped["n"] = [len(x) for x in grouped["multiple"].values]

    if grouped.empty:
        return grouped
    
    # Sort by intensity (most intense first) and number of isotopes (most isotopes first)
    grouped = grouped.sort_values(by=["intensity", "n"], ascending=False).reset_index(
        drop=True
    )
    return grouped
