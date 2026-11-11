from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .io_utils import atomic_pickle_dump


def compute_msa_features(output_paths: Dict[str, Any]) -> Tuple[int, int]:
    """Compute Meff-weighted per-position features from a compressed MSA and write a (L, 28) float32 matrix."""
    compressed_path_value = output_paths.get("compressed_alignment")
    if not compressed_path_value:
        raise FileNotFoundError("output_paths['compressed_alignment'] missing")
    compressed_path = Path(str(compressed_path_value))

    preferred_weights = [
        output_paths.get("row_weights_theta0p70_path"),
        output_paths.get("row_weights_theta0p80_path"),
        str(compressed_path.with_suffix("")) + "_row_weights_theta0p70.npy",
        str(compressed_path.with_suffix("")) + "_row_weights_theta0p80.npy",
        str(compressed_path.with_suffix("")) + "_row_weights.npy",
    ]
    weights_path = None
    for candidate in preferred_weights:
        if candidate and Path(str(candidate)).exists():
            weights_path = Path(str(candidate))
            break
    if weights_path is None:
        raise FileNotFoundError("Row weights not found for MSA context generation")

    context_path = Path(
        output_paths.get("msa_context")
        or compressed_path.with_name(compressed_path.stem + "_context.pkl")
    )
    context_path.parent.mkdir(parents=True, exist_ok=True)

    with open(compressed_path, "rb") as handle:
        matrix = np.asarray(pickle.load(handle), dtype=np.int32)
    if matrix.ndim != 2:
        raise ValueError(f"Compressed MSA must be 2D; got {matrix.shape}")

    weights = np.load(str(weights_path)).astype(np.float64, copy=False).ravel()
    if weights.shape[0] != matrix.shape[0]:
        raise ValueError(f"Row weights length {weights.shape[0]} does not match MSA rows {matrix.shape[0]}")

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Row weights sum to zero; cannot compute weighted features")

    num_rows, length = matrix.shape
    anchor_row = matrix[0]
    features = np.zeros((length, 28), dtype=np.float32)

    for col_index in range(length):
        column_tokens = matrix[:, col_index]
        counts = np.bincount(column_tokens, weights=weights, minlength=21).astype(np.float64)
        gap_weight = float(counts[0])
        nongap_weights = counts[1:]
        nongap_total = float(np.sum(nongap_weights))
        gap_frequency = gap_weight / total_weight
        occupancy = 1.0 - gap_frequency

        if nongap_total > 0.0:
            probabilities = nongap_weights / nongap_total
            positive = probabilities > 0
            entropy_bits = float(-(probabilities[positive] * np.log2(probabilities[positive])).sum())
            inv_sum_p2 = float(1.0 / max(float(np.sum(probabilities * probabilities)), 1e-12))
            aa_freq = probabilities
            mode_index = int(np.argmax(nongap_weights))
            mode_token = mode_index + 1
            p_mode = float(aa_freq[mode_index])
        else:
            aa_freq = np.zeros((20,), dtype=np.float64)
            entropy_bits = 0.0
            inv_sum_p2 = 0.0
            mode_token = 0
            p_mode = 0.0

        anchor_token = int(anchor_row[col_index])
        p_anchor = float(aa_freq[anchor_token - 1]) if 1 <= anchor_token <= 20 else 0.0
        is_anchor_mode = 1.0 if (anchor_token != 0 and anchor_token == mode_token) else 0.0
        delta_mode_anchor = float(p_mode - p_anchor)

        features[col_index, 0:20] = aa_freq.astype(np.float32)
        features[col_index, 20] = float(gap_frequency)
        features[col_index, 21] = float(entropy_bits)
        features[col_index, 22] = float(inv_sum_p2)
        features[col_index, 23] = float(occupancy)
        features[col_index, 24] = float(p_anchor)
        features[col_index, 25] = float(1.0 - p_anchor)
        features[col_index, 26] = float(is_anchor_mode)
        features[col_index, 27] = float(delta_mode_anchor)

    atomic_pickle_dump(context_path, features)
    output_paths["msa_context"] = str(context_path)
    output_paths["context_features"] = str(context_path)
    return int(length), 28
