from __future__ import annotations
import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import tensorflow as tf

logger = logging.getLogger("protein-experiments")


@dataclass
class ConditioningMeta:
    """Container describing conditioning status, requested and resolved types, per-modality slices, and total context dimension."""
    enabled: bool
    requested_types: List[str]
    resolved_types: List[str]
    slices: Dict[str, Tuple[int, int]]
    context_dim: int


def load_matrix(path_text: Optional[str]) -> Optional[np.ndarray]:
    """Load a pickled two-dimensional array and return a float32 copy or None when missing or invalid."""
    if not path_text:
        return None
    matrix_path = Path(str(path_text))
    if not matrix_path.exists():
        return None
    with open(matrix_path, "rb") as handle:
        array = np.asarray(pickle.load(handle))
    if array.ndim != 2 or array.size == 0:
        return None
    return array.astype(np.float32, copy=False)


def normalize_types(requested: Sequence[str]) -> List[str]:
    """Normalize requested modality names to lowercase tokens and drop empty entries."""
    tokens: List[str] = []
    for value in requested:
        token = str(value).strip().lower()
        if token:
            tokens.append(token)
    return tokens


def load_conditioning(
    output_paths: Dict[str, Any],
    *,
    enabled: bool = True,
    requested_types: Sequence[str] = ("msa",),
) -> Tuple[Optional[Any], ConditioningMeta]:
    """Load and concatenate requested conditioning modalities, returning a context tensor or array and a metadata record."""
    normalized = normalize_types(requested_types)
    if not normalized:
        normalized = ["msa"]
    if not enabled:
        meta = ConditioningMeta(
            enabled=False,
            requested_types=normalized,
            resolved_types=[],
            slices={},
            context_dim=0,
        )
        return None, meta
    sources: List[Tuple[str, np.ndarray]] = []
    if "msa" in normalized:
        msa_matrix = load_matrix(output_paths.get("msa_context") or output_paths.get("context_features"))
        if msa_matrix is not None:
            sources.append(("msa", msa_matrix))
    if "structure" in normalized:
        structure_matrix = load_matrix(output_paths.get("structure_context"))
        if structure_matrix is not None:
            sources.append(("structure", structure_matrix))
    if "text" in normalized:
        text_matrix = load_matrix(output_paths.get("text_context"))
        if text_matrix is not None:
            sources.append(("text", text_matrix))
    if not sources:
        checked_keys = ["msa_context", "context_features", "structure_context", "text_context"]
        logger.warning(
            "Conditioning requested but no sources were found. Checked keys: %s",
            ",".join(checked_keys),
        )
        meta = ConditioningMeta(
            enabled=False,
            requested_types=normalized,
            resolved_types=[],
            slices={},
            context_dim=0,
        )
        return None, meta
    first_length = int(sources[0][1].shape[0])
    for modality_name, matrix in sources:
        if int(matrix.shape[0]) != first_length:
            raise ValueError(
                f"Conditioning source '{modality_name}' has length {matrix.shape[0]} but expected {first_length}"
            )
    slices: Dict[str, Tuple[int, int]] = {}
    matrices: List[np.ndarray] = []
    resolved: List[str] = []
    start = 0
    for modality_name, matrix in sources:
        width = int(matrix.shape[1])
        slices[modality_name] = (start, start + width)
        start += width
        matrices.append(matrix)
        resolved.append(modality_name)
    context_matrix = np.concatenate(matrices, axis=1)
    context_dim = int(context_matrix.shape[1])
    meta = ConditioningMeta(
        enabled=True,
        requested_types=normalized,
        resolved_types=resolved,
        slices=slices,
        context_dim=context_dim,
    )
    return tf.convert_to_tensor(context_matrix, dtype=tf.float32), meta


def meta_as_dict(meta: ConditioningMeta) -> Dict[str, Any]:
    """Convert a ConditioningMeta instance into a JSON-serializable dictionary."""
    return asdict(meta)
