from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def atomic_write_text(path: PathLike, text: str) -> None:
    """
    Atomically write a text file (write tmp then rename) with explicit UTF-8 encoding.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # We append ".tmp" to the existing suffix, e.g., "file.csv.tmp", then atomically replace.
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def atomic_write_json(path: PathLike, payload: Dict[str, Any], *, indent: int = 2) -> None:
    """
    Atomically write a JSON file (write tmp then rename).

    Notes:
        - ensure_ascii=False preserves non-ASCII characters literally, and atomic_write_text
          ensures the file is UTF-8 encoded on disk for cross-platform consistency.
    """
    atomic_write_text(path, json.dumps(payload, indent=indent, ensure_ascii=False))


def atomic_pickle_dump(path: PathLike, obj: Any) -> None:
    """
    Atomically write a pickle file (write tmp then rename).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(obj, handle)
    tmp.replace(p)


def atomic_write_csv_df(path: PathLike, df: pd.DataFrame) -> None:
    """
    Atomically write a CSV from a DataFrame (write tmp then rename) with UTF-8 encoding.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(p)


def atomic_write_numpy_array(path: PathLike, array: np.ndarray) -> None:
    """
    Atomically write a NumPy array to .npy (write tmp then rename).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, np.asarray(array))
    tmp.replace(p)
