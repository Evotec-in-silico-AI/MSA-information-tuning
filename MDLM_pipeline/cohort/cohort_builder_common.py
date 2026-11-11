from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prody import calcMSAOccupancy, calcShannonEntropy


AA_TO_TOKEN: Dict[str, int] = {
    "-": 0,
    "A": 1,
    "E": 2,
    "L": 3,
    "M": 4,
    "C": 5,
    "D": 6,
    "F": 7,
    "G": 8,
    "H": 9,
    "K": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "W": 16,
    "Y": 17,
    "T": 18,
    "V": 19,
    "I": 20,
}


@dataclass(frozen=True)
class BuilderPaths:
    """Hold canonical builder output directories."""
    repo_root: Path
    cohort_dir: Path
    compiled_dir: Path
    cpu_dir: Path
    gpu_dir: Path
    final_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class ResolvedSample:
    """Store resolved per-sample paths and cohort metadata."""
    sample_id: str
    protein_name: str
    family: str
    run_root: str
    sequence: str
    anchor_sequence: str
    corruption_rate_init: float
    candidate_index: int
    seed: int
    candidate_seed: int
    p10_target: float
    control_type: str
    control_rep: int
    shard_group: str
    shard_path: str
    model_dir: str
    compressed_msa_pkl_path: str
    compression_json_path: str
    trace_path: str
    mask_path: str
    commit_hist_path: str
    nll_cond_path: str
    entropy_cond_path: str
    jsd_cond_uncond_path: str
    commit_step_index_path: str


@dataclass(frozen=True)
class BuildManifest:
    """Record build provenance."""
    schema_version: str
    created_utc: str
    cohort_xlsx_path: str
    cohort_sha256: str
    cohort_rows: int
    unique_proteins: int
    python_version: str
    prody_version: str
    pyarrow_version: str


class BuilderLogger:
    """Write JSONL logs."""
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        record = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": str(event),
            **{str(k): payload[k] for k in payload},
        }
        line = json.dumps(record, ensure_ascii=False)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def build_paths(repo_root: Path) -> BuilderPaths:
    """Create and return all builder output directories."""
    cohort_dir = repo_root / "cohort"
    compiled_dir = cohort_dir / "compiled"
    cpu_dir = compiled_dir / "cpu"
    gpu_dir = compiled_dir / "gpu"
    final_dir = compiled_dir / "final"
    checkpoints_dir = compiled_dir / "checkpoints"
    logs_dir = compiled_dir / "logs"
    for path in [compiled_dir, cpu_dir, gpu_dir, final_dir, checkpoints_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return BuilderPaths(
        repo_root=repo_root,
        cohort_dir=cohort_dir,
        compiled_dir=compiled_dir,
        cpu_dir=cpu_dir,
        gpu_dir=gpu_dir,
        final_dir=final_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
    )


def compute_sha256(path: Path) -> str:
    """Compute SHA256 for a file."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically."""
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def atomic_write_parquet(table: pa.Table, path: Path) -> None:
    """Write a parquet table atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def ensure_columns_present(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    """Validate required DataFrame columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def read_cohort_by_sample(cohort_xlsx_path: Path) -> pd.DataFrame:
    """Read cohort.xlsx by_sample sheet."""
    df = pd.read_excel(Path(cohort_xlsx_path), sheet_name="by_sample", engine="openpyxl")
    ensure_columns_present(
        df,
        [
            "sample_id",
            "protein_name",
            "family",
            "run_root",
            "sequence",
            "anchor_sequence",
            "corruption_rate_init",
            "candidate_index",
            "seed",
            "candidate_seed",
            "p10_target",
            "control_type",
            "control_rep",
            "shard_group",
        ],
        label="cohort.xlsx by_sample",
    )
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].nunique() != len(df):
        raise ValueError("cohort.xlsx by_sample has non-unique sample_id values")
    return df


def build_manifest(cohort_xlsx_path: Path, cohort_df: pd.DataFrame) -> BuildManifest:
    """Build provenance metadata for this run."""
    import prody
    return BuildManifest(
        schema_version="cohort_builder_v1",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cohort_xlsx_path=str(Path(cohort_xlsx_path).resolve()),
        cohort_sha256=compute_sha256(Path(cohort_xlsx_path)),
        cohort_rows=int(len(cohort_df)),
        unique_proteins=int(cohort_df["protein_name"].nunique()),
        python_version=sys.version,
        prody_version=getattr(prody, "__version__", "unknown"),
        pyarrow_version=pa.__version__,
    )


def write_build_manifest(paths: BuilderPaths, manifest: BuildManifest) -> None:
    """Write the build manifest JSON."""
    atomic_write_json(paths.compiled_dir / "manifest.json", dataclasses.asdict(manifest))


def checkpoint_path(paths: BuilderPaths, stage: str, sample_id: str) -> Path:
    """Return per-sample checkpoint path."""
    stage_dir = paths.checkpoints_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir / f"{sample_id}.json"


def is_checkpoint_complete(paths: BuilderPaths, stage: str, sample_id: str) -> bool:
    """Return True when stage checkpoint exists."""
    return checkpoint_path(paths, stage, sample_id).exists()


def mark_checkpoint_complete(paths: BuilderPaths, stage: str, sample_id: str, payload: Dict[str, Any]) -> None:
    """Write per-sample checkpoint JSON."""
    atomic_write_json(checkpoint_path(paths, stage, sample_id), payload)


def build_cpu_output_paths(paths: BuilderPaths, protein_name: str, sample_id: str) -> Dict[str, Path]:
    """Return CPU stage output paths for a sample."""
    base = paths.cpu_dir
    return {
        "per_site": base / "per_site" / f"protein_name={protein_name}" / f"sample_id={sample_id}.parquet",
        "edit_events": base / "edit_events" / f"protein_name={protein_name}" / f"sample_id={sample_id}.parquet",
        "per_step": base / "per_step" / f"protein_name={protein_name}" / f"sample_id={sample_id}.parquet",
    }


def build_gpu_output_paths(paths: BuilderPaths, protein_name: str, sample_id: str) -> Dict[str, Path]:
    """Return GPU stage output paths for a sample."""
    base = paths.gpu_dir
    return {
        "per_site_conf": base / "per_site_conf" / f"protein_name={protein_name}" / f"sample_id={sample_id}.parquet",
    }


def build_sidecar_dir(run_root: str, sample_id: str) -> Path:
    """Return fold sidecar directory for a sample."""
    return Path(run_root) / "fold" / "samples" / str(sample_id)


def token_matrix_to_char_array(encoded: np.ndarray) -> np.ndarray:
    """Convert 0..20 tokens to ProDy-compatible char array."""
    lut = np.array(
        [
            b"-",
            b"A",
            b"E",
            b"L",
            b"M",
            b"C",
            b"D",
            b"F",
            b"G",
            b"H",
            b"K",
            b"N",
            b"P",
            b"Q",
            b"R",
            b"S",
            b"W",
            b"Y",
            b"T",
            b"V",
            b"I",
        ],
        dtype="S1",
    )
    flat = np.asarray(encoded, dtype=np.int64).ravel()
    flat = np.clip(flat, 0, 20)
    out = lut[flat]
    return out.reshape(np.asarray(encoded).shape)


def compute_msa_site_metrics_from_compressed_msa(compressed_msa_tokens: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute occupancy, entropy_bits, bits/site, and effective alphabet per column."""
    chars = token_matrix_to_char_array(compressed_msa_tokens)
    occupancy = np.asarray(calcMSAOccupancy(chars, occ="res", count=False), dtype=np.float64)
    entropy_bits = np.asarray(calcShannonEntropy(chars, ambiguity=True, omitgaps=True), dtype=np.float64)
    bits_site = np.log2(20.0) - entropy_bits
    effective_alphabet = np.power(2.0, entropy_bits)
    return occupancy, entropy_bits, bits_site, effective_alphabet


def read_mask_vector(mask_path: str) -> Optional[np.ndarray]:
    """Read mask vector from npy."""
    try:
        arr = np.load(str(mask_path))
        return np.asarray(arr, dtype=np.uint8).ravel()
    except Exception:
        return None


def read_numpy_vector(path: str, dtype: Any) -> Optional[np.ndarray]:
    """Read a 1D numpy vector from npy."""
    try:
        arr = np.load(str(path))
        return np.asarray(arr, dtype=dtype).ravel()
    except Exception:
        return None


def read_trace_npz(trace_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Read trace npz into a dict of arrays."""
    try:
        data = np.load(str(trace_path), allow_pickle=True)
        return {k: np.asarray(data[k]) for k in data.files}
    except Exception:
        return None


def list_plddt_slot_paths(sidecar_dir: Path) -> List[Path]:
    """List pLDDT slot vector files."""
    return sorted(Path(sidecar_dir).glob("slot*_plddt.npy"))


def list_pae_slot_paths(sidecar_dir: Path) -> List[Path]:
    """List PAE slot matrix files."""
    return sorted(Path(sidecar_dir).glob("slot*_pae.npz"))


def compute_plddt_mean_std_from_slots(plddt_paths: Sequence[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute per-residue mean and std of pLDDT across slots."""
    if not plddt_paths:
        return None, None
    vectors: List[np.ndarray] = []
    for path in plddt_paths:
        try:
            vectors.append(np.asarray(np.load(str(path)), dtype=np.float32).ravel())
        except Exception:
            continue
    if not vectors:
        return None, None
    stack = np.stack(vectors, axis=0)
    return np.mean(stack, axis=0).astype(np.float32), np.std(stack, axis=0).astype(np.float32)


def compute_pae_flattened_mean_std_from_slots(pae_paths: Sequence[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute per-residue mean and std of flattened PAE across slots."""
    if not pae_paths:
        return None, None
    vectors: List[np.ndarray] = []
    for path in pae_paths:
        try:
            with np.load(str(path)) as data:
                mat = np.asarray(data["pae"], dtype=np.float32)
            row_mean = np.mean(mat, axis=1)
            col_mean = np.mean(mat, axis=0)
            per_residue = 0.5 * (row_mean + col_mean)
            vectors.append(per_residue.astype(np.float32))
        except Exception:
            continue
    if not vectors:
        return None, None
    stack = np.stack(vectors, axis=0)
    return np.mean(stack, axis=0).astype(np.float32), np.std(stack, axis=0).astype(np.float32)


def build_mutation_vectors(anchor_sequence: str, sequence: str, length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-residue mutation indicators."""
    anchor = (anchor_sequence or "").strip()
    seq = (sequence or "").strip()
    anchor_arr = np.full((length,), "?", dtype=object)
    seq_arr = np.full((length,), "?", dtype=object)
    is_mut = np.zeros((length,), dtype=np.uint8)
    for idx in range(length):
        if idx < len(anchor):
            anchor_arr[idx] = anchor[idx]
        if idx < len(seq):
            seq_arr[idx] = seq[idx]
        is_mut[idx] = 1 if anchor_arr[idx] != seq_arr[idx] else 0
    return anchor_arr, seq_arr, is_mut


def read_merged_csvs(run_root: Path, pattern: str) -> List[pd.DataFrame]:
    """Read merged CSVs matching a pattern."""
    merged_dir = Path(run_root) / "merged"
    if not merged_dir.exists():
        return []
    frames: List[pd.DataFrame] = []
    for path in sorted(merged_dir.glob(pattern)):
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)
    return frames


def build_resolver_tables_for_run_root(run_root: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged manifest and results tables for a run root."""
    rr = Path(run_root)
    manifest_parts = read_merged_csvs(rr, "*_manifest_enriched.csv")
    if not manifest_parts:
        manifest_parts = read_merged_csvs(rr, "*_manifest.csv")
    results_parts = read_merged_csvs(rr, "*_results_enriched.csv")
    if not results_parts:
        results_parts = read_merged_csvs(rr, "*_results.csv")
    manifest_df = pd.concat(manifest_parts, axis=0, ignore_index=True) if manifest_parts else pd.DataFrame()
    results_df = pd.concat(results_parts, axis=0, ignore_index=True) if results_parts else pd.DataFrame()
    if not manifest_df.empty and "sample_id" in manifest_df.columns:
        manifest_df["sample_id"] = manifest_df["sample_id"].astype(str)
    if not results_df.empty and "sample_id" in results_df.columns:
        results_df["sample_id"] = results_df["sample_id"].astype(str)
    return manifest_df, results_df


def resolve_required_paths(cohort_row: pd.Series, manifest_df: pd.DataFrame, results_df: pd.DataFrame) -> Optional[ResolvedSample]:
    """Resolve a sample using merged manifest and results rows."""
    sample_id = str(cohort_row["sample_id"])
    if manifest_df.empty or "sample_id" not in manifest_df.columns:
        return None
    if results_df.empty or "sample_id" not in results_df.columns:
        return None
    manifest_matches = manifest_df[manifest_df["sample_id"] == sample_id]
    results_matches = results_df[results_df["sample_id"] == sample_id]
    if manifest_matches.empty or results_matches.empty:
        return None
    manifest_row = manifest_matches.iloc[-1]
    results_row = results_matches.iloc[-1]
    shard_path = str(manifest_row.get("shard_path", "") or "")
    model_dir = str(manifest_row.get("model_dir", "") or "")
    compressed_msa = str(manifest_row.get("compressed_msa_pkl_path", manifest_row.get("compressed_msa", "")) or "")
    compression_json = str(manifest_row.get("compression_json_path", manifest_row.get("compression_json", "")) or "")
    trace_path = str(results_row.get("trace_path", "") or "")
    mask_path = str(results_row.get("mask_path", "") or "")
    commit_hist_path = str(results_row.get("commit_hist_path", "") or "")
    nll_cond_path = str(results_row.get("nll_cond_path", "") or "")
    entropy_cond_path = str(results_row.get("entropy_cond_path", "") or "")
    jsd_cond_uncond_path = str(results_row.get("jsd_cond_uncond_path", "") or "")
    commit_step_index_path = str(results_row.get("commit_step_index_path", "") or "")
    if not shard_path and model_dir:
        shard_path = str(Path(model_dir).parent)
    if not shard_path or not model_dir or not compressed_msa:
        return None
    return ResolvedSample(
        sample_id=sample_id,
        protein_name=str(cohort_row.get("protein_name", "") or ""),
        family=str(cohort_row.get("family", "") or ""),
        run_root=str(cohort_row.get("run_root", "") or ""),
        sequence=str(cohort_row.get("sequence", "") or ""),
        anchor_sequence=str(cohort_row.get("anchor_sequence", "") or ""),
        corruption_rate_init=float(cohort_row.get("corruption_rate_init", float("nan"))),
        candidate_index=int(cohort_row.get("candidate_index", 0)),
        seed=int(cohort_row.get("seed", 0)),
        candidate_seed=int(cohort_row.get("candidate_seed", 0)),
        p10_target=float(cohort_row.get("p10_target", float("nan"))),
        control_type=str(cohort_row.get("control_type", "") or ""),
        control_rep=int(cohort_row.get("control_rep", 0)),
        shard_group=str(cohort_row.get("shard_group", "") or ""),
        shard_path=shard_path,
        model_dir=model_dir,
        compressed_msa_pkl_path=compressed_msa,
        compression_json_path=compression_json,
        trace_path=trace_path,
        mask_path=mask_path,
        commit_hist_path=commit_hist_path,
        nll_cond_path=nll_cond_path,
        entropy_cond_path=entropy_cond_path,
        jsd_cond_uncond_path=jsd_cond_uncond_path,
        commit_step_index_path=commit_step_index_path,
    )


def resolve_cohort(cohort_xlsx_path: Path, logger: BuilderLogger) -> Dict[str, ResolvedSample]:
    """Resolve all samples by sample_id using merged tables per run_root."""
    by_sample_df = read_cohort_by_sample(cohort_xlsx_path)
    resolved: Dict[str, ResolvedSample] = {}
    for run_root, group_df in by_sample_df.groupby("run_root"):
        manifest_df, results_df = build_resolver_tables_for_run_root(str(run_root))
        logger.log(
            "resolver_run_root_loaded",
            {
                "run_root": str(run_root),
                "cohort_rows": int(len(group_df)),
                "manifest_rows": int(len(manifest_df)) if not manifest_df.empty else 0,
                "results_rows": int(len(results_df)) if not results_df.empty else 0,
            },
        )
        for _, row in group_df.iterrows():
            sample_id = str(row["sample_id"])
            resolved_sample = resolve_required_paths(row, manifest_df, results_df)
            if resolved_sample is None:
                continue
            resolved[sample_id] = resolved_sample
    logger.log("resolver_complete", {"resolved_samples": int(len(resolved)), "total_samples": int(len(by_sample_df))})
    return resolved
