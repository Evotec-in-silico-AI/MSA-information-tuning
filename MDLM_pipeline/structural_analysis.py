from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from hashlib import blake2b
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from chai_lab.chai1 import run_inference

SHUTDOWN_REQUESTED = False


def parse_cli() -> argparse.Namespace:
    """
    Parse command line arguments that identify the run and configure folding/runtime behavior.
    """
    parser = argparse.ArgumentParser(
        description="Folding worker: process new rows from the run's fold queue, then exit."
    )
    parser.add_argument("--name", required=True, help="Protein accession (e.g., P00330)")
    parser.add_argument("--database", required=True, help="BLAST database name (e.g., nr)")
    parser.add_argument("--steps", required=True, type=int, help="Max training steps for the run (e.g., 4000)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device string passed to Chai-1 (default: cuda:0)")
    parser.add_argument("--num_trunk_recycles", type=int, default=3, help="Chai-1 trunk recycles per call (default: 3)")
    parser.add_argument("--num_diffn_timesteps", type=int, default=200, help="Chai-1 diffusion timesteps (default: 200)")
    parser.add_argument("--num_diffn_samples", type=int, default=5, help="Model samples per sequence (default: 5)")
    parser.add_argument("--tmscore_bin", default="TMscore", help="Path or name of TMscore binary (default: TMscore)")
    parser.add_argument("--tmscore_timeout_sec", type=int, default=120, help="Timeout per TMscore call (sec)")
    parser.add_argument(
        "--diffusion_activity_grace_sec",
        type=int,
        default=10,
        help="If queue file was modified within this window, abort to avoid clashing with diffusion.",
    )
    parser.add_argument(
        "--lock_fail_exit_code",
        type=int,
        default=40,
        help="Exit code when a lock or queue activity conflict prevents running (default: 40).",
    )
    parser.add_argument(
        "--conflict_exit_code",
        type=int,
        default=41,
        help="Exit code when the queue changes during this job (default: 41).",
    )
    parser.add_argument(
        "--preflight_exit_code",
        type=int,
        default=42,
        help="Exit code when prerequisites like TMscore are missing (default: 42).",
    )
    parser.add_argument(
        "--max_runtime_sec",
        type=int,
        default=0,
        help="Optional hard runtime budget; if approaching limit, stop starting new rows and exit cleanly.",
    )
    return parser.parse_args()


def resolve_paths(name: str, database: str, steps: int) -> Dict[str, Path]:
    """
    Resolve all required filesystem paths for the run given name/database/steps.
    """
    root = Path("runs") / name / database / f"steps_{int(steps)}"
    merged_dir = root / "merged"
    fold_metrics_dir = root / "fold" / "metrics"
    fold_samples_dir = root / "fold" / "samples"
    final_dir = merged_dir
    queue_csv = merged_dir / f"{name}_fold_queue.csv"
    ledger_parquet = fold_metrics_dir / "candidates.parquet"
    failures_jsonl = fold_metrics_dir / "tmscore_failures.jsonl"
    run_meta_json = fold_metrics_dir / "run_meta.json"
    merged_xlsx = final_dir / f"{name}_fold_queue_merged.xlsx"
    merged_csv = final_dir / f"{name}_fold_queue_merged.csv"
    fold_lock = fold_metrics_dir / ".fold_lock"
    return {
        "root": root,
        "merged_dir": merged_dir,
        "fold_metrics_dir": fold_metrics_dir,
        "fold_samples_dir": fold_samples_dir,
        "final_dir": final_dir,
        "queue_csv": queue_csv,
        "ledger_parquet": ledger_parquet,
        "failures_jsonl": failures_jsonl,
        "run_meta_json": run_meta_json,
        "merged_xlsx": merged_xlsx,
        "merged_csv": merged_csv,
        "fold_lock": fold_lock,
    }


def ensure_directories(paths: Dict[str, Path]) -> None:
    """
    Ensure required output directories exist before writing results.
    """
    paths["fold_metrics_dir"].mkdir(parents=True, exist_ok=True)
    paths["final_dir"].mkdir(parents=True, exist_ok=True)
    paths["fold_samples_dir"].mkdir(parents=True, exist_ok=True)


def acquire_fold_lock(lock_path: Path, exit_code_on_fail: int) -> Optional[int]:
    """
    Acquire a non-blocking, per-run exclusive lock for folding. If the lock cannot be acquired, return the exit code.
    """
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as fh:
            fh.write(f"pid={os.getpid()} utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        return None
    except FileExistsError:
        print(f"Another fold job appears to be running (lock present: {lock_path}). Exiting to avoid conflicts.")
        return int(exit_code_on_fail)
    except Exception as exc:
        print(f"Unexpected error when acquiring lock {lock_path}: {exc}. Exiting to avoid conflicts.")
        return int(exit_code_on_fail)


def release_fold_lock(lock_path: Path) -> None:
    """
    Best-effort lock release for the folding job lock.
    """
    try:
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass


def detect_recent_diffusion_activity(queue_csv: Path, grace_seconds: int) -> bool:
    """
    Detect whether the diffusion process likely wrote to the queue CSV very recently.
    """
    if not queue_csv.exists():
        return False
    try:
        mtime = queue_csv.stat().st_mtime
    except Exception:
        return False
    now = time.time()
    return (now - mtime) <= float(grace_seconds)


def queue_snapshot(queue_csv: Path, df: pd.DataFrame) -> Tuple[int, float, int]:
    """
    Capture queue size, modification time, and row count to detect mid-run changes.
    """
    try:
        size = queue_csv.stat().st_size if queue_csv.exists() else 0
        mtime = queue_csv.stat().st_mtime if queue_csv.exists() else 0.0
    except Exception:
        size = 0
        mtime = 0.0
    rows = int(len(df))
    return size, mtime, rows


def queue_changed_since(queue_csv: Path, snap: Tuple[int, float, int]) -> bool:
    """
    Check if the queue file changed compared to the provided snapshot by size/mtime and row count.
    """
    size0, mtime0, rows0 = snap
    try:
        size1 = queue_csv.stat().st_size if queue_csv.exists() else 0
        mtime1 = queue_csv.stat().st_mtime if queue_csv.exists() else 0.0
    except Exception:
        return True
    if size1 != size0 or abs(mtime1 - mtime0) > 1e-6:
        return True
    try:
        df_now = pd.read_csv(queue_csv)
        rows1 = int(len(df_now))
    except Exception:
        return True
    return rows1 != rows0


def load_queue(queue_csv: Path) -> pd.DataFrame:
    """
    Load the consolidated fold queue CSV produced by the diffusion pipeline wrapper.
    """
    if not queue_csv.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(queue_csv)
    except Exception as exc:
        raise RuntimeError(f"Failed to read fold queue CSV: {queue_csv} ({exc})")
    return df


def load_ledger(ledger_parquet: Path) -> pd.DataFrame:
    """
    Load or initialize the append-only ledger of folded samples.
    """
    if not ledger_parquet.exists():
        cols = [
            "sample_id",
            "fold_status",
            "plddt_mean",
            "plddt_median",
            "plddt_std",
            "plddt_min",
            "plddt_max",
            "plddt_model_means_json",
            "pae_mean",
            "pae_median",
            "pae_std",
            "pae_min",
            "pae_max",
            "pae_model_means_json",
            "ptm_mean",
            "ptm_std",
            "iptm_mean",
            "iptm_std",
            "aggregate_mean",
            "aggregate_std",
            "tm_status",
            "tm_success_count",
            "tm_fail_count",
            "tm_fail_pairs",
            "tm_mean",
            "tm_std",
            "tm_max",
            "rmsd_mean",
            "rmsd_min",
            "tm_best_pair_i",
            "tm_best_pair_j",
            "fold_time_sec",
            "tmscore_time_sec",
            "n_model_slots",
            "plddt_across_slot_std_mean",
            "plddt_across_slot_std_median",
            "sidecar_dir",
        ]
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_parquet(ledger_parquet)
    except Exception as exc:
        raise RuntimeError(f"Failed to read ledger parquet: {ledger_parquet} ({exc})")
    return df


def anti_join_backlog(queue_df: pd.DataFrame, ledger_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify backlog rows that have not yet been folded, based on unique sample_id.
    """
    if queue_df.empty:
        return pd.DataFrame()
    if "sample_id" not in queue_df.columns:
        raise ValueError("Fold queue is missing required column: sample_id")
    done = set(ledger_df["sample_id"].astype(str)) if not ledger_df.empty else set()
    mask = ~queue_df["sample_id"].astype(str).isin(done)
    backlog = queue_df.loc[mask].copy()
    return backlog


def write_run_meta(run_meta_json: Path, args: argparse.Namespace) -> None:
    """
    Write a small run-level metadata file for folding provenance and reproducibility using atomic replacement.
    """
    payload = {
        "chai1_defaults": {
            "num_trunk_recycles": int(args.num_trunk_recycles),
            "num_diffn_timesteps": int(args.num_diffn_timesteps),
            "num_diffn_samples": int(args.num_diffn_samples),
            "device": str(args.device),
        },
        "tmscore": {
            "bin": str(args.tmscore_bin),
            "timeout_sec": int(args.tmscore_timeout_sec),
        },
        "env": {
            "python": sys.version,
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_is_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = run_meta_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(run_meta_json)


def write_failures_jsonl(failures_jsonl: Path, records: List[Dict]) -> None:
    """
    Append TMscore failure records to a JSONL file for diagnostic auditing.
    """
    if not records:
        return
    with failures_jsonl.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_fasta(sequence: str, fasta_path: Path) -> None:
    """
    Write a single-sequence FASTA file at the specified location.
    """
    text = f">protein|name=seq\n{str(sequence).strip()}\n"
    fasta_path.write_text(text)


def convert_to_numpy(x: object) -> Optional[np.ndarray]:
    """
    Convert a tensor-like object to a NumPy array if possible.
    """
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    except Exception:
        return None


def run_chai1_fold(
    fasta_file: Path,
    output_dir: Path,
    device: str,
    num_trunk_recycles: int,
    num_diffn_timesteps: int,
    num_diffn_samples: int,
    seed: Optional[int],
) -> object:
    """
    Run Chai-1 folding for a single-chain FASTA and return the candidates object.
    """
    candidates = run_inference(
        fasta_file=fasta_file,
        output_dir=output_dir,
        use_esm_embeddings=True,
        num_trunk_recycles=int(num_trunk_recycles),
        num_diffn_timesteps=int(num_diffn_timesteps),
        num_diffn_samples=int(num_diffn_samples),
        device=str(device),
        seed=int(seed) if seed is not None else None,
    )
    return candidates


def list_model_products(model_dir: Path, num_models: int) -> Tuple[List[Path], List[Path]]:
    """
    List expected CIF and NPZ files for a run_inference() call in the given directory.
    """
    cif_paths = []
    npz_paths = []
    for i in range(int(num_models)):
        cif_paths.append(model_dir / f"pred.model_idx_{i}.cif")
        npz_paths.append(model_dir / f"scores.model_idx_{i}.npz")
    return cif_paths, npz_paths


def extract_plddt_pae_arrays(candidates_obj: object) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Extract raw pLDDT and PAE arrays from the candidates object as NumPy arrays.
    """
    plddt_np = convert_to_numpy(getattr(candidates_obj, "plddt", None))
    pae_np = convert_to_numpy(getattr(candidates_obj, "pae", None))
    return plddt_np, pae_np


def summarize_plddt_plausible(plddt_np: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Summarize pLDDT arrays (models x L) into global statistics across models.
    """
    summary = {
        "plddt_model_means": [],
        "plddt_mean": np.nan,
        "plddt_median": np.nan,
        "plddt_std": np.nan,
        "plddt_min": np.nan,
        "plddt_max": np.nan,
    }
    if isinstance(plddt_np, np.ndarray) and plddt_np.ndim == 2 and plddt_np.size > 0:
        per_model_means = np.nanmean(plddt_np, axis=1)
        per_model_medians = np.nanmedian(plddt_np, axis=1)
        per_model_mins = np.nanmin(plddt_np, axis=1)
        per_model_maxs = np.nanmax(plddt_np, axis=1)
        summary["plddt_model_means"] = per_model_means.tolist()
        summary["plddt_mean"] = float(np.nanmean(per_model_means))
        summary["plddt_median"] = float(np.nanmedian(per_model_medians))
        summary["plddt_std"] = float(np.nanstd(per_model_means))
        summary["plddt_min"] = float(np.nanmin(per_model_mins))
        summary["plddt_max"] = float(np.nanmax(per_model_maxs))
    return summary


def summarize_pae_plausible(pae_np: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Summarize PAE arrays (models x L x L) into global statistics across models.
    """
    summary = {
        "pae_model_means": [],
        "pae_mean": np.nan,
        "pae_median": np.nan,
        "pae_std": np.nan,
        "pae_min": np.nan,
        "pae_max": np.nan,
    }
    if isinstance(pae_np, np.ndarray) and pae_np.ndim == 3 and pae_np.size > 0:
        flat = pae_np.reshape(pae_np.shape[0], -1)
        per_model_means = np.nanmean(flat, axis=1)
        per_model_medians = np.nanmedian(flat, axis=1)
        per_model_mins = np.nanmin(flat, axis=1)
        per_model_maxs = np.nanmax(flat, axis=1)
        summary["pae_model_means"] = per_model_means.tolist()
        summary["pae_mean"] = float(np.nanmean(per_model_means))
        summary["pae_median"] = float(np.nanmedian(per_model_medians))
        summary["pae_std"] = float(np.nanstd(per_model_means))
        summary["pae_min"] = float(np.nanmin(per_model_mins))
        summary["pae_max"] = float(np.nanmax(per_model_maxs))
    return summary


def compute_mean_std_list(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and std for a list of floats, returning NaNs for empty input.
    """
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def extract_npz_scores(npz_paths: List[Path]) -> Tuple[float, float, float, float, float, float]:
    """
    Extract PTM, iPTM, and aggregate_score from per-model NPZ files and aggregate mean/std across models.
    """
    ptm_values = []
    iptm_values = []
    agg_values = []
    for p in npz_paths:
        if not p.exists():
            continue
        try:
            data = np.load(str(p))
            if "ptm" in data:
                arr = np.asarray(data["ptm"])
                ptm_values.append(float(arr.item() if arr.size == 1 else arr.ravel()[0]))
            if "iptm" in data:
                arr = np.asarray(data["iptm"])
                iptm_values.append(float(arr.item() if arr.size == 1 else arr.ravel()[0]))
            if "aggregate_score" in data:
                arr = np.asarray(data["aggregate_score"])
                agg_values.append(float(arr.item() if arr.size == 1 else arr.ravel()[0]))
        except Exception:
            continue
    ptm_mean, ptm_std = compute_mean_std_list(ptm_values)
    iptm_mean, iptm_std = compute_mean_std_list(iptm_values)
    agg_mean, agg_std = compute_mean_std_list(agg_values)
    return ptm_mean, ptm_std, iptm_mean, iptm_std, agg_mean, agg_std


def run_tmscore_pair(
    tmscore_bin: str,
    cand_cif: Path,
    anch_cif: Path,
    timeout_sec: int,
) -> Tuple[bool, float, float, Dict]:
    """
    Run TMscore for a candidate-anchor pair and return (ok, tm, rmsd, fail_record).
    """
    cmd = [tmscore_bin, str(cand_cif), str(anch_cif), "-outfmt", "2"]
    fail_record = {}
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(timeout_sec),
            check=False,
        )
        out = proc.stdout or ""
        rc = int(proc.returncode)
    except subprocess.TimeoutExpired:
        fail_record = {
            "reason": "timeout",
            "cmd": " ".join(cmd),
            "stdout_excerpt": "",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return False, 0.0, 0.0, fail_record
    except Exception as exc:
        fail_record = {
            "reason": "subprocess_error",
            "cmd": " ".join(cmd),
            "stdout_excerpt": str(exc),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return False, 0.0, 0.0, fail_record
    tm, rmsd = parse_tmscore_outfmt2(out)
    explicit_fail = bool(re.search(r"(fail|failed|cannot|error)", out, re.IGNORECASE)) and (tm == 0.0 and rmsd == 0.0)
    all_zero = (tm == 0.0 and rmsd == 0.0)
    nonfinite = not (np.isfinite(tm) and np.isfinite(rmsd))
    if rc != 0 or explicit_fail or all_zero or nonfinite:
        reason = (
            "nonfinite" if nonfinite
            else ("all_zero" if all_zero else ("explicit_fail_msg" if explicit_fail else "nonzero_exit"))
        )
        fail_record = {
            "reason": reason,
            "cmd": " ".join(cmd),
            "stdout_excerpt": out[:300],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return False, tm, rmsd, fail_record
    return True, tm, rmsd, {}


def parse_tmscore_outfmt2(stdout: str) -> Tuple[float, float]:
    """
    Parse TMscore -outfmt 2 output to extract TM2 and RMSD values, with a conservative fallback.
    """
    tm = 0.0
    rmsd = 0.0
    lines = stdout.splitlines()
    header = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            cols = re.split(r"\s+", s.lstrip("#").strip())
            header = {name: idx for idx, name in enumerate(cols)}
            continue
        if header:
            cols = re.split(r"\s+", s.strip())
            tm_idx = header.get("TM2")
            rmsd_idx = header.get("RMSD")
            if tm_idx is not None and rmsd_idx is not None and len(cols) > max(tm_idx, rmsd_idx):
                try:
                    tm_val = float(cols[tm_idx])
                    rmsd_val = float(cols[rmsd_idx])
                    tm = tm_val
                    rmsd = rmsd_val
                    return tm, rmsd
                except Exception:
                    continue
    m_tm = re.search(r"TM-score\s*=\s*([0-9]*\.?[0-9]+)", stdout)
    m_rmsd = re.search(r"RMSD\s*=\s*([0-9]*\.?[0-9]+)", stdout)
    if m_tm:
        try:
            tm = float(m_tm.group(1))
        except Exception:
            tm = 0.0
    if m_rmsd:
        try:
            rmsd = float(m_rmsd.group(1))
        except Exception:
            rmsd = 0.0
    return tm, rmsd


def aggregate_tm_5x5(
    sample_id: str,
    tmscore_bin: str,
    tmscore_timeout_sec: int,
    cand_cifs: List[Path],
    anch_cifs: List[Path],
    failures_jsonl: Path,
) -> Dict[str, object]:
    """
    Run TMscore over the full grid of candidate-anchor models, apply failure policy, and compute success-only summaries.
    """
    results = []
    failure_records = []
    for i, cand in enumerate(cand_cifs):
        for j, anch in enumerate(anch_cifs):
            ok, tm, rmsd, fail = run_tmscore_pair(tmscore_bin, cand, anch, tmscore_timeout_sec)
            rec = {"i": i, "j": j, "tm": tm, "rmsd": rmsd, "ok": bool(ok)}
            results.append(rec)
            if fail:
                fail["sample_id"] = sample_id
                fail["anchor_model_idx"] = int(j)
                fail["candidate_model_idx"] = int(i)
                failure_records.append(fail)
    if failure_records:
        write_failures_jsonl(failures_jsonl, failure_records)
    successes = [r for r in results if r["ok"]]
    fails = [r for r in results if not r["ok"]]
    if len(successes) == 0:
        return {
            "tm_status": "all_fail",
            "tm_success_count": 0,
            "tm_fail_count": len(fails),
            "tm_fail_pairs": json.dumps([(r["i"], r["j"]) for r in fails]),
            "tm_mean": float("nan"),
            "tm_std": float("nan"),
            "tm_max": float("nan"),
            "rmsd_mean": float("nan"),
            "rmsd_min": float("nan"),
            "tm_best_pair_i": np.nan,
            "tm_best_pair_j": np.nan,
        }
    tm_vals = np.asarray([r["tm"] for r in successes], dtype=np.float64)
    rmsd_vals = np.asarray([r["rmsd"] for r in successes], dtype=np.float64)
    best_idx = int(np.argmax(tm_vals))
    best_tm = tm_vals[best_idx]
    cand_i, anch_j = successes[best_idx]["i"], successes[best_idx]["j"]
    return {
        "tm_status": "partial_success" if len(successes) < (len(cand_cifs) * len(anch_cifs)) else "all_success",
        "tm_success_count": len(successes),
        "tm_fail_count": len(fails),
        "tm_fail_pairs": json.dumps([(r["i"], r["j"]) for r in fails]),
        "tm_mean": float(np.nanmean(tm_vals)),
        "tm_std": float(np.nanstd(tm_vals)),
        "tm_max": float(best_tm),
        "rmsd_mean": float(np.nanmean(rmsd_vals)),
        "rmsd_min": float(np.nanmin(rmsd_vals)),
        "tm_best_pair_i": int(cand_i),
        "tm_best_pair_j": int(anch_j),
    }


def fold_anchor_once(
    anchor_sequence: str,
    device: str,
    num_trunk_recycles: int,
    num_diffn_timesteps: int,
    num_diffn_samples: int,
    seed: Optional[int],
) -> Tuple[tempfile.TemporaryDirectory, List[Path]]:
    """
    Fold the anchor sequence once for this invocation and return temporary dir handle and CIF paths.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix="anchor_", dir=os.environ.get("TMPDIR"))
    tmp = Path(tmpdir.name)
    fasta_path = tmp / "anchor.fasta"
    make_fasta(anchor_sequence, fasta_path)
    out_dir = tmp / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    _ = run_chai1_fold(
        fasta_file=fasta_path,
        output_dir=out_dir,
        device=device,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        seed=seed,
    )
    cif_paths, _npz_paths_ignore = list_model_products(out_dir, num_diffn_samples)
    return tmpdir, cif_paths


def derive_seed_from_sample_id(sample_id: str) -> int:
    """
    Derive a stable 32-bit seed from sample_id for reproducibility.
    """
    h = blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") % (2**32 - 1)


def compute_plddt_across_slot_std(plddt_np: Optional[np.ndarray]) -> Tuple[float, float, int]:
    """
    Compute across-slot per-residue pLDDT standard deviation summary (mean and median).
    """
    if not isinstance(plddt_np, np.ndarray) or plddt_np.ndim != 2 or plddt_np.size == 0:
        return float("nan"), float("nan"), 0
    stacked = np.asarray(plddt_np, dtype=np.float32)
    std_per_res = np.nanstd(stacked, axis=0)
    return float(np.nanmean(std_per_res)), float(np.nanmedian(std_per_res)), int(stacked.shape[0])


def write_residue_map_default(sample_sidecar_dir: Path, sequence: str) -> Optional[Path]:
    """
    Write residue_map.json using a simple 1:1 mapping to chain A when missing.
    """
    residue_map_path = sample_sidecar_dir / "residue_map.json"
    if residue_map_path.exists():
        return residue_map_path
    L = int(len(sequence or ""))
    if L <= 0:
        return None
    mapping = {}
    for idx in range(L):
        mapping[str(idx)] = {"chain_id": "A", "residue_index": int(idx + 1), "pae_index": int(idx)}
    try:
        residue_map_path.write_text(json.dumps(mapping, indent=2))
        return residue_map_path
    except Exception:
        return None


def write_chai1_provenance(sample_sidecar_dir: Path, args: argparse.Namespace, seed_value: Optional[int], n_slots: int) -> Optional[Path]:
    """
    Write chai1_provenance.json with run configuration and environment fields.
    """
    payload = {
        "chai1_model_name": "Chai-1",
        "chai1_model_version": "unspecified",
        "chai1_weights_sha256": "",
        "chai1_seed": int(seed_value) if seed_value is not None else None,
        "n_slots": int(n_slots),
        "ranking_metric": "none",
        "device": str(args.device),
        "num_trunk_recycles": int(args.num_trunk_recycles),
        "num_diffn_timesteps": int(args.num_diffn_timesteps),
        "num_diffn_samples": int(args.num_diffn_samples),
        "container_digest": "",
        "cuda_version": getattr(torch.version, "cuda", ""),
        "cudnn_version": "",
        "gpu_sku": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "driver_version": "",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = sample_sidecar_dir / "chai1_provenance.json"
    try:
        out_path.write_text(json.dumps(payload, indent=2))
        return out_path
    except Exception:
        return None


def write_plddt_pae_sidecars(sample_sidecar_dir: Path, plddt_np: Optional[np.ndarray], pae_np: Optional[np.ndarray]) -> Tuple[int, Optional[int]]:
    """
    Persist per-slot sidecars: slot{k}_plddt.npy and slot{k}_pae.npz. Return (n_slots, sequence_length).
    """
    n_slots = 0
    seq_len = None
    if isinstance(plddt_np, np.ndarray) and plddt_np.ndim == 2 and plddt_np.size > 0:
        n_slots = int(plddt_np.shape[0])
        seq_len = int(plddt_np.shape[1])
        for k in range(n_slots):
            plddt_vec = np.asarray(plddt_np[k], dtype=np.float32)
            np.save(sample_sidecar_dir / f"slot{k}_plddt.npy", plddt_vec)
    if isinstance(pae_np, np.ndarray) and pae_np.ndim == 3 and pae_np.size > 0:
        if n_slots == 0:
            n_slots = int(pae_np.shape[0])
        if seq_len is None:
            seq_len = int(pae_np.shape[1])
        for k in range(int(pae_np.shape[0])):
            pae_mat = np.asarray(pae_np[k], dtype=np.float32)
            np.savez_compressed(sample_sidecar_dir / f"slot{k}_pae.npz", pae=pae_mat)
    return n_slots, seq_len


def write_sample_metrics_json(
    sample_sidecar_dir: Path,
    sample_id: str,
    plddt_summary: Dict[str, float],
    pae_summary: Dict[str, float],
    tm_block: Dict[str, object],
    n_model_slots: int,
    plddt_std_mean: float,
    plddt_std_median: float,
    residue_map_path: Optional[Path],
    provenance_path: Optional[Path],
) -> None:
    """
    Write per-sample metrics.json for quick inspection linking summaries and sidecars.
    """
    payload = {
        "sample_id": sample_id,
        "plddt_summary": plddt_summary,
        "pae_summary": pae_summary,
        "tm_block": tm_block,
        "n_model_slots": int(n_model_slots),
        "plddt_across_slot_std_mean": float(plddt_std_mean),
        "plddt_across_slot_std_median": float(plddt_std_median),
        "residue_map_path": str(residue_map_path) if residue_map_path else "",
        "chai1_provenance_path": str(provenance_path) if provenance_path else "",
    }
    try:
        (sample_sidecar_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def fold_candidate_and_summarize(
    sequence: str,
    seed: Optional[int],
    device: str,
    num_trunk_recycles: int,
    num_diffn_timesteps: int,
    num_diffn_samples: int,
) -> Tuple[
    TemporaryDirectory,
    List[Path],
    Dict[str, float],
    Dict[str, float],
    Tuple[float, float, float, float, float, float],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Fold a candidate sequence and return CIFs, summaries, Chai-1 NPZ score summaries, and raw arrays.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix="cand_", dir=os.environ.get("TMPDIR"))
    tmp = Path(tmpdir.name)
    fasta_path = tmp / "seq.fasta"
    make_fasta(sequence, fasta_path)
    out_dir = tmp / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_obj = run_chai1_fold(
        fasta_file=fasta_path,
        output_dir=out_dir,
        device=device,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        seed=seed,
    )
    plddt_np, pae_np = extract_plddt_pae_arrays(candidates_obj)
    plddt_summary = summarize_plddt_plausible(plddt_np)
    pae_summary = summarize_pae_plausible(pae_np)
    cif_paths, npz_paths = list_model_products(out_dir, num_diffn_samples)
    ptm_mean, ptm_std, iptm_mean, iptm_std, agg_mean, agg_std = extract_npz_scores(npz_paths)
    return (
        tmpdir,
        cif_paths,
        plddt_summary,
        pae_summary,
        (ptm_mean, ptm_std, iptm_mean, iptm_std, agg_mean, agg_std),
        plddt_np,
        pae_np,
    )


def safe_concat_frames(existing: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate existing and new_df on the union of columns, handling empty sides.
    """
    union_cols = list(dict.fromkeys(list(existing.columns) + list(new_df.columns)))
    ex = existing.reindex(columns=union_cols)
    nd = new_df.reindex(columns=union_cols)
    if ex.empty and nd.empty:
        return nd.reset_index(drop=True)
    if ex.empty:
        return nd.reset_index(drop=True)
    if nd.empty:
        return ex.reset_index(drop=True)
    return pd.concat([ex, nd], axis=0, ignore_index=True)


def append_ledger_row_atomic(ledger_parquet: Path, row: Dict[str, object]) -> None:
    """
    Append one metrics row to the ledger parquet via atomic replacement.
    """
    existing = load_ledger(ledger_parquet)
    new_df = pd.DataFrame([row])
    if not existing.empty:
        dtype_map = existing.dtypes.to_dict()
        for col, dt in dtype_map.items():
            if col in new_df.columns:
                try:
                    new_df[col] = new_df[col].astype(dt)
                except Exception:
                    pass
        combined = safe_concat_frames(existing, new_df)
    else:
        combined = new_df
    tmp = ledger_parquet.with_suffix(".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(ledger_parquet)


def write_final_outputs_atomic(
    merged_queue_csv: Path,
    ledger_parquet: Path,
    out_xlsx: Path,
    out_csv: Path,
) -> None:
    """
    Merge queue with ledger on sample_id and write final XLSX and CSV artifacts using atomic replacement.
    """
    queue_df = load_queue(merged_queue_csv)
    if queue_df.empty:
        return
    ledger_df = load_ledger(ledger_parquet)
    if ledger_df.empty:
        merged = queue_df.copy()
    else:
        merged = queue_df.merge(ledger_df, on="sample_id", how="left")
    tmp_csv = out_csv.with_suffix(".tmp.csv")
    merged.to_csv(tmp_csv, index=False)
    tmp_csv.replace(out_csv)
    tmp_xlsx = out_xlsx.with_suffix(".tmp.xlsx")
    with pd.ExcelWriter(tmp_xlsx, engine="openpyxl") as writer:
        merged.to_excel(writer, index=False, sheet_name="fold_queue_merged")
    Path(tmp_xlsx).replace(out_xlsx)


def validate_queue_columns(queue_df: pd.DataFrame) -> None:
    """
    Validate the presence of required columns in the fold queue for folding.
    """
    required = {"sample_id", "sequence", "anchor_sequence"}
    missing = [c for c in required if c not in queue_df.columns]
    if missing:
        raise ValueError(f"Fold queue is missing required columns: {missing}")


def tmscore_available(tmscore_bin: str) -> bool:
    """
    Check if TMscore binary is available on PATH or at the given location.
    """
    if os.path.isabs(tmscore_bin) and Path(tmscore_bin).exists():
        return True
    return shutil.which(tmscore_bin) is not None


def handle_signal(signum, frame) -> None:
    """
    Mark shutdown requested so the runner finishes the current row safely and exits.
    """
    del frame
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"Signal {signum} received; will finish current row and exit.")


def metric_is_in_range(value, lo, hi):
    """
    Return True when value is within [lo, hi]. Missing/NaN values are treated as acceptable.
    """
    try:
        if value is None:
            return True
        v = float(value)
        if isinstance(v, float) and (v != v):
            return True
        return (v >= float(lo)) and (v <= float(hi))
    except Exception:
        return False


def metric_is_nonneg(value, lo=None, hi=None):
    """
    Return True when value >= 0. The extra parameters are ignored to keep a uniform signature.
    """
    try:
        if value is None:
            return True
        v = float(value)
        if isinstance(v, float) and (v != v):
            return True
        return v >= 0.0
    except Exception:
        return False


def sanitize_metric_ranges(ledger_row: dict) -> dict:
    """
    Validate and sanitize metric ranges in a folding ledger row, setting out-of-range to NaN.
    """
    status = str(ledger_row.get("fold_status", "ok"))
    checks = [
        ("plddt_mean", 0.0, 100.0, metric_is_in_range),
        ("plddt_median", 0.0, 100.0, metric_is_in_range),
        ("plddt_min", 0.0, 100.0, metric_is_in_range),
        ("plddt_max", 0.0, 100.0, metric_is_in_range),
        ("pae_mean", 0.0, float("inf"), metric_is_nonneg),
        ("pae_median", 0.0, float("inf"), metric_is_nonneg),
        ("pae_min", 0.0, float("inf"), metric_is_nonneg),
        ("pae_max", 0.0, float("inf"), metric_is_nonneg),
        ("ptm_mean", 0.0, 1.0, metric_is_in_range),
        ("iptm_mean", 0.0, 1.0, metric_is_in_range),
        ("aggregate_mean", 0.0, 1.0, metric_is_in_range),
        ("tm_mean", 0.0, 1.0, metric_is_in_range),
        ("tm_max", 0.0, 1.0, metric_is_in_range),
        ("rmsd_mean", 0.0, float("inf"), metric_is_nonneg),
        ("rmsd_min", 0.0, float("inf"), metric_is_nonneg),
    ]
    any_out_of_range = False
    for key, lo, hi, fn in checks:
        try:
            if key in ledger_row and not fn(ledger_row[key], lo, hi):
                ledger_row[key] = float("nan")
                any_out_of_range = True
        except Exception:
            ledger_row[key] = float("nan")
            any_out_of_range = True
    if any_out_of_range and status == "ok":
        ledger_row["fold_status"] = "fail: out_of_range"
    return ledger_row


def main() -> None:
    """
    Entry point: acquire lock, detect conflicts, fold new samples for a run, then exit.
    """
    args = parse_cli()
    paths = resolve_paths(args.name, args.database, args.steps)
    ensure_directories(paths)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    if not tmscore_available(args.tmscore_bin):
        print("TMscore binary not found. Exiting.")
        sys.exit(args.preflight_exit_code)
    if detect_recent_diffusion_activity(paths["queue_csv"], args.diffusion_activity_grace_sec):
        print("Diffusion activity detected recently on the queue. Exiting to avoid conflicts.")
        sys.exit(args.lock_fail_exit_code)
    lock_fail_code = acquire_fold_lock(paths["fold_lock"], args.lock_fail_exit_code)
    if lock_fail_code is not None:
        sys.exit(lock_fail_code)
    job_start_ts = time.time()
    queue_df = pd.DataFrame()
    queue_snap = (0, 0.0, 0)
    try:
        queue_df = load_queue(paths["queue_csv"])
        if queue_df.empty:
            print("No fold queue found or queue is empty. Exiting.")
            sys.exit(0)
        validate_queue_columns(queue_df)
        queue_snap = queue_snapshot(paths["queue_csv"], queue_df)
        ledger_df = load_ledger(paths["ledger_parquet"])
        backlog_df = anti_join_backlog(queue_df, ledger_df)
        if backlog_df.empty:
            print("No new samples to fold. Writing merged outputs and exiting.")
            try:
                if not queue_changed_since(paths["queue_csv"], queue_snap):
                    write_final_outputs_atomic(
                        merged_queue_csv=paths["queue_csv"],
                        ledger_parquet=paths["ledger_parquet"],
                        out_xlsx=paths["merged_xlsx"],
                        out_csv=paths["merged_csv"],
                    )
                else:
                    print("Queue changed during job; skipping final merge to avoid conflict.")
                    sys.exit(args.conflict_exit_code)
            except Exception as exc:
                print(f"Warning: Failed to write final outputs: {exc}")
            sys.exit(0)
        write_run_meta(paths["run_meta_json"], args)
        anchor_sequence = str(queue_df["anchor_sequence"].iloc[0])
        anchor_seed = derive_seed_from_sample_id(f"{args.name}:{args.database}:{args.steps}")
        anchor_tmp, anchor_cifs = fold_anchor_once(
            anchor_sequence=anchor_sequence,
            device=args.device,
            num_trunk_recycles=args.num_trunk_recycles,
            num_diffn_timesteps=args.num_diffn_timesteps,
            num_diffn_samples=args.num_diffn_samples,
            seed=int(anchor_seed),
        )
        anchor_tmp_handle = anchor_tmp
        try:
            for _, row in backlog_df.iterrows():
                if SHUTDOWN_REQUESTED:
                    print("Shutdown requested; stopping before starting another row.")
                    break
                if args.max_runtime_sec and (time.time() - job_start_ts) > max(0, int(args.max_runtime_sec) - 600):
                    print("Approaching runtime limit; stopping before starting another row.")
                    break
                sample_id = str(row["sample_id"])
                sequence = str(row["sequence"])
                seed_val = None
                try:
                    if "candidate_seed" in row and pd.notna(row["candidate_seed"]):
                        seed_val = int(row["candidate_seed"])
                    elif "seed" in row and pd.notna(row["seed"]):
                        seed_val = int(row["seed"])
                    else:
                        seed_val = derive_seed_from_sample_id(sample_id)
                except Exception:
                    seed_val = derive_seed_from_sample_id(sample_id)
                fold_status = "ok"
                plddt_sum = {}
                pae_sum = {}
                ptm_m = ptm_s = iptm_m = iptm_s = agg_m = agg_s = float("nan")
                n_models_expected = int(args.num_diffn_samples)
                default_fail_pairs = json.dumps([(i, j) for i in range(n_models_expected) for j in range(n_models_expected)])
                tm_block = {
                    "tm_status": "all_fail",
                    "tm_success_count": 0,
                    "tm_fail_count": int(n_models_expected * n_models_expected),
                    "tm_fail_pairs": default_fail_pairs,
                    "tm_mean": float("nan"),
                    "tm_std": float("nan"),
                    "tm_max": float("nan"),
                    "rmsd_mean": float("nan"),
                    "rmsd_min": float("nan"),
                    "tm_best_pair_i": np.nan,
                    "tm_best_pair_j": np.nan,
                }
                fold_time = 0.0
                tmscore_time = 0.0
                tmp_handle = None
                plddt_np = None
                pae_np = None
                try:
                    t0 = time.time()
                    (
                        tmp_handle,
                        cand_cifs,
                        plddt_sum,
                        pae_sum,
                        (ptm_m, ptm_s, iptm_m, iptm_s, agg_m, agg_s),
                        plddt_np,
                        pae_np,
                    ) = fold_candidate_and_summarize(
                        sequence=sequence,
                        seed=seed_val,
                        device=args.device,
                        num_trunk_recycles=args.num_trunk_recycles,
                        num_diffn_timesteps=args.num_diffn_timesteps,
                        num_diffn_samples=args.num_diffn_samples,
                    )
                    fold_time = time.time() - t0
                    missing = [str(p) for p in cand_cifs if not Path(p).exists()]
                    if missing:
                        raise FileNotFoundError(f"Candidate CIFs missing: {missing}")
                    t1 = time.time()
                    tm_block = aggregate_tm_5x5(
                        sample_id=sample_id,
                        tmscore_bin=args.tmscore_bin,
                        tmscore_timeout_sec=args.tmscore_timeout_sec,
                        cand_cifs=cand_cifs,
                        anch_cifs=anchor_cifs,
                        failures_jsonl=paths["failures_jsonl"],
                    )
                    tmscore_time = time.time() - t1
                except Exception as exc:
                    fold_status = f"fail: {exc}"
                    fold_time = float("nan")
                    tmscore_time = float("nan")
                    tm_block = {
                        "tm_status": "all_fail",
                        "tm_success_count": 0,
                        "tm_fail_count": int(n_models_expected * n_models_expected),
                        "tm_fail_pairs": default_fail_pairs,
                        "tm_mean": float("nan"),
                        "tm_std": float("nan"),
                        "tm_max": float("nan"),
                        "rmsd_mean": float("nan"),
                        "rmsd_min": float("nan"),
                        "tm_best_pair_i": np.nan,
                        "tm_best_pair_j": np.nan,
                    }
                finally:
                    try:
                        if tmp_handle is not None:
                            tmp_handle.cleanup()
                    except Exception:
                        pass
                sample_sidecar_dir = paths["fold_samples_dir"] / sample_id
                sample_sidecar_dir.mkdir(parents=True, exist_ok=True)
                n_model_slots_value, seq_len_value = write_plddt_pae_sidecars(sample_sidecar_dir, plddt_np, pae_np)
                plddt_std_mean, plddt_std_median, n_slots_for_std = compute_plddt_across_slot_std(plddt_np)
                residue_map_path = write_residue_map_default(sample_sidecar_dir, sequence)
                provenance_path = write_chai1_provenance(
                    sample_sidecar_dir, args, seed_val, n_model_slots_value or args.num_diffn_samples
                )
                write_sample_metrics_json(
                    sample_sidecar_dir=sample_sidecar_dir,
                    sample_id=sample_id,
                    plddt_summary=plddt_sum,
                    pae_summary=pae_sum,
                    tm_block=tm_block,
                    n_model_slots=n_model_slots_value or args.num_diffn_samples,
                    plddt_std_mean=plddt_std_mean,
                    plddt_std_median=plddt_std_median,
                    residue_map_path=residue_map_path,
                    provenance_path=provenance_path,
                )
                ledger_row = {
                    "sample_id": sample_id,
                    "fold_status": fold_status,
                    "plddt_mean": plddt_sum.get("plddt_mean", float("nan")),
                    "plddt_median": plddt_sum.get("plddt_median", float("nan")),
                    "plddt_std": plddt_sum.get("plddt_std", float("nan")),
                    "plddt_min": plddt_sum.get("plddt_min", float("nan")),
                    "plddt_max": plddt_sum.get("plddt_max", float("nan")),
                    "plddt_model_means_json": json.dumps(plddt_sum.get("plddt_model_means", [])),
                    "pae_mean": pae_sum.get("pae_mean", float("nan")),
                    "pae_median": pae_sum.get("pae_median", float("nan")),
                    "pae_std": pae_sum.get("pae_std", float("nan")),
                    "pae_min": pae_sum.get("pae_min", float("nan")),
                    "pae_max": pae_sum.get("pae_max", float("nan")),
                    "pae_model_means_json": json.dumps(pae_sum.get("pae_model_means", [])),
                    "ptm_mean": ptm_m,
                    "ptm_std": ptm_s,
                    "iptm_mean": iptm_m,
                    "iptm_std": iptm_s,
                    "aggregate_mean": agg_m,
                    "aggregate_std": agg_s,
                    "tm_status": tm_block["tm_status"],
                    "tm_success_count": tm_block["tm_success_count"],
                    "tm_fail_count": tm_block["tm_fail_count"],
                    "tm_fail_pairs": tm_block["tm_fail_pairs"],
                    "tm_mean": tm_block["tm_mean"],
                    "tm_std": tm_block["tm_std"],
                    "tm_max": tm_block["tm_max"],
                    "rmsd_mean": tm_block["rmsd_mean"],
                    "rmsd_min": tm_block["rmsd_min"],
                    "tm_best_pair_i": tm_block["tm_best_pair_i"],
                    "tm_best_pair_j": tm_block["tm_best_pair_j"],
                    "fold_time_sec": float(fold_time),
                    "tmscore_time_sec": float(tmscore_time),
                    "n_model_slots": int(n_model_slots_value or args.num_diffn_samples),
                    "plddt_across_slot_std_mean": float(plddt_std_mean),
                    "plddt_across_slot_std_median": float(plddt_std_median),
                    "sidecar_dir": str(sample_sidecar_dir),
                }
                ledger_row = sanitize_metric_ranges(ledger_row)
                try:
                    append_ledger_row_atomic(paths["ledger_parquet"], ledger_row)
                except Exception as exc:
                    print(f"Error writing ledger row for {sample_id}: {exc}")
                    sys.exit(43)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    print(
                        f"folded sample_id={sample_id} "
                        f"plddt_mean={ledger_row['plddt_mean']:.2f} "
                        f"ptm_mean={ledger_row['ptm_mean']:.3f} "
                        f"tm_max={ledger_row['tm_max']:.3f} "
                        f"tm_status={ledger_row['tm_status']} "
                        f"fold_s={ledger_row['fold_time_sec']:.1f} "
                        f"tmscore_s={ledger_row['tmscore_time_sec']:.1f}"
                    )
                except Exception:
                    pass
        finally:
            try:
                anchor_tmp_handle.cleanup()
            except Exception:
                pass
    finally:
        if SHUTDOWN_REQUESTED:
            print("Shutdown requested; skipping final merge to avoid conflicts and exiting.")
            release_fold_lock(paths["fold_lock"])
            sys.exit(args.conflict_exit_code)
    if detect_recent_diffusion_activity(paths["queue_csv"], args.diffusion_activity_grace_sec):
        print("Queue was modified recently; skipping final merge to avoid conflicts. Exiting.")
        release_fold_lock(paths["fold_lock"])
        sys.exit(args.lock_fail_exit_code)
    if queue_changed_since(paths["queue_csv"], queue_snap):
        print("Queue changed during job; skipping final merge to avoid conflict.")
        release_fold_lock(paths["fold_lock"])
        sys.exit(args.conflict_exit_code)
    try:
        write_final_outputs_atomic(
            merged_queue_csv=paths["queue_csv"],
            ledger_parquet=paths["ledger_parquet"],
            out_xlsx=paths["merged_xlsx"],
            out_csv=paths["merged_csv"],
        )
    except Exception as exc:
        print(f"Warning: Failed to write final outputs: {exc}")
    print("Folding worker completed successfully.")
    release_fold_lock(paths["fold_lock"])


if __name__ == "__main__":
    main()