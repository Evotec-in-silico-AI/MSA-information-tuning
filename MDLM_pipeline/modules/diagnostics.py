from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from prody import calcMSAOccupancy, calcShannonEntropy
from .io_utils import atomic_write_csv_df


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON into a dictionary; return an empty dict if the file is missing or invalid."""
    try:
        p = Path(str(path))
        if not p.exists():
            return {}
        payload = json.loads(p.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame; return an empty frame on failure."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame; return an empty frame on failure."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def resolve_child(base_csv_path: Path, maybe_rel: str) -> Path:
    """Resolve an absolute or relative path against a base CSV location."""
    value = str(maybe_rel or "").strip()
    if not value:
        return Path("")
    p = Path(value)
    if p.is_absolute():
        return p
    return base_csv_path.parent / p


def tokens_to_char_array(encoded: np.ndarray) -> np.ndarray:
    """Convert a 0..20 token matrix to a byte-character array for ProDy metrics."""
    lut = np.array(
        [b"-", b"A", b"E", b"L", b"M", b"C", b"D", b"F", b"G", b"H", b"K", b"N", b"P", b"Q", b"R", b"S", b"W", b"Y", b"T", b"V", b"I"],
        dtype="S1",
    )
    flat = encoded.ravel().astype(np.int64)
    flat = np.clip(flat, 0, 20)
    out = lut[flat]
    return out.reshape(encoded.shape)


def msa_summary_from_pkl(pkl_path: str) -> Dict[str, Any]:
    """Compute summary statistics from a pickled compressed MSA using ProDy for entropy and occupancy."""
    with open(pkl_path, "rb") as f:
        X = np.asarray(pickle.load(f), dtype=np.int32)
    if X.ndim != 2:
        return {}
    n_rows, n_cols = X.shape
    gap_frac = float((X == 0).mean())
    chars = tokens_to_char_array(X)
    occupancy = np.asarray(calcMSAOccupancy(chars, occ="res", count=False), dtype=np.float64)
    entropy_bits = np.asarray(calcShannonEntropy(chars, ambiguity=True, omitgaps=True), dtype=np.float64)
    bits_site = np.log2(20.0) - entropy_bits
    bits_mean = float(np.mean(bits_site)) if bits_site.size else float("nan")
    eff_alpha_mean = float(np.mean(2.0**entropy_bits)) if entropy_bits.size else float("nan")
    occ_mean = float(np.mean(occupancy)) if occupancy.size else float("nan")
    return {
        "msa_n_rows": int(n_rows),
        "msa_n_cols": int(n_cols),
        "msa_gap_fraction": gap_frac,
        "msa_bits_per_site_mean": bits_mean,
        "msa_effective_alphabet_mean": eff_alpha_mean,
        "msa_occupancy_mean": occ_mean,
    }


def compute_msa_rate_metrics(compressed_msa_pkl_path: str) -> Dict[str, Any]:
    """Compute rate-like information metrics from a pickled compressed MSA."""
    if not compressed_msa_pkl_path:
        return {}
    p = Path(str(compressed_msa_pkl_path))
    if not p.exists():
        return {}
    try:
        summary = msa_summary_from_pkl(str(p))
    except Exception:
        summary = {}
    if not summary:
        return {}
    return {
        **summary,
        "rate_bits_per_site_mean": summary.get("msa_bits_per_site_mean", np.nan),
        "rate_effective_alphabet_mean": summary.get("msa_effective_alphabet_mean", np.nan),
        "rate_occupancy_mean": summary.get("msa_occupancy_mean", np.nan),
        "rate_gap_fraction": summary.get("msa_gap_fraction", np.nan),
    }


def enrich_merged_manifest(run_root: Path, family: str) -> Path:
    """Enrich a merged manifest with compression, MSA, model-conditioning, and training metadata."""
    run_root = Path(run_root)
    base_path = run_root / "merged" / f"{family}_manifest.csv"
    out_path = run_root / "merged" / f"{family}_manifest_enriched.csv"
    if not base_path.exists():
        return out_path
    try:
        df = pd.read_csv(base_path)
    except pd.errors.EmptyDataError:
        return out_path
    except Exception:
        return out_path
    if df.empty:
        atomic_write_csv_df(out_path, df)
        return out_path
    extras: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        comp_json = load_json(str(row.get("compression_json_path", ""))) or load_json(str(row.get("compression_json", "")))
        msa_stats: Dict[str, Any] = {}
        try:
            msa_stats = msa_summary_from_pkl(str(row.get("compressed_msa_pkl_path", "")))
        except Exception:
            msa_stats = {}
        conditioning_dict: Dict[str, Any] = {}
        cfg_path = Path(str(row.get("model_dir", ""))) / "config.json"
        if cfg_path.exists():
            cfg_dict = load_json(str(cfg_path))
            conditioning_dict = cfg_dict.get("conditioning", {}) if isinstance(cfg_dict, dict) else {}
        training: Dict[str, Any] = {}
        tsp = Path(str(row.get("training_summary_json_path", "")))
        if not tsp.exists():
            model_dir = Path(str(row.get("model_dir", "")))
            tsp = model_dir / "training_summary.json"
        if tsp.exists():
            training = load_json(str(tsp))
        extras.append(
            {
                "comp_bits_site_p10_target": comp_json.get("bits_site_p10_target", np.nan),
                "comp_bits_site_p10_achieved": comp_json.get("bits_site_p10_achieved", np.nan),
                "comp_bits_site_p10_tolerance": comp_json.get("bits_site_p10_tolerance", np.nan),
                "comp_row_weights_theta0p70_path": comp_json.get("row_weights_theta0p70_path", ""),
                "comp_row_weights_theta0p80_path": comp_json.get("row_weights_theta0p80_path", ""),
                "comp_meff_theta0p70": comp_json.get("meff_theta0p70", np.nan),
                "comp_meff_fraction_theta0p70": comp_json.get("meff_fraction_theta0p70", np.nan),
                "comp_meff_theta0p80": comp_json.get("meff_theta0p80", np.nan),
                "comp_meff_fraction_theta0p80": comp_json.get("meff_fraction_theta0p80", np.nan),
                "comp_coverage_path": comp_json.get("coverage_path", ""),
                "comp_entropy_bits_path": comp_json.get("entropy_bits_path", ""),
                "model_conditioning_enabled": bool(conditioning_dict.get("enabled", True)) if isinstance(conditioning_dict, dict) else True,
                "model_conditioning_requested": ",".join(conditioning_dict.get("requested_types", []) or []) if isinstance(conditioning_dict, dict) else "",
                "model_conditioning_resolved": ",".join(conditioning_dict.get("resolved_types", []) or []) if isinstance(conditioning_dict, dict) else "",
                "model_conditioning_dim": conditioning_dict.get("context_dim", np.nan) if isinstance(conditioning_dict, dict) else np.nan,
                "model_row_weights_theta_used": conditioning_dict.get("row_weights_theta_used", np.nan) if isinstance(conditioning_dict, dict) else np.nan,
                "train_total_steps_run": training.get("outcome", {}).get("total_steps_run", np.nan) if isinstance(training, dict) else np.nan,
                "train_early_stopped": training.get("outcome", {}).get("early_stopped", np.nan) if isinstance(training, dict) else np.nan,
                "train_stop_reason": training.get("outcome", {}).get("stop_reason", "") if isinstance(training, dict) else "",
                "train_best_step": training.get("best", {}).get("best_step", np.nan) if isinstance(training, dict) else np.nan,
                "val_loss_masked_best": training.get("best", {}).get("best_val_loss_masked", np.nan) if isinstance(training, dict) else np.nan,
                "rate_bits_per_site_mean": msa_stats.get("msa_bits_per_site_mean", np.nan),
                "rate_effective_alphabet_mean": msa_stats.get("msa_effective_alphabet_mean", np.nan),
                "rate_occupancy_mean": msa_stats.get("msa_occupancy_mean", np.nan),
                "rate_gap_fraction": msa_stats.get("msa_gap_fraction", np.nan),
                **msa_stats,
            }
        )
    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(extras)], axis=1)
    atomic_write_csv_df(out_path, out_df)
    return out_path


def read_info_theory_by_sample(info_dir: Path) -> pd.DataFrame:
    """Read info_theory/by_sample_info.csv if present; return an empty frame otherwise."""
    csv_path = info_dir / "by_sample_info.csv"
    return safe_read_csv(csv_path)


def enrich_results_df(
    results_df: pd.DataFrame,
    results_csv_path: Path,
    run_root: Path,
    ledger_path: Optional[Path] = None,
    info_theory_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Return an enriched results DataFrame by merging optional info-theory and structure metrics."""
    if results_df is None or results_df.empty:
        return results_df
    merged = results_df.copy()
    if info_theory_dir is None:
        info_theory_dir = run_root / "info_theory"
    info_df = read_info_theory_by_sample(info_theory_dir)
    if not info_df.empty and "sample_id" in merged.columns and "sample_id" in info_df.columns:
        keep = [
            "sample_id",
            "edited_token_count",
            "sum_neglogp_cond_nats",
            "sum_neglogp_uncond_nats",
            "info_total_nats",
            "info_total_bits",
            "info_bits_per_site",
            "sequence_length",
        ]
        info_keep = [c for c in keep if c in info_df.columns]
        merged = merged.merge(info_df[info_keep], on="sample_id", how="left")
    if ledger_path is None:
        ledger_path = run_root / "fold" / "metrics" / "candidates.parquet"
    ledger_df = safe_read_parquet(ledger_path)
    if not ledger_df.empty and "sample_id" in merged.columns and "sample_id" in ledger_df.columns:
        struct_add = [c for c in ledger_df.columns if c != "sample_id"]
        merged = merged.merge(ledger_df[["sample_id"] + struct_add], on="sample_id", how="left")
    passthroughs = [
        "trace_path",
        "mask_path",
        "commit_hist_path",
        "invalid_token_resamples",
        "anchor_mismatch_violations",
    ]
    for c in passthroughs:
        if c not in merged.columns and c in results_df.columns:
            merged[c] = results_df[c]
    return merged


def enrich_results_in_place(
    run_root: Path,
    family: str,
    results_glob: str = "experiments/{family}/**/results.csv",
    ledger_path: Optional[str] = None,
    info_theory_dir: Optional[str] = None,
) -> List[Path]:
    """Write results_enriched.csv alongside each results.csv for the requested family and return written paths."""
    out_paths: List[Path] = []
    run_root = Path(run_root).resolve()
    pattern = results_glob.format(family=family)
    results_csvs = sorted((run_root / "").glob(pattern))
    ledger_default = run_root / "fold" / "metrics" / "candidates.parquet"
    ledger_p = Path(ledger_path) if ledger_path else ledger_default
    info_dir = Path(info_theory_dir) if info_theory_dir else (run_root / "info_theory")
    for csv_path in results_csvs:
        if not csv_path.is_file() or csv_path.name.lower() != "results.csv":
            continue
        df = safe_read_csv(csv_path)
        if df.empty:
            continue
        enriched = enrich_results_df(
            results_df=df,
            results_csv_path=csv_path,
            run_root=run_root,
            ledger_path=ledger_p,
            info_theory_dir=info_dir,
        )
        out_path = csv_path.with_name("results_enriched.csv")
        try:
            atomic_write_csv_df(out_path, enriched)
            out_paths.append(out_path)
        except Exception:
            try:
                enriched.to_csv(out_path, index=False)
                out_paths.append(out_path)
            except Exception:
                continue
    return out_paths
