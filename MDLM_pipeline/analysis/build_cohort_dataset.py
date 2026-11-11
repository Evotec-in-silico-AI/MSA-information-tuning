#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for consolidation.
    """
    parser = argparse.ArgumentParser(description="Merge fold, MSA, and latent analysis into a single XLSX with metrics.")
    parser.add_argument("--runs_root", type=str, required=True, help="Root directory containing run folders.")
    parser.add_argument("--msa_tag", type=str, default="compressed_msa", help="MSA statistics tag.")
    parser.add_argument("--latent_tag", type=str, default="model_latents", help="Latent analysis tag.")
    parser.add_argument("--out_xlsx", type=str, default="cohort/cohort.xlsx", help="Output XLSX path.")
    return parser.parse_args()


def find_fold_queue_merged_xlsx_paths(runs_root: Path) -> List[Path]:
    """
    Return a list of fold_queue_merged XLSX files under runs_root.
    """
    return sorted(runs_root.glob("**/merged/*_fold_queue_merged.xlsx"))


def derive_run_root_from_fold_path(fold_path: Path) -> Optional[Path]:
    """
    Infer the run root directory from a fold_queue_merged path.
    """
    try:
        idx = fold_path.parts.index("merged")
    except ValueError:
        return None
    return Path(*fold_path.parts[:idx])


def read_excel_file(path: Path) -> pd.DataFrame:
    """
    Read an Excel file with openpyxl.
    """
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        return pd.DataFrame()


def get_protein_name_from_fold_df(fold_df: pd.DataFrame, fold_path: Path) -> Optional[str]:
    """
    Return the protein name from fold dataframe or from path elements.
    """
    if "protein_name" in fold_df.columns and not fold_df["protein_name"].dropna().empty:
        return str(fold_df["protein_name"].dropna().iloc[0])
    parts = list(fold_path.parts)
    if "runs" in parts:
        try:
            idx = parts.index("runs")
            return parts[idx + 1]
        except Exception:
            return None
    return None


def build_msa_by_sample_path(run_root: Path, protein_name: str, msa_tag: str) -> Path:
    """
    Return the exact path for the MSA by_sample XLSX.
    """
    return run_root / "msa_statistics" / msa_tag / f"{protein_name}_{msa_tag}_by_sample.xlsx"


def build_latent_by_sample_path(run_root: Path, protein_name: str, latent_tag: str) -> Path:
    """
    Return the exact path for the latent by_sample XLSX.
    """
    return run_root / "latent_analysis" / latent_tag / f"{protein_name}_{latent_tag}_by_sample.xlsx"


def path_like_columns(columns: Sequence[str]) -> List[str]:
    """
    Return columns that look like file system or lineage paths.
    """
    out: List[str] = []
    for col in columns:
        cl = col.lower()
        if cl.endswith("_path") or "path" in cl or col == "sidecar_dir":
            out.append(col)
    return out


def drop_fold_paths_and_extras(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop path/lineage columns and detailed fold extras while keeping sequences.
    """
    if df.empty:
        return df
    drop_cols = set(path_like_columns(df.columns))
    drop_cols.update(
        {
            "fold_time_sec",
            "tmscore_time_sec",
            "n_model_slots",
            "tm_best_pair_i",
            "tm_best_pair_j",
        }
    )
    keep_cols = {"sequence", "anchor_sequence"}
    actual_drops = [c for c in drop_cols if c in df.columns and c not in keep_cols]
    return df.drop(columns=actual_drops) if actual_drops else df


def select_msa_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant MSA columns while dropping lineage/path fields.
    """
    if df.empty:
        return df
    path_like = [c for c in df.columns if c.lower().endswith("_path") or "path" in c.lower() or c == "sidecar_dir"]
    keep_keys = {"sample_id", "p10_target", "p10_achieved", "family", "control_rep", "outer_label"}
    keep_prefixes = [
        "entropy_bits_",
        "bits_site_",
        "gap_fraction",
        "effective_length",
        "meff",
        "meff_fraction",
        "n_sequences",
        "alignment_length",
        "anchor_pid_",
        "mi_apc_",
        "msa_graph_",
        "prop_columns_gt_",
        "pid_pair_",
        "snr_",
        "singleton_fraction_",
        "row_weight_entropy_",
    ]
    out_cols: List[str] = []
    for c in df.columns:
        if c in path_like:
            continue
        if c in keep_keys:
            out_cols.append(c)
            continue
        if any(c.startswith(p) for p in keep_prefixes):
            out_cols.append(c)
    out_cols = sorted(set(out_cols))
    return df[out_cols].copy() if out_cols else pd.DataFrame()


def select_latent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant latent columns while dropping lineage/path fields.
    """
    if df.empty:
        return df
    drop_paths = set(path_like_columns(df.columns))
    drop_paths.update(
        {
            "latent_path",
            "anchor_latent_path",
            "model_dir",
            "training_curve_csv_path",
            "training_summary_json_path",
            "shard_path",
            "manifest_path",
        }
    )
    keep_keys = {"sample_id", "p10_target", "p10_achieved"}
    keep_prefixes = [
        "lat_",
        "latent_",
        "anchor_latent_",
        "dist_",
        "knn_",
        "hubness_",
        "pca",
        "spec_",
        "condition",
        "rank",
        "twonn",
        "effective_rank",
        "stable_rank",
    ]
    out_cols: List[str] = []
    for c in df.columns:
        if c in drop_paths:
            continue
        if c in keep_keys:
            out_cols.append(c)
            continue
        if any(c.startswith(p) for p in keep_prefixes):
            out_cols.append(c)
    out_cols_unique = sorted(set(out_cols))
    return df[out_cols_unique].copy() if out_cols_unique else pd.DataFrame()


def merge_sample_tables_on_id(fold_df: pd.DataFrame, msa_df: pd.DataFrame, lat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fold, MSA, and latent frames on sample_id.
    """
    if fold_df.empty or "sample_id" not in fold_df.columns:
        return pd.DataFrame()
    merged = fold_df.copy()
    merged["sample_id"] = merged["sample_id"].astype(str)
    if not msa_df.empty and "sample_id" in msa_df.columns:
        right = msa_df.copy()
        right["sample_id"] = right["sample_id"].astype(str)
        msa_add = [c for c in right.columns if c != "sample_id" and c not in merged.columns]
        if msa_add:
            merged = merged.merge(right[["sample_id"] + msa_add], on="sample_id", how="left")
    if not lat_df.empty and "sample_id" in lat_df.columns:
        right = lat_df.copy()
        right["sample_id"] = right["sample_id"].astype(str)
        lat_add = [c for c in right.columns if c != "sample_id" and c not in merged.columns]
        if lat_add:
            merged = merged.merge(right[["sample_id"] + lat_add], on="sample_id", how="left")
    return merged


def add_binary_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary events and score columns derived from pLDDT [0,100] and pTM [0,1].
    """
    out = df.copy()
    if "plddt_median" in out.columns:
        s_plddt = pd.to_numeric(out["plddt_median"], errors="coerce")
        out["score_plddt"] = s_plddt
        with np.errstate(invalid="ignore"):
            out["label_plddt70"] = (s_plddt >= 70.0).astype(int)
            out["label_plddt90"] = (s_plddt >= 90.0).astype(int)
    else:
        out["score_plddt"] = np.nan
        out["label_plddt70"] = np.nan
        out["label_plddt90"] = np.nan
    if "ptm_mean" in out.columns:
        s_ptm = np.clip(pd.to_numeric(out["ptm_mean"], errors="coerce"), 0.0, 1.0)
        out["score_ptm01"] = s_ptm
        with np.errstate(invalid="ignore"):
            out["label_ptm0p7"] = (s_ptm >= 0.70).astype(int)
            out["label_ptm0p8"] = (s_ptm >= 0.80).astype(int)
    else:
        out["score_ptm01"] = np.nan
        out["label_ptm0p7"] = np.nan
        out["label_ptm0p8"] = np.nan
    return out


def parse_p10_from_outer_label(value: object) -> float:
    """
    Parse a numeric p10 bits/site value from an outer_label string like 'p10_2p70' or 'p10_2p70__null_r0'.
    """
    if value is None:
        return float("nan")
    s = str(value)
    m = re.search(r"p10_([0-9]+p[0-9]+)", s)
    if not m:
        return float("nan")
    token = m.group(1)
    token_norm = token.replace("p", ".")
    try:
        return float(token_norm)
    except Exception:
        return float("nan")


def family_rank_value(family: object) -> int:
    """
    Map family string to a stable ordering key for target vs null interleave.
    """
    s = str(family or "").lower()
    if "msa_p10_sweep_null" in s:
        return 1
    if "msa_p10_sweep" in s:
        return 0
    return 2


def reorder_within_protein_for_p10_and_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder rows within each protein by ascending p10 target, interleaving target and matching null control(s).
    """
    if df.empty:
        return df
    if "protein_name" not in df.columns:
        return df
    frames: List[pd.DataFrame] = []
    for protein, g in df.groupby("protein_name", sort=False):
        gg = g.copy()
        if "outer_label" in gg.columns:
            p10_from_outer = gg["outer_label"].apply(parse_p10_from_outer_label)
        else:
            p10_from_outer = pd.Series([np.nan] * gg.shape[0], index=gg.index)
        p10_from_col = pd.to_numeric(gg.get("p10_target"), errors="coerce")
        gg["order_p10"] = p10_from_outer.where(p10_from_outer.notna(), p10_from_col)
        gg["order_fam"] = gg.get("family").apply(family_rank_value) if "family" in gg.columns else 2
        if "control_rep" in gg.columns:
            gg["order_rep"] = pd.to_numeric(gg["control_rep"], errors="coerce").fillna(0).astype(int)
        else:
            gg["order_rep"] = 0
        gg = gg.sort_values(["order_p10", "order_fam", "order_rep"], kind="mergesort")
        gg = gg.drop(columns=["order_p10", "order_fam", "order_rep"])
        frames.append(gg)
    return pd.concat(frames, axis=0, ignore_index=True)


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute quantile bin edges for values in [0,1].
    """
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, q)
    edges[0] = 0.0
    edges[-1] = 1.0
    return np.unique(edges)


def compute_brier_score(prob: np.ndarray, label: np.ndarray) -> float:
    """
    Return the Brier score for binary labels and probabilities.
    """
    return float(np.mean((prob - label) ** 2)) if prob.size and label.size else float("nan")


def compute_ece_for_group(df: pd.DataFrame, score_col: str, label_col: str, n_bins: int) -> Tuple[float, float, int]:
    """
    Compute ECE and Brier for a group using equal-count quantile bins.
    """
    valid = df[[score_col, label_col]].dropna()
    if valid.empty:
        return float("nan"), float("nan"), 0
    scores_raw = valid[score_col].astype(float).to_numpy()
    if score_col == "score_plddt":
        scores = np.clip(scores_raw / 100.0, 0.0, 1.0)
    else:
        scores = np.clip(scores_raw, 0.0, 1.0)
    labels = valid[label_col].astype(int).to_numpy()
    if np.all(scores == scores[0]):
        return float(abs(labels.mean() - scores.mean())), compute_brier_score(scores, labels), 1
    edges = quantile_bin_edges(scores, n_bins)
    if edges.size < 2:
        return float(abs(labels.mean() - scores.mean())), compute_brier_score(scores, labels), 1
    ece = 0.0
    for i in range(edges.size - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (scores >= lo) & (scores <= hi) if i == 0 else (scores > lo) & (scores <= hi)
        if not np.any(mask):
            continue
        acc = float(labels[mask].mean())
        conf = float(scores[mask].mean())
        prop = float(np.mean(mask))
        ece += abs(acc - conf) * prop
    return float(ece), compute_brier_score(scores, labels), int(edges.size - 1)


def compute_auc_for_group(df: pd.DataFrame, score_col: str, label_col: str) -> float:
    """
    Compute ROC AUC using a rank-based estimate.
    """
    valid = df[[score_col, label_col]].dropna()
    if valid.empty:
        return float("nan")
    y = valid[label_col].astype(int).to_numpy()
    s_raw = valid[score_col].astype(float).to_numpy()
    if score_col == "score_plddt":
        s = np.clip(s_raw / 100.0, 0.0, 1.0)
    else:
        s = np.clip(s_raw, 0.0, 1.0)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[pos].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def group_keys_for_metrics(df: pd.DataFrame) -> List[List[str]]:
    """
    Return grouping key sets for metrics computation.
    """
    keys: List[List[str]] = []
    base = ["run_root", "protein_name"] if "protein_name" in df.columns else ["run_root"]
    keys.append(base)
    fam: List[str] = []
    if "family" in df.columns:
        fam.append("family")
    if "control_rep" in df.columns:
        fam.append("control_rep")
    if "p10_target" in df.columns:
        keys.append([*base, "p10_target"])
    if fam:
        keys.append([*base, *fam])
    return keys


def compute_calibration_by_group(df: pd.DataFrame, ece_bins: int) -> pd.DataFrame:
    """
    Compute ECE, Brier, and AUC per group.
    """
    key_sets = group_keys_for_metrics(df)
    rows: List[Dict[str, object]] = []
    pairs = [
        ("score_plddt", "label_plddt70", "plddt_to_plddt70"),
        ("score_plddt", "label_plddt90", "plddt_to_plddt90"),
        ("score_plddt", "label_ptm0p8", "plddt_to_ptm0p8"),
        ("score_plddt", "label_ptm0p7", "plddt_to_ptm0p7"),
    ]
    for keys in key_sets:
        if not all(k in df.columns for k in keys):
            continue
        for gvals, gdf in df.groupby(keys, dropna=False, sort=False):
            for score_col, label_col, pair_name in pairs:
                if score_col not in gdf.columns or label_col not in gdf.columns:
                    continue
                ece, brier, bins_used = compute_ece_for_group(gdf, score_col, label_col, int(ece_bins))
                auc = compute_auc_for_group(gdf, score_col, label_col)
                rec: Dict[str, object] = {}
                if isinstance(gvals, tuple):
                    for i, kname in enumerate(keys):
                        rec[kname] = gvals[i]
                else:
                    rec[keys[0]] = gvals
                rec.update(
                    {
                        "pair": pair_name,
                        "score_name": score_col,
                        "label_name": label_col,
                        "n": int(gdf.shape[0]),
                        "bins_used": int(bins_used),
                        "ece": float(ece),
                        "brier": float(brier),
                        "auc": float(auc),
                    }
                )
                rows.append(rec)
    return pd.DataFrame(rows)


def compute_coverage_by_group(df: pd.DataFrame, k_values: Sequence[int]) -> pd.DataFrame:
    """
    Compute coverage@K per group.
    """
    key_sets = group_keys_for_metrics(df)
    rows: List[Dict[str, object]] = []
    events = [("label_plddt70", "plddt70"), ("label_plddt90", "plddt90"), ("label_ptm0p8", "ptm0p8"), ("label_ptm0p7", "ptm0p7")]
    scores = [("score_plddt", "plddt_median"), ("score_ptm01", "ptm_mean")]
    for keys in key_sets:
        if not all(k in df.columns for k in keys):
            continue
        for gvals, gdf in df.groupby(keys, dropna=False, sort=False):
            for score_col, score_name in scores:
                if score_col not in gdf.columns:
                    continue
                for label_col, label_name in events:
                    if label_col not in gdf.columns:
                        continue
                    valid = gdf[[score_col, label_col]].dropna().sort_values(score_col, ascending=False)
                    if valid.empty:
                        for k in k_values:
                            rows.append(
                                {
                                    **({kname: (gvals[i] if isinstance(gvals, tuple) else gvals) for i, kname in enumerate(keys)}),
                                    "score_name": score_name,
                                    "event_name": label_name,
                                    "K": int(k),
                                    "coverage_at_k": float("nan"),
                                    "n_candidates": 0,
                                    "n_success_total": 0,
                                }
                            )
                        continue
                    labels = valid[label_col].astype(int).to_numpy()
                    n = labels.size
                    tot = int(labels.sum())
                    for k in k_values:
                        kk = int(min(max(1, int(k)), n))
                        cov = float(np.mean(labels[:kk])) if kk > 0 else float("nan")
                        rows.append(
                            {
                                **({kname: (gvals[i] if isinstance(gvals, tuple) else gvals) for i, kname in enumerate(keys)}),
                                "score_name": score_name,
                                "event_name": label_name,
                                "K": int(k),
                                "coverage_at_k": float(cov),
                                "n_candidates": int(n),
                                "n_success_total": int(tot),
                            }
                        )
    return pd.DataFrame(rows)


def make_dictionary_dataframe() -> pd.DataFrame:
    """
    Return a simple data dictionary for workbook columns.
    """
    rows = [
        {"column": "run_root", "description": "Run root directory for the protein."},
        {"column": "protein_name", "description": "Protein identifier used in the run."},
        {"column": "family", "description": "Experiment family"},
        {"column": "control_rep", "description": "Null replicate index."},
        {"column": "outer_label", "description": "Shard label including target and null patterns."},
        {"column": "sample_id", "description": "Unique candidate identifier."},
        {"column": "p10_target", "description": "Requested bits/site p10 target for the shard."},
        {"column": "p10_achieved", "description": "Achieved bits/site p10 for the shard."},
        {"column": "plddt_median", "description": "Median pLDDT in [0,100]."},
        {"column": "ptm_mean", "description": "Mean pTM in [0,1]."},
        {"column": "label_plddt70", "description": "1 if median pLDDT ≥ 70."},
        {"column": "label_plddt90", "description": "1 if median pLDDT ≥ 90."},
        {"column": "label_ptm0p7", "description": "1 if mean pTM ≥ 0.70."},
        {"column": "label_ptm0p8", "description": "1 if mean pTM ≥ 0.80."},
        {"column": "score_plddt", "description": "Median pLDDT used as a score in [0,100]."},
        {"column": "score_ptm01", "description": "Mean pTM used as a score in [0,1]."},
    ]
    return pd.DataFrame(rows)


def write_workbook_openpyxl(
    out_xlsx: Path,
    by_sample: pd.DataFrame,
    calibration: pd.DataFrame,
    coverage: pd.DataFrame,
    globals_payload: Dict[str, object],
    audit_payload: Dict[str, object],
    dictionary_df: pd.DataFrame,
) -> None:
    """
    Write multiple sheets to a single XLSX file using openpyxl.
    """
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xlw:
        by_sample.to_excel(xlw, sheet_name="by_sample", index=False)
        calibration.to_excel(xlw, sheet_name="calibration_by_group", index=False)
        coverage.to_excel(xlw, sheet_name="coverage_by_group", index=False)
        pd.DataFrame([globals_payload]).to_excel(xlw, sheet_name="globals", index=False)
        pd.DataFrame([audit_payload]).to_excel(xlw, sheet_name="audit", index=False)
        dictionary_df.to_excel(xlw, sheet_name="dictionary", index=False)


def write_figure_globals_json(out_xlsx: Path, by_sample: pd.DataFrame) -> Path:
    """
    Persist figure globals JSON alongside the cohort workbook.
    """
    fig: Dict[str, object] = {}
    if "p10_target" in by_sample.columns:
        p10v = pd.to_numeric(by_sample["p10_target"], errors="coerce").to_numpy()
        p10v = p10v[np.isfinite(p10v)]
        if p10v.size > 0:
            fig["p10_targets_unique_sorted"] = sorted(list(set([float(x) for x in p10v])))
    fig["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    fig["version"] = "1.3.0"
    path_json = out_xlsx.with_suffix(".fig_globals.json")
    with open(path_json, "w") as f:
        json.dump(fig, f, indent=2)
    return path_json


def consolidate_runs(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object], Dict[str, object]]:
    """
    Consolidate all run roots into final dataframes and payloads.
    """
    runs_root = Path(args.runs_root).resolve()
    fold_paths = find_fold_queue_merged_xlsx_paths(runs_root)
    compiled_by_sample: List[pd.DataFrame] = []
    diagnostics_records: List[Dict[str, object]] = []
    for fold_xlsx in fold_paths:
        run_root = derive_run_root_from_fold_path(fold_xlsx)
        if run_root is None:
            continue
        fold_df = read_excel_file(fold_xlsx)
        if fold_df.empty or "sample_id" not in fold_df.columns:
            continue
        fold_df = fold_df.drop_duplicates(subset=["sample_id"], keep="last")
        fold_df["run_root"] = str(run_root)
        fold_df = drop_fold_paths_and_extras(fold_df)
        protein_name = get_protein_name_from_fold_df(fold_df, fold_xlsx)
        if not protein_name:
            raise RuntimeError(f"Could not resolve protein_name for fold: {fold_xlsx}")
        msa_path = build_msa_by_sample_path(run_root, protein_name, args.msa_tag)
        latent_path = build_latent_by_sample_path(run_root, protein_name, args.latent_tag)
        if not msa_path.exists():
            raise FileNotFoundError(f"MSA by_sample not found: {msa_path}")
        if not latent_path.exists():
            raise FileNotFoundError(f"Latent by_sample not found: {latent_path}")
        msa_df_raw = read_excel_file(msa_path)
        lat_df_raw = read_excel_file(latent_path)
        msa_df = select_msa_columns(msa_df_raw)
        lat_df = select_latent_columns(lat_df_raw)
        merged = merge_sample_tables_on_id(fold_df, msa_df, lat_df)
        if not merged.empty and "protein_name" not in merged.columns:
            merged["protein_name"] = protein_name
        compiled_by_sample.append(merged)
        diagnostics_records.append(
            {
                "run_root": str(run_root),
                "protein_name": protein_name,
                "fold_rows": int(fold_df.shape[0]),
                "msa_rows": int(msa_df_raw.shape[0]),
                "latent_rows": int(lat_df_raw.shape[0]),
            }
        )
    if not compiled_by_sample:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}
    by_sample = pd.concat(compiled_by_sample, axis=0, ignore_index=True)
    if "sample_id" in by_sample.columns:
        by_sample = by_sample.drop_duplicates(subset=["sample_id"], keep="last")
    if "plddt_median" in by_sample.columns:
        plddt_med = pd.to_numeric(by_sample["plddt_median"], errors="coerce")
        if np.nanquantile(plddt_med, 0.95) <= 1.0:
            for c in ["plddt_mean", "plddt_median", "plddt_min", "plddt_max", "plddt_std"]:
                if c in by_sample.columns:
                    by_sample[c] = pd.to_numeric(by_sample[c], errors="coerce") * 100.0
    by_sample = add_binary_labels(by_sample)
    by_sample = reorder_within_protein_for_p10_and_null(by_sample)
    calibration = compute_calibration_by_group(by_sample, ece_bins=10)
    coverage = compute_coverage_by_group(by_sample, k_values=[1, 5, 10])
    globals_payload = {
        "runs_root": str(runs_root),
        "msa_tag": args.msa_tag,
        "latent_tag": args.latent_tag,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "k_values": "1,5,10",
        "ece_bins": 10,
        "version": "1.3.0",
    }
    audit_payload = {
        "n_run_roots": int(len(diagnostics_records)),
        "n_by_sample_rows": int(by_sample.shape[0]),
        "n_calibration_rows": int(calibration.shape[0]),
        "n_coverage_rows": int(coverage.shape[0]),
        "diagnostics_records": diagnostics_records,
    }
    return by_sample, calibration, coverage, globals_payload, audit_payload


def main() -> int:
    """
    Entry point for consolidation CLI.
    """
    args = parse_arguments()
    out_xlsx = Path(args.out_xlsx).resolve()
    by_sample, calibration, coverage, globals_payload, audit_payload = consolidate_runs(args)
    if by_sample.empty:
        print("No data to write.")
        return 0
    dictionary_df = make_dictionary_dataframe()
    write_workbook_openpyxl(out_xlsx, by_sample, calibration, coverage, globals_payload, audit_payload, dictionary_df)
    write_figure_globals_json(out_xlsx, by_sample)
    print(f"Wrote XLSX: {out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
