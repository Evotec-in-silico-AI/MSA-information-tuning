import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from figure_common import convert_archived_columns


@dataclass(frozen=True)
class Figure7Config:
    """Configure Figure 7 generation and exports."""

    data_dir_name: str = "folding_results"
    jackhmmer_filename: str = "chai1_structural_comparison_jackhmmer.csv"
    target_file_map: Tuple[Tuple[str, str], ...] = (
        ("0", "chai1_structural_comparison_below_folding_budget_0.csv"),
        ("2", "chai1_structural_comparison_below_folding_budget_0p02.csv"),
        ("5", "chai1_structural_comparison_below_folding_budget_0p05.csv"),
        ("7.5", "chai1_structural_comparison_below_folding_budget_0p075.csv"),
        ("10", "chai1_structural_comparison_below_folding_budget_0p1.csv"),
        ("12.5", "chai1_structural_comparison_below_folding_budget_0p125.csv"),
        ("15", "chai1_structural_comparison_below_folding_budget_0p15.csv"),
        ("20", "chai1_structural_comparison_below_folding_budget_0p2.csv"),
        ("25", "chai1_structural_comparison_below_folding_budget_0p25.csv"),
        ("30", "chai1_structural_comparison_below_folding_budget_0p3.csv")
    )
    analysis_target_order: Tuple[str, ...] = (
        "0",
        "2",
        "5",
        "7.5",
        "10",
        "12.5",
        "15",
        "20",
        "25",
        "30"
    )
    common_target_matched_set_target_order: Tuple[str, ...] = (
        "0",
        "2",
        "5",
        "7.5",
        "10",
        "12.5",
        "15",
        "20",
        "25",
        "30"
    )
    output_dir_name: str = "figure7_and_supplementary_figure5_outputs"
    output_xlsx_name: str = "figure7_and_supplementary_figure5_source_data.xlsx"
    excel_chunk_rows: int = 1_000_000
    use_recomputed_model_qc_means: bool = True
    ptm_positive_threshold: float = 0.0
    ptm_primary_operational_threshold: float = 0.03
    ptm_threshold_grid: Tuple[float, ...] = (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05)
    bootstrap_iterations: int = 10000
    bootstrap_ci_level: float = 0.95
    bootstrap_seed: int = 7
    bootstrap_min_n: int = 3


@dataclass(frozen=True)
class Figure7Paths:
    """Resolve input and output paths for Figure 7."""

    working_dir: Path
    data_dir: Path
    output_dir: Path
    output_xlsx_path: Path

class SheetNamePolicy:
    """Enforce Excel-safe sheet names and uniqueness."""

    def __init__(self, max_len: int = 31) -> None:
        """Create a sheet naming policy."""
        self.max_len = int(max_len)
        self.invalid_chars_pattern = re.compile(r"[:\\/?*\[\]]")
        self.used: Dict[str, int] = {}

    def make_safe(self, proposed: str) -> str:
        """Return a safe, unique sheet name."""
        cleaned = self.invalid_chars_pattern.sub("_", str(proposed)).strip()
        if not cleaned:
            cleaned = "Sheet"
        cleaned = cleaned[: self.max_len]
        cleaned = cleaned.rstrip("'").lstrip("'")
        if not cleaned:
            cleaned = "Sheet"
        base = cleaned
        if base not in self.used:
            self.used[base] = 1
            return base
        counter = self.used[base]
        while True:
            suffix = f"_{counter}"
            keep = max(1, self.max_len - len(suffix))
            candidate = (base[:keep] + suffix)[: self.max_len]
            if candidate not in self.used:
                self.used[base] = counter + 1
                self.used[candidate] = 1
                return candidate
            counter += 1


class ExcelNotebookWriter:
    """Write a multi-sheet XLSX notebook."""

    def __init__(self, xlsx_path: Path, chunk_rows: int) -> None:
        """Initialize the writer."""
        self.xlsx_path = Path(xlsx_path)
        self.chunk_rows = int(chunk_rows)
        self.sheet_policy = SheetNamePolicy(max_len=31)
        self.inventory_rows: List[Dict[str, Any]] = []

    def write(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Write all tables to the notebook."""
        self.xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(self.xlsx_path, engine="openpyxl") as writer:
            for base_name, dataframe in tables.items():
                self.write_dataframe_chunked(writer, base_name, dataframe)
            inventory_df = pd.DataFrame(self.inventory_rows)
            self.write_dataframe_chunked(writer, "export_inventory", inventory_df)

    def write_dataframe_chunked(self, writer: pd.ExcelWriter, base_name: str, dataframe: pd.DataFrame) -> None:
        """Write a dataframe as one or more sheets."""
        if dataframe is None or dataframe.shape[0] == 0:
            return
        n_rows = int(dataframe.shape[0])
        if n_rows <= self.chunk_rows:
            sheet_name = self.sheet_policy.make_safe(base_name)
            dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
            self.inventory_rows.append({"sheet_base": base_name, "rows": n_rows, "parts": 1})
            return
        parts = int(math.ceil(n_rows / float(self.chunk_rows)))
        for part_index in range(parts):
            lo = int(part_index * self.chunk_rows)
            hi = int(min((part_index + 1) * self.chunk_rows, n_rows))
            part_df = dataframe.iloc[lo:hi].copy()
            sheet_name = self.sheet_policy.make_safe(f"{base_name}_p{part_index + 1}")
            part_df.to_excel(writer, sheet_name=sheet_name, index=False)
            self.inventory_rows.append({"sheet_base": base_name, "rows": n_rows, "parts": parts})


class Figure7StatsEngine:
    """Compute the minimal paired statistics used in Figure 7."""

    def paired_ttest(self, values: np.ndarray) -> float:
        """Compute a paired t-test against zero."""
        clean_values = values[np.isfinite(values)]
        if clean_values.size < 3:
            return float("nan")
        zeros = np.zeros(clean_values.size, dtype=float)
        result = stats.ttest_rel(clean_values, zeros, nan_policy="omit")
        return float(result.pvalue)

    def paired_wilcoxon(self, values: np.ndarray) -> float:
        """Compute a paired Wilcoxon test against zero."""
        clean_values = values[np.isfinite(values)]
        if clean_values.size < 3 or np.allclose(clean_values, 0.0):
            return float("nan")
        zeros = np.zeros(clean_values.size, dtype=float)
        try:
            result = stats.wilcoxon(clean_values, zeros, zero_method="wilcox", correction=False, alternative="two-sided")
        except Exception:
            return float("nan")
        return float(result.pvalue)

    def bootstrap_mean_ci(self, values: np.ndarray, ci_level: float, iterations: int, seed: int, min_n: int) -> Tuple[float, float]:
        """Compute a percentile bootstrap confidence interval for the mean."""
        clean_values = values[np.isfinite(values)]
        iteration_count = int(iterations)
        minimum_count = int(min_n)
        confidence_level = float(ci_level)
        if iteration_count <= 0:
            raise ValueError("bootstrap_iterations must be positive")
        if minimum_count < 2:
            raise ValueError("bootstrap_min_n must be at least 2")
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError("bootstrap_ci_level must be between 0 and 1")
        if clean_values.size < minimum_count:
            return float("nan"), float("nan")
        rng = np.random.default_rng(int(seed))
        bootstrap_means = np.empty(iteration_count, dtype=float)
        batch_size = max(1, min(iteration_count, int(max(1, 1000000 // clean_values.size))))
        for start_index in range(0, iteration_count, batch_size):
            stop_index = min(start_index + batch_size, iteration_count)
            sample_indices = rng.integers(0, clean_values.size, size=(stop_index - start_index, clean_values.size))
            bootstrap_means[start_index:stop_index] = clean_values[sample_indices].mean(axis=1)
        alpha = 1.0 - confidence_level
        low = float(np.quantile(bootstrap_means, alpha / 2.0))
        high = float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0))
        return low, high


class Figure7PlotStyle:
    """Provide fixed colors and plotting helpers for Figure 7."""

    def __init__(self) -> None:
        """Initialize plotting style."""
        self.color_map = {
            "reference": "#3F3F3F",
            "all_pairs": "#2F6B55",
            "common_target_matched_set": "#7A557A",
            "delta_tm_score": "#2F6B55",
            "delta_rmsd": "#7A557A",
            "delta_ptm": "#8A3F62",
            "meff": "#5F5F5F",
            "raw_depth": "#7A705C",
            "depth_fraction_of_jackhmmer": "#3F3F3F",
            "neutral_light": "#BDBDBD",
            "neutral_dark": "#1F1F1F",
        }

    def get_color(self, key: str) -> str:
        """Return a color value for a key."""
        return self.color_map.get(str(key), "#444444")

    def safe_filename(self, text_value: str) -> str:
        """Return a filesystem-safe filename stem."""
        cleaned = re.sub(r"[^\w\s\-\.\(\)=/]+", "_", str(text_value))
        cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
        cleaned = cleaned.replace("/", "_over_")
        return cleaned


class Figure7DataLoader:
    """Load and normalize Figure 7 input tables."""

    def __init__(self, config: Figure7Config) -> None:
        """Initialize loader."""
        self.config = config

    def load_csv(self, path: Path) -> pd.DataFrame:
        """Load one CSV path."""
        return pd.read_csv(path)

    def target_file_map_dict(self) -> Dict[str, str]:
        """Return the explicit target-file mapping as a dictionary."""
        return {key: value for key, value in self.config.target_file_map}

    def target_label_to_percent(self, target_label: str) -> float:
        """Convert a target label like 7.5 to a numeric percent."""
        return float(str(target_label))

    def validate_target_file_map(self, paths: Figure7Paths) -> None:
        """Validate the explicit target-file mapping."""
        mapping = self.target_file_map_dict()
        if len(mapping) == 0:
            raise ValueError("target_file_map cannot be empty")
        if len(mapping) != len(self.config.target_file_map):
            raise ValueError("target_file_map contains duplicate target labels")
        analysis_labels = list(self.config.analysis_target_order)
        if set(analysis_labels) != set(mapping.keys()):
            raise ValueError("analysis_target_order must match target_file_map keys exactly")
        common_labels = set(self.config.common_target_matched_set_target_order)
        if not common_labels.issubset(set(analysis_labels)):
            raise ValueError("common_target_matched_set_target_order must be a subset of analysis_target_order")
        file_names = list(mapping.values())
        if len(file_names) != len(set(file_names)):
            raise ValueError("target_file_map contains duplicate file names")
        numeric_order = [self.target_label_to_percent(label) for label in analysis_labels]
        if any(b <= a for a, b in zip(numeric_order[:-1], numeric_order[1:])):
            raise ValueError("analysis_target_order must be strictly increasing")
        if any(b <= a for a, b in zip(self.config.ptm_threshold_grid[:-1], self.config.ptm_threshold_grid[1:])):
            raise ValueError("ptm_threshold_grid must be strictly increasing")
        primary_threshold = float(self.config.ptm_primary_operational_threshold)
        if primary_threshold < min(self.config.ptm_threshold_grid) or primary_threshold > max(self.config.ptm_threshold_grid):
            raise ValueError("ptm_primary_operational_threshold must lie within ptm_threshold_grid")
        required_paths = [paths.data_dir / self.config.jackhmmer_filename]
        required_paths.extend(paths.data_dir / file_name for file_name in file_names)
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Missing required input files: {missing_paths}")

    def build_input_inventory(self, paths: Figure7Paths) -> pd.DataFrame:
        """Build input file inventory from the explicit target mapping."""
        rows: List[Dict[str, Any]] = []
        jack_path = paths.data_dir / self.config.jackhmmer_filename
        rows.append({
            "source_kind": "jackhmmer",
            "target_label": "jackhmmer",
            "target_percent": float("nan"),
            "path": str(jack_path),
            "exists": bool(jack_path.exists()),
        })
        for target_label in self.config.analysis_target_order:
            file_name = self.target_file_map_dict()[target_label]
            tuned_path = paths.data_dir / file_name
            rows.append({
                "source_kind": "below_folding_budget",
                "target_label": str(target_label),
                "target_percent": float(self.target_label_to_percent(target_label)),
                "path": str(tuned_path),
                "exists": bool(tuned_path.exists()),
            })
        return pd.DataFrame(rows)

    def model_metric_columns(self, prefix: str) -> List[str]:
        """Return per-model metric columns for a prefix."""
        return [f"{prefix}_for_model_{index}" for index in range(1, 6)]

    def common_numeric_columns(self) -> List[str]:
        """Return shared numeric columns across baseline and tuned CSVs."""
        return [
            "num_sequences_in_alignment",
            "mean_tm_score",
            "std_dev_tm_score",
            "mean_rmsd",
            "std_dev_rmsd",
            "mean_ptm",
            "std_dev_ptm",
        ] + self.model_metric_columns("tm_score") + self.model_metric_columns("rmsd") + self.model_metric_columns("ptm")

    def tuned_numeric_columns(self) -> List[str]:
        """Return tuned-only numeric columns."""
        return [
            "row_drop_limit_frac",
            "meff_threshold",
            "start_rows",
            "snr_start",
            "min_rows_kept",
            "folding_budget_bps",
            "target_fraction_below_folding_budget",
            "tolerance",
            "selection_delta",
            "rows",
            "snr",
            "pid_mean_pct",
            "pid_sd_pct",
            "pid_pair_count",
            "bps_min",
            "bps_p1",
            "bps_p10",
            "bps_median",
            "bps_mean",
            "gap_fraction",
            "projected_columns",
            "columns_below_folding_budget",
            "fraction_columns_below_folding_budget",
            "meff_theta_0p70",
            "meff_fraction_theta_0p70",
            "path_index",
        ]

    def add_numeric_columns(self, df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Convert selected columns to numeric if present."""
        out = df.copy()
        for column in columns:
            if column in out.columns:
                out[column] = pd.to_numeric(out[column], errors="coerce")
        return out

    def required_jackhmmer_columns(self) -> List[str]:
        """Return required columns for the baseline file."""
        return [
            "pdb_id",
            "num_sequences_in_alignment",
            "mean_tm_score",
            "mean_rmsd",
            "mean_ptm",
        ] + self.model_metric_columns("tm_score") + self.model_metric_columns("rmsd") + self.model_metric_columns("ptm")

    def required_tuned_columns(self) -> List[str]:
        """Return required columns for tuned comparison files."""
        return self.required_jackhmmer_columns() + [
            "msa_id",
            "status",
            "target_fraction_below_folding_budget",
            "tolerance",
            "selection_delta",
            "rows",
            "snr",
            "pid_mean_pct",
            "pid_sd_pct",
            "pid_pair_count",
            "bps_min",
            "bps_p1",
            "bps_p10",
            "bps_median",
            "bps_mean",
            "gap_fraction",
            "projected_columns",
            "columns_below_folding_budget",
            "fraction_columns_below_folding_budget",
            "meff_theta_0p70",
            "meff_fraction_theta_0p70",
            "path_index",
            "direction",
            "message",
        ]

    def validate_jackhmmer_schema(self, raw_df: pd.DataFrame) -> None:
        """Validate required baseline columns."""
        missing = [column for column in self.required_jackhmmer_columns() if column not in raw_df.columns]
        if missing:
            raise ValueError(f"Missing required JackHMMER columns: {missing}")

    def validate_tuned_schema(self, raw_df: pd.DataFrame) -> None:
        """Validate required tuned-file columns."""
        missing = [column for column in self.required_tuned_columns() if column not in raw_df.columns]
        if missing:
            raise ValueError(f"Missing required tuned columns: {missing}")

    def build_raw_count_record(self, source_kind: str, target_label: str, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """Build one raw-row count record."""
        return {
            "source_kind": str(source_kind),
            "target_label": str(target_label),
            "raw_rows_loaded": int(raw_df.shape[0]),
        }

    def classify_zero_failure_row(self, row: pd.Series) -> bool:
        """Return whether a row is an all-zero structural failure."""
        tm_columns = [column for column in self.model_metric_columns("tm_score") if column in row.index]
        rmsd_columns = [column for column in self.model_metric_columns("rmsd") if column in row.index]
        tm_values = pd.to_numeric(pd.Series([row.get(column) for column in tm_columns]), errors="coerce").to_numpy(dtype=float) if tm_columns else np.array([], dtype=float)
        rmsd_values = pd.to_numeric(pd.Series([row.get(column) for column in rmsd_columns]), errors="coerce").to_numpy(dtype=float) if rmsd_columns else np.array([], dtype=float)
        mean_tm = pd.to_numeric(pd.Series([row.get("mean_tm_score", np.nan)]), errors="coerce").to_numpy(dtype=float)[0]
        mean_rmsd = pd.to_numeric(pd.Series([row.get("mean_rmsd", np.nan)]), errors="coerce").to_numpy(dtype=float)[0]
        tm_all_zero = bool(tm_values.size > 0 and np.all(np.isfinite(tm_values)) and np.allclose(tm_values, 0.0))
        rmsd_all_zero = bool(rmsd_values.size > 0 and np.all(np.isfinite(rmsd_values)) and np.allclose(rmsd_values, 0.0))
        mean_tm_zero = bool(np.isfinite(mean_tm) and math.isclose(float(mean_tm), 0.0, abs_tol=1.0e-12))
        mean_rmsd_zero = bool(np.isfinite(mean_rmsd) and math.isclose(float(mean_rmsd), 0.0, abs_tol=1.0e-12))
        return bool(tm_all_zero or rmsd_all_zero or mean_tm_zero or mean_rmsd_zero)

    def model_validity_mask(self, row: pd.Series) -> np.ndarray:
        """Return a validity mask for the five model outputs."""
        tm_values = np.array([pd.to_numeric(row.get(column), errors="coerce") for column in self.model_metric_columns("tm_score")], dtype=float)
        rmsd_values = np.array([pd.to_numeric(row.get(column), errors="coerce") for column in self.model_metric_columns("rmsd")], dtype=float)
        valid_mask = np.isfinite(tm_values) & np.isfinite(rmsd_values)
        valid_mask &= ~(np.isclose(tm_values, 0.0) & np.isclose(rmsd_values, 0.0))
        return valid_mask

    def recompute_model_qc_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute summary metrics after model-level QC."""
        out = df.copy()
        recomputed_tm_values: List[float] = []
        recomputed_rmsd_values: List[float] = []
        recomputed_ptm_values: List[float] = []
        valid_tm_counts: List[int] = []
        valid_rmsd_counts: List[int] = []
        valid_ptm_counts: List[int] = []
        for _, row in out.iterrows():
            valid_mask = self.model_validity_mask(row)
            tm_values = np.array([pd.to_numeric(row.get(column), errors="coerce") for column in self.model_metric_columns("tm_score")], dtype=float)
            rmsd_values = np.array([pd.to_numeric(row.get(column), errors="coerce") for column in self.model_metric_columns("rmsd")], dtype=float)
            ptm_values = np.array([pd.to_numeric(row.get(column), errors="coerce") for column in self.model_metric_columns("ptm")], dtype=float)
            tm_clean = tm_values[valid_mask]
            rmsd_clean = rmsd_values[valid_mask]
            ptm_clean = ptm_values[valid_mask & np.isfinite(ptm_values)]
            recomputed_tm_values.append(float(np.mean(tm_clean)) if tm_clean.size > 0 else float("nan"))
            recomputed_rmsd_values.append(float(np.mean(rmsd_clean)) if rmsd_clean.size > 0 else float("nan"))
            recomputed_ptm_values.append(float(np.mean(ptm_clean)) if ptm_clean.size > 0 else float("nan"))
            valid_tm_counts.append(int(tm_clean.size))
            valid_rmsd_counts.append(int(rmsd_clean.size))
            valid_ptm_counts.append(int(ptm_clean.size))
        out["mean_tm_score_recomputed"] = recomputed_tm_values
        out["mean_rmsd_recomputed"] = recomputed_rmsd_values
        out["mean_ptm_recomputed"] = recomputed_ptm_values
        out["n_valid_tm_models"] = valid_tm_counts
        out["n_valid_rmsd_models"] = valid_rmsd_counts
        out["n_valid_ptm_models"] = valid_ptm_counts
        return out

    def normalize_jackhmmer(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize JackHMMER rows and collect stage summaries."""
        self.validate_jackhmmer_schema(raw_df)
        out = self.add_numeric_columns(raw_df.copy(), self.common_numeric_columns())
        information_columns = tuple(column_name for column_name in ("folding_budget_bps", "bps_min", "bps_p1", "bps_p10", "bps_median", "bps_mean") if column_name in out.columns)
        if information_columns:
            out = convert_archived_columns(out, information_columns=information_columns)
        out["pdb_id"] = out["pdb_id"].astype(str)
        out["source_kind"] = "jackhmmer"
        out["target_label"] = "jackhmmer"
        out["target_percent"] = np.nan
        out["status"] = "ok"
        out = self.recompute_model_qc_metrics(out)
        out["passed_status_gate"] = True
        out["passed_nonzero_structural_gate"] = ~out.apply(self.classify_zero_failure_row, axis=1)
        out["passed_summary_gate"] = out[["mean_tm_score_recomputed", "mean_rmsd_recomputed", "mean_ptm_recomputed"]].notna().all(axis=1)
        out["valid_structural_row"] = out["passed_nonzero_structural_gate"] & out["passed_summary_gate"]
        out["invalid_reason_primary"] = ""
        out.loc[~out["passed_nonzero_structural_gate"], "invalid_reason_primary"] = "all_zero_structural_metrics"
        out.loc[out["passed_nonzero_structural_gate"] & ~out["passed_summary_gate"], "invalid_reason_primary"] = "missing_summary_metrics"
        excluded = out.loc[~out["valid_structural_row"]].copy()
        valid = out.loc[out["valid_structural_row"]].copy()
        stage_summary = pd.DataFrame([
            {
                "source_kind": "jackhmmer",
                "target_label": "jackhmmer",
                "target_percent": float("nan"),
                "raw_rows_loaded": int(out.shape[0]),
                "status_ok_rows": int(out["passed_status_gate"].sum()),
                "nonzero_structural_rows": int(out["passed_nonzero_structural_gate"].sum()),
                "summary_complete_rows": int(out["passed_summary_gate"].sum()),
                "valid_rows": int(valid.shape[0]),
                "excluded_rows": int(excluded.shape[0]),
                "excluded_status_not_ok": int(0),
                "excluded_all_zero_structural_metrics": int((excluded["invalid_reason_primary"] == "all_zero_structural_metrics").sum()),
                "excluded_missing_summary_metrics": int((excluded["invalid_reason_primary"] == "missing_summary_metrics").sum()),
            }
        ])
        return valid, excluded, stage_summary

    def normalize_tuned(self, raw_df: pd.DataFrame, target_label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize tuned rows and collect stage summaries."""
        self.validate_tuned_schema(raw_df)
        numeric_columns = self.common_numeric_columns() + self.tuned_numeric_columns()
        out = self.add_numeric_columns(raw_df.copy(), numeric_columns)
        information_columns = tuple(column_name for column_name in ("folding_budget_bps", "bps_min", "bps_p1", "bps_p10", "bps_median", "bps_mean") if column_name in out.columns)
        if information_columns:
            out = convert_archived_columns(out, information_columns=information_columns)
        out["pdb_id"] = out["pdb_id"].astype(str)
        out["msa_id"] = out["msa_id"].astype(str)
        out["source_kind"] = "below_folding_budget"
        out["target_label"] = str(target_label)
        out["target_percent"] = float(self.target_label_to_percent(target_label))
        out["target_percentage"] = out["target_percent"]
        out["status"] = out["status"].fillna("unknown").astype(str)
        out["direction"] = out["direction"].fillna("").astype(str)
        out["message"] = out["message"].fillna("").astype(str)
        out = self.recompute_model_qc_metrics(out)
        out["passed_status_gate"] = out["status"].str.lower() == "ok"
        out["passed_nonzero_structural_gate"] = ~out.apply(self.classify_zero_failure_row, axis=1)
        out["passed_summary_gate"] = out[["mean_tm_score_recomputed", "mean_rmsd_recomputed", "mean_ptm_recomputed"]].notna().all(axis=1)
        out["valid_structural_row"] = out["passed_status_gate"] & out["passed_nonzero_structural_gate"] & out["passed_summary_gate"]
        out["invalid_reason_primary"] = ""
        out.loc[~out["passed_status_gate"], "invalid_reason_primary"] = "status_not_ok"
        out.loc[out["passed_status_gate"] & ~out["passed_nonzero_structural_gate"], "invalid_reason_primary"] = "all_zero_structural_metrics"
        out.loc[out["passed_status_gate"] & out["passed_nonzero_structural_gate"] & ~out["passed_summary_gate"], "invalid_reason_primary"] = "missing_summary_metrics"
        excluded = out.loc[~out["valid_structural_row"]].copy()
        valid = out.loc[out["valid_structural_row"]].copy()
        stage_summary = pd.DataFrame([
            {
                "source_kind": "below_folding_budget",
                "target_label": str(target_label),
                "target_percent": float(self.target_label_to_percent(target_label)),
                "raw_rows_loaded": int(out.shape[0]),
                "status_ok_rows": int(out["passed_status_gate"].sum()),
                "nonzero_structural_rows": int((out["passed_status_gate"] & out["passed_nonzero_structural_gate"]).sum()),
                "summary_complete_rows": int((out["passed_status_gate"] & out["passed_nonzero_structural_gate"] & out["passed_summary_gate"]).sum()),
                "valid_rows": int(valid.shape[0]),
                "excluded_rows": int(excluded.shape[0]),
                "excluded_status_not_ok": int((excluded["invalid_reason_primary"] == "status_not_ok").sum()),
                "excluded_all_zero_structural_metrics": int((excluded["invalid_reason_primary"] == "all_zero_structural_metrics").sum()),
                "excluded_missing_summary_metrics": int((excluded["invalid_reason_primary"] == "missing_summary_metrics").sum()),
            }
        ])
        return valid, excluded, stage_summary


class Figure7Analyzer:
    """Build analysis tables for Figure 7."""

    def __init__(self, config: Figure7Config) -> None:
        """Initialize analyzer."""
        self.config = config
        self.stats_engine = Figure7StatsEngine()

    def target_label_to_percent(self, target_label: str) -> float:
        """Convert a target label like 7.5 to a numeric percent."""
        return float(str(target_label))

    def metric_column(self, base_name: str) -> str:
        """Return the metric column name according to config."""
        if bool(self.config.use_recomputed_model_qc_means):
            return f"{base_name}_recomputed"
        return base_name

    def bootstrap_seed_for_summary(self, summary_label: str, target_value: str, metric_name: str) -> int:
        """Return a stable per-summary bootstrap seed."""
        seed_text = f"{int(self.config.bootstrap_seed)}|{summary_label}|{target_value}|{metric_name}"
        digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2 ** 32)

    def build_mean_delta_record(self, summary_label: str, target_label: str, target_percent: float, metric_name: str, values: np.ndarray) -> Dict[str, Any]:
        """Build one mean-delta summary record with bootstrap confidence intervals."""
        clean_values = values[np.isfinite(values)]
        seed = self.bootstrap_seed_for_summary(str(summary_label), str(target_label), str(metric_name))
        ci_low, ci_high = self.stats_engine.bootstrap_mean_ci(
            clean_values,
            ci_level=float(self.config.bootstrap_ci_level),
            iterations=int(self.config.bootstrap_iterations),
            seed=int(seed),
            min_n=int(self.config.bootstrap_min_n),
        )
        mean_delta = float(np.mean(clean_values)) if clean_values.size > 0 else float("nan")
        return {
            "summary_label": str(summary_label),
            "target_label": str(target_label),
            "target_percent": float(target_percent),
            "metric": str(metric_name),
            "n_pairs": int(clean_values.size),
            "mean_delta": mean_delta,
            "mean_delta_ci_low": float(ci_low),
            "mean_delta_ci_high": float(ci_high),
            "mean_delta_ci_level": float(self.config.bootstrap_ci_level),
            "bootstrap_iterations": int(self.config.bootstrap_iterations),
            "bootstrap_seed": int(seed),
        }

    def build_exclusion_summary_by_target(self, excluded_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize exclusions by source, target, and reason."""
        if excluded_df.shape[0] == 0:
            return pd.DataFrame()
        out = (
            excluded_df.groupby(["source_kind", "target_label", "invalid_reason_primary"], sort=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values(["source_kind", "target_label", "invalid_reason_primary"])
        )
        return out

    def build_pair_rows(self, jack_valid_df: pd.DataFrame, tuned_valid_df: pd.DataFrame) -> pd.DataFrame:
        """Build all valid pair rows for each target."""
        if jack_valid_df.shape[0] == 0 or tuned_valid_df.shape[0] == 0:
            return pd.DataFrame()
        tm_metric = self.metric_column("mean_tm_score")
        rmsd_metric = self.metric_column("mean_rmsd")
        ptm_metric = self.metric_column("mean_ptm")
        jack_core = jack_valid_df[[
            "pdb_id",
            tm_metric,
            rmsd_metric,
            ptm_metric,
            "num_sequences_in_alignment",
            "n_valid_tm_models",
            "n_valid_rmsd_models",
            "n_valid_ptm_models",
        ]].copy()
        if jack_core["pdb_id"].duplicated().any():
            raise ValueError("JackHMMER baseline contains duplicate pdb_id values after QC filtering")
        jack_core = jack_core.rename(columns={
            tm_metric: "jack_mean_tm_score_qc",
            rmsd_metric: "jack_mean_rmsd_qc",
            ptm_metric: "jack_mean_ptm_qc",
            "num_sequences_in_alignment": "jack_depth",
            "n_valid_tm_models": "jack_n_valid_tm_models",
            "n_valid_rmsd_models": "jack_n_valid_rmsd_models",
            "n_valid_ptm_models": "jack_n_valid_ptm_models",
        })
        pair_frames: List[pd.DataFrame] = []
        for target_label in self.config.analysis_target_order:
            tuned_sub = tuned_valid_df.loc[tuned_valid_df["target_label"] == str(target_label)].copy()
            if tuned_sub.shape[0] == 0:
                continue
            if tuned_sub["pdb_id"].duplicated().any():
                raise ValueError(f"Tuned target {target_label} contains duplicate pdb_id values after QC filtering")
            tuned_columns = [
                "pdb_id",
                "target_label",
                "target_percent",
                tm_metric,
                rmsd_metric,
                ptm_metric,
                "num_sequences_in_alignment",
                "meff_theta_0p70",
                "meff_fraction_theta_0p70",
                "status",
                "n_valid_tm_models",
                "n_valid_rmsd_models",
                "n_valid_ptm_models",
                "path_index",
                "direction",
                "message",
                "fraction_columns_below_folding_budget",
            ]
            tuned_core = tuned_sub[tuned_columns].copy()
            tuned_core = tuned_core.rename(columns={
                tm_metric: "tuned_mean_tm_score_qc",
                rmsd_metric: "tuned_mean_rmsd_qc",
                ptm_metric: "tuned_mean_ptm_qc",
                "num_sequences_in_alignment": "tuned_depth",
                "meff_theta_0p70": "tuned_meff_theta_0p70",
                "meff_fraction_theta_0p70": "tuned_meff_fraction_theta_0p70",
                "status": "tuned_status",
                "n_valid_tm_models": "tuned_n_valid_tm_models",
                "n_valid_rmsd_models": "tuned_n_valid_rmsd_models",
                "n_valid_ptm_models": "tuned_n_valid_ptm_models",
                "path_index": "tuned_path_index",
                "direction": "tuned_direction",
                "message": "tuned_message",
                "fraction_columns_below_folding_budget": "achieved_fraction_below_2bps",
            })
            paired = jack_core.merge(tuned_core, on="pdb_id", how="inner", validate="one_to_one")
            if paired.shape[0] == 0:
                continue
            paired["target_percentage"] = paired["target_percent"]
            paired["delta_tm_score"] = paired["tuned_mean_tm_score_qc"] - paired["jack_mean_tm_score_qc"]
            paired["delta_rmsd"] = paired["jack_mean_rmsd_qc"] - paired["tuned_mean_rmsd_qc"]
            paired["delta_ptm"] = paired["tuned_mean_ptm_qc"] - paired["jack_mean_ptm_qc"]
            paired["delta_depth_raw"] = paired["tuned_depth"] - paired["jack_depth"]
            paired["depth_fraction_of_jackhmmer"] = paired["tuned_depth"] / paired["jack_depth"].replace(0, np.nan)
            pair_frames.append(paired)
        if len(pair_frames) == 0:
            return pd.DataFrame()
        out = pd.concat(pair_frames, ignore_index=True)
        return out

    def build_common_target_matched_set_membership(self, paired_df: pd.DataFrame) -> pd.DataFrame:
        """Build the common-target matched set membership table."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        required_targets = set(str(label) for label in self.config.common_target_matched_set_target_order)
        records: List[Dict[str, Any]] = []
        for pdb_id, sub_df in paired_df.groupby("pdb_id", sort=False):
            present_targets = set(sub_df["target_label"].astype(str).tolist())
            if required_targets.issubset(present_targets):
                records.append({
                    "pdb_id": str(pdb_id),
                    "common_target_matched_set_member": True,
                    "required_target_count": int(len(required_targets)),
                    "present_target_count": int(len(present_targets)),
                })
        return pd.DataFrame(records)

    def build_common_target_matched_set_pair_rows(self, paired_df: pd.DataFrame, membership_df: pd.DataFrame) -> pd.DataFrame:
        """Restrict paired rows to common-target matched set membership."""
        if paired_df.shape[0] == 0 or membership_df.shape[0] == 0:
            return pd.DataFrame()
        out = paired_df.merge(
            membership_df[["pdb_id", "common_target_matched_set_member"]],
            on="pdb_id",
            how="inner",
            validate="many_to_one",
        )
        return out

    def build_delta_summary_by_target(self, paired_df: pd.DataFrame, summary_label: str) -> pd.DataFrame:
        """Build mean delta summaries by Target percentage."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for target_label in self.config.analysis_target_order:
            sub_df = paired_df.loc[paired_df["target_label"] == str(target_label)].copy()
            for metric_name in ["delta_tm_score", "delta_rmsd", "delta_ptm"]:
                values = pd.to_numeric(sub_df.get(metric_name), errors="coerce").to_numpy(dtype=float)
                rows.append(self.build_mean_delta_record(
                    str(summary_label),
                    str(target_label),
                    float(self.target_label_to_percent(target_label)),
                    str(metric_name),
                    values,
                ))
        return pd.DataFrame(rows)

    def build_paired_stats_by_target(self, paired_df: pd.DataFrame, summary_label: str) -> pd.DataFrame:
        """Build paired t-test and Wilcoxon stats by Target percentage."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for target_label in self.config.analysis_target_order:
            sub_df = paired_df.loc[paired_df["target_label"] == str(target_label)].copy()
            for metric_name in ["delta_tm_score", "delta_rmsd", "delta_ptm"]:
                values = pd.to_numeric(sub_df.get(metric_name), errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                rows.append({
                    "summary_label": str(summary_label),
                    "target_label": str(target_label),
                    "target_percent": float(self.target_label_to_percent(target_label)),
                    "metric": str(metric_name),
                    "n_pairs": int(values.size),
                    "mean_delta": float(np.mean(values)) if values.size > 0 else float("nan"),
                    "paired_t_p_value": float(self.stats_engine.paired_ttest(values)),
                    "paired_wilcoxon_p_value": float(self.stats_engine.paired_wilcoxon(values)),
                })
        return pd.DataFrame(rows)

    def build_primary_ptm_gated_subset(self, paired_df: pd.DataFrame) -> pd.DataFrame:
        """Return the subset passing the primary ΔpTM threshold."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        threshold = float(self.config.ptm_primary_operational_threshold)
        out = paired_df.loc[pd.to_numeric(paired_df["delta_ptm"], errors="coerce") > threshold].copy()
        return out

    def build_primary_ptm_gated_delta_summary_by_target(self, paired_df: pd.DataFrame, summary_label: str) -> pd.DataFrame:
        """Build per-target mean delta summaries after the primary ΔpTM gate."""
        gated_df = self.build_primary_ptm_gated_subset(paired_df)
        out = self.build_delta_summary_by_target(gated_df, summary_label)
        if out.shape[0] == 0:
            return out
        out["ptm_threshold"] = float(self.config.ptm_primary_operational_threshold)
        return out

    def build_primary_ptm_gated_paired_stats_by_target(self, paired_df: pd.DataFrame, summary_label: str) -> pd.DataFrame:
        """Build per-target paired stats after the primary ΔpTM gate."""
        gated_df = self.build_primary_ptm_gated_subset(paired_df)
        out = self.build_paired_stats_by_target(gated_df, summary_label)
        if out.shape[0] == 0:
            return out
        out["ptm_threshold"] = float(self.config.ptm_primary_operational_threshold)
        return out

    def build_delta_distribution_source(self, paired_df: pd.DataFrame, summary_label: str) -> pd.DataFrame:
        """Build long-form delta values for violin plots."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        for metric_name in ["delta_tm_score", "delta_rmsd", "delta_ptm"]:
            sub_df = paired_df[["pdb_id", "target_label", "target_percent", metric_name]].copy()
            sub_df[metric_name] = pd.to_numeric(sub_df[metric_name], errors="coerce")
            sub_df = sub_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
            for _, row in sub_df.iterrows():
                rows.append({
                    "summary_label": str(summary_label),
                    "pdb_id": str(row["pdb_id"]),
                    "target_label": str(row["target_label"]),
                    "target_percent": float(row["target_percent"]),
                    "metric": str(metric_name),
                    "value": float(row[metric_name]),
                })
        return pd.DataFrame(rows)

    def build_target_property_source(self, paired_df: pd.DataFrame, summary_label: str) -> pd.DataFrame:
        """Build long-form target-property values for violin plots."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        property_map = {
            "meff": "tuned_meff_theta_0p70",
            "raw_depth": "tuned_depth",
            "depth_fraction_of_jackhmmer": "depth_fraction_of_jackhmmer",
        }
        rows: List[Dict[str, Any]] = []
        for metric_name, column_name in property_map.items():
            sub_df = paired_df[["pdb_id", "target_label", "target_percent", column_name]].copy()
            sub_df[column_name] = pd.to_numeric(sub_df[column_name], errors="coerce")
            sub_df = sub_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
            for _, row in sub_df.iterrows():
                rows.append({
                    "summary_label": str(summary_label),
                    "pdb_id": str(row["pdb_id"]),
                    "target_label": str(row["target_label"]),
                    "target_percent": float(row["target_percent"]),
                    "metric": str(metric_name),
                    "value": float(row[column_name]),
                })
        return pd.DataFrame(rows)

    def build_ptm_threshold_summary(self, paired_df: pd.DataFrame, summary_label: str) -> pd.DataFrame:
        """Build descriptive mean-delta summaries across the pTM threshold grid."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        total_rows = int(paired_df.shape[0])
        for threshold in self.config.ptm_threshold_grid:
            threshold_df = paired_df.loc[pd.to_numeric(paired_df["delta_ptm"], errors="coerce") > float(threshold)].copy()
            retained_fraction = float(threshold_df.shape[0] / float(total_rows)) if total_rows > 0 else float("nan")
            for metric_name in ["delta_tm_score", "delta_rmsd", "delta_ptm"]:
                values = pd.to_numeric(threshold_df.get(metric_name), errors="coerce").to_numpy(dtype=float)
                seed = self.bootstrap_seed_for_summary(str(summary_label), f"ptm_threshold={float(threshold):.12g}", str(metric_name))
                ci_low, ci_high = self.stats_engine.bootstrap_mean_ci(
                    values,
                    ci_level=float(self.config.bootstrap_ci_level),
                    iterations=int(self.config.bootstrap_iterations),
                    seed=int(seed),
                    min_n=int(self.config.bootstrap_min_n),
                )
                clean_values = values[np.isfinite(values)]
                rows.append({
                    "summary_label": str(summary_label),
                    "ptm_threshold": float(threshold),
                    "metric": str(metric_name),
                    "n_pairs": int(clean_values.size),
                    "retained_fraction_of_pairs": float(retained_fraction),
                    "mean_delta": float(np.mean(clean_values)) if clean_values.size > 0 else float("nan"),
                    "mean_delta_ci_low": float(ci_low),
                    "mean_delta_ci_high": float(ci_high),
                    "mean_delta_ci_level": float(self.config.bootstrap_ci_level),
                    "bootstrap_iterations": int(self.config.bootstrap_iterations),
                    "bootstrap_seed": int(seed),
                })
        return pd.DataFrame(rows)

    def build_ptm_threshold_paired_stats(self, paired_df: pd.DataFrame, summary_label: str) -> pd.DataFrame:
        """Build paired stats across the pTM threshold grid."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        total_rows = int(paired_df.shape[0])
        for threshold in self.config.ptm_threshold_grid:
            threshold_df = paired_df.loc[pd.to_numeric(paired_df["delta_ptm"], errors="coerce") > float(threshold)].copy()
            retained_fraction = float(threshold_df.shape[0] / float(total_rows)) if total_rows > 0 else float("nan")
            for metric_name in ["delta_tm_score", "delta_rmsd", "delta_ptm"]:
                values = pd.to_numeric(threshold_df.get(metric_name), errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                rows.append({
                    "summary_label": str(summary_label),
                    "ptm_threshold": float(threshold),
                    "metric": str(metric_name),
                    "n_pairs": int(values.size),
                    "retained_fraction_of_pairs": float(retained_fraction),
                    "mean_delta": float(np.mean(values)) if values.size > 0 else float("nan"),
                    "paired_t_p_value": float(self.stats_engine.paired_ttest(values)),
                    "paired_wilcoxon_p_value": float(self.stats_engine.paired_wilcoxon(values)),
                })
        return pd.DataFrame(rows)

    def build_scatter_source(self, paired_df: pd.DataFrame) -> pd.DataFrame:
        """Build descriptive scatter source rows."""
        if paired_df.shape[0] == 0:
            return pd.DataFrame()
        columns = ["pdb_id", "target_label", "target_percent", "delta_ptm", "delta_tm_score", "delta_rmsd"]
        out = paired_df[columns].copy()
        for column in ["delta_ptm", "delta_tm_score", "delta_rmsd", "target_percent"]:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        return out


class Figure7Plotter:
    """Render Figure 7 panels and save figures."""

    def __init__(self, config: Figure7Config, paths: Figure7Paths, style: Figure7PlotStyle) -> None:
        """Initialize plotter."""
        self.config = config
        self.paths = paths
        self.style = style

    def apply_publication_style(self) -> None:
        """Apply clean publication style defaults."""
        sns.set_style("white")
        mpl.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.0,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

    def apply_axis_style(self, ax: plt.Axes) -> None:
        """Apply per-axis styling."""
        ax.grid(False)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=True,
            top=False,
            left=True,
            right=False,
            labelbottom=True,
            labelleft=True,
        )
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        sns.despine(ax=ax, top=True, right=True)
    def save_figure(self, fig: plt.Figure, stem: str) -> None:
        """Save one figure as SVG and PNG."""
        fig.savefig(self.paths.output_dir / f"{stem}.svg", dpi=300)
        fig.savefig(self.paths.output_dir / f"{stem}.png", dpi=300)
        plt.close(fig)

    def ordered_target_labels(self) -> List[str]:
        """Return the configured Target percentage label order."""
        return list(self.config.analysis_target_order)

    def plot_delta_summary(self, summary_df: pd.DataFrame, color_key: str, summary_label: str, metric: str, stem: str, ylabel: str) -> None:
        """Plot mean delta by Target percentage."""
        sub_df = summary_df.loc[(summary_df["summary_label"] == str(summary_label)) & (summary_df["metric"] == str(metric))].sort_values("target_percent")
        if sub_df.shape[0] == 0:
            return
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        x_values = sub_df["target_percent"].to_numpy(dtype=float)
        y_values = sub_df["mean_delta"].to_numpy(dtype=float)
        color = self.style.get_color(color_key)
        ax.axhline(0.0, color=self.style.get_color("neutral_dark"), linewidth=0.8)
        if {"mean_delta_ci_low", "mean_delta_ci_high"}.issubset(set(sub_df.columns)):
            low_values = sub_df["mean_delta_ci_low"].to_numpy(dtype=float)
            high_values = sub_df["mean_delta_ci_high"].to_numpy(dtype=float)
            valid_mask = np.isfinite(y_values) & np.isfinite(low_values) & np.isfinite(high_values)
            if np.any(valid_mask):
                ax.vlines(
                    x_values[valid_mask],
                    low_values[valid_mask],
                    high_values[valid_mask],
                    color=color,
                    linewidth=0.9,
                    alpha=0.85,
                )
                ax.plot(x_values, y_values, marker="o", color=color)
            else:
                ax.plot(x_values, y_values, marker="o", color=color)
        else:
            ax.plot(x_values, y_values, marker="o", color=color)
        ax.set_xlabel("Target percentage")
        ax.set_ylabel(ylabel)
        self.apply_axis_style(ax)
        fig.tight_layout()
        self.save_figure(fig, stem)

    def plot_delta_distribution(self, distribution_df: pd.DataFrame, color_key: str, summary_label: str, metric: str, stem: str, ylabel: str) -> None:
        """Plot delta violin distributions by Target percentage."""
        sub_df = distribution_df.loc[(distribution_df["summary_label"] == str(summary_label)) & (distribution_df["metric"] == str(metric))].copy()
        if sub_df.shape[0] == 0:
            return
        order = self.ordered_target_labels()
        sub_df["target_label"] = pd.Categorical(sub_df["target_label"].astype(str), categories=order, ordered=True)
        sub_df = sub_df.sort_values("target_label")
        fig, ax = plt.subplots(figsize=(7.8, 3.8))
        sns.violinplot(
            ax=ax,
            data=sub_df,
            x="target_label",
            y="value",
            color=self.style.get_color(color_key),
            cut=0,
            inner=None,
            linewidth=0.9,
        )
        ax.axhline(0.0, color=self.style.get_color("neutral_dark"), linewidth=0.8)
        ax.set_xlabel("Target percentage")
        ax.set_ylabel(ylabel)
        self.apply_axis_style(ax)
        fig.tight_layout()
        self.save_figure(fig, stem)

    def plot_target_property_violin(self, property_df: pd.DataFrame, color_key: str, summary_label: str, metric: str, stem: str, ylabel: str) -> None:
        """Plot target-property violin distributions by Target percentage."""
        sub_df = property_df.loc[(property_df["summary_label"] == str(summary_label)) & (property_df["metric"] == str(metric))].copy()
        if sub_df.shape[0] == 0:
            return
        order = self.ordered_target_labels()
        sub_df["target_label"] = pd.Categorical(sub_df["target_label"].astype(str), categories=order, ordered=True)
        sub_df = sub_df.sort_values("target_label")
        fig, ax = plt.subplots(figsize=(7.8, 3.8))
        sns.violinplot(
            ax=ax,
            data=sub_df,
            x="target_label",
            y="value",
            color=self.style.get_color(color_key),
            cut=0,
            inner=None,
            linewidth=0.9,
        )
        ax.set_xlabel("Target percentage")
        ax.set_ylabel(ylabel)
        self.apply_axis_style(ax)
        fig.tight_layout()
        self.save_figure(fig, stem)

    def plot_ptm_threshold_summary(self, threshold_df: pd.DataFrame, color_key: str, summary_label: str, metric: str, stem: str, ylabel: str) -> None:
        """Plot mean delta across the pTM threshold grid."""
        sub_df = threshold_df.loc[(threshold_df["summary_label"] == str(summary_label)) & (threshold_df["metric"] == str(metric))].sort_values("ptm_threshold")
        if sub_df.shape[0] == 0:
            return
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        x_values = sub_df["ptm_threshold"].to_numpy(dtype=float)
        y_values = sub_df["mean_delta"].to_numpy(dtype=float)
        color = self.style.get_color(color_key)
        ax.axhline(0.0, color=self.style.get_color("neutral_dark"), linewidth=0.8)
        if {"mean_delta_ci_low", "mean_delta_ci_high"}.issubset(set(sub_df.columns)):
            low_values = sub_df["mean_delta_ci_low"].to_numpy(dtype=float)
            high_values = sub_df["mean_delta_ci_high"].to_numpy(dtype=float)
            valid_mask = np.isfinite(y_values) & np.isfinite(low_values) & np.isfinite(high_values)
            if np.any(valid_mask):
                ax.vlines(
                    x_values[valid_mask],
                    low_values[valid_mask],
                    high_values[valid_mask],
                    color=color,
                    linewidth=0.9,
                    alpha=0.85,
                )
                ax.plot(x_values, y_values, marker="o", color=color)
            else:
                ax.plot(x_values, y_values, marker="o", color=color)
        else:
            ax.plot(x_values, y_values, marker="o", color=color)
        ax.set_xlabel("ΔpTM gate threshold")
        ax.set_ylabel(ylabel)
        self.apply_axis_style(ax)
        fig.tight_layout()
        self.save_figure(fig, stem)

    def plot_scatter(self, source_df: pd.DataFrame, x_column: str, y_column: str, stem: str, xlabel: str, ylabel: str) -> None:
        """Plot a descriptive scatter plot."""
        if source_df.shape[0] == 0:
            return
        fig, ax = plt.subplots(figsize=(5.0, 3.8))
        ax.scatter(
            source_df[x_column].to_numpy(dtype=float),
            source_df[y_column].to_numpy(dtype=float),
            s=16,
            color=self.style.get_color("reference"),
            alpha=0.55,
            edgecolors="none",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        self.apply_axis_style(ax)
        fig.tight_layout()
        self.save_figure(fig, stem)


class Figure7Runner:
    """Coordinate data loading, analysis, plotting, and XLSX export."""

    def __init__(self, config: Figure7Config) -> None:
        """Initialize runner."""
        self.config = config
        self.paths = self.build_paths(config)
        self.style = Figure7PlotStyle()
        self.data_loader = Figure7DataLoader(config)
        self.analyzer = Figure7Analyzer(config)
        self.plotter = Figure7Plotter(config, self.paths, self.style)

    def build_paths(self, config: Figure7Config) -> Figure7Paths:
        """Resolve paths relative to the working directory."""
        working_dir = Path(".").resolve()
        data_dir = working_dir / str(config.data_dir_name)
        output_dir = working_dir / str(config.output_dir_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_xlsx_path = output_dir / str(config.output_xlsx_name)
        return Figure7Paths(
            working_dir=working_dir,
            data_dir=data_dir,
            output_dir=output_dir,
            output_xlsx_path=output_xlsx_path,
        )

    def validate_config(self) -> None:
        """Validate config settings for Figure 7."""
        if not self.paths.data_dir.exists():
            raise FileNotFoundError(f"Missing data directory: {self.paths.data_dir}")
        if int(self.config.excel_chunk_rows) <= 0:
            raise ValueError("excel_chunk_rows must be positive")
        if int(self.config.bootstrap_iterations) <= 0:
            raise ValueError("bootstrap_iterations must be positive")
        if float(self.config.bootstrap_ci_level) <= 0.0 or float(self.config.bootstrap_ci_level) >= 1.0:
            raise ValueError("bootstrap_ci_level must be between 0 and 1")
        if int(self.config.bootstrap_min_n) < 2:
            raise ValueError("bootstrap_min_n must be at least 2")
        self.data_loader.validate_target_file_map(self.paths)

    def build_readme_table(self) -> pd.DataFrame:
        """Create a README table describing outputs and assumptions."""
        rows: List[Dict[str, Any]] = []
        rows.append({"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        rows.append({"key": "inputs", "value": "folding_results/chai1_structural_comparison_jackhmmer.csv + explicit below_folding_budget target mapping"})
        rows.append({"key": "figure_outputs", "value": "Figure 7a-f and Supplementary Figure 5a-b only; Figure 7g and Supplementary Figure 5c-d are generated by separate scripts."})
        rows.append({"key": "target_percentage_labels", "value": json.dumps(list(self.config.analysis_target_order))})
        rows.append({"key": "target_file_map", "value": json.dumps({key: value for key, value in self.config.target_file_map})})
        rows.append({"key": "common_target_matched_set_labels", "value": json.dumps(list(self.config.common_target_matched_set_target_order))})
        rows.append({"key": "summary_metric_source", "value": "recomputed per-row means from valid model-level outputs" if bool(self.config.use_recomputed_model_qc_means) else "file-provided summary means"})
        rows.append({"key": "benefit_directionality", "value": "positive means tuned MSA is better: delta_tm_score=tuned-jack, delta_rmsd=jack-tuned, delta_ptm=tuned-jack"})
        rows.append({"key": "paired_tests", "value": json.dumps(["paired_t", "paired_wilcoxon"])})
        rows.append({"key": "ptm_positive_threshold", "value": str(self.config.ptm_positive_threshold)})
        rows.append({"key": "ptm_primary_operational_threshold", "value": str(self.config.ptm_primary_operational_threshold)})
        rows.append({"key": "ptm_threshold_grid", "value": json.dumps(list(self.config.ptm_threshold_grid))})
        rows.append({"key": "bootstrap_ci", "value": json.dumps({"level": float(self.config.bootstrap_ci_level), "iterations": int(self.config.bootstrap_iterations), "minimum_n": int(self.config.bootstrap_min_n), "base_seed": int(self.config.bootstrap_seed)})})
        rows.append({"key": "color_schema", "value": json.dumps(self.style.color_map)})
        rows.append({"key": "color_semantics", "value": "delta and target-property plots use metric-specific colors"})
        return pd.DataFrame(rows)

    def build_notebook_tables(
        self,
        input_inventory_df: pd.DataFrame,
        raw_counts_df: pd.DataFrame,
        stage_summary_df: pd.DataFrame,
        jack_valid_df: pd.DataFrame,
        tuned_valid_all_df: pd.DataFrame,
        excluded_all_df: pd.DataFrame,
        exclusion_summary_df: pd.DataFrame,
        paired_all_df: pd.DataFrame,
        membership_df: pd.DataFrame,
        paired_common_df: pd.DataFrame,
        delta_summary_all_df: pd.DataFrame,
        delta_summary_common_df: pd.DataFrame,
        paired_stats_all_df: pd.DataFrame,
        paired_stats_common_df: pd.DataFrame,
        delta_summary_ptm_gated_all_df: pd.DataFrame,
        delta_summary_ptm_gated_common_df: pd.DataFrame,
        paired_stats_ptm_gated_all_df: pd.DataFrame,
        paired_stats_ptm_gated_common_df: pd.DataFrame,
        delta_distribution_all_df: pd.DataFrame,
        delta_distribution_common_df: pd.DataFrame,
        target_property_all_df: pd.DataFrame,
        target_property_common_df: pd.DataFrame,
        ptm_threshold_all_df: pd.DataFrame,
        ptm_threshold_common_df: pd.DataFrame,
        ptm_threshold_stats_all_df: pd.DataFrame,
        ptm_threshold_stats_common_df: pd.DataFrame,
        scatter_source_df: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Build the exported workbook table dictionary."""
        figure7a_source_df = delta_summary_common_df[
            (delta_summary_common_df["summary_label"] == "common_target_matched_set")
            & (delta_summary_common_df["metric"] == "delta_tm_score")
        ].copy()
        figure7b_source_df = delta_summary_common_df[
            (delta_summary_common_df["summary_label"] == "common_target_matched_set")
            & (delta_summary_common_df["metric"] == "delta_rmsd")
        ].copy()
        figure7c_source_df = ptm_threshold_common_df[
            (ptm_threshold_common_df["summary_label"] == "common_target_matched_set")
            & (ptm_threshold_common_df["metric"] == "delta_tm_score")
        ].copy()
        figure7d_source_df = ptm_threshold_common_df[
            (ptm_threshold_common_df["summary_label"] == "common_target_matched_set")
            & (ptm_threshold_common_df["metric"] == "delta_rmsd")
        ].copy()
        figure7e_source_df = delta_summary_ptm_gated_common_df[
            (delta_summary_ptm_gated_common_df["summary_label"] == "common_target_matched_set_ptm_gated")
            & (delta_summary_ptm_gated_common_df["metric"] == "delta_tm_score")
        ].copy()
        figure7f_source_df = delta_summary_ptm_gated_common_df[
            (delta_summary_ptm_gated_common_df["summary_label"] == "common_target_matched_set_ptm_gated")
            & (delta_summary_ptm_gated_common_df["metric"] == "delta_rmsd")
        ].copy()
        suppfig5a_source_df = target_property_common_df[
            (target_property_common_df["summary_label"] == "common_target_matched_set")
            & (target_property_common_df["metric"] == "raw_depth")
        ].copy()
        suppfig5b_source_df = target_property_common_df[
            (target_property_common_df["summary_label"] == "common_target_matched_set")
            & (target_property_common_df["metric"] == "meff")
        ].copy()
        return {
            "readme": self.build_readme_table(),
            "input_inventory": input_inventory_df,
            "raw_counts": raw_counts_df,
            "stage_summary": stage_summary_df,
            "jack_valid": jack_valid_df,
            "tuned_valid_all": tuned_valid_all_df,
            "excluded_all": excluded_all_df,
            "exclude_summary": exclusion_summary_df,
            "common_target_matched_members": membership_df,
            "paired_rows_common_target": paired_common_df,
            "delta_summary_common_target": delta_summary_common_df,
            "paired_stats_common_target": paired_stats_common_df,
            "delta_summary_ptm_gated_common": delta_summary_ptm_gated_common_df,
            "paired_stats_ptm_gated_common": paired_stats_ptm_gated_common_df,
            "target_props_common_target": target_property_common_df,
            "ptm_threshold_common_target": ptm_threshold_common_df,
            "ptm_threshold_stats_common": ptm_threshold_stats_common_df,
            "figure7a_source": figure7a_source_df,
            "figure7b_source": figure7b_source_df,
            "figure7c_source": figure7c_source_df,
            "figure7d_source": figure7d_source_df,
            "figure7e_source": figure7e_source_df,
            "figure7f_source": figure7f_source_df,
            "suppfig5a_source": suppfig5a_source_df,
            "suppfig5b_source": suppfig5b_source_df,
        }

    def render_plots(
        self,
        delta_summary_common_df: pd.DataFrame,
        delta_summary_ptm_gated_common_df: pd.DataFrame,
        target_property_common_df: pd.DataFrame,
        ptm_threshold_common_df: pd.DataFrame,
    ) -> None:
        """Render Figure 7a-f and Supplementary Figure 5a-b."""
        self.plotter.plot_delta_summary(
            delta_summary_common_df,
            "delta_tm_score",
            "common_target_matched_set",
            "delta_tm_score",
            "figure7a_delta_tm_score_common_target_matched_set",
            "ΔTM-score",
        )
        self.plotter.plot_delta_summary(
            delta_summary_common_df,
            "delta_rmsd",
            "common_target_matched_set",
            "delta_rmsd",
            "figure7b_delta_rmsd_common_target_matched_set",
            "ΔRMSD",
        )
        self.plotter.plot_ptm_threshold_summary(
            ptm_threshold_common_df,
            "delta_tm_score",
            "common_target_matched_set",
            "delta_tm_score",
            "figure7c_ptm_threshold_effect_common_target_matched_set_delta_tm_score",
            "ΔTM-score",
        )
        self.plotter.plot_ptm_threshold_summary(
            ptm_threshold_common_df,
            "delta_rmsd",
            "common_target_matched_set",
            "delta_rmsd",
            "figure7d_ptm_threshold_effect_common_target_matched_set_delta_rmsd",
            "ΔRMSD",
        )
        self.plotter.plot_delta_summary(
            delta_summary_ptm_gated_common_df,
            "delta_tm_score",
            "common_target_matched_set_ptm_gated",
            "delta_tm_score",
            "figure7e_delta_tm_score_ptm_gated_common_target_matched_set",
            "ΔTM-score",
        )
        self.plotter.plot_delta_summary(
            delta_summary_ptm_gated_common_df,
            "delta_rmsd",
            "common_target_matched_set_ptm_gated",
            "delta_rmsd",
            "figure7f_delta_rmsd_ptm_gated_common_target_matched_set",
            "ΔRMSD",
        )
        self.plotter.plot_target_property_violin(
            target_property_common_df,
            "raw_depth",
            "common_target_matched_set",
            "raw_depth",
            "suppfig5a_raw_depth_by_target_percentage_common_target_matched_set",
            "Raw MSA depth",
        )
        self.plotter.plot_target_property_violin(
            target_property_common_df,
            "meff",
            "common_target_matched_set",
            "meff",
            "suppfig5b_meff_by_target_percentage_common_target_matched_set",
            "Meff",
        )

    def run(self) -> None:
        """Run the full Figure 7 pipeline."""
        self.validate_config()
        self.plotter.apply_publication_style()
        input_inventory_df = self.data_loader.build_input_inventory(self.paths)
        raw_count_rows: List[Dict[str, Any]] = []
        stage_summary_frames: List[pd.DataFrame] = []
        excluded_frames: List[pd.DataFrame] = []
        tuned_valid_frames: List[pd.DataFrame] = []
        start_time = time.time()
        jack_path = self.paths.data_dir / self.config.jackhmmer_filename
        jack_raw_df = self.data_loader.load_csv(jack_path)
        raw_count_rows.append(self.data_loader.build_raw_count_record("jackhmmer", "jackhmmer", jack_raw_df))
        jack_valid_df, jack_excluded_df, jack_stage_summary_df = self.data_loader.normalize_jackhmmer(jack_raw_df)
        stage_summary_frames.append(jack_stage_summary_df)
        if jack_excluded_df.shape[0] > 0:
            excluded_frames.append(jack_excluded_df)
        for target_label in self.config.analysis_target_order:
            file_name = self.data_loader.target_file_map_dict()[target_label]
            tuned_path = self.paths.data_dir / file_name
            tuned_raw_df = self.data_loader.load_csv(tuned_path)
            raw_count_rows.append(self.data_loader.build_raw_count_record("below_folding_budget", str(target_label), tuned_raw_df))
            tuned_valid_df, tuned_excluded_df, tuned_stage_summary_df = self.data_loader.normalize_tuned(tuned_raw_df, target_label=str(target_label))
            tuned_valid_frames.append(tuned_valid_df)
            stage_summary_frames.append(tuned_stage_summary_df)
            if tuned_excluded_df.shape[0] > 0:
                excluded_frames.append(tuned_excluded_df)
        tuned_valid_all_df = pd.concat(tuned_valid_frames, ignore_index=True) if len(tuned_valid_frames) > 0 else pd.DataFrame()
        excluded_all_df = pd.concat(excluded_frames, ignore_index=True) if len(excluded_frames) > 0 else pd.DataFrame()
        raw_counts_df = pd.DataFrame(raw_count_rows)
        stage_summary_df = pd.concat(stage_summary_frames, ignore_index=True) if len(stage_summary_frames) > 0 else pd.DataFrame()
        exclusion_summary_df = self.analyzer.build_exclusion_summary_by_target(excluded_all_df)
        paired_all_df = self.analyzer.build_pair_rows(jack_valid_df, tuned_valid_all_df)
        membership_df = self.analyzer.build_common_target_matched_set_membership(paired_all_df)
        paired_common_df = self.analyzer.build_common_target_matched_set_pair_rows(paired_all_df, membership_df)
        delta_summary_all_df = self.analyzer.build_delta_summary_by_target(paired_all_df, "all_pairs")
        delta_summary_common_df = self.analyzer.build_delta_summary_by_target(paired_common_df, "common_target_matched_set")
        paired_stats_all_df = self.analyzer.build_paired_stats_by_target(paired_all_df, "all_pairs")
        paired_stats_common_df = self.analyzer.build_paired_stats_by_target(paired_common_df, "common_target_matched_set")
        delta_summary_ptm_gated_all_df = self.analyzer.build_primary_ptm_gated_delta_summary_by_target(paired_all_df, "all_pairs_ptm_gated")
        delta_summary_ptm_gated_common_df = self.analyzer.build_primary_ptm_gated_delta_summary_by_target(paired_common_df, "common_target_matched_set_ptm_gated")
        paired_stats_ptm_gated_all_df = self.analyzer.build_primary_ptm_gated_paired_stats_by_target(paired_all_df, "all_pairs_ptm_gated")
        paired_stats_ptm_gated_common_df = self.analyzer.build_primary_ptm_gated_paired_stats_by_target(paired_common_df, "common_target_matched_set_ptm_gated")
        delta_distribution_all_df = self.analyzer.build_delta_distribution_source(paired_all_df, "all_pairs")
        delta_distribution_common_df = self.analyzer.build_delta_distribution_source(paired_common_df, "common_target_matched_set")
        target_property_all_df = self.analyzer.build_target_property_source(paired_all_df, "all_pairs")
        target_property_common_df = self.analyzer.build_target_property_source(paired_common_df, "common_target_matched_set")
        ptm_threshold_all_df = self.analyzer.build_ptm_threshold_summary(paired_all_df, "all_pairs")
        ptm_threshold_common_df = self.analyzer.build_ptm_threshold_summary(paired_common_df, "common_target_matched_set")
        ptm_threshold_stats_all_df = self.analyzer.build_ptm_threshold_paired_stats(paired_all_df, "all_pairs")
        ptm_threshold_stats_common_df = self.analyzer.build_ptm_threshold_paired_stats(paired_common_df, "common_target_matched_set")
        scatter_source_df = self.analyzer.build_scatter_source(paired_all_df)
        self.render_plots(
            delta_summary_common_df,
            delta_summary_ptm_gated_common_df,
            target_property_common_df,
            ptm_threshold_common_df,
        )
        notebook_tables = self.build_notebook_tables(
            input_inventory_df,
            raw_counts_df,
            stage_summary_df,
            jack_valid_df,
            tuned_valid_all_df,
            excluded_all_df,
            exclusion_summary_df,
            paired_all_df,
            membership_df,
            paired_common_df,
            delta_summary_all_df,
            delta_summary_common_df,
            paired_stats_all_df,
            paired_stats_common_df,
            delta_summary_ptm_gated_all_df,
            delta_summary_ptm_gated_common_df,
            paired_stats_ptm_gated_all_df,
            paired_stats_ptm_gated_common_df,
            delta_distribution_all_df,
            delta_distribution_common_df,
            target_property_all_df,
            target_property_common_df,
            ptm_threshold_all_df,
            ptm_threshold_common_df,
            ptm_threshold_stats_all_df,
            ptm_threshold_stats_common_df,
            scatter_source_df,
        )
        writer = ExcelNotebookWriter(self.paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows))
        writer.write(notebook_tables)
        elapsed = float(time.time() - start_time)
        print(f"Figure 7 build complete in {elapsed:.1f}s")


def main() -> None:
    """Generate Figure 7 plots and a companion XLSX notebook."""
    config = Figure7Config()
    runner = Figure7Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
