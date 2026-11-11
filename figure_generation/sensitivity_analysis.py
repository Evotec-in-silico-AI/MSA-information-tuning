import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from figure_common import ExcelNotebookWriter
from figure_common import operational_bits_site_threshold
from figure_common import add_threshold_condition_columns
from figure_common import convert_archived_columns
from figure_common import benjamini_hochberg
from figure_common import require_columns
from figure_common import safe_mean
from figure_common import safe_sample_sd


@dataclass(frozen=True)
class PairedSensitivityConfig:
    """Configure paired sensitivity analysis."""

    cohort_xlsx_name: str = "cohort.xlsx"
    cohort_sheet_name: str = "by_sample"
    output_dir_name: str = "sensitivity_analyses"
    output_xlsx_name: str = "paired_sensitivity_analysis.xlsx"
    bits_site_min_threshold: float = operational_bits_site_threshold
    expected_overlap_protein_count: int = 20
    excel_chunk_rows: int = 1000000


@dataclass(frozen=True)
class PairedSensitivityPaths:
    """Resolve paired sensitivity paths."""

    root_dir: Path
    cohort_xlsx_path: Path
    output_dir: Path
    output_xlsx_path: Path

    @classmethod
    def from_config(cls, config: PairedSensitivityConfig) -> "PairedSensitivityPaths":
        """Build paths from the current directory."""
        root_dir = Path(".").resolve()
        output_dir = root_dir / str(config.output_dir_name)
        return cls(root_dir, root_dir / str(config.cohort_xlsx_name), output_dir, output_dir / str(config.output_xlsx_name))

    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SensitivityMetric:
    """Describe one sensitivity-analysis metric."""

    figure: str
    panel: str
    column: str
    label: str


@dataclass
class PairedSensitivityAnalyzer:
    """Compute paired-overlap sensitivity analyses."""

    config: PairedSensitivityConfig

    def get_figure3_metrics(self) -> List[SensitivityMetric]:
        """Return Figure 3 metrics in panel order."""
        return [
            SensitivityMetric("figure3", "figure3c", "msa_effective_alphabet_mean", "Effective alphabet size (mean)"),
            SensitivityMetric("figure3", "figure3d", "meff_theta0p70", "Meff (θ=0.70)"),
            SensitivityMetric("figure3", "figure3e", "mi_apc_topq_mean", "MI-APC (top-q mean)"),
            SensitivityMetric("figure3", "figure3f", "msa_graph_laplacian_lambda2", "MSA graph connectivity (λ2)"),
            SensitivityMetric("figure3", "figure3g", "msa_graph_lcc_fraction", "LCC fraction"),
            SensitivityMetric("figure3", "figure3h", "msa_graph_edge_density", "MSA graph edge density"),
            SensitivityMetric("figure3", "figure3i", "msa_graph_apl_lcc", "Avg path length (LCC)"),
            SensitivityMetric("figure3", "figure3j", "msa_graph_bridge_fraction", "Bridge fraction"),
            SensitivityMetric("figure3", "figure3k", "msa_graph_articulation_fraction", "Articulation fraction"),
        ]

    def get_figure4_metrics(self) -> List[SensitivityMetric]:
        """Return Figure 4 metrics in panel order."""
        return [
            SensitivityMetric("figure4", "figure4c", "lat_knn_radius_k5_median", "Local kNN radius (k=5, median)"),
            SensitivityMetric("figure4", "figure4d", "lat_knn_clustering_mean", "kNN graph clustering"),
            SensitivityMetric("figure4", "figure4e", "lat_knn_laplacian_lambda2", "kNN graph connectivity (λ2)"),
            SensitivityMetric("figure4", "figure4f", "lat_hubness_std", "Hubness (std)"),
            SensitivityMetric("figure4", "figure4g", "lat_knn_degree_gini", "kNN degree inequality (Gini)"),
            SensitivityMetric("figure4", "figure4h", "dist_to_anchor", "Distance to anchor"),
            SensitivityMetric("figure4", "figure4i", "lat_sphericity_pr_norm", "Sphericity (PR/d)"),
            SensitivityMetric("figure4", "figure4j", "lat_effective_rank", "Effective rank"),
        ]

    def get_metrics(self) -> List[SensitivityMetric]:
        """Return all metrics."""
        return self.get_figure3_metrics() + self.get_figure4_metrics()

    def read_cohort_by_sample(self, paths: PairedSensitivityPaths) -> pd.DataFrame:
        """Read and validate the cohort workbook."""
        if not paths.cohort_xlsx_path.exists():
            raise FileNotFoundError(f"Missing cohort workbook: {paths.cohort_xlsx_path}")
        cohort = pd.read_excel(paths.cohort_xlsx_path, sheet_name=str(self.config.cohort_sheet_name), engine="openpyxl")
        required_columns = ["sample_id", "protein_name", "outer_label", "control_type", "bits_site_min"]
        required_columns.extend([metric.column for metric in self.get_metrics()])
        require_columns(cohort, list(dict.fromkeys(required_columns)), "cohort workbook")
        out = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type"]:
            out[column_name] = out[column_name].astype(str)
        for column_name in ["bits_site_min"] + [metric.column for metric in self.get_metrics()]:
            out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
        return convert_archived_columns(out, information_columns=("bits_site_min",), effective_alphabet_columns=("msa_effective_alphabet_mean",))

    def compute_protein_condition_means(self, cohort: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
        """Compute protein-level means by condition."""
        required_columns = ["protein_name", "condition4"] + list(metric_columns)
        require_columns(cohort, required_columns, "condition-labeled cohort")
        base = cohort[required_columns].copy()
        for column_name in metric_columns:
            base[column_name] = pd.to_numeric(base[column_name], errors="coerce")
        return base.groupby(["protein_name", "condition4"], sort=False)[list(metric_columns)].mean(numeric_only=True).reset_index()

    def find_overlap_proteins(self, protein_summary: pd.DataFrame) -> List[str]:
        """Return proteins represented in both targeted regimes."""
        targeted = protein_summary[protein_summary["condition4"].isin(["below targeted", "above targeted"])].copy()
        condition_counts = targeted.groupby("protein_name")["condition4"].nunique()
        overlap_proteins = sorted(condition_counts[condition_counts == 2].index.astype(str).tolist())
        if not overlap_proteins:
            raise ValueError("No proteins were represented in both targeted regimes")
        expected_count = int(self.config.expected_overlap_protein_count)
        if expected_count > 0 and len(overlap_proteins) != expected_count:
            raise ValueError(f"Expected {expected_count} paired-overlap proteins, found {len(overlap_proteins)}")
        return overlap_proteins

    def build_paired_input_table(self, protein_summary: pd.DataFrame, overlap_proteins: Sequence[str], metric_columns: Sequence[str]) -> pd.DataFrame:
        """Build paired-overlap protein means."""
        paired = protein_summary[
            protein_summary["protein_name"].isin(list(overlap_proteins))
            & protein_summary["condition4"].isin(["below targeted", "above targeted"])
        ].copy()
        paired = paired[["protein_name", "condition4"] + list(metric_columns)].sort_values(["protein_name", "condition4"])
        paired.insert(0, "threshold_bits_site_min", float(self.config.bits_site_min_threshold))
        return paired.reset_index(drop=True)

    def compute_paired_tests(self, paired_input: pd.DataFrame, metrics: Sequence[SensitivityMetric]) -> pd.DataFrame:
        """Compute paired t-tests for each metric."""
        rows: List[Dict[str, Any]] = []
        for metric in metrics:
            table = paired_input.pivot(index="protein_name", columns="condition4", values=metric.column)
            if "below targeted" not in table.columns or "above targeted" not in table.columns:
                raise ValueError(f"Missing targeted conditions for metric {metric.column}")
            paired_values = table[["below targeted", "above targeted"]].apply(pd.to_numeric, errors="coerce").dropna()
            below_values = paired_values["below targeted"].to_numpy(dtype=float)
            above_values = paired_values["above targeted"].to_numpy(dtype=float)
            result = self.compute_single_paired_test(above_values, below_values)
            rows.append(
                {
                    "threshold_bits_site_min": float(self.config.bits_site_min_threshold),
                    "figure": metric.figure,
                    "panel": metric.panel,
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "n_paired_proteins": int(paired_values.shape[0]),
                    "mean_below": safe_mean(below_values),
                    "sd_below": safe_sample_sd(below_values),
                    "mean_above": safe_mean(above_values),
                    "sd_above": safe_sample_sd(above_values),
                    "mean_below_minus_above": safe_mean(below_values - above_values),
                    "mean_above_minus_below": safe_mean(above_values - below_values),
                    "median_below_minus_above": self.safe_median(below_values - above_values),
                    "sd_below_minus_above": safe_sample_sd(below_values - above_values),
                    "paired_t_stat_above_minus_below": result["t_statistic"],
                    "paired_t_p_value_two_sided": result["p_value"],
                    "test": "paired_t_two_sided",
                    "analysis_unit": "paired_protein_level_condition_mean",
                    "status": result["status"],
                }
            )
        output = pd.DataFrame(rows)
        output = self.add_q_values(output)
        output["direction"] = np.where(output["mean_below_minus_above"] > 0.0, "below_higher", np.where(output["mean_below_minus_above"] < 0.0, "above_higher", "no_difference"))
        return output

    def add_q_values(self, paired_tests: pd.DataFrame) -> pd.DataFrame:
        """Add BH q-values and correction metadata."""
        output = paired_tests.copy()
        output["q_value_all_metrics"] = benjamini_hochberg(output["paired_t_p_value_two_sided"].to_numpy(dtype=float))
        output["q_value_all_metrics_correction_family"] = "figure3_figure4_paired_overlap_metrics"
        output["n_tests_all_metrics"] = int(output.shape[0])
        output["q_value_within_figure"] = np.nan
        output["q_value_within_figure_correction_family"] = ""
        output["n_tests_within_figure"] = np.nan
        for figure_name in sorted(output["figure"].dropna().unique().tolist()):
            mask = output["figure"] == figure_name
            output.loc[mask, "q_value_within_figure"] = benjamini_hochberg(output.loc[mask, "paired_t_p_value_two_sided"].to_numpy(dtype=float))
            output.loc[mask, "q_value_within_figure_correction_family"] = f"{figure_name}_paired_overlap_metrics"
            output.loc[mask, "n_tests_within_figure"] = int(mask.sum())
        return output

    def compute_single_paired_test(self, above_values: np.ndarray, below_values: np.ndarray) -> Dict[str, Any]:
        """Compute one paired t-test."""
        if above_values.size != below_values.size:
            raise ValueError("Paired arrays must have equal length")
        if above_values.size < 2:
            return {"t_statistic": float("nan"), "p_value": float("nan"), "status": "insufficient_pairs"}
        differences = above_values - below_values
        finite_differences = differences[np.isfinite(differences)]
        if finite_differences.size < 2:
            return {"t_statistic": float("nan"), "p_value": float("nan"), "status": "insufficient_finite_pairs"}
        if np.allclose(finite_differences, 0.0, atol=0.0, rtol=0.0):
            return {"t_statistic": 0.0, "p_value": 1.0, "status": "all_differences_zero"}
        test_result = stats.ttest_rel(above_values, below_values, nan_policy="omit")
        return {"t_statistic": float(test_result.statistic), "p_value": float(test_result.pvalue), "status": "ok"}

    def compute_correlation_outputs(self, paired_input: pd.DataFrame, metrics: Sequence[SensitivityMetric]) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """Compute correlation matrices and correlation deltas."""
        matrix_tables: Dict[str, pd.DataFrame] = {}
        summary_rows: List[Dict[str, Any]] = []
        pair_rows: List[Dict[str, Any]] = []
        for figure_name in ["figure3", "figure4"]:
            figure_metrics = [metric for metric in metrics if metric.figure == figure_name]
            metric_columns = [metric.column for metric in figure_metrics]
            metric_labels = [metric.label for metric in figure_metrics]
            correlation_by_tag: Dict[str, pd.DataFrame] = {}
            for condition_name, condition_tag in [("below targeted", "below"), ("above targeted", "above")]:
                subset = paired_input[paired_input["condition4"] == condition_name].copy()
                for column_name in metric_columns:
                    subset[column_name] = pd.to_numeric(subset[column_name], errors="coerce")
                correlation = subset[metric_columns].corr(method="spearman", min_periods=2)
                correlation.index = metric_labels
                correlation.columns = metric_labels
                correlation_by_tag[condition_tag] = correlation
                matrix_tables[f"{figure_name}_corr_{condition_tag}"] = correlation.reset_index().rename(columns={"index": "metric"})
                summary_rows.append(self.summarize_correlation_matrix(figure_name, condition_tag, correlation, int(subset["protein_name"].nunique())))
            pair_rows.extend(self.build_correlation_pair_delta_rows(figure_name, correlation_by_tag["below"], correlation_by_tag["above"]))
        return matrix_tables, pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)

    def summarize_correlation_matrix(self, figure_name: str, condition_tag: str, correlation: pd.DataFrame, protein_count: int) -> Dict[str, Any]:
        """Summarize one correlation matrix."""
        values = self.upper_triangle_values(correlation)
        finite_values = values[np.isfinite(values)]
        absolute_values = np.abs(finite_values)
        return {
            "threshold_bits_site_min": float(self.config.bits_site_min_threshold),
            "figure": figure_name,
            "condition": condition_tag,
            "n_proteins": int(protein_count),
            "n_metric_pairs": int(finite_values.size),
            "mean_spearman": safe_mean(finite_values),
            "median_spearman": self.safe_median(finite_values),
            "mean_abs_spearman": safe_mean(absolute_values),
            "median_abs_spearman": self.safe_median(absolute_values),
            "positive_fraction": safe_mean((finite_values > 0.0).astype(float)) if finite_values.size else float("nan"),
        }

    def build_correlation_pair_delta_rows(self, figure_name: str, below_correlation: pd.DataFrame, above_correlation: pd.DataFrame) -> List[Dict[str, Any]]:
        """Build long-format correlation-delta rows."""
        rows: List[Dict[str, Any]] = []
        labels = list(below_correlation.index)
        for first_index in range(len(labels)):
            for second_index in range(first_index + 1, len(labels)):
                below_value = float(below_correlation.iloc[first_index, second_index])
                above_value = float(above_correlation.iloc[first_index, second_index])
                rows.append(
                    {
                        "threshold_bits_site_min": float(self.config.bits_site_min_threshold),
                        "figure": figure_name,
                        "metric_a": labels[first_index],
                        "metric_b": labels[second_index],
                        "spearman_below": below_value,
                        "spearman_above": above_value,
                        "abs_spearman_below": abs(below_value) if np.isfinite(below_value) else float("nan"),
                        "abs_spearman_above": abs(above_value) if np.isfinite(above_value) else float("nan"),
                        "spearman_below_minus_above": below_value - above_value,
                        "abs_spearman_below_minus_above": (abs(below_value) - abs(above_value)) if np.isfinite(below_value) and np.isfinite(above_value) else float("nan"),
                    }
                )
        return rows

    def build_metric_dictionary(self, metrics: Sequence[SensitivityMetric]) -> pd.DataFrame:
        """Build metric dictionary."""
        return pd.DataFrame([{"figure": metric.figure, "panel": metric.panel, "metric": metric.column, "metric_label": metric.label} for metric in metrics])

    def build_overlap_table(self, protein_summary: pd.DataFrame, overlap_proteins: Sequence[str]) -> pd.DataFrame:
        """Build protein overlap table."""
        overlap_set = set(str(protein_name) for protein_name in overlap_proteins)
        rows: List[Dict[str, Any]] = []
        for protein_name in sorted(protein_summary["protein_name"].dropna().astype(str).unique().tolist()):
            subset = protein_summary[protein_summary["protein_name"] == protein_name]
            conditions = sorted(subset["condition4"].astype(str).unique().tolist())
            rows.append(
                {
                    "threshold_bits_site_min": float(self.config.bits_site_min_threshold),
                    "protein_name": protein_name,
                    "has_below_targeted": int("below targeted" in conditions),
                    "has_above_targeted": int("above targeted" in conditions),
                    "included_in_paired_overlap": int(protein_name in overlap_set),
                    "conditions_present": ";".join(conditions),
                }
            )
        return pd.DataFrame(rows)

    def build_readme_table(self, cohort_rows: int, protein_summary: pd.DataFrame, overlap_proteins: Sequence[str]) -> pd.DataFrame:
        """Build workbook readme."""
        targeted = protein_summary[protein_summary["condition4"].isin(["below targeted", "above targeted"])]
        below_count = int(targeted[targeted["condition4"] == "below targeted"]["protein_name"].nunique())
        above_count = int(targeted[targeted["condition4"] == "above targeted"]["protein_name"].nunique())
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "input_workbook", "value": str(self.config.cohort_xlsx_name)},
            {"key": "input_sheet", "value": str(self.config.cohort_sheet_name)},
            {"key": "cohort_rows_loaded", "value": str(int(cohort_rows))},
            {"key": "threshold_bits_site_min", "value": str(float(self.config.bits_site_min_threshold))},
            {"key": "condition_logic", "value": "figure_common threshold labels; matched controls inherit targeted regime"},
            {"key": "analysis_subset", "value": "targeted MSAs only; proteins represented in both below and above targeted strata"},
            {"key": "below_targeted_protein_count_all_available", "value": str(below_count)},
            {"key": "above_targeted_protein_count_all_available", "value": str(above_count)},
            {"key": "paired_overlap_protein_count", "value": str(len(overlap_proteins))},
            {"key": "paired_test", "value": "two-sided paired t-test on protein-level condition means; statistic uses above minus below"},
            {"key": "q_value_within_figure", "value": "Benjamini-Hochberg correction within each figure metric family"},
            {"key": "q_value_all_metrics", "value": "Benjamini-Hochberg correction across Figure 3 and Figure 4 paired-overlap metrics"},
            {"key": "correlations", "value": "Spearman correlations across paired-overlap protein-level means, computed separately below and above"},
        ]
        return pd.DataFrame(rows)

    def upper_triangle_values(self, correlation: pd.DataFrame) -> np.ndarray:
        """Return upper-triangle values excluding the diagonal."""
        values = correlation.to_numpy(dtype=float, copy=True)
        indices = np.triu_indices_from(values, k=1)
        return values[indices]

    def safe_median(self, values: Sequence[float]) -> float:
        """Return finite median or NaN."""
        value_array = np.asarray(values, dtype=float)
        finite_values = value_array[np.isfinite(value_array)]
        return float(np.median(finite_values)) if finite_values.size else float("nan")

    def run_analysis(self, cohort: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run complete paired sensitivity analysis."""
        cohort_with_conditions = add_threshold_condition_columns(cohort, threshold=float(self.config.bits_site_min_threshold))
        metrics = self.get_metrics()
        metric_columns = [metric.column for metric in metrics]
        require_columns(cohort_with_conditions, metric_columns, "condition-labeled cohort")
        protein_summary = self.compute_protein_condition_means(cohort_with_conditions, metric_columns)
        overlap_proteins = self.find_overlap_proteins(protein_summary)
        paired_input = self.build_paired_input_table(protein_summary, overlap_proteins, metric_columns)
        paired_tests = self.compute_paired_tests(paired_input, metrics)
        correlation_matrices, correlation_summary, correlation_pair_deltas = self.compute_correlation_outputs(paired_input, metrics)
        tables: Dict[str, pd.DataFrame] = {
            "readme": self.build_readme_table(int(cohort.shape[0]), protein_summary, overlap_proteins),
            "metric_dictionary": self.build_metric_dictionary(metrics),
            "paired_tests": paired_tests,
            "paired_protein_means": paired_input,
            "protein_overlap": self.build_overlap_table(protein_summary, overlap_proteins),
            "correlation_summary": correlation_summary,
            "correlation_pair_deltas": correlation_pair_deltas,
        }
        tables.update(correlation_matrices)
        return tables


@dataclass
class PairedSensitivityRunner:
    """Coordinate paired sensitivity export."""

    config: PairedSensitivityConfig

    def run(self) -> None:
        """Run analysis and write workbook."""
        paths = PairedSensitivityPaths.from_config(self.config)
        paths.ensure_dirs()
        analyzer = PairedSensitivityAnalyzer(self.config)
        cohort = analyzer.read_cohort_by_sample(paths)
        tables = analyzer.run_analysis(cohort)
        writer = ExcelNotebookWriter(paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        print(f"Wrote paired sensitivity analysis to {paths.output_xlsx_path}")



@dataclass(frozen=True)
class SensitivityAnalysisConfig:
    """Configure sensitivity-analysis notebooks."""

    cohort_xlsx_name: str = "cohort.xlsx"
    cohort_sheet_name: str = "by_sample"
    output_dir_name: str = "sensitivity_analyses"
    paired_output_xlsx_name: str = "paired_sensitivity_analysis.xlsx"
    threshold_output_xlsx_name: str = "threshold_boundary_sensitivity.xlsx"
    msa_snr_output_xlsx_name: str = "msa_snr_metric_sensitivity.xlsx"
    meff_depth_output_xlsx_name: str = "meff_depth_sensitivity.xlsx"
    primary_threshold: float = operational_bits_site_threshold
    threshold_grid: Tuple[float, ...] = (0.8, 0.9, operational_bits_site_threshold, 1.0, 1.1, 1.2)
    expected_overlap_protein_count: int = 20
    softmax_temperature: float = 1.0
    protein_block_subsample_replicates: int = 1000
    protein_block_subsample_fraction: float = 0.80
    protein_block_subsample_seed: int = 17
    excel_chunk_rows: int = 1000000


@dataclass(frozen=True)
class SensitivityAnalysisPaths:
    """Resolve sensitivity-analysis paths."""

    cohort_xlsx_path: Path
    output_dir: Path
    threshold_output_xlsx_path: Path
    msa_snr_output_xlsx_path: Path
    meff_depth_output_xlsx_path: Path

    @classmethod
    def from_config(cls, config: SensitivityAnalysisConfig) -> "SensitivityAnalysisPaths":
        """Build paths from the current directory."""
        root_dir = Path(".").resolve()
        output_dir = root_dir / str(config.output_dir_name)
        return cls(
            root_dir / str(config.cohort_xlsx_name),
            output_dir,
            output_dir / str(config.threshold_output_xlsx_name),
            output_dir / str(config.msa_snr_output_xlsx_name),
            output_dir / str(config.meff_depth_output_xlsx_name),
        )

    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class MsaLevelTableBuilder:
    """Build MSA-level tables from cohort rows."""

    config: SensitivityAnalysisConfig

    def read_cohort_by_sample(self, cohort_xlsx_path: Path) -> pd.DataFrame:
        """Read and validate cohort rows."""
        if not cohort_xlsx_path.exists():
            raise FileNotFoundError(f"Missing cohort workbook: {cohort_xlsx_path}")
        required_columns = ["sample_id", "protein_name", "outer_label", "control_type", "bits_site_min", "snr_used", "pid_pair_mean", "max_protein_id", "plddt_mean", "tm_mean"]
        cohort = pd.read_excel(cohort_xlsx_path, sheet_name=str(self.config.cohort_sheet_name), engine="openpyxl")
        require_columns(cohort, required_columns, "cohort workbook")
        out = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type"]:
            out[column_name] = out[column_name].astype(str)
        for column_name in ["bits_site_min", "snr_used", "pid_pair_mean", "max_protein_id", "plddt_mean", "tm_mean", "meff_theta0p70", "n_sequences"]:
            if column_name in out.columns:
                out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
        return convert_archived_columns(out, information_columns=("bits_site_min",))

    def compute_msa_level_table(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Aggregate candidate rows to one row per protein and MSA."""
        df = cohort.copy()
        if "meff_theta0p70" not in df.columns:
            df["meff_theta0p70"] = np.nan
        if "n_sequences" not in df.columns:
            df["n_sequences"] = np.nan
        df["novelty_pct"] = 100.0 - pd.to_numeric(df["max_protein_id"], errors="coerce")
        df["novelty_frac"] = df["novelty_pct"] / 100.0
        df["novelty_plddt_score"] = pd.to_numeric(df["plddt_mean"], errors="coerce") * df["novelty_frac"]
        df["novelty_tm_score"] = pd.to_numeric(df["tm_mean"], errors="coerce") * df["novelty_frac"]
        grouped = df.groupby(["protein_name", "outer_label"], sort=False)
        table = grouped.agg(
            control_type=("control_type", "first"),
            bits_site_min=("bits_site_min", "first"),
            snr_used=("snr_used", "first"),
            pid_pair_mean=("pid_pair_mean", "first"),
            n_candidates=("sample_id", "size"),
            max_plddt=("plddt_mean", "max"),
            max_tm=("tm_mean", "max"),
            max_novelty=("novelty_pct", "max"),
            median_novelty=("novelty_pct", "median"),
            max_novelty_plddt_score=("novelty_plddt_score", "max"),
            max_novelty_tm_score=("novelty_tm_score", "max"),
            meff_theta0p70=("meff_theta0p70", "first"),
            n_sequences=("n_sequences", "first"),
            bits_site_min_nunique=("bits_site_min", "nunique"),
            snr_used_nunique=("snr_used", "nunique"),
            pid_pair_mean_nunique=("pid_pair_mean", "nunique"),
        ).reset_index()
        table["pid_pair_std"] = self.compute_pid_pair_std(table["pid_pair_mean"], table["snr_used"])
        return self.add_metric_columns(table)

    def compute_pid_pair_std(self, pid_pair_mean: pd.Series, snr_used: pd.Series) -> pd.Series:
        """Derive pairwise-identity standard deviation."""
        mean_values = pd.to_numeric(pid_pair_mean, errors="coerce").to_numpy(dtype=float)
        snr_values = pd.to_numeric(snr_used, errors="coerce").to_numpy(dtype=float)
        output = np.full(mean_values.shape, np.nan, dtype=float)
        valid_mask = np.isfinite(mean_values) & np.isfinite(snr_values) & (snr_values > 0.0)
        output[valid_mask] = mean_values[valid_mask] / snr_values[valid_mask]
        return pd.Series(output, index=pid_pair_mean.index, dtype=float)

    def add_metric_columns(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add row-coherence and scale-comparator columns."""
        out = table.copy()
        out["primary_msa_snr"] = pd.to_numeric(out["snr_used"], errors="coerce")
        out["negative_pid_pair_std"] = -pd.to_numeric(out["pid_pair_std"], errors="coerce")
        out["rank_composite_pid_mean_minus_std"] = self.compute_rank_composite(out)
        return out

    def compute_rank_composite(self, table: pd.DataFrame) -> pd.Series:
        """Compute within-protein rank composite."""
        output = pd.Series(np.nan, index=table.index, dtype=float)
        for protein_name, subset in table.groupby("protein_name", sort=False):
            mean_rank = pd.to_numeric(subset["pid_pair_mean"], errors="coerce").rank(method="average", ascending=True)
            std_rank = pd.to_numeric(subset["pid_pair_std"], errors="coerce").rank(method="average", ascending=True)
            output.loc[subset.index] = mean_rank - std_rank
        return output


@dataclass
class ThresholdBoundarySensitivityAnalyzer:
    """Compute threshold-grid sensitivity for Figure 3 and Figure 4 metrics."""

    config: SensitivityAnalysisConfig

    def get_metrics(self) -> List[SensitivityMetric]:
        """Return Figure 3 and Figure 4 metrics."""
        paired_config = PairedSensitivityConfig(
            cohort_xlsx_name=self.config.cohort_xlsx_name,
            cohort_sheet_name=self.config.cohort_sheet_name,
            output_dir_name=self.config.output_dir_name,
            output_xlsx_name=self.config.paired_output_xlsx_name,
            bits_site_min_threshold=float(self.config.primary_threshold),
            expected_overlap_protein_count=int(self.config.expected_overlap_protein_count),
            excel_chunk_rows=int(self.config.excel_chunk_rows),
        )
        return PairedSensitivityAnalyzer(paired_config).get_metrics()

    def read_cohort_by_sample(self, paths: SensitivityAnalysisPaths) -> pd.DataFrame:
        """Read and validate threshold-sensitivity rows."""
        if not paths.cohort_xlsx_path.exists():
            raise FileNotFoundError(f"Missing cohort workbook: {paths.cohort_xlsx_path}")
        cohort = pd.read_excel(paths.cohort_xlsx_path, sheet_name=str(self.config.cohort_sheet_name), engine="openpyxl")
        required_columns = ["sample_id", "protein_name", "outer_label", "control_type", "bits_site_min"]
        required_columns.extend([metric.column for metric in self.get_metrics()])
        require_columns(cohort, list(dict.fromkeys(required_columns)), "cohort workbook")
        out = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type"]:
            out[column_name] = out[column_name].astype(str)
        for column_name in ["bits_site_min"] + [metric.column for metric in self.get_metrics()]:
            out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
        return convert_archived_columns(
            out,
            information_columns=("bits_site_min",),
            effective_alphabet_columns=("msa_effective_alphabet_mean",),
        )

    def protein_summary(self, cohort: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
        """Compute Figure-style protein-level condition means."""
        group_columns = ["protein_name", "condition4", "target_stratum"]
        require_columns(cohort, group_columns + list(metric_columns), "condition-labeled cohort")
        base = cohort[group_columns + list(metric_columns)].copy()
        for column_name in metric_columns:
            base[column_name] = pd.to_numeric(base[column_name], errors="coerce")
        return base.groupby(group_columns, sort=False)[list(metric_columns)].mean(numeric_only=True).reset_index()

    def condition_values(self, table: pd.DataFrame, condition_name: str, metric_column: str) -> np.ndarray:
        """Return finite values for one condition."""
        values = pd.to_numeric(table.loc[table["condition4"] == condition_name, metric_column], errors="coerce").to_numpy(dtype=float)
        return values[np.isfinite(values)]

    def welch_test(self, group_a_values: np.ndarray, group_b_values: np.ndarray) -> Dict[str, Any]:
        """Compute one Welch test."""
        if group_a_values.size < 3 or group_b_values.size < 3:
            return {"t_statistic": float("nan"), "p_value": float("nan"), "status": "insufficient_values"}
        result = stats.ttest_ind(group_a_values, group_b_values, equal_var=False, nan_policy="omit")
        return {"t_statistic": float(result.statistic), "p_value": float(result.pvalue), "status": "ok"}


    def hedges_g(self, group_a_values: Sequence[float], group_b_values: Sequence[float]) -> float:
        """Compute Hedges g."""
        a_values = np.asarray(group_a_values, dtype=float)
        b_values = np.asarray(group_b_values, dtype=float)
        a_values = a_values[np.isfinite(a_values)]
        b_values = b_values[np.isfinite(b_values)]
        n_a = float(a_values.size)
        n_b = float(b_values.size)
        if n_a < 2 or n_b < 2:
            return float("nan")
        variance_a = float(np.var(a_values, ddof=1))
        variance_b = float(np.var(b_values, ddof=1))
        denominator = n_a + n_b - 2.0
        numerator = (n_a - 1.0) * variance_a + (n_b - 1.0) * variance_b
        if denominator <= 0.0 or numerator <= 0.0:
            return float("nan")
        pooled_sd = math.sqrt(numerator / denominator)
        correction = 1.0 - (3.0 / (4.0 * (n_a + n_b) - 9.0)) if (n_a + n_b) > 2.0 else 1.0
        return float(correction * ((float(np.mean(a_values)) - float(np.mean(b_values))) / pooled_sd))

    def unpaired_tests(self, table: pd.DataFrame, metrics: Sequence[SensitivityMetric], threshold: float) -> pd.DataFrame:
        """Compute unpaired threshold contrasts."""
        rows: List[Dict[str, Any]] = []
        contrasts = [("target_below_vs_above", "below targeted", "above targeted", True), ("matched_control_below_vs_above", "below matched control", "above matched control", False)]
        for metric in metrics:
            for contrast_name, group_a_name, group_b_name, reported in contrasts:
                group_a = self.condition_values(table, group_a_name, metric.column)
                group_b = self.condition_values(table, group_b_name, metric.column)
                result = self.welch_test(group_a, group_b)
                rows.append({
                    "threshold_bits_site_min": float(threshold),
                    "figure": metric.figure,
                    "panel": metric.panel,
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "contrast": contrast_name,
                    "reported_in_figure": bool(reported),
                    "test": "welch_two_sided",
                    "analysis_unit": "protein_level_condition_mean",
                    "group_a": group_a_name,
                    "n_a": int(group_a.size),
                    "mean_a": safe_mean(group_a),
                    "sd_a": safe_sample_sd(group_a),
                    "group_b": group_b_name,
                    "n_b": int(group_b.size),
                    "mean_b": safe_mean(group_b),
                    "sd_b": safe_sample_sd(group_b),
                    "mean_a_minus_b": safe_mean(group_a) - safe_mean(group_b),
                    "p_value": result["p_value"],
                    "t_statistic": result["t_statistic"],
                    "effect_name": "hedges_g",
                    "effect": self.hedges_g(group_a, group_b),
                    "status": result["status"],
                })
        output = self.add_unpaired_q_values(pd.DataFrame(rows))
        output["direction"] = np.where(output["mean_a_minus_b"] > 0.0, "below_higher", np.where(output["mean_a_minus_b"] < 0.0, "above_higher", "no_difference"))
        return output

    def add_unpaired_q_values(self, table: pd.DataFrame) -> pd.DataFrame:
        """Add Figure-style BH q-values to unpaired rows."""
        output = table.copy()
        output["q_value_bh"] = np.nan
        output["bh_threshold"] = 0.05
        output["reject_fdr_bh"] = False
        output["correction_family"] = ""
        output["n_tests_in_family"] = np.nan
        for threshold_value in sorted(output["threshold_bits_site_min"].dropna().unique().tolist()):
            threshold_mask = output["threshold_bits_site_min"] == threshold_value
            for metric_name in sorted(output.loc[threshold_mask, "metric"].dropna().unique().tolist()):
                metric_mask = threshold_mask & (output["metric"] == metric_name)
                q_values = benjamini_hochberg(output.loc[metric_mask, "p_value"].to_numpy(dtype=float))
                output.loc[metric_mask, "q_value_bh"] = q_values
                output.loc[metric_mask, "reject_fdr_bh"] = pd.to_numeric(output.loc[metric_mask, "q_value_bh"], errors="coerce") <= 0.05
                output.loc[metric_mask, "correction_family"] = "within_metric_targeted_and_matched_control_below_vs_above"
                output.loc[metric_mask, "n_tests_in_family"] = int(metric_mask.sum())
        return output

    def count_condition(self, table: pd.DataFrame, condition_name: str) -> int:
        """Count proteins in one condition."""
        return int(table.loc[table["condition4"] == condition_name, "protein_name"].nunique())

    def protein_count_row(self, table: pd.DataFrame, threshold: float) -> Dict[str, Any]:
        """Build one protein-count row."""
        return {
            "threshold_bits_site_min": float(threshold),
            "below_targeted_protein_count": self.count_condition(table, "below targeted"),
            "above_targeted_protein_count": self.count_condition(table, "above targeted"),
            "below_matched_control_protein_count": self.count_condition(table, "below matched control"),
            "above_matched_control_protein_count": self.count_condition(table, "above matched control"),
        }


    def sign_stability(self, unpaired_table: pd.DataFrame) -> pd.DataFrame:
        """Summarize direction stability relative to the primary threshold."""
        target_rows = unpaired_table[unpaired_table["contrast"] == "target_below_vs_above"].copy()
        primary_rows = target_rows[np.isclose(target_rows["threshold_bits_site_min"].to_numpy(dtype=float), float(self.config.primary_threshold))].copy()
        rows: List[Dict[str, Any]] = []
        for row in primary_rows.itertuples(index=False):
            subset = target_rows[(target_rows["figure"] == getattr(row, "figure")) & (target_rows["metric"] == getattr(row, "metric"))]
            directions = subset["direction"].astype(str).to_numpy()
            primary_direction = str(getattr(row, "direction"))
            p_values = pd.to_numeric(subset["p_value"], errors="coerce").to_numpy(dtype=float)
            q_values = pd.to_numeric(subset["q_value_bh"], errors="coerce").to_numpy(dtype=float)
            rows.append({
                "figure": str(getattr(row, "figure")),
                "panel": str(getattr(row, "panel")),
                "metric": str(getattr(row, "metric")),
                "metric_label": str(getattr(row, "metric_label")),
                "contrast": "target_below_vs_above",
                "primary_threshold": float(self.config.primary_threshold),
                "primary_direction": primary_direction,
                "n_thresholds_tested": int(subset.shape[0]),
                "n_matching_primary_direction": int(np.sum(directions == primary_direction)),
                "fraction_matching_primary_direction": float(np.mean(directions == primary_direction)) if directions.size else float("nan"),
                "min_p_value": self.safe_min(p_values),
                "max_p_value": self.safe_max(p_values),
                "min_q_value_bh": self.safe_min(q_values),
                "max_q_value_bh": self.safe_max(q_values),
                "n_thresholds_q_lt_0p05": int(np.sum(q_values < 0.05)),
                "fraction_thresholds_q_lt_0p05": float(np.mean(q_values < 0.05)) if q_values.size else float("nan"),
            })
        return pd.DataFrame(rows)

    def run_analysis(self, cohort: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run threshold-boundary sensitivity."""
        metrics = self.get_metrics()
        metric_columns = [metric.column for metric in metrics]
        unpaired_tables: List[pd.DataFrame] = []
        count_rows: List[Dict[str, Any]] = []
        for threshold in self.config.threshold_grid:
            labeled = add_threshold_condition_columns(cohort, threshold=float(threshold))
            summary = self.protein_summary(labeled, metric_columns)
            count_rows.append(self.protein_count_row(summary, float(threshold)))
            unpaired_tables.append(self.unpaired_tests(summary, metrics, float(threshold)))
        unpaired = pd.concat(unpaired_tables, axis=0, ignore_index=True)
        primary_reference = unpaired[np.isclose(unpaired["threshold_bits_site_min"].to_numpy(dtype=float), float(self.config.primary_threshold))].copy()
        return {
            "readme": self.readme_table(int(cohort.shape[0])),
            "threshold_grid": pd.DataFrame({"threshold_bits_site_min": list(self.config.threshold_grid)}),
            "metric_dictionary": pd.DataFrame([{"figure": metric.figure, "panel": metric.panel, "metric": metric.column, "metric_label": metric.label} for metric in metrics]),
            "protein_counts_by_threshold": pd.DataFrame(count_rows),
            "unpaired_sensitivity": unpaired,
            "sign_stability_summary": self.sign_stability(unpaired),
            "primary_threshold_reference": primary_reference,
        }

    def readme_table(self, cohort_rows: int) -> pd.DataFrame:
        """Build workbook readme."""
        return pd.DataFrame([
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "input_workbook", "value": str(self.config.cohort_xlsx_name)},
            {"key": "input_sheet", "value": str(self.config.cohort_sheet_name)},
            {"key": "cohort_rows_loaded", "value": str(int(cohort_rows))},
            {"key": "analysis_name", "value": "threshold_boundary_sensitivity"},
            {"key": "primary_threshold_bits_site_min", "value": str(float(self.config.primary_threshold))},
            {"key": "threshold_grid", "value": ",".join([f"{value:.6g}" for value in self.config.threshold_grid])},
            {"key": "condition_logic", "value": "Regime labels are reassigned at each threshold with matched controls inheriting targeted regime."},
            {"key": "analysis_unit", "value": "Protein-level condition means."},
            {"key": "statistics", "value": "Welch two-sided tests with BH correction within each metric across targeted and matched-control contrasts."},
        ])

    def safe_median(self, values: Sequence[float]) -> float:
        """Return finite median or NaN."""
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(np.median(finite)) if finite.size else float("nan")

    def safe_min(self, values: Sequence[float]) -> float:
        """Return finite minimum or NaN."""
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(np.min(finite)) if finite.size else float("nan")

    def safe_max(self, values: Sequence[float]) -> float:
        """Return finite maximum or NaN."""
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(np.max(finite)) if finite.size else float("nan")


@dataclass
class MsaSnrMetricSensitivityAnalyzer:
    """Compute MSA-SNR metric-choice sensitivity."""

    config: SensitivityAnalysisConfig

    def metric_definitions(self) -> pd.DataFrame:
        """Return metric definitions."""
        return pd.DataFrame([
            {"metric": "primary_msa_snr", "source_column": "snr_used", "metric_type": "row_coherence", "definition": "Mean no-gap pairwise identity divided by no-gap pairwise-identity standard deviation."},
            {"metric": "pid_pair_mean", "source_column": "pid_pair_mean", "metric_type": "row_coherence_component", "definition": "Mean no-gap pairwise identity."},
            {"metric": "negative_pid_pair_std", "source_column": "pid_pair_std", "metric_type": "row_coherence_component", "definition": "Negative pairwise-identity standard deviation."},
            {"metric": "rank_composite_pid_mean_minus_std", "source_column": "pid_pair_mean,pid_pair_std", "metric_type": "row_coherence", "definition": "Within-protein rank(PID mean) minus rank(PID standard deviation)."},
            {"metric": "meff_theta0p70", "source_column": "meff_theta0p70", "metric_type": "scale_comparator", "definition": "Meff at theta 0.70."},
            {"metric": "n_sequences", "source_column": "n_sequences", "metric_type": "scale_comparator", "definition": "Retained MSA depth."},
        ])

    def metric_names(self) -> List[str]:
        """Return sensitivity metric names."""
        return self.metric_definitions()["metric"].astype(str).tolist()

    def readout_names(self) -> List[str]:
        """Return readout names."""
        return ["max_plddt", "max_tm", "max_novelty_plddt_score", "max_novelty_tm_score", "max_novelty"]
    def readout_family(self, readout_name: str) -> str:
        """Return the readout family label."""
        if str(readout_name) in ["max_plddt", "max_tm"]:
            return "figure2_primary"
        return "extended_novelty_frontier"

    def compute_peak_locations(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute per-protein peak location for each metric."""
        rows: List[Dict[str, Any]] = []
        for metric_name in self.metric_names():
            for protein_name, subset in table.groupby("protein_name", sort=False):
                values = pd.to_numeric(subset[metric_name], errors="coerce").to_numpy(dtype=float)
                bits = pd.to_numeric(subset["bits_site_min"], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(values) & np.isfinite(bits)
                if not valid.any():
                    continue
                peak_value = float(np.max(values[valid]))
                positions = np.where((values == peak_value) & valid)[0]
                selected = subset.iloc[int(positions[0])]
                rows.append({"metric": metric_name, "protein_name": str(protein_name), "peak_bits_site_min": float(selected["bits_site_min"]), "peak_metric_value": peak_value, "peak_outer_label": str(selected["outer_label"]), "peak_control_type": str(selected["control_type"]), "n_tied_peak_rows": int(positions.size)})
        peaks = pd.DataFrame(rows)
        if peaks.empty:
            return peaks
        return self.add_softmax_weights(peaks)

    def add_softmax_weights(self, peaks: pd.DataFrame) -> pd.DataFrame:
        """Add metric-specific softmax weights."""
        output = peaks.copy()
        output["softmax_weight"] = np.nan
        temperature = float(self.config.softmax_temperature)
        if temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        for metric_name, subset in output.groupby("metric", sort=False):
            values = pd.to_numeric(subset["peak_metric_value"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values)
            if not valid.any():
                continue
            weights = np.full(values.shape, np.nan, dtype=float)
            finite_values = values[valid]
            metric_weights = np.exp((finite_values - float(np.max(finite_values))) / temperature)
            weight_sum = float(np.sum(metric_weights))
            if weight_sum > 0.0:
                weights[valid] = metric_weights / weight_sum
            output.loc[subset.index, "softmax_weight"] = weights
        return output

    def peak_summary(self, peaks: pd.DataFrame) -> pd.DataFrame:
        """Summarize peak locations by metric."""
        rows: List[Dict[str, Any]] = []
        for metric_name, subset in peaks.groupby("metric", sort=False):
            bits = pd.to_numeric(subset["peak_bits_site_min"], errors="coerce").to_numpy(dtype=float)
            weights = pd.to_numeric(subset["softmax_weight"], errors="coerce").to_numpy(dtype=float)
            finite_bits = bits[np.isfinite(bits)]
            rows.append({
                "metric": str(metric_name),
                "n_proteins": int(finite_bits.size),
                "unweighted_mean_peak_bits_site_min": safe_mean(finite_bits),
                "unweighted_sample_sd_peak_bits_site_min": safe_sample_sd(finite_bits),
                "unweighted_population_sd_peak_bits_site_min": float(np.std(finite_bits, ddof=0)) if finite_bits.size else float("nan"),
                "median_peak_bits_site_min": self.safe_median(finite_bits),
                "fraction_peak_below_primary_threshold": safe_mean((finite_bits < float(self.config.primary_threshold)).astype(float)) if finite_bits.size else float("nan"),
                "weighted_mean_peak_bits_site_min": self.weighted_mean(bits, weights),
                "weighted_sd_peak_bits_site_min": self.weighted_sd(bits, weights),
                "softmax_temperature": float(self.config.softmax_temperature),
            })
        return pd.DataFrame(rows)

    def structural_correlations(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute metric-readout correlations."""
        rows: List[Dict[str, Any]] = []
        for group_name, subset in self.group_specs(table):
            for metric_name in self.metric_names():
                for readout_name in self.readout_names():
                    rho, p_value, count = self.spearman(subset[metric_name], subset[readout_name])
                    rows.append({"analysis_group": group_name, "metric": metric_name, "readout": readout_name, "readout_family": self.readout_family(readout_name), "n_msa": count, "spearman_rho": rho, "p_value": p_value})
        return pd.DataFrame(rows)

    def validate_protein_block_subsample_inputs(self, table: pd.DataFrame) -> None:
        """Validate protein-block subsampling inputs."""
        required_columns = ["protein_name", "control_type"] + self.metric_names() + self.readout_names()
        require_columns(table, required_columns, "MSA-level protein-block subsampling table")
        replicate_count = int(self.config.protein_block_subsample_replicates)
        protein_fraction = float(self.config.protein_block_subsample_fraction)
        if replicate_count < 1:
            raise ValueError("protein_block_subsample_replicates must be at least 1")
        if protein_fraction <= 0.0 or protein_fraction > 1.0:
            raise ValueError("protein_block_subsample_fraction must be greater than 0 and less than or equal to 1")
        protein_count = int(table["protein_name"].dropna().astype(str).nunique())
        if protein_count < 2:
            raise ValueError("At least two proteins are required for protein-block subsampling")
        int(self.config.protein_block_subsample_seed)

    def protein_block_subsample_retained_count(self, protein_count: int) -> int:
        """Return retained protein count for subsampling."""
        if int(protein_count) < 2:
            raise ValueError("protein_count must be at least 2")
        protein_fraction = float(self.config.protein_block_subsample_fraction)
        retained_count = int(round(float(protein_count) * protein_fraction))
        retained_count = max(1, min(int(protein_count), retained_count))
        return retained_count

    def select_subsampled_proteins(self, protein_names: Sequence[str], retained_count: int, random_seed: int) -> List[str]:
        """Select one deterministic protein subset."""
        names = np.asarray(list(protein_names), dtype=object)
        if int(retained_count) > names.size:
            raise ValueError("retained_count cannot exceed available protein count")
        rng = np.random.default_rng(int(random_seed))
        selected = rng.choice(names, size=int(retained_count), replace=False)
        return sorted(str(value) for value in selected.tolist())

    def protein_block_subsample_correlations(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute protein-block subsampling correlation stability."""
        self.validate_protein_block_subsample_inputs(table)
        rows: List[Dict[str, Any]] = []
        protein_names = sorted(table["protein_name"].dropna().astype(str).unique().tolist())
        protein_count = len(protein_names)
        retained_count = self.protein_block_subsample_retained_count(protein_count)
        base_seed = int(self.config.protein_block_subsample_seed)
        replicate_count = int(self.config.protein_block_subsample_replicates)
        protein_fraction = float(self.config.protein_block_subsample_fraction)
        for replicate_index in range(replicate_count):
            replicate_seed = int(base_seed + replicate_index)
            retained_proteins = self.select_subsampled_proteins(protein_names, retained_count, replicate_seed)
            subset = table[table["protein_name"].astype(str).isin(retained_proteins)].copy()
            for group_name, group_table in self.group_specs(subset):
                for metric_name in self.metric_names():
                    for readout_name in self.readout_names():
                        rho, p_value, count = self.spearman(group_table[metric_name], group_table[readout_name])
                        rows.append({
                            "analysis_group": str(group_name),
                            "metric": str(metric_name),
                            "readout": str(readout_name),
                            "readout_family": self.readout_family(readout_name),
                            "replicate_index": int(replicate_index),
                            "protein_fraction_retained": float(protein_fraction),
                            "n_proteins_total": int(protein_count),
                            "n_proteins_retained": int(retained_count),
                            "n_msa": int(count),
                            "spearman_rho": float(rho),
                            "p_value": float(p_value),
                            "random_seed": int(replicate_seed),
                        })
        return pd.DataFrame(rows)

    def protein_block_subsample_summary(self, subsample_correlations: pd.DataFrame, full_correlations: pd.DataFrame) -> pd.DataFrame:
        """Summarize protein-block subsampling stability."""
        required_subsample_columns = ["analysis_group", "metric", "readout", "readout_family", "protein_fraction_retained", "spearman_rho"]
        required_full_columns = ["analysis_group", "metric", "readout", "spearman_rho"]
        require_columns(subsample_correlations, required_subsample_columns, "protein-block subsample correlations")
        require_columns(full_correlations, required_full_columns, "full structural correlations")
        rows: List[Dict[str, Any]] = []
        grouped = subsample_correlations.groupby(["analysis_group", "metric", "readout"], sort=False)
        for group_key, subset in grouped:
            analysis_group = str(group_key[0])
            metric_name = str(group_key[1])
            readout_name = str(group_key[2])
            rho_values = pd.to_numeric(subset["spearman_rho"], errors="coerce").to_numpy(dtype=float)
            finite_rho = rho_values[np.isfinite(rho_values)]
            full_rho = self.lookup_full_spearman_rho(full_correlations, analysis_group, metric_name, readout_name)
            protein_fraction = self.safe_first_numeric(subset["protein_fraction_retained"])
            readout_family = str(subset["readout_family"].iloc[0]) if subset.shape[0] else self.readout_family(readout_name)
            rows.append({
                "analysis_group": analysis_group,
                "metric": metric_name,
                "readout": readout_name,
                "readout_family": readout_family,
                "full_spearman_rho": float(full_rho),
                "n_replicates": int(subset.shape[0]),
                "n_finite_replicates": int(finite_rho.size),
                "protein_fraction_retained": float(protein_fraction),
                "mean_subsample_rho": safe_mean(finite_rho),
                "sd_subsample_rho": safe_sample_sd(finite_rho),
                "q05_subsample_rho": self.safe_quantile(finite_rho, 0.05),
                "median_subsample_rho": self.safe_median(finite_rho),
                "q95_subsample_rho": self.safe_quantile(finite_rho, 0.95),
                "min_subsample_rho": self.safe_minimum(finite_rho),
                "max_subsample_rho": self.safe_maximum(finite_rho),
                "fraction_same_sign_as_full": self.fraction_same_sign(finite_rho, full_rho),
                "fraction_positive": safe_mean((finite_rho > 0.0).astype(float)) if finite_rho.size else float("nan"),
                "max_abs_deviation_from_full": self.max_abs_deviation(finite_rho, full_rho),
            })
        return pd.DataFrame(rows)

    def lookup_full_spearman_rho(self, full_correlations: pd.DataFrame, analysis_group: str, metric_name: str, readout_name: str) -> float:
        """Return the matching full-cohort Spearman rho."""
        match = full_correlations[
            (full_correlations["analysis_group"].astype(str) == str(analysis_group))
            & (full_correlations["metric"].astype(str) == str(metric_name))
            & (full_correlations["readout"].astype(str) == str(readout_name))
        ].copy()
        if match.shape[0] != 1:
            return float("nan")
        values = pd.to_numeric(match["spearman_rho"], errors="coerce").to_numpy(dtype=float)
        return float(values[0]) if values.size else float("nan")

    def safe_first_numeric(self, values: Sequence[float]) -> float:
        """Return first finite numeric value or NaN."""
        value_array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        finite_values = value_array[np.isfinite(value_array)]
        return float(finite_values[0]) if finite_values.size else float("nan")

    def safe_quantile(self, values: Sequence[float], quantile: float) -> float:
        """Return finite quantile or NaN."""
        value_array = np.asarray(values, dtype=float)
        finite_values = value_array[np.isfinite(value_array)]
        if finite_values.size == 0:
            return float("nan")
        return float(np.quantile(finite_values, float(quantile)))

    def safe_minimum(self, values: Sequence[float]) -> float:
        """Return finite minimum or NaN."""
        value_array = np.asarray(values, dtype=float)
        finite_values = value_array[np.isfinite(value_array)]
        return float(np.min(finite_values)) if finite_values.size else float("nan")

    def safe_maximum(self, values: Sequence[float]) -> float:
        """Return finite maximum or NaN."""
        value_array = np.asarray(values, dtype=float)
        finite_values = value_array[np.isfinite(value_array)]
        return float(np.max(finite_values)) if finite_values.size else float("nan")

    def fraction_same_sign(self, values: Sequence[float], reference_value: float) -> float:
        """Return fraction with the same nonzero sign as reference."""
        value_array = np.asarray(values, dtype=float)
        finite_values = value_array[np.isfinite(value_array)]
        if finite_values.size == 0 or not np.isfinite(reference_value):
            return float("nan")
        reference_sign = float(np.sign(reference_value))
        if reference_sign == 0.0:
            return float("nan")
        return float(np.mean(np.sign(finite_values) == reference_sign))

    def max_abs_deviation(self, values: Sequence[float], reference_value: float) -> float:
        """Return maximum absolute deviation from reference."""
        value_array = np.asarray(values, dtype=float)
        finite_values = value_array[np.isfinite(value_array)]
        if finite_values.size == 0 or not np.isfinite(reference_value):
            return float("nan")
        return float(np.max(np.abs(finite_values - float(reference_value))))

    def metric_rank_correlations(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute rank correlations among metrics."""
        names = [name for name in self.metric_names() if name in table.columns]
        rows: List[Dict[str, Any]] = []
        for first_index in range(len(names)):
            for second_index in range(first_index + 1, len(names)):
                metric_a = names[first_index]
                metric_b = names[second_index]
                rho, p_value, count = self.spearman(table[metric_a], table[metric_b])
                rows.append({"metric_a": metric_a, "metric_b": metric_b, "n_msa": count, "spearman_rho": rho, "p_value": p_value})
        return pd.DataFrame(rows)

    def primary_peak_overlap(self, peaks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compare alternative peaks with primary MSA-SNR peaks."""
        primary = peaks[peaks["metric"] == "primary_msa_snr"][["protein_name", "peak_outer_label", "peak_bits_site_min"]].copy()
        primary = primary.rename(columns={"peak_outer_label": "primary_peak_outer_label", "peak_bits_site_min": "primary_peak_bits_site_min"})
        rows: List[Dict[str, Any]] = []
        for row in peaks.itertuples(index=False):
            metric_name = str(getattr(row, "metric"))
            if metric_name == "primary_msa_snr":
                continue
            match = primary[primary["protein_name"] == str(getattr(row, "protein_name"))]
            if match.empty:
                continue
            primary_label = str(match.iloc[0]["primary_peak_outer_label"])
            primary_bits = float(match.iloc[0]["primary_peak_bits_site_min"])
            alternative_bits = float(getattr(row, "peak_bits_site_min"))
            rows.append({"metric": metric_name, "protein_name": str(getattr(row, "protein_name")), "primary_peak_outer_label": primary_label, "alternative_peak_outer_label": str(getattr(row, "peak_outer_label")), "same_peak_as_primary": int(str(getattr(row, "peak_outer_label")) == primary_label), "primary_peak_bits_site_min": primary_bits, "alternative_peak_bits_site_min": alternative_bits, "absolute_peak_bits_site_min_difference": abs(alternative_bits - primary_bits)})
        overlap = pd.DataFrame(rows)
        if overlap.empty:
            return overlap, pd.DataFrame()
        summary_rows: List[Dict[str, Any]] = []
        for metric_name, subset in overlap.groupby("metric", sort=False):
            differences = pd.to_numeric(subset["absolute_peak_bits_site_min_difference"], errors="coerce").to_numpy(dtype=float)
            same_peak = pd.to_numeric(subset["same_peak_as_primary"], errors="coerce").to_numpy(dtype=float)
            summary_rows.append({"metric": str(metric_name), "n_proteins": int(subset.shape[0]), "same_peak_fraction": safe_mean(same_peak[np.isfinite(same_peak)]), "mean_absolute_peak_bits_site_min_difference": safe_mean(differences[np.isfinite(differences)]), "median_absolute_peak_bits_site_min_difference": self.safe_median(differences)})
        return overlap, pd.DataFrame(summary_rows)

    def run_analysis(self, cohort: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run MSA-SNR metric sensitivity."""
        builder = MsaLevelTableBuilder(self.config)
        table = builder.compute_msa_level_table(cohort)
        peaks = self.compute_peak_locations(table)
        peak_summary = self.peak_summary(peaks) if not peaks.empty else pd.DataFrame()
        correlations = self.structural_correlations(table)
        subsample_correlations = self.protein_block_subsample_correlations(table)
        subsample_summary = self.protein_block_subsample_summary(subsample_correlations, correlations)
        overlap, overlap_summary = self.primary_peak_overlap(peaks)
        return {
            "readme": self.readme_table(int(cohort.shape[0]), int(table.shape[0])),
            "metric_definitions": self.metric_definitions(),
            "msa_level_input": table,
            "peak_locations_by_metric": peaks,
            "peak_location_summary": peak_summary,
            "structural_correlations": correlations,
            "protein_block_subsample_corrs": subsample_correlations,
            "protein_block_subsample_summary": subsample_summary,
            "metric_rank_correlations": self.metric_rank_correlations(table),
            "primary_peak_overlap": overlap,
            "primary_peak_overlap_summary": overlap_summary,
            "scale_comparator_correlations": correlations[correlations["metric"].isin(["meff_theta0p70", "n_sequences"])].copy().reset_index(drop=True),
        }

    def group_specs(self, table: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """Return analysis groups."""
        groups: List[Tuple[str, pd.DataFrame]] = [("all", table)]
        for control_type, subset in table.groupby(table["control_type"].astype(str).str.lower(), sort=False):
            groups.append((str(control_type), subset))
        return groups

    def spearman(self, x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, float, int]:
        """Compute Spearman correlation."""
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        valid = np.isfinite(x_array) & np.isfinite(y_array)
        count = int(valid.sum())
        if count < 10:
            return float("nan"), float("nan"), count
        result = stats.spearmanr(x_array[valid], y_array[valid])
        return float(result.statistic), float(result.pvalue), count

    def weighted_mean(self, values: Sequence[float], weights: Sequence[float]) -> float:
        """Return weighted mean or NaN."""
        value_array = np.asarray(values, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array >= 0.0)
        if not valid.any():
            return float("nan")
        weight_sum = float(np.sum(weight_array[valid]))
        if weight_sum <= 0.0:
            return float("nan")
        return float(np.sum(value_array[valid] * weight_array[valid]) / weight_sum)

    def weighted_sd(self, values: Sequence[float], weights: Sequence[float]) -> float:
        """Return weighted population standard deviation or NaN."""
        mean_value = self.weighted_mean(values, weights)
        if not np.isfinite(mean_value):
            return float("nan")
        value_array = np.asarray(values, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array >= 0.0)
        weight_sum = float(np.sum(weight_array[valid]))
        if weight_sum <= 0.0:
            return float("nan")
        variance = float(np.sum(weight_array[valid] * (value_array[valid] - mean_value) ** 2) / weight_sum)
        return float(math.sqrt(max(variance, 0.0)))

    def safe_median(self, values: Sequence[float]) -> float:
        """Return finite median or NaN."""
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(np.median(finite)) if finite.size else float("nan")

    def readme_table(self, cohort_rows: int, msa_rows: int) -> pd.DataFrame:
        """Build workbook readme."""
        return pd.DataFrame([
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "input_workbook", "value": str(self.config.cohort_xlsx_name)},
            {"key": "input_sheet", "value": str(self.config.cohort_sheet_name)},
            {"key": "cohort_rows_loaded", "value": str(int(cohort_rows))},
            {"key": "msa_rows", "value": str(int(msa_rows))},
            {"key": "analysis_name", "value": "msa_snr_metric_sensitivity"},
            {"key": "msa_unit", "value": "protein_name plus outer_label"},
            {"key": "primary_threshold_bits_site_min", "value": str(float(self.config.primary_threshold))},
            {"key": "pid_pair_std", "value": "Derived as pid_pair_mean divided by snr_used."},
            {"key": "readout_family", "value": "max_plddt and max_tm are figure2_primary; max_novelty_plddt_score, max_novelty_tm_score, and max_novelty are extended_novelty_frontier."},
            {"key": "protein_block_subsample_analysis", "value": "Repeated protein-block subsampling of MSA-level metric-readout correlations."},
            {"key": "protein_block_subsample_scope", "value": "All metrics returned by metric_names against all readouts returned by readout_names across all existing analysis groups."},
            {"key": "protein_block_subsample_fraction", "value": str(float(self.config.protein_block_subsample_fraction))},
            {"key": "protein_block_subsample_replicates", "value": str(int(self.config.protein_block_subsample_replicates))},
            {"key": "protein_block_subsample_seed", "value": str(int(self.config.protein_block_subsample_seed))},
        ])


@dataclass
class MeffDepthSensitivityAnalyzer:
    """Compute Meff and depth scale-comparator sensitivity."""

    config: SensitivityAnalysisConfig

    def read_cohort_by_sample(self, paths: SensitivityAnalysisPaths) -> pd.DataFrame:
        """Read cohort rows and require scale columns."""
        builder = MsaLevelTableBuilder(self.config)
        cohort = builder.read_cohort_by_sample(paths.cohort_xlsx_path)
        require_columns(cohort, ["meff_theta0p70", "n_sequences"], "cohort workbook")
        return cohort

    def variable_definitions(self) -> pd.DataFrame:
        """Return variable definitions."""
        return pd.DataFrame([
            {"variable": "primary_msa_snr", "variable_type": "row_coherence", "definition": "MSA-SNR used in the main analysis."},
            {"variable": "negative_pid_pair_std", "variable_type": "row_coherence_component", "definition": "Negative pairwise-identity standard deviation."},
            {"variable": "pid_pair_mean", "variable_type": "row_coherence_component", "definition": "Mean pairwise identity."},
            {"variable": "bits_site_min", "variable_type": "information_budget", "definition": "Minimum bits per site across retained MSA columns."},
            {"variable": "meff_theta0p70", "variable_type": "scale_comparator", "definition": "Meff at theta 0.70."},
            {"variable": "n_sequences", "variable_type": "scale_comparator", "definition": "Retained MSA depth."},
            {"variable": "max_plddt", "variable_type": "figure2_primary_readout", "definition": "Maximum mean pLDDT across generated candidates for an MSA."},
            {"variable": "max_tm", "variable_type": "figure2_primary_readout", "definition": "Maximum TM-score across generated candidates for an MSA."},
            {"variable": "max_novelty_plddt_score", "variable_type": "extended_novelty_frontier_readout", "definition": "Maximum candidate-level pLDDT multiplied by novelty fraction."},
            {"variable": "max_novelty_tm_score", "variable_type": "extended_novelty_frontier_readout", "definition": "Maximum candidate-level TM-score multiplied by novelty fraction."},
            {"variable": "max_novelty", "variable_type": "extended_novelty_frontier_readout", "definition": "Maximum generated-candidate novelty percentage."},
        ])

    def predictor_names(self) -> List[str]:
        """Return predictors for structural correlations."""
        return ["primary_msa_snr", "negative_pid_pair_std", "pid_pair_mean", "bits_site_min", "meff_theta0p70", "n_sequences"]

    def scale_correlation_names(self) -> List[str]:
        """Return variables for scale-correlation analysis."""
        return ["primary_msa_snr", "negative_pid_pair_std", "pid_pair_mean", "bits_site_min", "meff_theta0p70", "n_sequences"]

    def readout_names(self) -> List[str]:
        """Return structural readout names."""
        return ["max_plddt", "max_tm", "max_novelty_plddt_score", "max_novelty_tm_score", "max_novelty"]
    def readout_family(self, readout_name: str) -> str:
        """Return the readout family label."""
        if str(readout_name) in ["max_plddt", "max_tm"]:
            return "figure2_primary"
        return "extended_novelty_frontier"

    def group_specs(self, table: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """Return analysis groups."""
        groups: List[Tuple[str, pd.DataFrame]] = [("all", table)]
        for control_type, subset in table.groupby(table["control_type"].astype(str).str.lower(), sort=False):
            groups.append((str(control_type), subset))
        return groups

    def scale_correlations(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute correlations among scale and coherence variables."""
        variables = self.scale_correlation_names()
        rows: List[Dict[str, Any]] = []
        for group_name, subset in self.group_specs(table):
            for first_index in range(len(variables)):
                for second_index in range(first_index + 1, len(variables)):
                    variable_a = variables[first_index]
                    variable_b = variables[second_index]
                    rho, p_value, count = self.spearman(subset[variable_a], subset[variable_b])
                    rows.append({"analysis_group": group_name, "variable_a": variable_a, "variable_b": variable_b, "n_msa": count, "spearman_rho": rho, "p_value": p_value})
        return pd.DataFrame(rows)

    def structural_correlations(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute predictor-readout Spearman correlations."""
        rows: List[Dict[str, Any]] = []
        for group_name, subset in self.group_specs(table):
            for predictor in self.predictor_names():
                for readout in self.readout_names():
                    rho, p_value, count = self.spearman(subset[predictor], subset[readout])
                    rows.append({"analysis_group": group_name, "predictor": predictor, "readout": readout, "readout_family": self.readout_family(readout), "n_msa": count, "spearman_rho": rho, "p_value": p_value})
        output = pd.DataFrame(rows)
        return self.add_groupwise_q_values(output, "p_value", "q_value_within_analysis_group")

    def partial_spearman_sensitivity(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute rank-residual partial Spearman sensitivities."""
        focal_specs = [
            ("primary_msa_snr", ["meff_theta0p70", "n_sequences"]),
            ("negative_pid_pair_std", ["meff_theta0p70", "n_sequences"]),
            ("meff_theta0p70", ["primary_msa_snr", "n_sequences"]),
            ("n_sequences", ["primary_msa_snr", "meff_theta0p70"]),
        ]
        rows: List[Dict[str, Any]] = []
        for group_name, subset in self.group_specs(table):
            for focal_predictor, covariates in focal_specs:
                for readout in self.readout_names():
                    rho, p_value, count = self.partial_spearman(subset, focal_predictor, readout, covariates)
                    rows.append({"analysis_group": group_name, "readout": readout, "readout_family": self.readout_family(readout), "focal_predictor": focal_predictor, "covariates": "+".join(covariates), "n_msa": count, "partial_spearman_rho": rho, "p_value": p_value})
        output = pd.DataFrame(rows)
        return self.add_groupwise_q_values(output, "p_value", "q_value_within_analysis_group")

    def scale_stratified_snr(self, table: pd.DataFrame) -> pd.DataFrame:
        """Compute SNR associations inside scale strata."""
        rows: List[Dict[str, Any]] = []
        for group_name, group_table in self.group_specs(table):
            for stratifier in ["meff_theta0p70", "n_sequences"]:
                stratified = self.add_tertiles(group_table, stratifier)
                for stratum, subset in stratified.groupby("scale_stratum", sort=False):
                    for focal_predictor in ["primary_msa_snr", "negative_pid_pair_std"]:
                        for readout in self.readout_names():
                            rho, p_value, count = self.spearman(subset[focal_predictor], subset[readout])
                            rows.append({"analysis_group": group_name, "stratifier": stratifier, "stratum": str(stratum), "focal_predictor": focal_predictor, "readout": readout, "readout_family": self.readout_family(readout), "n_msa": count, "spearman_rho": rho, "p_value": p_value})
        return pd.DataFrame(rows)

    def threshold_scale_balance(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Summarize scale variables across threshold labels."""
        rows: List[Dict[str, Any]] = []
        variables = ["primary_msa_snr", "meff_theta0p70", "n_sequences", "bits_site_min"]
        builder = MsaLevelTableBuilder(self.config)
        group_columns = ["protein_name", "outer_label"]
        for threshold in self.config.threshold_grid:
            labeled = add_threshold_condition_columns(cohort, threshold=float(threshold))
            require_columns(labeled, group_columns + ["condition4"], "threshold-labeled cohort")
            condition_table = labeled.groupby(group_columns, sort=False).agg(
                condition4=("condition4", "first"),
                condition4_nunique=("condition4", "nunique"),
            ).reset_index()
            max_condition_count = int(condition_table["condition4_nunique"].max())
            if max_condition_count != 1:
                raise ValueError("Each protein_name plus outer_label group must map to exactly one condition4 value")
            condition_table = condition_table[group_columns + ["condition4"]]
            msa_table = builder.compute_msa_level_table(labeled)
            msa_table = msa_table.merge(condition_table, on=group_columns, how="left", validate="one_to_one")
            require_columns(msa_table, ["condition4"] + variables, "MSA-level threshold table")
            if msa_table["condition4"].isna().any():
                raise ValueError("Missing condition4 labels after MSA-level threshold merge")
            protein_summary = msa_table.groupby(["protein_name", "condition4"], sort=False)[variables].mean(numeric_only=True).reset_index()
            for condition_name, subset in protein_summary.groupby("condition4", sort=False):
                row: Dict[str, Any] = {
                    "threshold_bits_site_min": float(threshold),
                    "condition4": str(condition_name),
                    "n_proteins": int(subset["protein_name"].nunique()),
                }
                for variable in variables:
                    values = pd.to_numeric(subset[variable], errors="coerce").to_numpy(dtype=float)
                    finite = values[np.isfinite(values)]
                    row[f"mean_{variable}"] = safe_mean(finite)
                    row[f"median_{variable}"] = self.safe_median(finite)
                rows.append(row)
        return pd.DataFrame(rows)

    def summary_interpretation(self, partial_table: pd.DataFrame, threshold_balance: pd.DataFrame) -> pd.DataFrame:
        """Build compact summary values."""
        all_rows = partial_table[partial_table["analysis_group"] == "all"].copy()
        rows = [
            self.partial_summary_row(all_rows, "primary_msa_snr"),
            self.partial_summary_row(all_rows, "negative_pid_pair_std"),
            self.partial_summary_row(all_rows, "meff_theta0p70"),
            self.partial_summary_row(all_rows, "n_sequences"),
            {"summary_key": "threshold_grid_min", "summary_value": float(np.min(np.asarray(self.config.threshold_grid, dtype=float)))},
            {"summary_key": "threshold_grid_max", "summary_value": float(np.max(np.asarray(self.config.threshold_grid, dtype=float)))},
            {"summary_key": "threshold_balance_rows", "summary_value": int(threshold_balance.shape[0])},
        ]
        return pd.DataFrame(rows)

    def partial_summary_row(self, table: pd.DataFrame, focal_predictor: str) -> Dict[str, Any]:
        """Summarize partial Spearman rows for one focal predictor."""
        subset = table[table["focal_predictor"] == focal_predictor].copy()
        rho_values = pd.to_numeric(subset["partial_spearman_rho"], errors="coerce").to_numpy(dtype=float)
        p_values = pd.to_numeric(subset["p_value"], errors="coerce").to_numpy(dtype=float)
        q_values = pd.to_numeric(subset["q_value_within_analysis_group"], errors="coerce").to_numpy(dtype=float)
        summary_value = f"{int(np.sum(rho_values > 0.0))}/{int(np.sum(np.isfinite(rho_values)))} positive; {int(np.sum(p_values < 0.05))}/{int(np.sum(np.isfinite(p_values)))} p<0.05; {int(np.sum(q_values < 0.05))}/{int(np.sum(np.isfinite(q_values)))} q<0.05"
        return {"summary_key": f"{focal_predictor}_partial_positive_and_significant_counts", "summary_value": summary_value}

    def run_analysis(self, cohort: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run Meff/depth sensitivity analysis."""
        builder = MsaLevelTableBuilder(self.config)
        table = builder.compute_msa_level_table(cohort)
        require_columns(table, ["meff_theta0p70", "n_sequences"], "MSA-level scale table")
        scale_correlations = self.scale_correlations(table)
        structural_correlations = self.structural_correlations(table)
        partial = self.partial_spearman_sensitivity(table)
        stratified = self.scale_stratified_snr(table)
        balance = self.threshold_scale_balance(cohort)
        return {
            "readme": self.readme_table(int(cohort.shape[0]), int(table.shape[0])),
            "variable_definitions": self.variable_definitions(),
            "msa_level_scale_table": table,
            "scale_correlations": scale_correlations,
            "structural_correlations": structural_correlations,
            "partial_spearman_sensitivity": partial,
            "scale_stratified_snr": stratified,
            "threshold_scale_balance": balance,
            "summary_interpretation": self.summary_interpretation(partial, balance),
        }

    def add_tertiles(self, table: pd.DataFrame, variable: str) -> pd.DataFrame:
        """Add tertile labels for one variable."""
        out = table.copy()
        values = pd.to_numeric(out[variable], errors="coerce")
        out["scale_stratum"] = pd.qcut(values.rank(method="first"), q=3, labels=["low", "middle", "high"], duplicates="drop")
        return out

    def partial_spearman(self, table: pd.DataFrame, focal_predictor: str, readout: str, covariates: Sequence[str]) -> Tuple[float, float, int]:
        """Compute rank-residual partial Spearman correlation."""
        columns = [focal_predictor, readout] + list(covariates)
        data = table[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        count = int(data.shape[0])
        if count < 10:
            return float("nan"), float("nan"), count
        ranked = data.rank(method="average")
        covariate_matrix = ranked[list(covariates)].to_numpy(dtype=float)
        design = np.column_stack([np.ones(count, dtype=float), covariate_matrix])
        focal_residuals = self.regression_residuals(ranked[focal_predictor].to_numpy(dtype=float), design)
        readout_residuals = self.regression_residuals(ranked[readout].to_numpy(dtype=float), design)
        result = stats.pearsonr(focal_residuals, readout_residuals)
        return float(result.statistic), float(result.pvalue), count

    def regression_residuals(self, y_values: np.ndarray, design: np.ndarray) -> np.ndarray:
        """Return least-squares residuals."""
        coefficients = np.linalg.lstsq(design, y_values, rcond=None)[0]
        return y_values - design @ coefficients

    def spearman(self, x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, float, int]:
        """Compute Spearman correlation."""
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        valid = np.isfinite(x_array) & np.isfinite(y_array)
        count = int(valid.sum())
        if count < 10:
            return float("nan"), float("nan"), count
        result = stats.spearmanr(x_array[valid], y_array[valid])
        return float(result.statistic), float(result.pvalue), count

    def safe_median(self, values: Sequence[float]) -> float:
        """Return finite median or NaN."""
        array = np.asarray(values, dtype=float)
        finite = array[np.isfinite(array)]
        return float(np.median(finite)) if finite.size else float("nan")

    def add_groupwise_q_values(self, table: pd.DataFrame, p_value_column: str, q_value_column: str) -> pd.DataFrame:
        """Add BH q-values within each analysis group."""
        output = table.copy()
        output[q_value_column] = np.nan
        output[f"{q_value_column}_correction_family"] = ""
        output[f"n_tests_{q_value_column}"] = np.nan
        for group_name in sorted(output["analysis_group"].dropna().astype(str).unique().tolist()):
            mask = output["analysis_group"].astype(str) == group_name
            p_values = pd.to_numeric(output.loc[mask, p_value_column], errors="coerce").to_numpy(dtype=float)
            output.loc[mask, q_value_column] = benjamini_hochberg(p_values)
            output.loc[mask, f"{q_value_column}_correction_family"] = f"{group_name}_analysis_group"
            output.loc[mask, f"n_tests_{q_value_column}"] = int(mask.sum())
        return output

    def readme_table(self, cohort_rows: int, msa_rows: int) -> pd.DataFrame:
        """Build workbook readme."""
        return pd.DataFrame([
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "input_workbook", "value": str(self.config.cohort_xlsx_name)},
            {"key": "input_sheet", "value": str(self.config.cohort_sheet_name)},
            {"key": "cohort_rows_loaded", "value": str(int(cohort_rows))},
            {"key": "msa_rows", "value": str(int(msa_rows))},
            {"key": "analysis_name", "value": "meff_depth_sensitivity"},
            {"key": "msa_unit", "value": "protein_name plus outer_label"},
            {"key": "scale_variables", "value": "meff_theta0p70 and n_sequences"},
            {"key": "partial_spearman_method", "value": "Rank-transform focal, readout, and covariates; regress focal and readout ranks on covariate ranks; correlate residuals."},
            {"key": "threshold_grid", "value": ",".join([f"{value:.6g}" for value in self.config.threshold_grid])},
            {"key": "summary_interpretation", "value": "Summarizes partial_spearman_sensitivity rows only."},
            {"key": "readout_family", "value": "max_plddt and max_tm are figure2_primary; max_novelty_plddt_score, max_novelty_tm_score, and max_novelty are extended_novelty_frontier."},
        ])


@dataclass
class SensitivityAnalysisRunner:
    """Coordinate sensitivity-analysis notebooks."""

    config: SensitivityAnalysisConfig

    def run(self) -> None:
        """Write all sensitivity notebooks."""
        threshold_values = np.asarray(self.config.threshold_grid, dtype=float)
        primary_matches = np.isclose(threshold_values, float(self.config.primary_threshold), rtol=0.0, atol=1e-12)
        if int(np.sum(primary_matches)) != 1:
            raise ValueError("threshold_grid must contain primary_threshold exactly once")
        paths = SensitivityAnalysisPaths.from_config(self.config)
        paths.ensure_dirs()
        self.run_paired_sensitivity()
        self.run_threshold_boundary(paths)
        self.run_msa_snr_metric(paths)
        self.run_meff_depth(paths)

    def run_paired_sensitivity(self) -> None:
        """Run the paired-overlap notebook."""
        paired_config = PairedSensitivityConfig(
            cohort_xlsx_name=str(self.config.cohort_xlsx_name),
            cohort_sheet_name=str(self.config.cohort_sheet_name),
            output_dir_name=str(self.config.output_dir_name),
            output_xlsx_name=str(self.config.paired_output_xlsx_name),
            bits_site_min_threshold=float(self.config.primary_threshold),
            expected_overlap_protein_count=int(self.config.expected_overlap_protein_count),
            excel_chunk_rows=int(self.config.excel_chunk_rows),
        )
        PairedSensitivityRunner(paired_config).run()

    def run_threshold_boundary(self, paths: SensitivityAnalysisPaths) -> None:
        """Run threshold-boundary notebook."""
        analyzer = ThresholdBoundarySensitivityAnalyzer(self.config)
        cohort = analyzer.read_cohort_by_sample(paths)
        tables = analyzer.run_analysis(cohort)
        writer = ExcelNotebookWriter(paths.threshold_output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        print(f"Wrote threshold-boundary sensitivity analysis to {paths.threshold_output_xlsx_path}")

    def run_msa_snr_metric(self, paths: SensitivityAnalysisPaths) -> None:
        """Run MSA-SNR metric notebook."""
        builder = MsaLevelTableBuilder(self.config)
        analyzer = MsaSnrMetricSensitivityAnalyzer(self.config)
        cohort = builder.read_cohort_by_sample(paths.cohort_xlsx_path)
        tables = analyzer.run_analysis(cohort)
        writer = ExcelNotebookWriter(paths.msa_snr_output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        print(f"Wrote MSA-SNR metric sensitivity analysis to {paths.msa_snr_output_xlsx_path}")

    def run_meff_depth(self, paths: SensitivityAnalysisPaths) -> None:
        """Run Meff/depth sensitivity notebook."""
        analyzer = MeffDepthSensitivityAnalyzer(self.config)
        cohort = analyzer.read_cohort_by_sample(paths)
        tables = analyzer.run_analysis(cohort)
        writer = ExcelNotebookWriter(paths.meff_depth_output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        print(f"Wrote Meff/depth sensitivity analysis to {paths.meff_depth_output_xlsx_path}")


def parse_float_list(text_value: str) -> Tuple[float, ...]:
    """Parse a comma-separated float list."""
    values: List[float] = []
    for token in str(text_value).replace(";", ",").split(","):
        stripped = token.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise ValueError("At least one threshold value is required")
    return tuple(values)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run sensitivity analyses for MSA information-regime analyses.")
    parser.add_argument("--cohort_xlsx", type=str, default="cohort.xlsx")
    parser.add_argument("--sheet", type=str, default="by_sample")
    parser.add_argument("--threshold", type=float, default=operational_bits_site_threshold)
    parser.add_argument("--thresholds", type=str, default="0.80,0.90,0.9720939469655248,1.00,1.10,1.20")
    parser.add_argument("--output_dir", type=str, default="sensitivity_analyses")
    parser.add_argument("--output_xlsx", type=str, default="paired_sensitivity_analysis.xlsx")
    parser.add_argument("--threshold_output_xlsx", type=str, default="threshold_boundary_sensitivity.xlsx")
    parser.add_argument("--msa_snr_output_xlsx", type=str, default="msa_snr_metric_sensitivity.xlsx")
    parser.add_argument("--meff_depth_output_xlsx", type=str, default="meff_depth_sensitivity.xlsx")
    parser.add_argument("--expected_overlap_proteins", type=int, default=20)
    parser.add_argument("--softmax_temperature", type=float, default=1.0)
    parser.add_argument("--protein_block_subsample_replicates", type=int, default=1000)
    parser.add_argument("--protein_block_subsample_fraction", type=float, default=0.80)
    parser.add_argument("--protein_block_subsample_seed", type=int, default=17)
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> SensitivityAnalysisConfig:
    """Build configuration from command-line arguments."""
    return SensitivityAnalysisConfig(
        cohort_xlsx_name=str(args.cohort_xlsx),
        cohort_sheet_name=str(args.sheet),
        output_dir_name=str(args.output_dir),
        paired_output_xlsx_name=str(args.output_xlsx),
        threshold_output_xlsx_name=str(args.threshold_output_xlsx),
        msa_snr_output_xlsx_name=str(args.msa_snr_output_xlsx),
        meff_depth_output_xlsx_name=str(args.meff_depth_output_xlsx),
        primary_threshold=float(args.threshold),
        threshold_grid=parse_float_list(str(args.thresholds)),
        expected_overlap_protein_count=int(args.expected_overlap_proteins),
        softmax_temperature=float(args.softmax_temperature),
        protein_block_subsample_replicates=int(args.protein_block_subsample_replicates),
        protein_block_subsample_fraction=float(args.protein_block_subsample_fraction),
        protein_block_subsample_seed=int(args.protein_block_subsample_seed),
    )


def main() -> None:
    """Run all sensitivity-analysis notebooks."""
    args = parse_args()
    config = build_config_from_args(args)
    runner = SensitivityAnalysisRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
