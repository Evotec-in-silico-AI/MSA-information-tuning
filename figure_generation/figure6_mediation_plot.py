import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GroupKFold, KFold

from figure_common import ConditionStyle
from figure_common import operational_bits_site_threshold
from figure_common import ExcelNotebookWriter
from figure_common import add_threshold_condition_columns
from figure_common import convert_archived_columns
from figure_common import require_columns
from figure_common import save_figure_pair



@dataclass(frozen=True)
class Figure6MediationConfig:
    """Configure Figure 6 and Supplementary Figure 4 mediation analysis."""

    cohort_xlsx_name: str = "cohort.xlsx"
    cohort_sheet_name: str = "by_sample"
    output_dir_name: str = "figure6_and_supplementary_figure4_outputs"
    output_xlsx_name: str = "figure6_and_supplementary_figure4_mediation_source_data.xlsx"
    excel_chunk_rows: int = 1000000
    bits_site_min_threshold: float = operational_bits_site_threshold
    source_variable: str = "bits_site_median"
    source_regime_variable: str = "bits_site_min"
    protein_name_column: str = "protein_name"
    outer_label_column: str = "outer_label"
    control_type_column: str = "control_type"
    sample_id_column: str = "sample_id"
    terminal_response_columns: Tuple[str, ...] = ("plddt_median", "pae_median")
    msa_mediator_columns: Tuple[str, ...] = (
        "mi_apc_topq_mean",
        "msa_effective_alphabet_mean",
        "msa_graph_apl_lcc",
        "msa_graph_articulation_fraction",
        "msa_graph_bridge_fraction",
        "msa_graph_edge_density",
        "msa_graph_laplacian_lambda2",
        "msa_graph_lcc_fraction",
    )
    latent_mediator_columns: Tuple[str, ...] = (
        "lat_effective_rank",
        "lat_hubness_std",
        "lat_knn_clustering_mean",
        "lat_knn_degree_gini",
        "lat_knn_laplacian_lambda2",
        "lat_knn_radius_k5_median",
        "lat_sphericity_pr_norm",
    )
    analysis_regimes: Tuple[str, ...] = ("low_target", "high_target")
    min_abs_serial_indirect: float = 0.005
    require_ci_exclusion_for_keep: bool = True
    required_cv_pass_folds: int = 4
    cv_folds: int = 5
    bootstrap_replicates: int = 5000
    bootstrap_seed: int = 42


@dataclass(frozen=True)
class Figure6MediationPaths:
    """Resolve input and output paths."""

    root_dir: Path
    cohort_xlsx_path: Path
    output_dir: Path
    output_xlsx_path: Path

    @classmethod
    def from_config(cls, config: Figure6MediationConfig) -> "Figure6MediationPaths":
        """Build paths from the current working directory."""
        root_dir = Path(".").resolve()
        output_dir = root_dir / str(config.output_dir_name)
        return cls(
            root_dir=root_dir,
            cohort_xlsx_path=root_dir / str(config.cohort_xlsx_name),
            output_dir=output_dir,
            output_xlsx_path=output_dir / str(config.output_xlsx_name),
        )

    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Figure6MediationStyle:
    """Provide mediation plot labels and colors."""

    condition_style: ConditionStyle

    def apply_publication_style(self) -> None:
        """Apply publication style defaults."""
        sns.set_style("white")
        mpl.rcParams.update(
            {
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
                "lines.linewidth": 1.8,
                "lines.markersize": 4.0,
                "savefig.dpi": 300,
                "figure.dpi": 120,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def apply_axis_style(self, axis: plt.Axes) -> None:
        """Apply axis styling."""
        axis.grid(False)
        axis.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False)
        axis.xaxis.set_ticks_position("bottom")
        axis.yaxis.set_ticks_position("left")
        sns.despine(ax=axis, top=True, right=True)

    def get_regime_color(self, regime: str) -> str:
        """Return color for a mediation regime."""
        condition_by_regime = {
            "low_target": "below targeted",
            "high_target": "above targeted",
        }
        return self.condition_style.get_color(condition_by_regime[str(regime)])

    def get_regime_display_label(self, regime: str) -> str:
        """Return display label for a mediation regime."""
        display_by_regime = {
            "low_target": "Low information · targeted",
            "high_target": "High information · targeted",
        }
        return display_by_regime.get(str(regime), str(regime))

    def get_regime_output_stem(self, regime: str) -> str:
        """Return output stem for manuscript target mediation panels."""
        stems_by_regime = {
            "low_target": "supplementary_figure4a_left_low_target_mediation_ranked_full_chain_effects",
            "high_target": "supplementary_figure4a_right_high_target_mediation_ranked_full_chain_effects",
        }
        return stems_by_regime[str(regime)]


class Figure6MediationLabels:
    """Provide raw-to-display label mappings."""

    def get_msa_label_map(self) -> Dict[str, str]:
        """Return MSA mediator labels."""
        return {
            "mi_apc_topq_mean": "Covariation strength (MI-APC)",
            "msa_effective_alphabet_mean": "MSA effective alphabet",
            "msa_graph_apl_lcc": "MSA graph avg path length",
            "msa_graph_articulation_fraction": "MSA articulation fraction",
            "msa_graph_bridge_fraction": "MSA bridge fraction",
            "msa_graph_edge_density": "MSA graph density",
            "msa_graph_laplacian_lambda2": "MSA graph connectivity (λ2)",
            "msa_graph_lcc_fraction": "MSA graph connectedness",
        }

    def get_latent_label_map(self) -> Dict[str, str]:
        """Return latent mediator labels."""
        return {
            "lat_effective_rank": "Latent effective rank",
            "lat_hubness_std": "Latent hubness",
            "lat_knn_clustering_mean": "Latent clustering",
            "lat_knn_degree_gini": "Latent degree inequality",
            "lat_knn_laplacian_lambda2": "Latent connectivity (λ2)",
            "lat_knn_radius_k5_median": "Latent local radius",
            "lat_sphericity_pr_norm": "Latent sphericity",
        }

    def get_endpoint_label_map(self) -> Dict[str, str]:
        """Return endpoint labels."""
        return {
            "pae_median": "Chai-1 PAE",
            "plddt_median": "Chai-1 pLDDT",
        }

    def get_msa_label(self, value: str) -> str:
        """Return display label for one MSA mediator."""
        return self.get_msa_label_map().get(str(value), str(value))

    def get_latent_label(self, value: str) -> str:
        """Return display label for one latent mediator."""
        return self.get_latent_label_map().get(str(value), str(value))

    def get_endpoint_label(self, value: str) -> str:
        """Return display label for one endpoint."""
        return self.get_endpoint_label_map().get(str(value), str(value))

    def build_path_label(self, msa_mediator: str, latent_mediator: str, terminal_response: str) -> str:
        """Build display label for one serial path."""
        return f"{self.get_msa_label(msa_mediator)} → {self.get_latent_label(latent_mediator)} → {self.get_endpoint_label(terminal_response)}"

    def build_label_dictionary(self) -> pd.DataFrame:
        """Build mediator and endpoint label dictionary."""
        rows: List[Dict[str, Any]] = []
        for raw_name, display_name in self.get_msa_label_map().items():
            rows.append({"category": "msa_mediator", "raw_name": raw_name, "display_name": display_name})
        for raw_name, display_name in self.get_latent_label_map().items():
            rows.append({"category": "latent_mediator", "raw_name": raw_name, "display_name": display_name})
        for raw_name, display_name in self.get_endpoint_label_map().items():
            rows.append({"category": "terminal_response", "raw_name": raw_name, "display_name": display_name})
        return pd.DataFrame(rows)


@dataclass
class Figure6MediationDataBuilder:
    """Build mediation input and block tables."""

    config: Figure6MediationConfig

    def read_cohort(self, paths: Figure6MediationPaths) -> pd.DataFrame:
        """Read and validate the cohort table."""
        if not paths.cohort_xlsx_path.exists():
            raise FileNotFoundError(f"Missing cohort workbook: {paths.cohort_xlsx_path}")
        cohort = pd.read_excel(paths.cohort_xlsx_path, sheet_name=str(self.config.cohort_sheet_name), engine="openpyxl")
        required_columns = [
            self.config.sample_id_column,
            self.config.protein_name_column,
            self.config.outer_label_column,
            self.config.control_type_column,
            self.config.source_regime_variable,
            self.config.source_variable,
        ]
        required_columns.extend(list(self.config.terminal_response_columns))
        required_columns.extend(list(self.config.msa_mediator_columns))
        required_columns.extend(list(self.config.latent_mediator_columns))
        require_columns(cohort, list(dict.fromkeys(required_columns)), "cohort workbook")
        if cohort[self.config.sample_id_column].astype(str).nunique() != int(cohort.shape[0]):
            raise ValueError("The cohort by-sample sheet has non-unique sample_id values")
        output = cohort.copy()
        for column_name in [self.config.sample_id_column, self.config.protein_name_column, self.config.outer_label_column, self.config.control_type_column]:
            output[column_name] = output[column_name].astype(str)
        numeric_columns = [self.config.source_regime_variable, self.config.source_variable]
        numeric_columns.extend(list(self.config.terminal_response_columns))
        numeric_columns.extend(list(self.config.msa_mediator_columns))
        numeric_columns.extend(list(self.config.latent_mediator_columns))
        for column_name in list(dict.fromkeys(numeric_columns)):
            output[column_name] = pd.to_numeric(output[column_name], errors="coerce")
        information_columns = tuple(column_name for column_name in (self.config.source_regime_variable, self.config.source_variable) if column_name in output.columns and ("bits_site" in column_name or "bits_per_site" in column_name))
        effective_columns = tuple(column_name for column_name in self.config.msa_mediator_columns if column_name == "msa_effective_alphabet_mean")
        return convert_archived_columns(output, information_columns=information_columns, effective_alphabet_columns=effective_columns)

    def add_regime_columns(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Add target-only mediation regime labels."""
        labeled = add_threshold_condition_columns(cohort, threshold=float(self.config.bits_site_min_threshold))
        regime_by_condition = {
            "below targeted": "low_target",
            "above targeted": "high_target",
            "below matched control": "low_control",
            "above matched control": "high_control",
        }
        labeled["regime"] = labeled["condition4"].map(regime_by_condition).astype(str)
        return labeled

    def build_input_table(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Build sample-level mediation input table."""
        selected_columns = [
            self.config.sample_id_column,
            self.config.protein_name_column,
            self.config.control_type_column,
            self.config.outer_label_column,
            "outer_label_base",
            "target_bits_site_min",
            "target_stratum",
            "condition4",
            "regime",
            self.config.source_regime_variable,
            self.config.source_variable,
        ]
        selected_columns.extend(list(self.config.terminal_response_columns))
        selected_columns.extend(list(self.config.msa_mediator_columns))
        selected_columns.extend(list(self.config.latent_mediator_columns))
        selected_columns = list(dict.fromkeys(selected_columns))
        output = cohort[selected_columns].copy()
        numeric_columns = ["target_bits_site_min", self.config.source_regime_variable, self.config.source_variable]
        numeric_columns.extend(list(self.config.terminal_response_columns))
        numeric_columns.extend(list(self.config.msa_mediator_columns))
        numeric_columns.extend(list(self.config.latent_mediator_columns))
        for column_name in list(dict.fromkeys(numeric_columns)):
            output[column_name] = pd.to_numeric(output[column_name], errors="coerce")
        output = output[output["regime"].isin(list(self.config.analysis_regimes))].copy()
        return output.reset_index(drop=True)

    def build_block_table(self, mediation_input: pd.DataFrame) -> pd.DataFrame:
        """Build block-level table with median aggregation."""
        group_columns = ["regime", self.config.protein_name_column, "outer_label_base"]
        invariant_columns = [self.config.control_type_column, "target_stratum", "condition4", "target_bits_site_min"]
        numeric_columns = [self.config.source_regime_variable, self.config.source_variable]
        numeric_columns.extend(list(self.config.terminal_response_columns))
        numeric_columns.extend(list(self.config.msa_mediator_columns))
        numeric_columns.extend(list(self.config.latent_mediator_columns))
        block_records: List[Dict[str, Any]] = []
        for group_key, block in mediation_input.groupby(group_columns, sort=False):
            first_row = block.iloc[0]
            for column_name in invariant_columns:
                unique_values = pd.Index(block[column_name].dropna()).unique().tolist()
                if len(unique_values) > 1:
                    raise ValueError(f"Block invariant violated for {column_name}: {group_key} has {unique_values}")
            record: Dict[str, Any] = {
                "regime": str(first_row["regime"]),
                self.config.protein_name_column: str(first_row[self.config.protein_name_column]),
                "outer_label_base": str(first_row["outer_label_base"]),
                self.config.control_type_column: str(first_row[self.config.control_type_column]),
                "condition4": str(first_row["condition4"]),
                "target_stratum": str(first_row["target_stratum"]),
                "target_bits_site_min": float(first_row["target_bits_site_min"]) if pd.notna(first_row["target_bits_site_min"]) else float("nan"),
                "block_n_samples": int(block.shape[0]),
            }
            for column_name in list(dict.fromkeys(numeric_columns)):
                values = pd.to_numeric(block[column_name], errors="coerce")
                record[column_name] = float(values.median()) if values.notna().any() else float("nan")
            block_records.append(record)
        output = pd.DataFrame(block_records)
        ordered_columns = [
            "regime",
            self.config.protein_name_column,
            "outer_label_base",
            self.config.control_type_column,
            "condition4",
            "target_stratum",
            "target_bits_site_min",
            "block_n_samples",
            self.config.source_regime_variable,
            self.config.source_variable,
        ]
        ordered_columns.extend(list(self.config.terminal_response_columns))
        ordered_columns.extend(list(self.config.msa_mediator_columns))
        ordered_columns.extend(list(self.config.latent_mediator_columns))
        return output[[column_name for column_name in ordered_columns if column_name in output.columns]].copy()


@dataclass
class Figure6MediationAnalyzer:
    """Run serial mediation-style pathway screens."""

    config: Figure6MediationConfig
    labels: Figure6MediationLabels

    def get_full_chain_columns(self) -> List[str]:
        """Return full-chain result columns."""
        return [
            "regime",
            "regime_display",
            "source_variable",
            "msa_mediator",
            "msa_mediator_display",
            "latent_mediator",
            "latent_mediator_display",
            "terminal_response",
            "terminal_response_display",
            "path_label",
            "n",
            "serial_indirect",
            "ci_serial_lo",
            "ci_serial_hi",
            "cv_pass_folds",
            "cv_total_folds",
            "abs_serial_indirect",
            "is_viable_full_chain",
        ]

    def run_full_chain_screen(self, block_table: pd.DataFrame) -> pd.DataFrame:
        """Evaluate all configured full-chain paths."""
        records: List[Dict[str, Any]] = []
        for regime in self.config.analysis_regimes:
            regime_table = block_table[block_table["regime"] == str(regime)].copy()
            if regime_table.empty:
                continue
            for msa_mediator in self.config.msa_mediator_columns:
                for latent_mediator in self.config.latent_mediator_columns:
                    for terminal_response in self.config.terminal_response_columns:
                        result = self.evaluate_two_mediator_chain(regime_table, self.config.source_variable, msa_mediator, latent_mediator, terminal_response)
                        serial_value = pd.to_numeric(pd.Series([result["serial_indirect"]]), errors="coerce").iloc[0]
                        record = {
                            "regime": str(regime),
                            "source_variable": str(self.config.source_variable),
                            "msa_mediator": str(msa_mediator),
                            "latent_mediator": str(latent_mediator),
                            "terminal_response": str(terminal_response),
                            **result,
                            "abs_serial_indirect": abs(float(serial_value)) if np.isfinite(serial_value) else float("nan"),
                        }
                        record["is_viable_full_chain"] = bool(self.row_passes_keep_rule(pd.Series(record)))
                        records.append(record)
        if not records:
            return pd.DataFrame(columns=self.get_full_chain_columns())
        output = pd.DataFrame(records)
        output = output.sort_values(
            ["regime", "terminal_response", "abs_serial_indirect", "msa_mediator", "latent_mediator"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return self.add_display_columns(output)

    def add_display_columns(self, results: pd.DataFrame) -> pd.DataFrame:
        """Add display columns to full-chain results."""
        output = results.copy()
        regime_display_map = {
            "low_target": "Low information · targeted",
            "high_target": "High information · targeted",
        }
        output["regime_display"] = output["regime"].map(regime_display_map).fillna(output["regime"])
        output["msa_mediator_display"] = output["msa_mediator"].map(self.labels.get_msa_label)
        output["latent_mediator_display"] = output["latent_mediator"].map(self.labels.get_latent_label)
        output["terminal_response_display"] = output["terminal_response"].map(self.labels.get_endpoint_label)
        output["path_label"] = output.apply(lambda row: self.labels.build_path_label(str(row["msa_mediator"]), str(row["latent_mediator"]), str(row["terminal_response"])), axis=1)
        return output[self.get_full_chain_columns()]

    def filter_viable_paths(self, full_chain_tested: pd.DataFrame) -> pd.DataFrame:
        """Return viable full-chain paths."""
        if full_chain_tested.empty:
            return pd.DataFrame(columns=self.get_full_chain_columns())
        output = full_chain_tested[full_chain_tested["is_viable_full_chain"].astype(bool)].copy()
        if output.empty:
            return pd.DataFrame(columns=self.get_full_chain_columns())
        output = output.sort_values(
            ["regime", "terminal_response", "abs_serial_indirect", "msa_mediator", "latent_mediator"],
            ascending=[True, True, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        return output[self.get_full_chain_columns()]

    def build_regime_endpoint_summary(self, viable_paths: pd.DataFrame) -> pd.DataFrame:
        """Build summary by regime and endpoint."""
        rows: List[Dict[str, Any]] = []
        for regime in self.config.analysis_regimes:
            for endpoint in self.config.terminal_response_columns:
                sub = viable_paths[(viable_paths["regime"] == str(regime)) & (viable_paths["terminal_response"] == str(endpoint))].copy()
                if sub.empty:
                    rows.append(
                        {
                            "regime": str(regime),
                            "regime_display": self.get_regime_display(regime),
                            "terminal_response": str(endpoint),
                            "terminal_response_display": self.labels.get_endpoint_label(endpoint),
                            "n_viable_full_chain_paths": 0,
                            "top_path_label": "",
                            "top_serial_indirect": float("nan"),
                            "top_ci_serial_lo": float("nan"),
                            "top_ci_serial_hi": float("nan"),
                            "summary_flag": "no_viable_path",
                        }
                    )
                    continue
                sub = sub.sort_values(["abs_serial_indirect", "msa_mediator", "latent_mediator"], ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
                top_row = sub.iloc[0]
                rows.append(
                    {
                        "regime": str(regime),
                        "regime_display": self.get_regime_display(regime),
                        "terminal_response": str(endpoint),
                        "terminal_response_display": self.labels.get_endpoint_label(endpoint),
                        "n_viable_full_chain_paths": int(sub.shape[0]),
                        "top_path_label": str(top_row["path_label"]),
                        "top_serial_indirect": float(top_row["serial_indirect"]),
                        "top_ci_serial_lo": float(top_row["ci_serial_lo"]),
                        "top_ci_serial_hi": float(top_row["ci_serial_hi"]),
                        "summary_flag": "viable_pathways_present",
                    }
                )
        return pd.DataFrame(rows)

    def get_regime_display(self, regime: str) -> str:
        """Return display label for regime."""
        return {"low_target": "Low information · targeted", "high_target": "High information · targeted"}.get(str(regime), str(regime))

    def row_passes_keep_rule(self, row: pd.Series) -> bool:
        """Return whether one pathway passes the viability filter."""
        effect_value = pd.to_numeric(pd.Series([row.get("serial_indirect", np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(effect_value):
            return False
        effect_pass = abs(float(effect_value)) >= float(self.config.min_abs_serial_indirect)
        ci_lo = pd.to_numeric(pd.Series([row.get("ci_serial_lo", np.nan)]), errors="coerce").iloc[0]
        ci_hi = pd.to_numeric(pd.Series([row.get("ci_serial_hi", np.nan)]), errors="coerce").iloc[0]
        if bool(self.config.require_ci_exclusion_for_keep):
            if not np.isfinite(ci_lo) or not np.isfinite(ci_hi):
                return False
            ci_pass = bool((float(ci_lo) > 0.0 and float(ci_hi) > 0.0) or (float(ci_lo) < 0.0 and float(ci_hi) < 0.0))
        else:
            ci_pass = True
        cv_pass_folds = int(pd.to_numeric(pd.Series([row.get("cv_pass_folds", 0)]), errors="coerce").fillna(0).iloc[0])
        cv_total_folds = int(pd.to_numeric(pd.Series([row.get("cv_total_folds", 0)]), errors="coerce").fillna(0).iloc[0])
        cv_pass = cv_total_folds > 0 and cv_pass_folds >= int(self.config.required_cv_pass_folds)
        return bool(effect_pass and ci_pass and cv_pass)

    def evaluate_two_mediator_chain(self, frame: pd.DataFrame, source_column: str, mediator_one_column: str, mediator_two_column: str, response_column: str) -> Dict[str, Any]:
        """Evaluate one two-mediator serial chain."""
        required_columns = [source_column, mediator_one_column, mediator_two_column, response_column, self.config.protein_name_column]
        analysis_frame = self.prepare_analysis_frame(frame, required_columns)
        row_count = int(analysis_frame.shape[0])
        if row_count < 10:
            return self.empty_result(row_count)
        point_estimate = self.compute_two_mediator_point_estimate(analysis_frame, source_column, mediator_one_column, mediator_two_column, response_column)
        ci_lo, ci_hi = self.bootstrap_two_mediator_ci(analysis_frame, source_column, mediator_one_column, mediator_two_column, response_column)
        cv_pass_folds, cv_total_folds = self.compute_cv_passes(analysis_frame, source_column, mediator_one_column, mediator_two_column, response_column, point_estimate)
        return {
            "n": row_count,
            "serial_indirect": point_estimate,
            "ci_serial_lo": ci_lo,
            "ci_serial_hi": ci_hi,
            "cv_pass_folds": cv_pass_folds,
            "cv_total_folds": cv_total_folds,
        }

    def prepare_analysis_frame(self, frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Return finite-only analysis frame."""
        output = frame[list(columns)].copy()
        for column_name in columns:
            if column_name == self.config.protein_name_column:
                output[column_name] = output[column_name].astype(str)
            else:
                output[column_name] = pd.to_numeric(output[column_name], errors="coerce")
        finite_mask = np.ones((output.shape[0],), dtype=bool)
        for column_name in columns:
            if column_name == self.config.protein_name_column:
                continue
            finite_mask &= np.isfinite(output[column_name].to_numpy(dtype=float, copy=False))
        return output.loc[finite_mask].copy().reset_index(drop=True)

    def empty_result(self, row_count: int) -> Dict[str, Any]:
        """Return empty result payload."""
        return {
            "n": int(row_count),
            "serial_indirect": float("nan"),
            "ci_serial_lo": float("nan"),
            "ci_serial_hi": float("nan"),
            "cv_pass_folds": 0,
            "cv_total_folds": 0,
        }

    def compute_two_mediator_point_estimate(self, frame: pd.DataFrame, source_column: str, mediator_one_column: str, mediator_two_column: str, response_column: str) -> float:
        """Compute standardized serial indirect effect."""
        source_to_msa = self.fit_standardized_beta(frame, [source_column], mediator_one_column, source_column)
        msa_to_latent = self.fit_standardized_beta(frame, [source_column, mediator_one_column], mediator_two_column, mediator_one_column)
        latent_to_response = self.fit_standardized_beta(frame, [source_column, mediator_one_column, mediator_two_column], response_column, mediator_two_column)
        if not np.isfinite(source_to_msa) or not np.isfinite(msa_to_latent) or not np.isfinite(latent_to_response):
            return float("nan")
        return float(source_to_msa * msa_to_latent * latent_to_response)

    def fit_standardized_beta(self, frame: pd.DataFrame, predictor_columns: Sequence[str], response_column: str, target_predictor: str) -> float:
        """Fit standardized linear model and return target coefficient."""
        x_matrix = frame[list(predictor_columns)].to_numpy(dtype=float, copy=False)
        y_vector = frame[response_column].to_numpy(dtype=float, copy=False)
        x_z = self.zscore_matrix(x_matrix)
        y_z = self.zscore_vector(y_vector)
        if x_z is None or y_z is None:
            return float("nan")
        design_matrix = np.column_stack([np.ones((x_z.shape[0],), dtype=float), x_z])
        try:
            beta_values = np.linalg.lstsq(design_matrix, y_z, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float("nan")
        target_index = list(predictor_columns).index(str(target_predictor)) + 1
        if target_index >= beta_values.size:
            return float("nan")
        return float(beta_values[target_index])

    def zscore_matrix(self, x_matrix: np.ndarray) -> Optional[np.ndarray]:
        """Return z-scored matrix or None for invalid predictors."""
        means = np.mean(x_matrix, axis=0)
        stds = np.std(x_matrix, axis=0, ddof=0)
        if np.any(~np.isfinite(stds)) or np.any(stds <= 0.0):
            return None
        return (x_matrix - means) / stds

    def zscore_vector(self, y_vector: np.ndarray) -> Optional[np.ndarray]:
        """Return z-scored vector or None for invalid response."""
        mean_value = float(np.mean(y_vector))
        std_value = float(np.std(y_vector, ddof=0))
        if not np.isfinite(std_value) or std_value <= 0.0:
            return None
        return (y_vector - mean_value) / std_value

    def bootstrap_two_mediator_ci(self, frame: pd.DataFrame, source_column: str, mediator_one_column: str, mediator_two_column: str, response_column: str) -> Tuple[float, float]:
        """Compute bootstrap percentile interval."""
        values = self.bootstrap_effect_values(frame, source_column, mediator_one_column, mediator_two_column, response_column)
        return self.quantile_ci(values)

    def bootstrap_effect_values(self, frame: pd.DataFrame, source_column: str, mediator_one_column: str, mediator_two_column: str, response_column: str) -> np.ndarray:
        """Bootstrap effect values over rows."""
        rng = np.random.default_rng(int(self.config.bootstrap_seed))
        row_count = int(frame.shape[0])
        values = np.full((int(self.config.bootstrap_replicates),), np.nan, dtype=float)
        for replicate_index in range(int(self.config.bootstrap_replicates)):
            sample_indices = rng.integers(0, row_count, size=row_count, endpoint=False)
            sample_frame = frame.iloc[sample_indices].reset_index(drop=True)
            values[replicate_index] = self.compute_two_mediator_point_estimate(sample_frame, source_column, mediator_one_column, mediator_two_column, response_column)
        return values[np.isfinite(values)]

    def quantile_ci(self, values: np.ndarray) -> Tuple[float, float]:
        """Return empirical 95 percent interval."""
        if values.size == 0:
            return float("nan"), float("nan")
        return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

    def compute_cv_passes(self, frame: pd.DataFrame, source_column: str, mediator_one_column: str, mediator_two_column: str, response_column: str, full_estimate: float) -> Tuple[int, int]:
        """Count grouped CV support folds."""
        fold_indices = self.build_fold_splits(frame)
        pass_count = 0
        for train_index in fold_indices:
            train_frame = frame.iloc[train_index].reset_index(drop=True)
            fold_estimate = self.compute_two_mediator_point_estimate(train_frame, source_column, mediator_one_column, mediator_two_column, response_column)
            if self.fold_passes(full_estimate, fold_estimate):
                pass_count += 1
        return int(pass_count), int(len(fold_indices))

    def build_fold_splits(self, frame: pd.DataFrame) -> List[np.ndarray]:
        """Build train-index folds grouped by protein when possible."""
        row_count = int(frame.shape[0])
        groups = frame[self.config.protein_name_column].astype(str).to_numpy(copy=False)
        unique_groups = pd.Index(groups).unique().tolist()
        if len(unique_groups) >= 2:
            split_count = int(min(self.config.cv_folds, len(unique_groups)))
            splitter = GroupKFold(n_splits=split_count)
            dummy_x = np.zeros((row_count, 1), dtype=float)
            dummy_y = np.zeros((row_count,), dtype=float)
            return [np.array(train_index, dtype=int) for train_index, test_index in splitter.split(dummy_x, dummy_y, groups=groups)]
        split_count = int(min(self.config.cv_folds, max(2, row_count)))
        splitter = KFold(n_splits=split_count, shuffle=True, random_state=int(self.config.bootstrap_seed))
        dummy_x = np.zeros((row_count, 1), dtype=float)
        return [np.array(train_index, dtype=int) for train_index, test_index in splitter.split(dummy_x)]

    def fold_passes(self, full_estimate: float, fold_estimate: float) -> bool:
        """Return whether one CV fold supports the full estimate."""
        if not np.isfinite(full_estimate) or not np.isfinite(fold_estimate):
            return False
        sign_match = float(np.sign(full_estimate)) == float(np.sign(fold_estimate))
        magnitude_pass = abs(float(fold_estimate)) >= float(self.config.min_abs_serial_indirect)
        return bool(sign_match and magnitude_pass)

    def build_tables(self, mediation_input: pd.DataFrame, block_table: pd.DataFrame, full_chain_tested: pd.DataFrame, viable_paths: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Build workbook tables."""
        summary = self.build_regime_endpoint_summary(viable_paths)
        return {
            "mediation_input_table": mediation_input,
            "mediation_block_table": block_table,
            "full_chain_tested": full_chain_tested,
            "full_chain_viable": viable_paths,
            "regime_endpoint_summary": summary,
            "mediation_label_dictionary": self.labels.build_label_dictionary(),
        }


@dataclass
class Figure6MediationPlotter:
    """Render manuscript mediation panels."""

    config: Figure6MediationConfig
    paths: Figure6MediationPaths
    style: Figure6MediationStyle

    def plot_ranked_full_chain_effects(self, viable_paths: pd.DataFrame) -> None:
        """Render target-regime ranked full-chain effect plots."""
        for regime in self.config.analysis_regimes:
            self.plot_ranked_full_chain_effects_for_regime(viable_paths, str(regime))

    def build_ranked_plot_table(self, viable_paths: pd.DataFrame, regime: str) -> pd.DataFrame:
        """Build ranked plot table for one regime."""
        subset = viable_paths[viable_paths["regime"] == str(regime)].copy()
        if subset.empty:
            return subset
        subset = subset.sort_values(["serial_indirect", "terminal_response", "path_label"], ascending=[True, True, True], kind="mergesort").reset_index(drop=True)
        subset["rank_within_regime_plot"] = np.arange(1, int(subset.shape[0]) + 1, dtype=int)
        return subset

    def plot_ranked_full_chain_effects_for_regime(self, viable_paths: pd.DataFrame, regime: str) -> None:
        """Render one target-regime ranked full-chain effect plot."""
        plot_table = self.build_ranked_plot_table(viable_paths, regime)
        output_stem = self.style.get_regime_output_stem(regime)
        display_label = self.style.get_regime_display_label(regime)
        if plot_table.empty:
            fig, axis = plt.subplots(figsize=(10.5, 3.4))
            axis.text(0.5, 0.5, "No viable full-chain pathways", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(display_label)
            self.style.apply_axis_style(axis)
            fig.tight_layout()
            save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)
            return
        figure_height = max(3.2, 0.58 * float(plot_table.shape[0]) + 1.6)
        fig, axis = plt.subplots(figsize=(11.4, figure_height))
        y_positions = np.arange(plot_table.shape[0], dtype=float)
        color_value = self.style.get_regime_color(regime)
        for row_index, row in plot_table.iterrows():
            y_value = float(y_positions[row_index])
            serial_value = float(row["serial_indirect"])
            ci_lo = float(row["ci_serial_lo"])
            ci_hi = float(row["ci_serial_hi"])
            axis.plot([ci_lo, ci_hi], [y_value, y_value], color=color_value, linewidth=1.7, alpha=0.95)
            axis.scatter([serial_value], [y_value], color=color_value, marker="o", s=48, zorder=3)
        axis.axvline(0.0, color="#555555", linewidth=0.8, alpha=0.85)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(plot_table["path_label"].tolist())
        pad_value = 0.6 if plot_table.shape[0] <= 3 else 0.4
        axis.set_ylim(float(y_positions[-1]) + pad_value, float(y_positions[0]) - pad_value)
        axis.set_xlabel("Serial indirect effect")
        axis.set_title(display_label)
        self.style.apply_axis_style(axis)
        fig.tight_layout()
        save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)


@dataclass
class Figure6MediationRunner:
    """Coordinate mediation analysis and output export."""

    config: Figure6MediationConfig

    def build_readme_table(self, mediation_input: pd.DataFrame, block_table: pd.DataFrame) -> pd.DataFrame:
        """Build workbook readme."""
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "cohort_input", "value": str(self.config.cohort_xlsx_name)},
            {"key": "cohort_sheet_name", "value": str(self.config.cohort_sheet_name)},
            {"key": "output_dir", "value": str(self.config.output_dir_name)},
            {"key": "threshold_bits_site_min", "value": str(float(self.config.bits_site_min_threshold))},
            {"key": "regime_thresholding", "value": "Regime thresholding is always based on bits_site_min; source_regime_variable is retained only for table export compatibility."},
            {"key": "analysis_design", "value": "Serial mediation-style screen using block-level median aggregation within targeted regimes"},
            {"key": "four_node_chain", "value": "bits_site_median -> Figure 3 MSA mediator -> Figure 4 latent mediator -> endpoint"},
            {"key": "analysis_unit", "value": "One block per regime, protein_name, and outer_label_base"},
            {"key": "analysis_regimes", "value": ", ".join(self.config.analysis_regimes)},
            {"key": "sample_level_rows", "value": str(int(mediation_input.shape[0]))},
            {"key": "block_level_rows", "value": str(int(block_table.shape[0]))},
            {"key": "source_variable", "value": str(self.config.source_variable)},
            {"key": "terminal_responses", "value": ", ".join(self.config.terminal_response_columns)},
            {"key": "msa_mediator_columns", "value": ", ".join(self.config.msa_mediator_columns)},
            {"key": "latent_mediator_columns", "value": ", ".join(self.config.latent_mediator_columns)},
            {"key": "viable_full_chain_definition", "value": f"abs(serial_indirect) >= {self.config.min_abs_serial_indirect}, percentile bootstrap CI excludes zero, and grouped CV passes >= {self.config.required_cv_pass_folds}"},
            {"key": "bootstrap_replicates", "value": str(int(self.config.bootstrap_replicates))},
            {"key": "bootstrap_seed", "value": str(int(self.config.bootstrap_seed))},
            {"key": "plot_outputs", "value": "Supplementary Figure 4a target-regime ranked full-chain effect panels only"},
        ]
        return pd.DataFrame(rows)

    def run(self) -> None:
        """Run mediation analysis and write outputs."""
        paths = Figure6MediationPaths.from_config(self.config)
        paths.ensure_dirs()
        labels = Figure6MediationLabels()
        style = Figure6MediationStyle(ConditionStyle())
        style.apply_publication_style()
        builder = Figure6MediationDataBuilder(self.config)
        analyzer = Figure6MediationAnalyzer(self.config, labels)
        plotter = Figure6MediationPlotter(self.config, paths, style)
        cohort = builder.read_cohort(paths)
        cohort = builder.add_regime_columns(cohort)
        mediation_input = builder.build_input_table(cohort)
        block_table = builder.build_block_table(mediation_input)
        full_chain_tested = analyzer.run_full_chain_screen(block_table)
        viable_paths = analyzer.filter_viable_paths(full_chain_tested)
        tables = {"readme": self.build_readme_table(mediation_input, block_table)}
        tables.update(analyzer.build_tables(mediation_input, block_table, full_chain_tested, viable_paths))
        plotter.plot_ranked_full_chain_effects(viable_paths)
        writer = ExcelNotebookWriter(paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        print(f"Wrote Figure 6 mediation outputs to {paths.output_dir}")



def main() -> None:
    """Run mediation analysis."""
    runner = Figure6MediationRunner(Figure6MediationConfig())
    runner.run()


if __name__ == "__main__":
    main()
