import math
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

from figure_common import ConditionStyle
from figure_common import operational_bits_site_threshold
from figure_common import ExcelNotebookWriter
from figure_common import add_threshold_condition_columns
from figure_common import convert_archived_columns
from figure_common import benjamini_hochberg
from figure_common import require_columns
from figure_common import safe_mean
from figure_common import safe_sample_sd


@dataclass(frozen=True)
class Figure3Config:
    """Configure Figure 3 outputs."""

    cohort_xlsx_name: str = "cohort.xlsx"
    cohort_sheet_name: str = "by_sample"
    output_dir_name: str = "figure3_outputs"
    output_xlsx_name: str = "figure3_source_data.xlsx"
    bps_min_threshold: float = operational_bits_site_threshold
    stats_fdr_alpha: float = 0.05
    excel_chunk_rows: int = 1000000


@dataclass(frozen=True)
class Figure3Paths:
    """Resolve input and output paths for Figure 3."""

    root_dir: Path
    cohort_xlsx_path: Path
    output_dir: Path
    output_xlsx_path: Path

    @classmethod
    def from_root_dir(cls, root_dir: Path, config: Figure3Config) -> "Figure3Paths":
        """Build paths from the working directory."""
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


@dataclass(frozen=True)
class Figure3Metric:
    """Describe one Figure 3 metric."""

    panel: str
    column: str
    label: str
    scale: str
    output_stem: str


@dataclass
class Figure3Analyzer:
    """Compute Figure 3 summaries, correlations, and statistics."""

    config: Figure3Config

    def get_metrics(self) -> List[Figure3Metric]:
        """Return the Figure 3 metric list."""
        return [
            Figure3Metric("figure3c", "msa_effective_alphabet_mean", "Effective alphabet size (mean)", "linear", "figure3c_effective_alphabet_size_mean"),
            Figure3Metric("figure3d", "meff_theta0p70", "Meff (θ=0.70)", "log", "figure3d_meff_theta0p70_logy"),
            Figure3Metric("figure3e", "mi_apc_topq_mean", "MI-APC (top-q mean)", "linear", "figure3e_mi_apc_topq_mean"),
            Figure3Metric("figure3f", "msa_graph_laplacian_lambda2", "MSA graph connectivity (λ2)", "log", "figure3f_msa_graph_connectivity_lambda2_logy"),
            Figure3Metric("figure3g", "msa_graph_lcc_fraction", "LCC fraction", "linear", "figure3g_lcc_fraction"),
            Figure3Metric("figure3h", "msa_graph_edge_density", "MSA graph edge density", "linear", "figure3h_msa_graph_edge_density"),
            Figure3Metric("figure3i", "msa_graph_apl_lcc", "Avg path length (LCC)", "linear", "figure3i_avg_path_length_lcc"),
            Figure3Metric("figure3j", "msa_graph_bridge_fraction", "Bridge fraction", "linear", "figure3j_bridge_fraction"),
            Figure3Metric("figure3k", "msa_graph_articulation_fraction", "Articulation fraction", "linear", "figure3k_articulation_fraction"),
        ]

    def read_cohort_by_sample(self, xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
        """Read and validate the cohort by-sample sheet."""
        metrics = self.get_metrics()
        required_columns = ["sample_id", "protein_name", "outer_label", "control_type", "bits_site_min"] + [metric.column for metric in metrics]
        required_columns.append("msa_graph_components")
        if not Path(xlsx_path).exists():
            raise FileNotFoundError(f"Missing cohort workbook: {xlsx_path}")
        cohort = pd.read_excel(xlsx_path, sheet_name=str(sheet_name), engine="openpyxl")
        require_columns(cohort, required_columns, "cohort workbook")
        out = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type"]:
            out[column_name] = out[column_name].astype(str)
        numeric_columns = ["bits_site_min", "msa_graph_components"] + [metric.column for metric in metrics]
        for column_name in numeric_columns:
            out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
        return convert_archived_columns(out, information_columns=("bits_site_min",), effective_alphabet_columns=("msa_effective_alphabet_mean",))

    def compute_protein_summary(self, cohort: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
        """Compute protein-level condition means."""
        columns = ["protein_name", "condition4", "target_stratum"] + list(metric_columns)
        base = cohort[columns].copy()
        for column_name in metric_columns:
            base[column_name] = pd.to_numeric(base[column_name], errors="coerce")
        summary = base.groupby(["protein_name", "condition4", "target_stratum"], sort=False)[list(metric_columns)].mean(numeric_only=True).reset_index()
        return summary

    def build_violin_values(self, protein_summary: pd.DataFrame, metrics: Sequence[Figure3Metric]) -> pd.DataFrame:
        """Build long-form Figure 3c-k violin source data."""
        rows: List[Dict[str, Any]] = []
        allowed_conditions = ["below targeted", "above targeted"]
        for metric in metrics:
            sub = protein_summary[protein_summary["condition4"].isin(allowed_conditions)].copy()
            sub[metric.column] = pd.to_numeric(sub[metric.column], errors="coerce")
            sub = sub[np.isfinite(sub[metric.column].to_numpy(dtype=float))].copy()
            for row in sub.itertuples(index=False):
                rows.append(
                    {
                        "panel": metric.panel,
                        "protein_name": str(getattr(row, "protein_name")),
                        "condition4": str(getattr(row, "condition4")),
                        "target_stratum": str(getattr(row, "target_stratum")),
                        "metric": metric.column,
                        "metric_label": metric.label,
                        "value": float(getattr(row, metric.column)),
                        "scale": metric.scale,
                    }
                )
        return pd.DataFrame(rows)

    def compute_hedges_g(self, group_a: Sequence[float], group_b: Sequence[float]) -> float:
        """Compute Hedges g for unpaired samples."""
        a_values = np.asarray(group_a, dtype=float)
        b_values = np.asarray(group_b, dtype=float)
        a_values = a_values[np.isfinite(a_values)]
        b_values = b_values[np.isfinite(b_values)]
        n_a = float(a_values.size)
        n_b = float(b_values.size)
        if n_a < 2 or n_b < 2:
            return float("nan")
        variance_a = float(np.var(a_values, ddof=1))
        variance_b = float(np.var(b_values, ddof=1))
        pooled_denominator = n_a + n_b - 2.0
        if pooled_denominator <= 0.0:
            return float("nan")
        pooled_numerator = (n_a - 1.0) * variance_a + (n_b - 1.0) * variance_b
        pooled_sd = math.sqrt(pooled_numerator / pooled_denominator) if pooled_numerator > 0.0 else 0.0
        if pooled_sd <= 0.0:
            return float("nan")
        cohens_d = (float(np.mean(a_values)) - float(np.mean(b_values))) / pooled_sd
        correction = 1.0 - (3.0 / (4.0 * (n_a + n_b) - 9.0)) if (n_a + n_b) > 2.0 else 1.0
        return float(correction * cohens_d)

    def compute_unpaired_stats(self, protein_summary: pd.DataFrame, metrics: Sequence[Figure3Metric]) -> pd.DataFrame:
        """Compute Figure 3c-k Welch tests with per-metric BH families."""
        rows: List[Dict[str, Any]] = []
        contrast_specs = [
            ("target_below_vs_above", "below targeted", "above targeted", True),
            ("matched_control_below_vs_above", "below matched control", "above matched control", False),
        ]
        for metric in metrics:
            contrast_rows: List[Dict[str, Any]] = []
            p_values: List[float] = []
            for contrast_name, group_a_name, group_b_name, reported_in_figure in contrast_specs:
                group_a_values = pd.to_numeric(protein_summary.loc[protein_summary["condition4"] == group_a_name, metric.column], errors="coerce").to_numpy(dtype=float)
                group_b_values = pd.to_numeric(protein_summary.loc[protein_summary["condition4"] == group_b_name, metric.column], errors="coerce").to_numpy(dtype=float)
                group_a_values = group_a_values[np.isfinite(group_a_values)]
                group_b_values = group_b_values[np.isfinite(group_b_values)]
                if group_a_values.size < 3 or group_b_values.size < 3:
                    p_value = float("nan")
                else:
                    result = stats.ttest_ind(group_a_values, group_b_values, equal_var=False, nan_policy="omit")
                    p_value = float(result.pvalue)
                p_values.append(p_value)
                contrast_rows.append(
                    {
                        "panel": metric.panel,
                        "metric": metric.column,
                        "metric_label": metric.label,
                        "contrast": contrast_name,
                        "reported_in_figure": bool(reported_in_figure),
                        "test": "welch_two_sided",
                        "analysis_unit": "protein_level_condition_mean",
                        "group_a": group_a_name,
                        "n_a": int(group_a_values.size),
                        "mean_a": safe_mean(group_a_values),
                        "sd_a": safe_sample_sd(group_a_values),
                        "group_b": group_b_name,
                        "n_b": int(group_b_values.size),
                        "mean_b": safe_mean(group_b_values),
                        "sd_b": safe_sample_sd(group_b_values),
                        "p_value": p_value,
                        "effect_name": "hedges_g",
                        "effect": self.compute_hedges_g(group_a_values, group_b_values),
                    }
                )
            q_values = benjamini_hochberg(p_values)
            for row_index, q_value in enumerate(q_values):
                contrast_rows[row_index]["q_value_bh"] = float(q_value) if np.isfinite(q_value) else float("nan")
                contrast_rows[row_index]["bh_threshold"] = float(self.config.stats_fdr_alpha)
                contrast_rows[row_index]["reject_fdr_bh"] = bool(np.isfinite(contrast_rows[row_index]["q_value_bh"]) and contrast_rows[row_index]["q_value_bh"] <= float(self.config.stats_fdr_alpha))
                contrast_rows[row_index]["correction_family"] = "within_metric_targeted_and_matched_control_below_vs_above"
                contrast_rows[row_index]["n_tests_in_family"] = int(len(contrast_specs))
            rows.extend(contrast_rows)
        return pd.DataFrame(rows)

    def compute_correlation_matrices(self, protein_summary: pd.DataFrame, metrics: Sequence[Figure3Metric]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute targeted below and above Spearman correlation matrices."""
        metric_columns = [metric.column for metric in metrics]
        labels = [metric.label for metric in metrics]
        below = protein_summary[protein_summary["condition4"] == "below targeted"].copy()
        above = protein_summary[protein_summary["condition4"] == "above targeted"].copy()
        below_corr = below[metric_columns].corr(method="spearman", min_periods=2)
        above_corr = above[metric_columns].corr(method="spearman", min_periods=2)
        below_corr.index = labels
        below_corr.columns = labels
        above_corr.index = labels
        above_corr.columns = labels
        return below_corr, above_corr

    def summarize_correlation_matrix(self, regime: str, corr: pd.DataFrame) -> Dict[str, Any]:
        """Summarize absolute upper-triangle Spearman correlations."""
        values = corr.to_numpy(dtype=float)
        mask = np.triu(np.ones_like(values, dtype=bool), k=1)
        pair_values = np.abs(values[mask])
        pair_values = pair_values[np.isfinite(pair_values)]
        return {
            "regime": str(regime),
            "n_features": int(corr.shape[0]),
            "n_feature_pairs": int(pair_values.size),
            "mean_abs_spearman_rho": float(np.mean(pair_values)) if pair_values.size else float("nan"),
            "median_abs_spearman_rho": float(np.median(pair_values)) if pair_values.size else float("nan"),
        }

    def compute_schematic_values(self, protein_summary: pd.DataFrame) -> pd.DataFrame:
        """Compute protein-level medians used by the Figure 3a schematic."""
        schematic_columns = [
            "msa_graph_components",
            "msa_graph_lcc_fraction",
            "msa_graph_edge_density",
            "msa_graph_laplacian_lambda2",
            "msa_graph_apl_lcc",
            "msa_graph_bridge_fraction",
            "msa_graph_articulation_fraction",
        ]
        rows: List[Dict[str, Any]] = []
        for condition_value, regime in [("below targeted", "below"), ("above targeted", "above")]:
            sub = protein_summary[protein_summary["condition4"] == condition_value].copy()
            row: Dict[str, Any] = {"regime": regime, "condition4": condition_value}
            for column_name in schematic_columns:
                values = pd.to_numeric(sub[column_name], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                row[column_name] = float(np.median(values)) if values.size else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)

    def build_metric_dictionary(self, metrics: Sequence[Figure3Metric]) -> pd.DataFrame:
        """Build a metric dictionary for Figure 3."""
        rows = []
        for metric in metrics:
            rows.append({"panel": metric.panel, "metric": metric.column, "metric_label": metric.label, "scale": metric.scale, "output_stem": metric.output_stem, "role": "violin"})
        rows.append({"panel": "figure3a", "metric": "msa_graph_components", "metric_label": "MSA graph components", "scale": "linear", "output_stem": "figure3a_msa_graph_schematic_targeted_below_vs_ge_1p0", "role": "schematic_only"})
        return pd.DataFrame(rows)

    def build_readme_table(self) -> pd.DataFrame:
        """Build a concise workbook manifest."""
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "cohort_input", "value": str(self.config.cohort_xlsx_name)},
            {"key": "cohort_sheet_name", "value": str(self.config.cohort_sheet_name)},
            {"key": "threshold_bits_site_min", "value": str(float(self.config.bps_min_threshold))},
            {"key": "violin_analysis_unit", "value": "protein-level means for targeted below versus targeted above"},
            {"key": "violin_orientation", "value": "horizontal_left"},
            {"key": "figure3b_analysis_unit", "value": "protein-level targeted condition means"},
            {"key": "figure3c_k_statistics", "value": "Welch two-sided tests with BH correction within each metric across targeted and matched-control below-vs-above contrasts"},
            {"key": "schematic_values", "value": "protein-level medians for targeted below and targeted above"},
        ]
        return pd.DataFrame(rows)


@dataclass
class Figure3Plotter:
    """Render Figure 3 plots."""

    config: Figure3Config
    paths: Figure3Paths
    style: ConditionStyle

    def apply_publication_style(self) -> None:
        """Apply the Figure 3 publication style."""
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

    def apply_axis_style(self, ax: plt.Axes) -> None:
        """Apply per-axis styling."""
        ax.grid(False)
        ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        sns.despine(ax=ax, top=True, right=True)

    def save_current_figure(self, fig: plt.Figure, output_stem: str) -> None:
        """Save a figure as SVG and PNG."""
        fig.tight_layout()
        fig.savefig(self.paths.output_dir / f"{output_stem}.svg", dpi=300)
        fig.savefig(self.paths.output_dir / f"{output_stem}.png", dpi=300)
        plt.close(fig)

    def build_correlation_cmap(self) -> mpl.colors.LinearSegmentedColormap:
        """Build the Okabe-Ito correlation colormap."""
        cmap = mpl.colors.LinearSegmentedColormap.from_list("okabe_ito_div", ["#E69F00", "#FFFFFF", "#0072B2"], N=256)
        cmap.set_bad(color="white")
        return cmap

    def save_condition_legend(self) -> None:
        """Save the standalone Figure 3 condition legend."""
        handles: List[Any] = []
        labels: List[str] = []
        for condition_label in self.style.get_order():
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=self.style.get_color(condition_label), markeredgecolor="none", markersize=9))
            labels.append(condition_label)
        fig = plt.figure(figsize=(4.4, 1.2))
        legend_obj = fig.legend(handles, labels, loc="center", frameon=False, ncol=2, handlelength=1.0, handletextpad=0.6, borderaxespad=0.2, columnspacing=1.2)
        fig.savefig(self.paths.output_dir / "legend_figure3_conditions.svg", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend_obj])
        fig.savefig(self.paths.output_dir / "legend_figure3_conditions.png", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend_obj])
        plt.close(fig)

    def save_correlation_colorbar(self) -> None:
        """Save the standalone Figure 3b colorbar."""
        cmap = self.build_correlation_cmap()
        fig, ax = plt.subplots(figsize=(1.2, 3.0))
        fig.subplots_adjust(left=0.55, right=0.85)
        norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
        scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, cax=ax)
        colorbar.set_label("Spearman ρ")
        ax.grid(False)
        fig.savefig(self.paths.output_dir / "legend_figure3b_spearman_correlation_colorbar.svg", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(self.paths.output_dir / "legend_figure3b_spearman_correlation_colorbar.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def plot_violin_metric(self, violin_values: pd.DataFrame, metric: Figure3Metric) -> None:
        """Plot one horizontal-left targeted-only violin panel."""
        source = violin_values[violin_values["metric"] == metric.column].copy()
        source = source[source["condition4"].isin(["below targeted", "above targeted"])].copy()
        source["value"] = pd.to_numeric(source["value"], errors="coerce")
        source = source[np.isfinite(source["value"].to_numpy(dtype=float))].copy()
        if source.empty:
            return
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        order = ["above targeted", "below targeted"]
        sns.violinplot(
            ax=ax,
            data=source,
            y="condition4",
            x="value",
            hue="condition4",
            order=order,
            palette=self.style.get_palette(),
            cut=0,
            inner=None,
            linewidth=0.8,
            dodge=False,
            legend=False,
        )
        axis_label = metric.label
        if metric.scale == "log":
            ax.set_xscale("log")
            axis_label = f"{metric.label} (log scale)"
        ax.set_xlabel(axis_label)
        ax.set_ylabel("")
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")
        if ax.legend_ is not None:
            ax.legend_.remove()
        self.apply_axis_style(ax)
        self.save_current_figure(fig, metric.output_stem)

    def plot_correlation_heatmap(self, corr: pd.DataFrame, title_text: str, output_stem: str) -> None:
        """Plot an upper-triangle Spearman correlation heatmap."""
        label_order = corr.index.tolist()
        corr_df = corr.loc[label_order, label_order]
        matrix = corr_df.to_numpy(dtype=float, copy=True)
        mask = np.tril(np.ones_like(matrix, dtype=bool))
        masked_matrix = np.ma.masked_where(mask, matrix)
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        ax.imshow(masked_matrix, cmap=self.build_correlation_cmap(), vmin=-1.0, vmax=1.0)
        ax.set_xticks(range(len(label_order)))
        ax.set_yticks(range(len(label_order)))
        ax.set_xticklabels(label_order, rotation=45, ha="right")
        ax.set_yticklabels(label_order)
        ax.set_title(title_text)
        ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        self.save_current_figure(fig, output_stem)

    def normalize_between(self, x: float, lo: float, hi: float) -> float:
        """Normalize x into [0,1] given bounds; clamp outside."""
        if not np.isfinite(x) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.5
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    def normalize_log_between(self, x: float, lo: float, hi: float, eps: float) -> float:
        """Normalize log10(x+eps) into [0,1] given bounds."""
        if not np.isfinite(x) or x < 0:
            return 0.0
        value = float(np.log10(float(x) + float(eps)))
        return self.normalize_between(value, lo, hi)

    def build_cluster_sizes(self, nodes_total: int, cluster_count: int, lcc_fraction: float) -> List[int]:
        """Allocate cluster sizes given a total node count and LCC fraction."""
        cluster_count_value = int(max(1, cluster_count))
        lcc_fraction_value = float(lcc_fraction) if np.isfinite(lcc_fraction) else 0.7
        lcc_fraction_value = float(np.clip(lcc_fraction_value, 0.05, 0.98))
        lcc_size = int(round(float(nodes_total) * lcc_fraction_value))
        lcc_size = int(np.clip(lcc_size, 2, nodes_total))
        remaining = int(nodes_total - lcc_size)
        if cluster_count_value == 1:
            return [nodes_total]
        other_count = int(cluster_count_value - 1)
        base_size = int(remaining // other_count) if other_count > 0 else 0
        sizes = [lcc_size] + [base_size] * other_count
        remainder = int(remaining - base_size * other_count)
        for index in range(remainder):
            size_index = 1 + (index % max(1, other_count))
            sizes[size_index] += 1
        return sizes

    def generate_cluster_centers(self, cluster_count: int) -> np.ndarray:
        """Generate fixed cluster centers on a circle."""
        cluster_count_value = int(max(1, cluster_count))
        if cluster_count_value == 1:
            return np.array([[0.0, 0.0]], dtype=float)
        angles = np.linspace(0.0, 2.0 * math.pi, cluster_count_value, endpoint=False)
        radius = 0.55
        centers = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1).astype(float)
        return centers

    def generate_points_from_clusters(self, sizes: Sequence[int], centers: np.ndarray, spread: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate points around cluster centers with a specified spread."""
        rng = np.random.default_rng(int(seed))
        points: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for index, size in enumerate(sizes):
            if int(size) <= 0:
                continue
            base = rng.normal(size=(int(size), 2)).astype(float)
            base *= float(spread)
            base += centers[int(index)]
            points.append(base)
            labels.append(np.full(int(size), int(index), dtype=int))
        if not points:
            return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=int)
        point_array = np.concatenate(points, axis=0)
        label_array = np.concatenate(labels, axis=0)
        return point_array, label_array

    def sample_edges(self, labels: np.ndarray, total_edges: int, bridge_share: float, seed: int) -> np.ndarray:
        """Sample undirected edges with a specified inter-cluster share."""
        rng = np.random.default_rng(int(seed))
        node_count = int(labels.size)
        if node_count <= 1 or total_edges <= 0:
            return np.zeros((0, 2), dtype=np.int32)
        total_edges_int = int(total_edges)
        inter_edges = int(round(float(total_edges_int) * float(np.clip(bridge_share, 0.0, 1.0))))
        intra_edges = int(max(0, total_edges_int - inter_edges))
        edges: List[Tuple[int, int]] = []
        unique_labels = np.unique(labels)
        indices_by_label: Dict[int, np.ndarray] = {int(label_value): np.where(labels == label_value)[0] for label_value in unique_labels}
        label_keys = list(indices_by_label.keys())
        if len(label_keys) == 1:
            inter_edges = 0
            intra_edges = total_edges_int
        if intra_edges > 0:
            for edge_index in range(intra_edges):
                label_key = int(rng.choice(label_keys))
                nodes = indices_by_label[label_key]
                if nodes.size < 2:
                    continue
                left_node, right_node = rng.choice(nodes, size=2, replace=False)
                left_index = int(left_node)
                right_index = int(right_node)
                if left_index == right_index:
                    continue
                if left_index > right_index:
                    left_index, right_index = right_index, left_index
                edges.append((left_index, right_index))
        if inter_edges > 0 and len(label_keys) >= 2:
            for edge_index in range(inter_edges):
                left_label, right_label = rng.choice(label_keys, size=2, replace=False)
                left_nodes = indices_by_label[int(left_label)]
                right_nodes = indices_by_label[int(right_label)]
                if left_nodes.size == 0 or right_nodes.size == 0:
                    continue
                left_index = int(rng.choice(left_nodes))
                right_index = int(rng.choice(right_nodes))
                if left_index == right_index:
                    continue
                if left_index > right_index:
                    left_index, right_index = right_index, left_index
                edges.append((left_index, right_index))
        if not edges:
            return np.zeros((0, 2), dtype=np.int32)
        unique_edges = list(dict.fromkeys(edges))
        return np.array(unique_edges, dtype=np.int32)

    def compute_shared_scale_factor(self, left_cloud: np.ndarray, right_cloud: np.ndarray, margin: float) -> float:
        """Compute one shared scale to fit both clouds within axes."""
        left_maximum = float(np.max(np.abs(left_cloud))) if left_cloud.size else 1.0
        right_maximum = float(np.max(np.abs(right_cloud))) if right_cloud.size else 1.0
        denominator = max(left_maximum, right_maximum, 1e-6)
        return float(margin) / float(denominator)

    def draw_msa_graph_panel(self, ax: plt.Axes, points: np.ndarray, edges: np.ndarray, shared_scale: float, node_color: str, edge_color: str, title: str, highlight_fraction: float, annotation: str) -> None:
        """Draw one MSA-graph cartoon panel."""
        scaled_points = points * float(shared_scale)
        for left_index, right_index in edges:
            ax.plot(
                [scaled_points[left_index, 0], scaled_points[right_index, 0]],
                [scaled_points[left_index, 1], scaled_points[right_index, 1]],
                color=edge_color,
                alpha=0.22,
                linewidth=0.6,
            )
        point_count = int(scaled_points.shape[0])
        rng = np.random.default_rng(int(1000 + point_count))
        highlight_count = int(max(0, round(float(highlight_fraction) * float(point_count))))
        highlight_count = int(min(highlight_count, point_count))
        highlight_indices = rng.choice(point_count, size=highlight_count, replace=False) if highlight_count > 0 else np.array([], dtype=int)
        ax.scatter(scaled_points[:, 0], scaled_points[:, 1], s=10.0, c=node_color, alpha=0.55, edgecolors="none")
        if highlight_indices.size:
            ax.scatter(scaled_points[highlight_indices, 0], scaled_points[highlight_indices, 1], s=28.0, c=node_color, alpha=0.90, edgecolors="none")
        if annotation:
            ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=8)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

    def plot_schematic(self, schematic_values: pd.DataFrame) -> None:
        """Build a two-panel MSA-graph concept figure with shared scale."""
        below_row = schematic_values[schematic_values["regime"] == "below"].iloc[0]
        above_row = schematic_values[schematic_values["regime"] == "above"].iloc[0]
        components_below = float(below_row["msa_graph_components"])
        components_above = float(above_row["msa_graph_components"])
        lcc_below = float(below_row["msa_graph_lcc_fraction"])
        lcc_above = float(above_row["msa_graph_lcc_fraction"])
        density_below = float(below_row["msa_graph_edge_density"])
        density_above = float(above_row["msa_graph_edge_density"])
        lambda2_below = float(below_row["msa_graph_laplacian_lambda2"])
        lambda2_above = float(above_row["msa_graph_laplacian_lambda2"])
        apl_below = float(below_row["msa_graph_apl_lcc"])
        apl_above = float(above_row["msa_graph_apl_lcc"])
        bridge_below = float(below_row["msa_graph_bridge_fraction"])
        bridge_above = float(above_row["msa_graph_bridge_fraction"])
        articulation_below = float(below_row["msa_graph_articulation_fraction"])
        articulation_above = float(above_row["msa_graph_articulation_fraction"])
        nodes_total = 240
        cluster_below = int(np.clip(int(round(components_below)) if np.isfinite(components_below) else 2, 1, 8))
        cluster_above = int(np.clip(int(round(components_above)) if np.isfinite(components_above) else 2, 1, 8))
        sizes_below = self.build_cluster_sizes(nodes_total, cluster_below, lcc_below)
        sizes_above = self.build_cluster_sizes(nodes_total, cluster_above, lcc_above)
        apl_lower = min([value for value in [apl_below, apl_above] if np.isfinite(value)] or [1.0])
        apl_upper = max([value for value in [apl_below, apl_above] if np.isfinite(value)] or [3.0])
        apln_below = self.normalize_between(apl_below, apl_lower, apl_upper)
        apln_above = self.normalize_between(apl_above, apl_lower, apl_upper)
        spread_below = float(0.20 + 0.18 * apln_below)
        spread_above = float(0.20 + 0.18 * apln_above)
        centers_below = self.generate_cluster_centers(cluster_below)
        centers_above = self.generate_cluster_centers(cluster_above)
        points_below, labels_below = self.generate_points_from_clusters(sizes_below, centers_below, spread_below, seed=101)
        points_above, labels_above = self.generate_points_from_clusters(sizes_above, centers_above, spread_above, seed=202)
        density_lower = min([value for value in [density_below, density_above] if np.isfinite(value)] or [0.0])
        density_upper = max([value for value in [density_below, density_above] if np.isfinite(value)] or [0.5])
        densityn_below = self.normalize_between(density_below, density_lower, density_upper)
        densityn_above = self.normalize_between(density_above, density_lower, density_upper)
        epsilon = 1e-12
        lambda2_values = [value for value in [lambda2_below, lambda2_above] if np.isfinite(value) and value >= 0]
        if lambda2_values:
            log_lambda2 = [float(np.log10(float(value) + epsilon)) for value in lambda2_values]
            lambda2_lower = float(min(log_lambda2))
            lambda2_upper = float(max(log_lambda2))
        else:
            lambda2_lower = -6.0
            lambda2_upper = 2.0
        lambda2n_below = self.normalize_log_between(lambda2_below, lambda2_lower, lambda2_upper, epsilon)
        lambda2n_above = self.normalize_log_between(lambda2_above, lambda2_lower, lambda2_upper, epsilon)
        base_edges = 380
        extra_edges = 520
        total_edges_below = int(base_edges + extra_edges * (0.65 * densityn_below + 0.35 * lambda2n_below))
        total_edges_above = int(base_edges + extra_edges * (0.65 * densityn_above + 0.35 * lambda2n_above))
        bridge_values = [value for value in [bridge_below, bridge_above] if np.isfinite(value) and value >= 0]
        if bridge_values:
            log_bridge = [float(np.log10(float(value) + epsilon)) for value in bridge_values]
            bridge_lower = float(min(log_bridge))
            bridge_upper = float(max(log_bridge))
        else:
            bridge_lower = -12.0
            bridge_upper = -1.0
        bridgen_below = self.normalize_log_between(bridge_below, bridge_lower, bridge_upper, epsilon)
        bridgen_above = self.normalize_log_between(bridge_above, bridge_lower, bridge_upper, epsilon)
        bridge_share_below = float(0.02 + 0.16 * bridgen_below + 0.08 * lambda2n_below)
        bridge_share_above = float(0.02 + 0.16 * bridgen_above + 0.08 * lambda2n_above)
        bridge_share_below = float(np.clip(bridge_share_below, 0.02, 0.28))
        bridge_share_above = float(np.clip(bridge_share_above, 0.02, 0.28))
        edges_below = self.sample_edges(labels_below, total_edges_below, bridge_share_below, seed=303)
        edges_above = self.sample_edges(labels_above, total_edges_above, bridge_share_above, seed=404)
        shared_scale = self.compute_shared_scale_factor(points_below, points_above, margin=0.92)
        articulation_values = [value for value in [articulation_below, articulation_above] if np.isfinite(value) and value >= 0]
        if articulation_values:
            log_articulation = [float(np.log10(float(value) + epsilon)) for value in articulation_values]
            articulation_lower = float(min(log_articulation))
            articulation_upper = float(max(log_articulation))
        else:
            articulation_lower = -12.0
            articulation_upper = -1.0
        articulationn_below = self.normalize_log_between(articulation_below, articulation_lower, articulation_upper, epsilon)
        articulationn_above = self.normalize_log_between(articulation_above, articulation_lower, articulation_upper, epsilon)
        highlight_below = float(np.clip(0.02 + 0.10 * articulationn_below, 0.0, 0.18))
        highlight_above = float(np.clip(0.02 + 0.10 * articulationn_above, 0.0, 0.18))
        node_color_below = self.style.get_color("below targeted")
        node_color_above = self.style.get_color("above targeted")
        edge_color_below = self.style.blend_hex_with_white(node_color_below, 0.70)
        edge_color_above = self.style.blend_hex_with_white(node_color_above, 0.70)
        annotation_below = f"density={density_below:.3f}, λ2={lambda2_below:.3g}\nAPL={apl_below:.2f}, bridge={bridge_below:.2g}\nartic={articulation_below:.3g}, LCC={lcc_below:.2f}"
        annotation_above = f"density={density_above:.3f}, λ2={lambda2_above:.3g}\nAPL={apl_above:.2f}, bridge={bridge_above:.2g}\nartic={articulation_above:.3g}, LCC={lcc_above:.2f}"
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True, sharey=True)
        threshold = float(self.config.bps_min_threshold)
        self.draw_msa_graph_panel(
            axes[0],
            points_below,
            edges_below,
            shared_scale,
            node_color_below,
            edge_color_below,
            f"Targeted < {threshold:.1f} bits/site_min",
            highlight_below,
            annotation_below,
        )
        self.draw_msa_graph_panel(
            axes[1],
            points_above,
            edges_above,
            shared_scale,
            node_color_above,
            edge_color_above,
            f"Targeted ≥ {threshold:.1f} bits/site_min",
            highlight_above,
            annotation_above,
        )
        self.save_current_figure(fig, "figure3a_msa_graph_schematic_targeted_below_vs_ge_1p0")

@dataclass
class Figure3Runner:
    """Coordinate Figure 3 analysis, plotting, and source-data export."""

    config: Figure3Config

    def run(self) -> None:
        """Run the Figure 3 build."""
        paths = Figure3Paths.from_root_dir(Path(".").resolve(), self.config)
        paths.ensure_dirs()
        style = ConditionStyle()
        analyzer = Figure3Analyzer(self.config)
        plotter = Figure3Plotter(self.config, paths, style)
        plotter.apply_publication_style()
        metrics = analyzer.get_metrics()
        cohort = analyzer.read_cohort_by_sample(paths.cohort_xlsx_path, self.config.cohort_sheet_name)
        cohort = add_threshold_condition_columns(cohort, threshold=float(self.config.bps_min_threshold))
        metric_columns = [metric.column for metric in metrics] + ["msa_graph_components"]
        protein_summary = analyzer.compute_protein_summary(cohort, metric_columns)
        violin_values = analyzer.build_violin_values(protein_summary, metrics)
        stats_table = analyzer.compute_unpaired_stats(protein_summary, metrics)
        corr_below, corr_above = analyzer.compute_correlation_matrices(protein_summary, metrics)
        corr_summary = pd.DataFrame(
            [
                analyzer.summarize_correlation_matrix("below targeted", corr_below),
                analyzer.summarize_correlation_matrix("above targeted", corr_above),
            ]
        )
        schematic_values = analyzer.compute_schematic_values(protein_summary)
        plotter.save_condition_legend()
        plotter.plot_schematic(schematic_values)
        plotter.plot_correlation_heatmap(corr_below, f"Targeted correlations (< {self.config.bps_min_threshold:.1f} bits/site_min)", "figure3b_left_targeted_below_1p0_msa_metric_correlations")
        plotter.plot_correlation_heatmap(corr_above, f"Targeted correlations (≥ {self.config.bps_min_threshold:.1f} bits/site_min)", "figure3b_right_targeted_ge_1p0_msa_metric_correlations")
        plotter.save_correlation_colorbar()
        for metric in metrics:
            plotter.plot_violin_metric(violin_values, metric)
        tables = {
            "readme": analyzer.build_readme_table(),
            "figure3a_schematic_values": schematic_values,
            "figure3b_corr_below_targeted": corr_below.reset_index().rename(columns={"index": "metric"}),
            "figure3b_corr_above_targeted": corr_above.reset_index().rename(columns={"index": "metric"}),
            "figure3b_correlation_summary": corr_summary,
            "figure3c_k_violin_values": violin_values,
            "figure3c_k_unpaired_welch": stats_table,
            "figure3_metric_dictionary": analyzer.build_metric_dictionary(metrics),
        }
        writer = ExcelNotebookWriter(paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        print(f"Wrote Figure 3 outputs to {paths.output_dir}")


def main() -> None:
    """Run the Figure 3 generator."""
    runner = Figure3Runner(Figure3Config())
    runner.run()


if __name__ == "__main__":
    main()
