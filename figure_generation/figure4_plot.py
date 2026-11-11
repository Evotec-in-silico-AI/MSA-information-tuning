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
class Figure4Config:
    """Configure Figure 4 outputs."""

    cohort_xlsx_name: str = "cohort.xlsx"
    cohort_sheet_name: str = "by_sample"
    output_dir_name: str = "figure4_outputs"
    output_xlsx_name: str = "figure4_source_data.xlsx"
    bps_min_threshold: float = operational_bits_site_threshold
    stats_fdr_alpha: float = 0.05
    excel_chunk_rows: int = 1000000


@dataclass(frozen=True)
class Figure4Metric:
    """Describe one Figure 4 metric."""

    panel: str
    column: str
    label: str
    scale: str
    output_stem: str


@dataclass(frozen=True)
class Figure4Paths:
    """Resolve Figure 4 input and output paths."""

    root_dir: Path
    cohort_xlsx_path: Path
    output_dir: Path
    output_xlsx_path: Path

    @classmethod
    def from_config(cls, config: Figure4Config) -> "Figure4Paths":
        """Build paths from the current working directory."""
        root_dir = Path(".").resolve()
        output_dir = root_dir / str(config.output_dir_name)
        return cls(root_dir, root_dir / str(config.cohort_xlsx_name), output_dir, output_dir / str(config.output_xlsx_name))

    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Figure4Analyzer:
    """Compute Figure 4 summaries and tests."""

    config: Figure4Config

    def get_metrics(self) -> List[Figure4Metric]:
        """Return the Figure 4 metric list."""
        return [
            Figure4Metric("figure4c", "lat_knn_radius_k5_median", "Local kNN radius (k=5, median)", "linear", "figure4c_local_knn_radius_k5_median"),
            Figure4Metric("figure4d", "lat_knn_clustering_mean", "kNN graph clustering", "linear", "figure4d_knn_graph_clustering"),
            Figure4Metric("figure4e", "lat_knn_laplacian_lambda2", "kNN graph connectivity (λ2)", "linear", "figure4e_knn_graph_connectivity_lambda2"),
            Figure4Metric("figure4f", "lat_hubness_std", "Hubness (std)", "linear", "figure4f_hubness_std"),
            Figure4Metric("figure4g", "lat_knn_degree_gini", "kNN degree inequality (Gini)", "linear", "figure4g_knn_degree_inequality_gini"),
            Figure4Metric("figure4h", "dist_to_anchor", "Distance to anchor", "linear", "figure4h_distance_to_anchor"),
            Figure4Metric("figure4i", "lat_sphericity_pr_norm", "Sphericity (PR/d)", "log", "figure4i_sphericity_pr_over_d_logy"),
            Figure4Metric("figure4j", "lat_effective_rank", "Effective rank", "log", "figure4j_effective_rank_logy"),
        ]

    def get_schematic_columns(self) -> List[str]:
        """Return columns used by the Figure 4a schematic."""
        return [
            "lat_knn_radius_k5_median",
            "lat_sphericity_pr_norm",
            "lat_knn_laplacian_lambda2",
            "lat_knn_clustering_mean",
            "lat_hubness_std",
            "lat_knn_degree_gini",
            "dist_to_anchor",
        ]

    def read_cohort(self, paths: Figure4Paths) -> pd.DataFrame:
        """Read and validate the cohort workbook."""
        metrics = self.get_metrics()
        required_columns = ["sample_id", "protein_name", "outer_label", "control_type", "bits_site_min"]
        required_columns.extend([metric.column for metric in metrics])
        required_columns.extend(self.get_schematic_columns())
        required_columns = list(dict.fromkeys(required_columns))
        if not paths.cohort_xlsx_path.exists():
            raise FileNotFoundError(f"Missing cohort workbook: {paths.cohort_xlsx_path}")
        cohort = pd.read_excel(paths.cohort_xlsx_path, sheet_name=str(self.config.cohort_sheet_name), engine="openpyxl")
        require_columns(cohort, required_columns, "cohort workbook")
        out = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type"]:
            out[column_name] = out[column_name].astype(str)
        for column_name in ["bits_site_min"] + [metric.column for metric in metrics] + self.get_schematic_columns():
            out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
        return convert_archived_columns(out, information_columns=("bits_site_min",))

    def compute_protein_summary(self, cohort: pd.DataFrame, metric_columns: Sequence[str]) -> pd.DataFrame:
        """Compute protein-level condition means."""
        base = cohort[["protein_name", "condition4", "target_stratum"] + list(metric_columns)].copy()
        for column_name in metric_columns:
            base[column_name] = pd.to_numeric(base[column_name], errors="coerce")
        return base.groupby(["protein_name", "condition4", "target_stratum"], sort=False)[list(metric_columns)].mean(numeric_only=True).reset_index()

    def build_violin_values(self, protein_summary: pd.DataFrame, metrics: Sequence[Figure4Metric]) -> pd.DataFrame:
        """Build long-form source values for Figure 4c-j."""
        rows: List[Dict[str, Any]] = []
        for metric in metrics:
            sub = protein_summary[protein_summary["condition4"].isin(["below targeted", "above targeted"])].copy()
            sub[metric.column] = pd.to_numeric(sub[metric.column], errors="coerce")
            sub = sub[np.isfinite(sub[metric.column].to_numpy(dtype=float))].copy()
            for row in sub.itertuples(index=False):
                rows.append({"panel": metric.panel, "protein_name": str(getattr(row, "protein_name")), "condition4": str(getattr(row, "condition4")), "target_stratum": str(getattr(row, "target_stratum")), "metric": metric.column, "metric_label": metric.label, "value": float(getattr(row, metric.column)), "scale": metric.scale})
        return pd.DataFrame(rows)

    def compute_hedges_g(self, group_a: Sequence[float], group_b: Sequence[float]) -> float:
        """Compute unpaired Hedges g."""
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
        pooled_numerator = (n_a - 1.0) * variance_a + (n_b - 1.0) * variance_b
        if pooled_denominator <= 0.0 or pooled_numerator <= 0.0:
            return float("nan")
        pooled_sd = math.sqrt(pooled_numerator / pooled_denominator)
        correction = 1.0 - (3.0 / (4.0 * (n_a + n_b) - 9.0)) if (n_a + n_b) > 2.0 else 1.0
        return float(correction * ((float(np.mean(a_values)) - float(np.mean(b_values))) / pooled_sd))

    def build_contrast_row(self, protein_summary: pd.DataFrame, metric: Figure4Metric, contrast_name: str, group_a_name: str, group_b_name: str, reported_in_figure: bool) -> Dict[str, Any]:
        """Build one unpaired contrast row."""
        group_a_values = pd.to_numeric(protein_summary.loc[protein_summary["condition4"] == group_a_name, metric.column], errors="coerce").to_numpy(dtype=float)
        group_b_values = pd.to_numeric(protein_summary.loc[protein_summary["condition4"] == group_b_name, metric.column], errors="coerce").to_numpy(dtype=float)
        group_a_values = group_a_values[np.isfinite(group_a_values)]
        group_b_values = group_b_values[np.isfinite(group_b_values)]
        if group_a_values.size < 3 or group_b_values.size < 3:
            p_value = float("nan")
        else:
            result = stats.ttest_ind(group_a_values, group_b_values, equal_var=False, nan_policy="omit")
            p_value = float(result.pvalue)
        return {"panel": metric.panel, "metric": metric.column, "metric_label": metric.label, "contrast": contrast_name, "reported_in_figure": bool(reported_in_figure), "test": "welch_two_sided", "analysis_unit": "protein_level_condition_mean", "group_a": group_a_name, "n_a": int(group_a_values.size), "mean_a": safe_mean(group_a_values), "sd_a": safe_sample_sd(group_a_values), "group_b": group_b_name, "n_b": int(group_b_values.size), "mean_b": safe_mean(group_b_values), "sd_b": safe_sample_sd(group_b_values), "p_value": p_value, "effect_name": "hedges_g", "effect": self.compute_hedges_g(group_a_values, group_b_values)}

    def compute_unpaired_stats(self, protein_summary: pd.DataFrame, metrics: Sequence[Figure4Metric]) -> pd.DataFrame:
        """Compute Figure 4c-j Welch tests with per-metric BH families."""
        rows: List[Dict[str, Any]] = []
        contrast_specs = [("target_below_vs_above", "below targeted", "above targeted", True), ("matched_control_below_vs_above", "below matched control", "above matched control", False)]
        for metric in metrics:
            contrast_rows = [self.build_contrast_row(protein_summary, metric, contrast_name, group_a_name, group_b_name, reported_in_figure) for contrast_name, group_a_name, group_b_name, reported_in_figure in contrast_specs]
            q_values = benjamini_hochberg([row["p_value"] for row in contrast_rows])
            for row_index, q_value in enumerate(q_values):
                contrast_rows[row_index]["q_value_bh"] = float(q_value) if np.isfinite(q_value) else float("nan")
                contrast_rows[row_index]["bh_threshold"] = float(self.config.stats_fdr_alpha)
                contrast_rows[row_index]["reject_fdr_bh"] = bool(np.isfinite(contrast_rows[row_index]["q_value_bh"]) and contrast_rows[row_index]["q_value_bh"] <= float(self.config.stats_fdr_alpha))
                contrast_rows[row_index]["correction_family"] = "within_metric_targeted_and_matched_control_below_vs_above"
                contrast_rows[row_index]["n_tests_in_family"] = int(len(contrast_specs))
            rows.extend(contrast_rows)
        return pd.DataFrame(rows)

    def compute_correlation_matrices(self, protein_summary: pd.DataFrame, metrics: Sequence[Figure4Metric]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute targeted-regime Spearman correlation matrices."""
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
        """Summarize absolute upper-triangle correlations."""
        values = corr.to_numpy(dtype=float)
        mask = np.triu(np.ones_like(values, dtype=bool), k=1)
        pair_values = np.abs(values[mask])
        pair_values = pair_values[np.isfinite(pair_values)]
        return {"regime": regime, "n_features": int(corr.shape[0]), "n_feature_pairs": int(pair_values.size), "mean_abs_spearman_rho": float(np.mean(pair_values)) if pair_values.size else float("nan"), "median_abs_spearman_rho": float(np.median(pair_values)) if pair_values.size else float("nan")}

    def compute_schematic_values(self, protein_summary: pd.DataFrame) -> pd.DataFrame:
        """Compute protein-level medians for Figure 4a."""
        rows: List[Dict[str, Any]] = []
        for condition_value, regime in [("below targeted", "below"), ("above targeted", "above")]:
            sub = protein_summary[protein_summary["condition4"] == condition_value].copy()
            row: Dict[str, Any] = {"regime": regime, "condition4": condition_value}
            for column_name in self.get_schematic_columns():
                values = pd.to_numeric(sub[column_name], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                row[column_name] = float(np.median(values)) if values.size else float("nan")
            rows.append(row)
        return pd.DataFrame(rows)

    def build_readme_table(self) -> pd.DataFrame:
        """Build a workbook manifest."""
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "cohort_input", "value": str(self.config.cohort_xlsx_name)},
            {"key": "cohort_sheet_name", "value": str(self.config.cohort_sheet_name)},
            {"key": "threshold_bits_site_min", "value": str(float(self.config.bps_min_threshold))},
            {"key": "violin_analysis_unit", "value": "protein-level means for targeted below versus targeted above"},
            {"key": "figure4c_j_statistics", "value": "Welch two-sided tests with BH correction within each metric across targeted and matched-control below-vs-above contrasts"},
            {"key": "schematic_values", "value": "protein-level medians for targeted below and targeted above"},
        ]
        return pd.DataFrame(rows)

    def build_metric_dictionary(self, metrics: Sequence[Figure4Metric]) -> pd.DataFrame:
        """Build a Figure 4 metric dictionary."""
        rows = [{"panel": metric.panel, "metric": metric.column, "metric_label": metric.label, "scale": metric.scale, "output_stem": metric.output_stem, "role": "violin"} for metric in metrics]
        rows.append({"panel": "figure4a", "metric": "schematic_parameters", "metric_label": "Latent-space schematic parameters", "scale": "mixed", "output_stem": "figure4a_latent_space_schematic_targeted_below_vs_ge_1p0", "role": "schematic_only"})
        return pd.DataFrame(rows)


@dataclass
class Figure4Plotter:
    """Render Figure 4 plots."""

    config: Figure4Config
    paths: Figure4Paths
    style: ConditionStyle

    def apply_publication_style(self) -> None:
        """Apply publication styling."""
        sns.set_style("white")
        mpl.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "lines.linewidth": 1.8, "lines.markersize": 4.0, "savefig.dpi": 300, "figure.dpi": 120, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})

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
        """Build the Figure 4 correlation colormap."""
        cmap = mpl.colors.LinearSegmentedColormap.from_list("okabe_ito_div", ["#E69F00", "#FFFFFF", "#0072B2"], N=256)
        cmap.set_bad(color="white")
        return cmap

    def save_condition_legend(self) -> None:
        """Save the standalone condition legend."""
        handles: List[Any] = []
        labels: List[str] = []
        for condition_label in self.style.get_order():
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=self.style.get_color(condition_label), markeredgecolor="none", markersize=9))
            labels.append(condition_label)
        fig = plt.figure(figsize=(4.4, 1.2))
        legend_obj = fig.legend(handles, labels, loc="center", frameon=False, ncol=2, handlelength=1.0, handletextpad=0.6, borderaxespad=0.2, columnspacing=1.2)
        fig.savefig(self.paths.output_dir / "legend_figure4_conditions.svg", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend_obj])
        fig.savefig(self.paths.output_dir / "legend_figure4_conditions.png", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend_obj])
        plt.close(fig)

    def save_correlation_colorbar(self) -> None:
        """Save the standalone correlation colorbar."""
        fig, ax = plt.subplots(figsize=(1.2, 3.0))
        fig.subplots_adjust(left=0.55, right=0.85)
        norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
        scalar_mappable = mpl.cm.ScalarMappable(cmap=self.build_correlation_cmap(), norm=norm)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, cax=ax)
        colorbar.set_label("Spearman ρ")
        ax.grid(False)
        fig.savefig(self.paths.output_dir / "legend_figure4b_spearman_correlation_colorbar.svg", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(self.paths.output_dir / "legend_figure4b_spearman_correlation_colorbar.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    def plot_violin_metric(self, violin_values: pd.DataFrame, metric: Figure4Metric) -> None:
        """Plot one horizontal targeted-only violin panel."""
        source = violin_values[violin_values["metric"] == metric.column].copy()
        source["value"] = pd.to_numeric(source["value"], errors="coerce")
        source = source[np.isfinite(source["value"].to_numpy(dtype=float))].copy()
        if source.empty:
            return
        fig, ax = plt.subplots(figsize=(5.8, 3.8))
        sns.violinplot(ax=ax, data=source, y="condition4", x="value", hue="condition4", order=["above targeted", "below targeted"], palette=self.style.get_palette(), cut=0, inner=None, linewidth=0.8, dodge=False, legend=False)
        axis_label = metric.label
        if metric.scale == "log":
            ax.set_xscale("log")
            axis_label = f"{metric.label} (log scale)"
        ax.set_xlabel(axis_label)
        ax.set_ylabel("")
        if ax.legend_ is not None:
            ax.legend_.remove()
        self.apply_axis_style(ax)
        self.save_current_figure(fig, metric.output_stem)

    def plot_correlation_heatmap(self, corr: pd.DataFrame, title_text: str, output_stem: str) -> None:
        """Plot an upper-triangle correlation heatmap."""
        label_order = corr.index.tolist()
        matrix = corr.loc[label_order, label_order].to_numpy(dtype=float, copy=True)
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
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
        self.save_current_figure(fig, output_stem)

    def normalize_between(self, value: float, lower: float, upper: float) -> float:
        """Normalize a scalar into the unit interval."""
        if not np.isfinite(value) or not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            return 0.5
        return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))

    def compute_knn_edges(self, points: np.ndarray, k_value: int, max_edges: int, rng: np.random.Generator) -> np.ndarray:
        """Return subsampled undirected edges from kNN candidates."""
        point_count = int(points.shape[0])
        if point_count <= 2:
            return np.zeros((0, 2), dtype=np.int32)
        neighbor_count = int(min(max(1, k_value), point_count - 1))
        candidates: List[Tuple[int, int, float]] = []
        for point_index in range(point_count):
            squared_distances = np.sum((points - points[point_index]) ** 2, axis=1)
            squared_distances[point_index] = np.inf
            neighbor_indices = np.argpartition(squared_distances, neighbor_count)[:neighbor_count]
            for neighbor_index in neighbor_indices:
                neighbor_value = int(neighbor_index)
                if point_index < neighbor_value:
                    candidates.append((point_index, neighbor_value, float(squared_distances[neighbor_value])))
        if not candidates:
            return np.zeros((0, 2), dtype=np.int32)
        candidates.sort(key=lambda item: item[2])
        take_count = int(min(len(candidates), int(max_edges)))
        pool = candidates[: min(len(candidates), max(10, 3 * take_count))]
        chosen = rng.choice(len(pool), size=take_count, replace=False) if len(pool) >= take_count else np.arange(len(pool))
        return np.array([[pool[index][0], pool[index][1]] for index in chosen], dtype=np.int32)

    def ellipse_radii_from_ratio(self, radius_scale: float, axis_ratio: float) -> Tuple[float, float]:
        """Compute ellipse radii with constant area."""
        base = float(max(0.05, radius_scale))
        ratio = float(max(1.0, axis_ratio))
        return base * math.sqrt(ratio), base / math.sqrt(ratio)

    def make_cloud_points(self, radius_x: float, radius_y: float, seed: int, point_count: int) -> np.ndarray:
        """Generate a seeded schematic point cloud."""
        rng = np.random.default_rng(int(seed))
        points = rng.normal(size=(int(point_count), 2)).astype(float)
        points[:, 0] *= float(radius_x)
        points[:, 1] *= float(radius_y)
        return points

    def compute_shared_scale_factor(self, left: np.ndarray, right: np.ndarray, margin: float) -> float:
        """Compute a shared schematic scale factor."""
        left_maximum = float(np.max(np.abs(left))) if left.size else 1.0
        right_maximum = float(np.max(np.abs(right))) if right.size else 1.0
        return float(margin) / max(left_maximum, right_maximum, 1e-6)

    def draw_schematic_panel(self, ax: plt.Axes, points: np.ndarray, edge_count: int, hub_strength: float, node_color: str, edge_color: str, title: str, distance_to_anchor: float) -> None:
        """Draw one latent schematic panel."""
        rng = np.random.default_rng(int(999 + int(edge_count)))
        edges = self.compute_knn_edges(points, k_value=10, max_edges=int(edge_count), rng=rng)
        for first_index, second_index in edges:
            ax.plot([points[first_index, 0], points[second_index, 0]], [points[first_index, 1], points[second_index, 1]], color=edge_color, alpha=0.25, linewidth=0.6)
        point_count = int(points.shape[0])
        hub_count = int(max(3, round(float(0.02 + 0.08 * hub_strength) * point_count)))
        hub_indices = rng.choice(point_count, size=hub_count, replace=False)
        ax.scatter(points[:, 0], points[:, 1], s=10.0, c=node_color, alpha=0.55, edgecolors="none")
        ax.scatter(points[hub_indices, 0], points[hub_indices, 1], s=10.0 * (2.8 + 3.2 * float(hub_strength)), c=node_color, alpha=0.90, edgecolors="none")
        ax.scatter([0.0], [0.0], s=20, c="#000000", alpha=0.85, marker="o")
        if np.isfinite(distance_to_anchor):
            ax.text(0.02, 0.02, f"median dist-to-anchor ≈ {distance_to_anchor:.2f}", transform=ax.transAxes, fontsize=8)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

    def plot_schematic(self, schematic_values: pd.DataFrame) -> None:
        """Plot Figure 4a schematic."""
        below_row = schematic_values[schematic_values["regime"] == "below"].iloc[0]
        above_row = schematic_values[schematic_values["regime"] == "above"].iloc[0]
        radius_below = float(below_row["lat_knn_radius_k5_median"])
        radius_above = float(above_row["lat_knn_radius_k5_median"])
        radius_maximum = max(radius_below, radius_above)
        radius_maximum = float(radius_maximum) if np.isfinite(radius_maximum) and radius_maximum > 0.0 else 1.0
        radius_scale_below = float(np.clip(radius_below / radius_maximum, 0.15, 1.0))
        radius_scale_above = float(np.clip(radius_above / radius_maximum, 0.15, 1.0))
        sphericity_below = float(below_row["lat_sphericity_pr_norm"])
        sphericity_above = float(above_row["lat_sphericity_pr_norm"])
        epsilon = 1e-12
        log_sphericity_below = float(np.log10(max(sphericity_below, epsilon))) if np.isfinite(sphericity_below) else float("nan")
        log_sphericity_above = float(np.log10(max(sphericity_above, epsilon))) if np.isfinite(sphericity_above) else float("nan")
        finite_logs = [value for value in [log_sphericity_below, log_sphericity_above] if np.isfinite(value)]
        sphericity_lower = min(finite_logs) if finite_logs else -6.0
        sphericity_upper = max(finite_logs) if finite_logs else -1.0
        axis_ratio_below = float(1.0 + (1.0 - self.normalize_between(log_sphericity_below, sphericity_lower, sphericity_upper)) * 2.5)
        axis_ratio_above = float(1.0 + (1.0 - self.normalize_between(log_sphericity_above, sphericity_lower, sphericity_upper)) * 2.5)
        radius_x_below, radius_y_below = self.ellipse_radii_from_ratio(radius_scale_below, axis_ratio_below)
        radius_x_above, radius_y_above = self.ellipse_radii_from_ratio(radius_scale_above, axis_ratio_above)
        lambda_below = float(below_row["lat_knn_laplacian_lambda2"])
        lambda_above = float(above_row["lat_knn_laplacian_lambda2"])
        clustering_below = float(below_row["lat_knn_clustering_mean"])
        clustering_above = float(above_row["lat_knn_clustering_mean"])
        lambda_lower = min([value for value in [lambda_below, lambda_above] if np.isfinite(value)] or [0.0])
        lambda_upper = max([value for value in [lambda_below, lambda_above] if np.isfinite(value)] or [1.0])
        clustering_lower = min([value for value in [clustering_below, clustering_above] if np.isfinite(value)] or [0.0])
        clustering_upper = max([value for value in [clustering_below, clustering_above] if np.isfinite(value)] or [1.0])
        edge_count_below = int(420 + 380 * (0.65 * self.normalize_between(lambda_below, lambda_lower, lambda_upper) + 0.35 * self.normalize_between(clustering_below, clustering_lower, clustering_upper)))
        edge_count_above = int(420 + 380 * (0.65 * self.normalize_between(lambda_above, lambda_lower, lambda_upper) + 0.35 * self.normalize_between(clustering_above, clustering_lower, clustering_upper)))
        hubness_below = float(below_row["lat_hubness_std"])
        hubness_above = float(above_row["lat_hubness_std"])
        gini_below = float(below_row["lat_knn_degree_gini"])
        gini_above = float(above_row["lat_knn_degree_gini"])
        hubness_lower = min([value for value in [hubness_below, hubness_above] if np.isfinite(value)] or [0.0])
        hubness_upper = max([value for value in [hubness_below, hubness_above] if np.isfinite(value)] or [1.0])
        gini_lower = min([value for value in [gini_below, gini_above] if np.isfinite(value)] or [0.0])
        gini_upper = max([value for value in [gini_below, gini_above] if np.isfinite(value)] or [1.0])
        hub_strength_below = float(np.clip(0.6 * self.normalize_between(hubness_below, hubness_lower, hubness_upper) + 0.4 * self.normalize_between(gini_below, gini_lower, gini_upper), 0.0, 1.0))
        hub_strength_above = float(np.clip(0.6 * self.normalize_between(hubness_above, hubness_lower, hubness_upper) + 0.4 * self.normalize_between(gini_above, gini_lower, gini_upper), 0.0, 1.0))
        cloud_below = self.make_cloud_points(radius_x_below, radius_y_below, seed=101, point_count=260)
        cloud_above = self.make_cloud_points(radius_x_above, radius_y_above, seed=202, point_count=260)
        shared_scale = self.compute_shared_scale_factor(cloud_below, cloud_above, margin=0.92)
        cloud_below *= shared_scale
        cloud_above *= shared_scale
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True, sharey=True)
        threshold = float(self.config.bps_min_threshold)
        below_color = self.style.get_color("below targeted")
        above_color = self.style.get_color("above targeted")
        self.draw_schematic_panel(axes[0], cloud_below, edge_count_below, hub_strength_below, below_color, self.style.blend_hex_with_white(below_color, 0.65), f"Targeted < {threshold:.1f} bits/site_min", float(below_row["dist_to_anchor"]))
        self.draw_schematic_panel(axes[1], cloud_above, edge_count_above, hub_strength_above, above_color, self.style.blend_hex_with_white(above_color, 0.65), f"Targeted ≥ {threshold:.1f} bits/site_min", float(above_row["dist_to_anchor"]))
        self.save_current_figure(fig, "figure4a_latent_space_schematic_targeted_below_vs_ge_1p0")


@dataclass
class Figure4Runner:
    """Coordinate Figure 4 generation."""

    config: Figure4Config

    def run(self) -> None:
        """Run the Figure 4 pipeline."""
        paths = Figure4Paths.from_config(self.config)
        paths.ensure_dirs()
        analyzer = Figure4Analyzer(self.config)
        style = ConditionStyle()
        plotter = Figure4Plotter(self.config, paths, style)
        plotter.apply_publication_style()
        metrics = analyzer.get_metrics()
        cohort = analyzer.read_cohort(paths)
        cohort = add_threshold_condition_columns(cohort, threshold=float(self.config.bps_min_threshold))
        metric_columns = list(dict.fromkeys([metric.column for metric in metrics] + analyzer.get_schematic_columns()))
        protein_summary = analyzer.compute_protein_summary(cohort, metric_columns)
        violin_values = analyzer.build_violin_values(protein_summary, metrics)
        stats_table = analyzer.compute_unpaired_stats(protein_summary, metrics)
        corr_below, corr_above = analyzer.compute_correlation_matrices(protein_summary, metrics)
        corr_summary = pd.DataFrame([analyzer.summarize_correlation_matrix("below targeted", corr_below), analyzer.summarize_correlation_matrix("above targeted", corr_above)])
        schematic_values = analyzer.compute_schematic_values(protein_summary)
        plotter.save_condition_legend()
        plotter.plot_schematic(schematic_values)
        plotter.plot_correlation_heatmap(corr_below, f"Targeted correlations (< {self.config.bps_min_threshold:.1f} bits/site_min)", "figure4b_left_targeted_below_1p0_latent_metric_correlations")
        plotter.plot_correlation_heatmap(corr_above, f"Targeted correlations (≥ {self.config.bps_min_threshold:.1f} bits/site_min)", "figure4b_right_targeted_ge_1p0_latent_metric_correlations")
        plotter.save_correlation_colorbar()
        for metric in metrics:
            plotter.plot_violin_metric(violin_values, metric)
        tables = {"readme": analyzer.build_readme_table(), "figure4a_schematic_values": schematic_values, "figure4b_corr_below_targeted": corr_below.reset_index().rename(columns={"index": "metric"}), "figure4b_corr_above_targeted": corr_above.reset_index().rename(columns={"index": "metric"}), "figure4b_correlation_summary": corr_summary, "figure4c_j_violin_values": violin_values, "figure4c_j_unpaired_welch": stats_table, "figure4_metric_dictionary": analyzer.build_metric_dictionary(metrics)}
        writer = ExcelNotebookWriter(paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        print(f"Wrote Figure 4 outputs to {paths.output_dir}")


def main() -> None:
    """Generate Figure 4 outputs."""
    runner = Figure4Runner(Figure4Config())
    runner.run()


if __name__ == "__main__":
    main()
