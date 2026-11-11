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

from figure_common import ExcelNotebookWriter
from figure_common import operational_bits_site_threshold
from figure_common import require_columns
from figure_common import convert_archived_columns
from figure_common import safe_mean
from figure_common import safe_sample_sd


@dataclass(frozen=True)
class Figure2Config:
    """Configure Figure 2 and Supplementary Figure 2 outputs."""

    cohort_xlsx_name: str = "cohort.xlsx"
    cohort_sheet_name: str = "by_sample"
    output_dir_name: str = "figure2_and_supplementary_figure2_outputs"
    output_xlsx_name: str = "figure2_and_supplementary_figure2_source_data.xlsx"
    bps_min_threshold: float = operational_bits_site_threshold
    softmax_temperature: float = 1.0
    excel_chunk_rows: int = 1000000
    pid_colormap_name: str = "cividis"


@dataclass(frozen=True)
class Figure2Paths:
    """Resolve input and output paths for Figure 2."""

    root_dir: Path
    cohort_xlsx_path: Path
    output_dir: Path
    output_xlsx_path: Path

    @classmethod
    def from_root_dir(cls, root_dir: Path, config: Figure2Config) -> "Figure2Paths":
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


class ControlTypeStyle:
    """Assign fixed colors to MSA control types."""

    def __init__(self) -> None:
        """Initialize the control-type palette."""
        self.palette = {
            "targeted": "#0072B2",
            "null_random": "#E69F00",
        }
        self.fallback_color = "#999999"

    def get_color(self, control_type: str) -> str:
        """Return a color for a control type."""
        key = str(control_type).lower()
        return self.palette.get(key, self.fallback_color)

    def get_present_order(self, present_values: Sequence[str]) -> List[str]:
        """Return control types in display order."""
        base_order = ["targeted", "null_random"]
        present_lower = [str(value).lower() for value in present_values]
        ordered_values = [value for value in base_order if value in present_lower]
        extra_values = sorted([value for value in set(present_lower) if value not in ordered_values])
        return ordered_values + extra_values


@dataclass
class Figure2Analyzer:
    """Compute Figure 2 and Supplementary Figure 2 source-data tables."""

    config: Figure2Config

    def read_cohort_by_sample(self, xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
        """Read and validate the cohort by-sample sheet."""
        required_columns = [
            "sample_id",
            "protein_name",
            "outer_label",
            "control_type",
            "bits_site_min",
            "snr_used",
            "pid_pair_mean",
            "max_protein_id",
            "plddt_mean",
            "tm_mean",
        ]
        if not Path(xlsx_path).exists():
            raise FileNotFoundError(f"Missing cohort workbook: {xlsx_path}")
        cohort = pd.read_excel(xlsx_path, sheet_name=str(sheet_name), engine="openpyxl")
        require_columns(cohort, required_columns, "cohort workbook")
        out = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type"]:
            out[column_name] = out[column_name].astype(str)
        numeric_columns = [
            "bits_site_min",
            "snr_used",
            "pid_pair_mean",
            "max_protein_id",
            "plddt_mean",
            "tm_mean",
            "meff_theta0p70",
            "n_sequences",
        ]
        for column_name in numeric_columns:
            if column_name in out.columns:
                out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
        return convert_archived_columns(out, information_columns=("bits_site_min",))

    def compute_pid_pair_std(self, pid_pair_mean: pd.Series, snr_used: pd.Series) -> pd.Series:
        """Derive pairwise-identity standard deviation from mean PID and MSA-SNR."""
        mean_values = pd.to_numeric(pid_pair_mean, errors="coerce")
        snr_values = pd.to_numeric(snr_used, errors="coerce")
        output = pd.Series(np.nan, index=mean_values.index, dtype=float)
        mean_array = mean_values.to_numpy(dtype=float)
        snr_array = snr_values.to_numpy(dtype=float)
        valid_mask = np.isfinite(mean_array) & np.isfinite(snr_array) & (snr_array > 0.0)
        output.loc[valid_mask] = mean_array[valid_mask] / snr_array[valid_mask]
        return output

    def compute_msa_level_table(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Aggregate candidate rows into one row per protein and MSA."""
        df = cohort.copy()
        if "meff_theta0p70" not in df.columns:
            df["meff_theta0p70"] = np.nan
        if "n_sequences" not in df.columns:
            df["n_sequences"] = np.nan
        df["novelty_pct"] = 100.0 - pd.to_numeric(df["max_protein_id"], errors="coerce")
        df["novelty_frac"] = df["novelty_pct"] / 100.0
        df["score_plddt"] = pd.to_numeric(df["plddt_mean"], errors="coerce") * df["novelty_frac"]
        df["score_tm"] = pd.to_numeric(df["tm_mean"], errors="coerce") * df["novelty_frac"]
        group_columns = ["protein_name", "outer_label"]
        grouped = df.groupby(group_columns, sort=False)
        msa_table = grouped.agg(
            control_type=("control_type", "first"),
            bits_site_min=("bits_site_min", "first"),
            snr_used=("snr_used", "first"),
            pid_pair_mean=("pid_pair_mean", "first"),
            n_candidates=("sample_id", "size"),
            max_plddt=("plddt_mean", "max"),
            max_tm=("tm_mean", "max"),
            max_novelty=("novelty_pct", "max"),
            median_novelty=("novelty_pct", "median"),
            max_score_plddt=("score_plddt", "max"),
            max_score_tm=("score_tm", "max"),
            meff_theta0p70=("meff_theta0p70", "first"),
            n_sequences=("n_sequences", "first"),
        ).reset_index()
        bits_unique = grouped["bits_site_min"].nunique().reset_index(name="bits_site_min_nunique")
        msa_table = msa_table.merge(bits_unique, on=group_columns, how="left", validate="one_to_one")
        msa_table["pid_pair_std"] = self.compute_pid_pair_std(msa_table["pid_pair_mean"], msa_table["snr_used"])
        return msa_table

    def compute_spearman_rho_p(self, x_values: Sequence[float], y_values: Sequence[float]) -> Tuple[float, float, int]:
        """Compute Spearman correlation with the finite-count threshold."""
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
        count_value = int(finite_mask.sum())
        if count_value < 10:
            return float("nan"), float("nan"), count_value
        rho_value, p_value = stats.spearmanr(x_array[finite_mask], y_array[finite_mask])
        return float(rho_value), float(p_value), count_value

    def compute_spearman_table(self, msa_table: pd.DataFrame) -> pd.DataFrame:
        """Compute pooled and control-type-specific Spearman correlations."""
        panel_specs = [
            ("figure2a", "snr_used", "max_plddt"),
            ("figure2b", "snr_used", "max_tm"),
            ("supplementary_figure2a", "meff_theta0p70", "max_plddt"),
            ("supplementary_figure2b", "meff_theta0p70", "max_tm"),
            ("supplementary_figure2c", "n_sequences", "max_plddt"),
            ("supplementary_figure2d", "n_sequences", "max_tm"),
        ]
        group_specs = [("all", msa_table)]
        for control_type, group_table in msa_table.groupby(msa_table["control_type"].astype(str).str.lower(), sort=False):
            group_specs.append((str(control_type), group_table))
        rows: List[Dict[str, Any]] = []
        for group_name, group_table in group_specs:
            for panel_name, x_column, y_column in panel_specs:
                rho_value, p_value, count_value = self.compute_spearman_rho_p(group_table[x_column], group_table[y_column])
                rows.append(
                    {
                        "panel": str(panel_name),
                        "analysis_group": str(group_name),
                        "x_column": str(x_column),
                        "y_column": str(y_column),
                        "n_msa": int(count_value),
                        "spearman_rho": float(rho_value),
                        "p_value": float(p_value),
                    }
                )
        return pd.DataFrame(rows)

    def compute_peak_location_table(self, cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute per-protein observed maximum MSA-SNR locations."""
        required_columns = ["protein_name", "outer_label", "control_type", "snr_used", "bits_site_min"]
        require_columns(cohort, required_columns, "Figure 2 peak-location cohort")
        df = cohort[required_columns].drop_duplicates().copy()
        df["snr_used"] = pd.to_numeric(df["snr_used"], errors="coerce")
        df["bits_site_min"] = pd.to_numeric(df["bits_site_min"], errors="coerce")
        rows: List[Dict[str, Any]] = []
        for protein_name, protein_table in df.groupby("protein_name", sort=False):
            snr_values = protein_table["snr_used"].to_numpy(dtype=float)
            bits_values = protein_table["bits_site_min"].to_numpy(dtype=float)
            if snr_values.size == 0:
                continue
            max_snr = float(np.nanmax(snr_values))
            if not np.isfinite(max_snr):
                continue
            tied_indices = np.where(snr_values == max_snr)[0]
            if tied_indices.size == 0:
                continue
            selected_index = int(tied_indices[0])
            selected_bits = float(bits_values[selected_index])
            if not np.isfinite(selected_bits):
                continue
            tied_bits = bits_values[tied_indices]
            tied_bits = tied_bits[np.isfinite(tied_bits)]
            rows.append(
                {
                    "protein_name": str(protein_name),
                    "bps_at_max_snr": selected_bits,
                    "snr_at_max": max_snr,
                    "n_msa_designs_at_maximum": int(tied_indices.size),
                    "n_distinct_bps_at_max_snr": int(pd.Series(tied_bits).nunique()) if tied_bits.size else 0,
                    "all_bps_at_max_snr": ";".join([f"{value:.12g}" for value in tied_bits]),
                }
            )
        peak_table = pd.DataFrame(rows)
        if peak_table.empty:
            return peak_table, pd.DataFrame()
        temperature = float(self.config.softmax_temperature)
        snr_array = peak_table["snr_at_max"].to_numpy(dtype=float)
        bits_array = peak_table["bps_at_max_snr"].to_numpy(dtype=float)
        weights = np.exp((snr_array - float(np.max(snr_array))) / temperature)
        weight_sum = float(np.sum(weights))
        if weight_sum > 0.0:
            normalized_weights = weights / weight_sum
            weighted_mean = float(np.sum(normalized_weights * bits_array))
            weighted_variance = float(np.sum(normalized_weights * (bits_array - weighted_mean) ** 2))
            weighted_sd = float(math.sqrt(weighted_variance))
        else:
            normalized_weights = np.full_like(weights, np.nan, dtype=float)
            weighted_mean = float("nan")
            weighted_sd = float("nan")
        peak_table["softmax_weight"] = normalized_weights
        threshold = float(self.config.bps_min_threshold)
        summary_table = pd.DataFrame(
            [
                {"metric": "n_proteins", "value": int(peak_table.shape[0])},
                {"metric": "unweighted_mean_bps", "value": float(np.mean(bits_array))},
                {"metric": "unweighted_sd_bps", "value": float(np.std(bits_array, ddof=0))},
                {"metric": "weighted_mean_bps", "value": weighted_mean},
                {"metric": "weighted_sd_bps", "value": weighted_sd},
                {"metric": "frac_bps_below_threshold", "value": float(np.mean(bits_array < threshold))},
                {"metric": "softmax_temperature", "value": float(temperature)},
            ]
        )
        return peak_table, summary_table

    def build_novelty_source_table(self, msa_table: pd.DataFrame) -> pd.DataFrame:
        """Build the MSA-level source table for Figure 2i."""
        threshold = float(self.config.bps_min_threshold)
        source = msa_table[["protein_name", "outer_label", "control_type", "bits_site_min", "max_novelty"]].copy()
        source["bits_site_min"] = pd.to_numeric(source["bits_site_min"], errors="coerce")
        source["max_novelty"] = pd.to_numeric(source["max_novelty"], errors="coerce")
        source = source[np.isfinite(source["bits_site_min"].to_numpy(dtype=float)) & np.isfinite(source["max_novelty"].to_numpy(dtype=float))].copy()
        source["threshold_group"] = np.where(source["bits_site_min"] < threshold, f"below {threshold:.1f}", f"at or above {threshold:.1f}")
        return source.reset_index(drop=True)

    def compute_novelty_welch_stats(self, novelty_source: pd.DataFrame) -> pd.DataFrame:
        """Compute the Figure 2i Welch test without multiple-testing correction."""
        threshold = float(self.config.bps_min_threshold)
        below_label = f"below {threshold:.1f}"
        above_label = f"at or above {threshold:.1f}"
        below_values = pd.to_numeric(novelty_source.loc[novelty_source["threshold_group"] == below_label, "max_novelty"], errors="coerce").to_numpy(dtype=float)
        above_values = pd.to_numeric(novelty_source.loc[novelty_source["threshold_group"] == above_label, "max_novelty"], errors="coerce").to_numpy(dtype=float)
        below_values = below_values[np.isfinite(below_values)]
        above_values = above_values[np.isfinite(above_values)]
        if below_values.size < 3 or above_values.size < 3:
            p_value = float("nan")
        else:
            result = stats.ttest_ind(below_values, above_values, equal_var=False, nan_policy="omit")
            p_value = float(result.pvalue)
        return pd.DataFrame(
            [
                {
                    "panel": "figure2i",
                    "test": "welch_two_sided",
                    "group_a": below_label,
                    "n_a": int(below_values.size),
                    "mean_a": safe_mean(below_values),
                    "sd_a": safe_sample_sd(below_values),
                    "group_b": above_label,
                    "n_b": int(above_values.size),
                    "mean_b": safe_mean(above_values),
                    "sd_b": safe_sample_sd(above_values),
                    "p_value": p_value,
                }
            ]
        )

    def build_readme_table(self) -> pd.DataFrame:
        """Build a concise workbook manifest."""
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "cohort_input", "value": str(self.config.cohort_xlsx_name)},
            {"key": "cohort_sheet_name", "value": str(self.config.cohort_sheet_name)},
            {"key": "msa_unit", "value": "protein_name plus outer_label"},
            {"key": "bps_min_threshold", "value": str(float(self.config.bps_min_threshold))},
            {"key": "softmax_temperature", "value": str(float(self.config.softmax_temperature))},
            {"key": "novelty_summary_unit", "value": "msa"},
            {"key": "pid_pair_std", "value": "Derived as pid_pair_mean divided by snr_used."},
            {"key": "figure2a_output", "value": "figure2a_snr_vs_max_plddt_msa"},
            {"key": "figure2b_output", "value": "figure2b_snr_vs_max_tm_msa"},
            {"key": "figure2c_output", "value": "figure2c_snr_vs_bits_site_min_pid_pair_mean"},
            {"key": "figure2d_output", "value": "figure2d_snr_vs_bits_site_min_pid_pair_std"},
            {"key": "figure2e_output", "value": "figure2e_bps_at_max_snr_unweighted"},
            {"key": "figure2f_output", "value": "figure2f_bps_at_max_snr_weighted"},
            {"key": "figure2g_output", "value": "figure2g_bits_site_min_vs_max_score_plddt_msa"},
            {"key": "figure2h_output", "value": "figure2h_bits_site_min_vs_max_score_tm_msa"},
            {"key": "figure2i_output", "value": "figure2i_max_novelty_below_vs_above_1p0"},
            {"key": "supplementary_figure2a_output", "value": "supplementary_figure2a_meff_theta0p70_vs_max_plddt_msa"},
            {"key": "supplementary_figure2b_output", "value": "supplementary_figure2b_meff_theta0p70_vs_max_tm_msa"},
            {"key": "supplementary_figure2c_output", "value": "supplementary_figure2c_n_sequences_vs_max_plddt_msa"},
            {"key": "supplementary_figure2d_output", "value": "supplementary_figure2d_n_sequences_vs_max_tm_msa"},
            {"key": "supplementary_figure2e_output", "value": "supplementary_figure2e_bps_at_max_snr_vs_max_snr"},
        ]
        return pd.DataFrame(rows)


@dataclass
class Figure2Plotter:
    """Render Figure 2 and Supplementary Figure 2 plots."""

    config: Figure2Config
    paths: Figure2Paths
    style: ControlTypeStyle

    def apply_publication_style(self) -> None:
        """Apply the Figure 2 publication style."""
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
                "lines.linewidth": 1.6,
                "lines.markersize": 4.0,
                "savefig.dpi": 300,
                "figure.dpi": 120,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def apply_axis_style(self, ax: plt.Axes) -> None:
        """Apply consistent axis styling."""
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

    def save_control_type_legend(self, present_values: Sequence[str]) -> None:
        """Save a standalone control-type legend."""
        order = self.style.get_present_order(present_values)
        handles: List[Any] = []
        labels: List[str] = []
        for key in order:
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=self.style.get_color(key), markeredgecolor="none", markersize=9))
            labels.append(key)
        fig = plt.figure(figsize=(4.0, 1.1))
        legend_obj = fig.legend(handles, labels, loc="center", frameon=False, ncol=2, handlelength=1.0, handletextpad=0.6, borderaxespad=0.2, columnspacing=1.2)
        fig.savefig(self.paths.output_dir / "legend_figure2_control_type.svg", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend_obj])
        fig.savefig(self.paths.output_dir / "legend_figure2_control_type.png", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend_obj])
        plt.close(fig)

    def save_colorbar_legend(self, cmap_name: str, vmin: float, vmax: float, label: str, output_stem: str) -> None:
        """Save a standalone continuous colorbar."""
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            return
        fig, ax = plt.subplots(figsize=(1.2, 3.0))
        fig.subplots_adjust(left=0.55, right=0.85, top=0.98, bottom=0.02)
        cmap = plt.get_cmap(cmap_name)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        colorbar = mpl.colorbar.ColorbarBase(ax=ax, cmap=cmap, norm=norm, orientation="vertical")
        colorbar.set_label(label, fontsize=10)
        colorbar.ax.tick_params(labelsize=9)
        fig.savefig(self.paths.output_dir / f"{output_stem}.svg", dpi=300, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(self.paths.output_dir / f"{output_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)

    def plot_scatter_control_type(self, source: pd.DataFrame, x_column: str, y_column: str, x_label: str, y_label: str, title: str, output_stem: str) -> None:
        """Plot a control-type-colored scatter without embedded legend."""
        plot_table = source[[x_column, y_column, "control_type"]].copy()
        plot_table[x_column] = pd.to_numeric(plot_table[x_column], errors="coerce")
        plot_table[y_column] = pd.to_numeric(plot_table[y_column], errors="coerce")
        finite_mask = np.isfinite(plot_table[x_column].to_numpy(dtype=float)) & np.isfinite(plot_table[y_column].to_numpy(dtype=float))
        plot_table = plot_table.loc[finite_mask].copy()
        if plot_table.empty:
            return
        colors = plot_table["control_type"].map(self.style.get_color)
        fig, ax = plt.subplots(figsize=(5.4, 3.8))
        ax.scatter(plot_table[x_column], plot_table[y_column], s=26.0, c=colors, alpha=0.70, edgecolors="none")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self.apply_axis_style(ax)
        self.save_current_figure(fig, output_stem)

    def plot_scatter_colormap(self, source: pd.DataFrame, x_column: str, y_column: str, color_column: str, x_label: str, y_label: str, title: str, output_stem: str) -> Tuple[float, float]:
        """Plot a continuous-color scatter without embedded colorbar."""
        plot_table = source[[x_column, y_column, color_column]].copy()
        for column_name in [x_column, y_column, color_column]:
            plot_table[column_name] = pd.to_numeric(plot_table[column_name], errors="coerce")
        finite_mask = np.isfinite(plot_table[x_column].to_numpy(dtype=float)) & np.isfinite(plot_table[y_column].to_numpy(dtype=float)) & np.isfinite(plot_table[color_column].to_numpy(dtype=float))
        plot_table = plot_table.loc[finite_mask].copy()
        if plot_table.empty:
            return float("nan"), float("nan")
        values = plot_table[color_column].to_numpy(dtype=float)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        cmap = plt.get_cmap(self.config.pid_colormap_name)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots(figsize=(5.8, 3.9))
        ax.scatter(plot_table[x_column], plot_table[y_column], s=24.0, c=values, cmap=cmap, norm=norm, alpha=0.80, edgecolors="none")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self.apply_axis_style(ax)
        self.save_current_figure(fig, output_stem)
        return vmin, vmax

    def plot_histogram_unweighted(self, values: Sequence[float], x_label: str, title: str, output_stem: str) -> None:
        """Plot an unweighted peak-location histogram."""
        value_array = np.asarray(values, dtype=float)
        value_array = value_array[np.isfinite(value_array)]
        if value_array.size == 0:
            return
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.hist(value_array, bins=18, color="#56B4E9", edgecolor="black", linewidth=0.6, alpha=0.9)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Protein count")
        ax.set_title(title)
        self.apply_axis_style(ax)
        self.save_current_figure(fig, output_stem)

    def plot_histogram_weighted(self, values: Sequence[float], weights: Sequence[float], x_label: str, title: str, output_stem: str) -> None:
        """Plot a weighted peak-location histogram."""
        value_array = np.asarray(values, dtype=float)
        weight_array = np.asarray(weights, dtype=float)
        finite_mask = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array >= 0.0)
        value_array = value_array[finite_mask]
        weight_array = weight_array[finite_mask]
        if value_array.size == 0:
            return
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        ax.hist(value_array, bins=18, weights=weight_array, color="#0072B2", edgecolor="black", linewidth=0.6, alpha=0.9)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Weighted protein density")
        ax.set_title(title)
        self.apply_axis_style(ax)
        self.save_current_figure(fig, output_stem)

    def plot_peak_strength_scatter(self, source: pd.DataFrame, output_stem: str) -> None:
        """Plot per-protein peak strength versus peak location."""
        plot_table = source[["bps_at_max_snr", "snr_at_max"]].copy()
        plot_table["bps_at_max_snr"] = pd.to_numeric(plot_table["bps_at_max_snr"], errors="coerce")
        plot_table["snr_at_max"] = pd.to_numeric(plot_table["snr_at_max"], errors="coerce")
        finite_mask = np.isfinite(plot_table["bps_at_max_snr"].to_numpy(dtype=float)) & np.isfinite(plot_table["snr_at_max"].to_numpy(dtype=float))
        plot_table = plot_table.loc[finite_mask]
        if plot_table.empty:
            return
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        ax.scatter(plot_table["bps_at_max_snr"], plot_table["snr_at_max"], s=30.0, color="#999999", alpha=0.75, edgecolors="none")
        ax.set_xlabel("bits/site(min) at max MSA-SNR")
        ax.set_ylabel("Max MSA-SNR")
        ax.set_title("Per-protein max-SNR strength vs location")
        self.apply_axis_style(ax)
        self.save_current_figure(fig, output_stem)

    def plot_max_novelty_violin(self, source: pd.DataFrame, output_stem: str) -> None:
        """Plot Figure 2i max novelty below versus above threshold."""
        plot_table = source[["threshold_group", "max_novelty"]].copy()
        threshold = float(self.config.bps_min_threshold)
        order = [f"below {threshold:.1f}", f"at or above {threshold:.1f}"]
        palette = {order[0]: "#56B4E9", order[1]: "#0072B2"}
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        sns.violinplot(
            data=plot_table,
            x="threshold_group",
            y="max_novelty",
            hue="threshold_group",
            order=order,
            hue_order=order,
            palette=palette,
            dodge=False,
            cut=0,
            inner="box",
            linewidth=0.9,
            saturation=1.0,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Max novelty")
        ax.set_title("Max novelty by MSA information threshold")
        self.apply_axis_style(ax)
        self.save_current_figure(fig, output_stem)


@dataclass
class Figure2Runner:
    """Coordinate Figure 2 analysis, plotting, and source-data export."""

    config: Figure2Config

    def run(self) -> None:
        """Run the Figure 2 and Supplementary Figure 2 build."""
        paths = Figure2Paths.from_root_dir(Path(".").resolve(), self.config)
        paths.ensure_dirs()
        analyzer = Figure2Analyzer(self.config)
        style = ControlTypeStyle()
        plotter = Figure2Plotter(self.config, paths, style)
        plotter.apply_publication_style()
        cohort = analyzer.read_cohort_by_sample(paths.cohort_xlsx_path, self.config.cohort_sheet_name)
        msa_table = analyzer.compute_msa_level_table(cohort)
        spearman_table = analyzer.compute_spearman_table(msa_table)
        peak_table, peak_summary = analyzer.compute_peak_location_table(cohort)
        novelty_source = analyzer.build_novelty_source_table(msa_table)
        novelty_stats = analyzer.compute_novelty_welch_stats(novelty_source)
        present_control_types = msa_table["control_type"].dropna().astype(str).unique().tolist()
        plotter.save_control_type_legend(present_control_types)
        plotter.plot_scatter_control_type(msa_table, "snr_used", "max_plddt", "MSA-SNR", "Max pLDDT", "MSA-SNR vs foldability (pLDDT)", "figure2a_snr_vs_max_plddt_msa")
        plotter.plot_scatter_control_type(msa_table, "snr_used", "max_tm", "MSA-SNR", "Max TM-score", "MSA-SNR vs foldability (TM-score)", "figure2b_snr_vs_max_tm_msa")
        mean_vmin, mean_vmax = plotter.plot_scatter_colormap(msa_table, "bits_site_min", "snr_used", "pid_pair_mean", "bits/site (min)", "MSA-SNR", "MSA-SNR vs bits/site(min) (color = PID mean)", "figure2c_snr_vs_bits_site_min_pid_pair_mean")
        plotter.save_colorbar_legend(self.config.pid_colormap_name, mean_vmin, mean_vmax, "PID mean", "legend_figure2c_pid_pair_mean_colorbar")
        std_vmin, std_vmax = plotter.plot_scatter_colormap(msa_table, "bits_site_min", "snr_used", "pid_pair_std", "bits/site (min)", "MSA-SNR", "MSA-SNR vs bits/site(min) (color = PID std)", "figure2d_snr_vs_bits_site_min_pid_pair_std")
        plotter.save_colorbar_legend(self.config.pid_colormap_name, std_vmin, std_vmax, "PID std", "legend_figure2d_pid_pair_std_colorbar")
        if not peak_table.empty:
            plotter.plot_histogram_unweighted(peak_table["bps_at_max_snr"], "bits/site(min) at max MSA-SNR", "Per-protein max-SNR location (unweighted)", "figure2e_bps_at_max_snr_unweighted")
            weighted_title = f"Per-protein max-SNR location (weighted, τ={self.config.softmax_temperature:.1f})"
            plotter.plot_histogram_weighted(peak_table["bps_at_max_snr"], peak_table["softmax_weight"], "bits/site(min) at max MSA-SNR", weighted_title, "figure2f_bps_at_max_snr_weighted")
            plotter.plot_peak_strength_scatter(peak_table, "supplementary_figure2e_bps_at_max_snr_vs_max_snr")
        plotter.plot_scatter_control_type(msa_table, "bits_site_min", "max_score_plddt", "bits/site (min)", "Max pLDDT×Novelty score", "Tradeoff score (max): pLDDT × novelty", "figure2g_bits_site_min_vs_max_score_plddt_msa")
        plotter.plot_scatter_control_type(msa_table, "bits_site_min", "max_score_tm", "bits/site (min)", "Max TM×Novelty score", "Tradeoff score (max): TM × novelty", "figure2h_bits_site_min_vs_max_score_tm_msa")
        plotter.plot_max_novelty_violin(novelty_source, "figure2i_max_novelty_below_vs_above_1p0")
        plotter.plot_scatter_control_type(msa_table, "meff_theta0p70", "max_plddt", "Meff (θ=0.70)", "Max pLDDT", "Meff vs foldability (pLDDT)", "supplementary_figure2a_meff_theta0p70_vs_max_plddt_msa")
        plotter.plot_scatter_control_type(msa_table, "meff_theta0p70", "max_tm", "Meff (θ=0.70)", "Max TM-score", "Meff vs foldability (TM-score)", "supplementary_figure2b_meff_theta0p70_vs_max_tm_msa")
        plotter.plot_scatter_control_type(msa_table, "n_sequences", "max_plddt", "Number of sequences", "Max pLDDT", "MSA depth vs foldability (pLDDT)", "supplementary_figure2c_n_sequences_vs_max_plddt_msa")
        plotter.plot_scatter_control_type(msa_table, "n_sequences", "max_tm", "Number of sequences", "Max TM-score", "MSA depth vs foldability (TM-score)", "supplementary_figure2d_n_sequences_vs_max_tm_msa")
        workbook_tables = {
            "readme": analyzer.build_readme_table(),
            "figure2a_b_snr_foldability": msa_table[["protein_name", "outer_label", "control_type", "snr_used", "max_plddt", "max_tm"]].copy(),
            "figure2_snr_spearman_stats": spearman_table,
            "figure2c_d_snr_bits_pid": msa_table[["protein_name", "outer_label", "control_type", "bits_site_min", "snr_used", "pid_pair_mean", "pid_pair_std"]].copy(),
            "figure2e_f_peak_by_protein": peak_table,
            "figure2e_f_peak_summary": peak_summary,
            "figure2g_h_tradeoff": msa_table[["protein_name", "outer_label", "control_type", "bits_site_min", "max_score_plddt", "max_score_tm", "max_plddt", "max_tm", "max_novelty"]].copy(),
            "figure2i_max_novelty": novelty_source,
            "figure2i_welch_stats": novelty_stats,
            "suppfig2a_d_meff_depth": msa_table[["protein_name", "outer_label", "control_type", "meff_theta0p70", "n_sequences", "max_plddt", "max_tm"]].copy(),
            "suppfig2e_peak_strength": peak_table[["protein_name", "bps_at_max_snr", "snr_at_max", "softmax_weight"]].copy() if not peak_table.empty else peak_table,
        }
        writer = ExcelNotebookWriter(paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(workbook_tables)
        print(f"Wrote Figure 2 outputs to {paths.output_dir}")


def main() -> None:
    """Run the Figure 2 and Supplementary Figure 2 generator."""
    runner = Figure2Runner(Figure2Config())
    runner.run()


if __name__ == "__main__":
    main()
