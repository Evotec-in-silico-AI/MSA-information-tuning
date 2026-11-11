import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from figure_common import ConditionStyle
from figure_common import operational_bits_site_threshold
from figure_common import ExcelNotebookWriter
from figure_common import add_threshold_condition_columns
from figure_common import convert_archived_columns
from figure_common import compute_spearman_with_finite_filter
from figure_common import require_columns
from figure_common import save_figure_pair


@dataclass(frozen=True)
class Figure1Config:
    """Configure Figure 1 and Supplementary Figure 1 outputs."""

    cohort_xlsx_name: str = "cohort.xlsx"
    data_summary_xlsx_name: str = "data_summary.xlsx"
    cohort_sheet_name: str = "by_sample"
    output_dir_name: str = "figure1_and_supplementary_figure1_outputs"
    output_xlsx_name: str = "figure1_and_supplementary_figure1_source_data.xlsx"
    excel_chunk_rows: int = 1000000
    bits_boundary_threshold: float = operational_bits_site_threshold
    supplementary_figure1a_protein_name: str = "P29972"
    supplementary_figure1b_protein_name: str = "P07740"
    si_scatter_color: str = "#AA3377"
    taxon_order: Tuple[str, ...] = (
        "Bacterium",
        "Animal",
        "Fungus",
        "Archaea",
        "Plant",
        "Virus/Phage",
        "Other",
        "Unknown",
    )
    functional_order: Tuple[str, ...] = (
        "Enzyme",
        "Binding/carrier/structural",
        "Membrane protein",
        "Regulatory/signaling",
        "Chaperone/proteostasis",
        "Reporter",
        "Unknown",
    )


@dataclass(frozen=True)
class Figure1Paths:
    """Resolve paths for Figure 1 inputs and outputs."""

    root_dir: Path
    cohort_xlsx_path: Path
    data_summary_xlsx_path: Path
    output_dir: Path
    output_xlsx_path: Path

    @classmethod
    def from_root_dir(cls, root_dir: Path, config: Figure1Config) -> "Figure1Paths":
        """Build paths from the working directory."""
        output_dir = root_dir / str(config.output_dir_name)
        return cls(
            root_dir=root_dir,
            cohort_xlsx_path=root_dir / str(config.cohort_xlsx_name),
            data_summary_xlsx_path=root_dir / str(config.data_summary_xlsx_name),
            output_dir=output_dir,
            output_xlsx_path=output_dir / str(config.output_xlsx_name),
        )

    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Figure1Analyzer:
    """Build Figure 1 and Supplementary Figure 1 source-data tables."""

    config: Figure1Config

    def read_data_summary(self, xlsx_path: Path) -> pd.DataFrame:
        """Read and validate the protein summary workbook."""
        required_columns = [
            "accession",
            "name",
            "species",
            "residues",
            "fig1_taxon_group",
            "fig1_functional_group",
        ]
        if not Path(xlsx_path).exists():
            raise FileNotFoundError(f"Missing data summary workbook: {xlsx_path}")
        summary = pd.read_excel(xlsx_path, engine="openpyxl")
        require_columns(summary, required_columns, "data summary workbook")
        out = summary.copy()
        out["residues"] = pd.to_numeric(out["residues"], errors="coerce")
        if out["residues"].isna().any():
            raise ValueError("Data summary workbook contains non-numeric residues values")
        for column_name in ["accession", "name", "species", "fig1_taxon_group", "fig1_functional_group"]:
            out[column_name] = out[column_name].astype(str).str.strip()
        return out

    def read_cohort_by_sample(self, xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
        """Read and validate the cohort by-sample sheet."""
        required_columns = [
            "sample_id",
            "protein_name",
            "outer_label",
            "control_type",
            "p10_used",
            "snr_used",
            "pid_pair_median",
            "bits_site_min",
        ]
        if not Path(xlsx_path).exists():
            raise FileNotFoundError(f"Missing cohort workbook: {xlsx_path}")
        cohort = pd.read_excel(xlsx_path, sheet_name=str(sheet_name), engine="openpyxl")
        require_columns(cohort, required_columns, "cohort workbook")
        out = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type"]:
            out[column_name] = out[column_name].astype(str)
        for column_name in ["p10_used", "snr_used", "pid_pair_median", "bits_site_min"]:
            out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
        return convert_archived_columns(out, information_columns=("p10_used", "bits_site_min"))

    def build_taxon_counts(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Figure 1a taxon-group counts."""
        rows: List[Dict[str, Any]] = []
        total_count = float(summary.shape[0])
        for category in self.config.taxon_order:
            count_value = int((summary["fig1_taxon_group"] == str(category)).sum())
            if count_value <= 0:
                continue
            rows.append({"taxon_group": str(category), "count": count_value, "fraction": float(count_value / total_count)})
        return pd.DataFrame(rows)

    def build_function_counts(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Figure 1b functional-group counts."""
        rows: List[Dict[str, Any]] = []
        total_count = float(summary.shape[0])
        for category in self.config.functional_order:
            count_value = int((summary["fig1_functional_group"] == str(category)).sum())
            if count_value <= 0:
                continue
            rows.append({"functional_group": str(category), "count": count_value, "fraction": float(count_value / total_count)})
        return pd.DataFrame(rows)

    def build_length_distribution(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Build the sorted protein-length source table."""
        columns = ["accession", "name", "species", "residues"]
        out = summary[columns].copy()
        out["residues"] = pd.to_numeric(out["residues"], errors="coerce")
        out = out.dropna(subset=["residues"]).sort_values("residues", kind="mergesort").reset_index(drop=True)
        out.insert(0, "rank_by_length", np.arange(1, int(out.shape[0]) + 1, dtype=int))
        return out

    def build_length_summary(self, summary: pd.DataFrame) -> pd.DataFrame:
        """Build length and cohort summary metrics."""
        residues = pd.to_numeric(summary["residues"], errors="coerce")
        rows = [
            {"metric": "n_proteins", "value": int(summary.shape[0])},
            {"metric": "n_unique_species", "value": int(summary["species"].astype(str).nunique())},
            {"metric": "n_taxon_groups", "value": int(summary["fig1_taxon_group"].astype(str).nunique())},
            {"metric": "n_function_groups", "value": int(summary["fig1_functional_group"].astype(str).nunique())},
            {"metric": "residues_min", "value": float(residues.min())},
            {"metric": "residues_q1", "value": float(residues.quantile(0.25))},
            {"metric": "residues_median", "value": float(residues.median())},
            {"metric": "residues_q3", "value": float(residues.quantile(0.75))},
            {"metric": "residues_max", "value": float(residues.max())},
        ]
        return pd.DataFrame(rows)

    def build_targeted_msa_subset(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Return deduplicated targeted MSA rows."""
        subset = cohort[cohort["control_type"].astype(str).str.lower() == "targeted"].copy()
        preferred_columns = ["protein_name", "outer_label", "control_type", "control_rep", "p10_used", "snr_used", "bits_site_min", "pid_pair_median"]
        available_columns = [column_name for column_name in preferred_columns if column_name in subset.columns]
        return subset[available_columns].drop_duplicates().reset_index(drop=True)

    def build_hero_point_table(self, targeted_msa: pd.DataFrame, protein_name: str) -> pd.DataFrame:
        """Build one fixed-protein Supplementary Figure 1 scatter table."""
        sub = targeted_msa[targeted_msa["protein_name"].astype(str) == str(protein_name)].copy()
        if sub.empty:
            raise ValueError(f"No targeted MSA rows found for protein_name={protein_name}")
        sub["p10_used"] = pd.to_numeric(sub["p10_used"], errors="coerce")
        sub["snr_used"] = pd.to_numeric(sub["snr_used"], errors="coerce")
        out = sub.dropna(subset=["p10_used", "snr_used"]).reset_index(drop=True)
        if out.empty:
            raise ValueError(f"No finite p10_used and snr_used rows found for protein_name={protein_name}")
        return out

    def build_hero_stats_table(self, negative_points: pd.DataFrame, positive_points: pd.DataFrame) -> pd.DataFrame:
        """Compute Spearman summaries for the fixed Supplementary Figure 1 examples."""
        rows = []
        rows.append(self.build_one_hero_stats("supplementary_figure1a", self.config.supplementary_figure1a_protein_name, negative_points))
        rows.append(self.build_one_hero_stats("supplementary_figure1b", self.config.supplementary_figure1b_protein_name, positive_points))
        return pd.DataFrame(rows)

    def build_one_hero_stats(self, panel: str, protein_name: str, point_table: pd.DataFrame) -> Dict[str, Any]:
        """Compute one fixed-protein Spearman summary."""
        rho_value, p_value, point_count = compute_spearman_with_finite_filter(point_table["snr_used"], point_table["p10_used"], minimum_count=3)
        return {
            "panel": str(panel),
            "protein_name": str(protein_name),
            "n_points": int(point_count),
            "spearman_rho_snr_vs_p10": float(rho_value),
            "p_value": float(p_value),
        }

    def compute_per_protein_spearman(self, cohort: pd.DataFrame, x_column: str, y_column: str, minimum_count: int = 3) -> pd.DataFrame:
        """Compute per-protein Spearman correlations after MSA-level deduplication."""
        dedupe_columns = ["protein_name", "outer_label", "control_type", "control_rep", x_column, y_column]
        available_columns = [column_name for column_name in dedupe_columns if column_name in cohort.columns]
        base = cohort[available_columns].drop_duplicates().copy()
        rows: List[Dict[str, Any]] = []
        for protein_name, sub in base.groupby("protein_name", dropna=False, sort=False):
            numeric = sub[[x_column, y_column]].apply(pd.to_numeric, errors="coerce").dropna()
            rho_value, p_value, point_count = compute_spearman_with_finite_filter(numeric[x_column], numeric[y_column], minimum_count=minimum_count)
            if point_count < int(minimum_count):
                continue
            rows.append({
                "protein_name": str(protein_name),
                "n_points": int(point_count),
                "rho": float(rho_value),
                "p_value": float(p_value),
                "x_column": str(x_column),
                "y_column": str(y_column),
            })
        return pd.DataFrame(rows)

    def compute_corr_of_corr(self, cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute Supplementary Figure 1c correlation-of-correlations tables."""
        snr = self.compute_per_protein_spearman(cohort, "snr_used", "p10_used", minimum_count=3)
        pid = self.compute_per_protein_spearman(cohort, "pid_pair_median", "p10_used", minimum_count=3)
        snr = snr.rename(columns={"rho": "rho_snr_p10", "p_value": "p_value_snr_p10", "n_points": "n_points_snr_p10"})
        pid = pid.rename(columns={"rho": "rho_pid_p10", "p_value": "p_value_pid_p10", "n_points": "n_points_pid_p10"})
        points = snr[["protein_name", "rho_snr_p10", "p_value_snr_p10", "n_points_snr_p10"]].merge(
            pid[["protein_name", "rho_pid_p10", "p_value_pid_p10", "n_points_pid_p10"]],
            on="protein_name",
            how="inner",
        )
        points = points.dropna(subset=["rho_snr_p10", "rho_pid_p10"]).reset_index(drop=True)
        rho_value, p_value, point_count = compute_spearman_with_finite_filter(points["rho_snr_p10"], points["rho_pid_p10"], minimum_count=3)
        stats_table = pd.DataFrame([
            {"metric": "spearman_rho", "value": float(rho_value)},
            {"metric": "p_value", "value": float(p_value)},
            {"metric": "n_proteins", "value": int(point_count)},
        ])
        return points, stats_table

    def compute_p10_vs_bits_site_min(self, cohort: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute Supplementary Figure 1d p10 versus minimum bits/site tables."""
        columns = ["sample_id", "protein_name", "outer_label", "control_type", "control_rep", "p10_used", "bits_site_min"]
        available_columns = [column_name for column_name in columns if column_name in cohort.columns]
        points = cohort[available_columns].drop_duplicates().copy()
        points["p10_used"] = pd.to_numeric(points["p10_used"], errors="coerce")
        points["bits_site_min"] = pd.to_numeric(points["bits_site_min"], errors="coerce")
        points = points.dropna(subset=["p10_used", "bits_site_min"]).reset_index(drop=True)
        rho_value, p_value, point_count = compute_spearman_with_finite_filter(points["p10_used"], points["bits_site_min"], minimum_count=3)
        stats_table = pd.DataFrame([
            {"metric": "spearman_rho", "value": float(rho_value)},
            {"metric": "p_value", "value": float(p_value)},
            {"metric": "n_points", "value": int(point_count)},
        ])
        return points, stats_table

    def build_bits_site_min_violin_tables(self, cohort_condition: pd.DataFrame, style: ConditionStyle) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build Supplementary Figure 1e violin source and summary tables."""
        columns = ["sample_id", "protein_name", "outer_label", "control_type", "condition4", "bits_site_min"]
        available_columns = [column_name for column_name in columns if column_name in cohort_condition.columns]
        points = cohort_condition[available_columns].copy()
        points["bits_site_min"] = pd.to_numeric(points["bits_site_min"], errors="coerce")
        points = points[points["condition4"].isin(style.get_order())].copy()
        points = points.dropna(subset=["bits_site_min"]).reset_index(drop=True)
        rows: List[Dict[str, Any]] = []
        for condition_name in style.get_order():
            sub = points[points["condition4"] == str(condition_name)]
            if sub.empty:
                continue
            rows.append({
                "condition4": str(condition_name),
                "n_points": int(sub.shape[0]),
                "mean_bits_site_min": float(sub["bits_site_min"].mean()),
                "median_bits_site_min": float(sub["bits_site_min"].median()),
                "q1_bits_site_min": float(sub["bits_site_min"].quantile(0.25)),
                "q3_bits_site_min": float(sub["bits_site_min"].quantile(0.75)),
            })
        return points, pd.DataFrame(rows)

    def build_readme_table(self) -> pd.DataFrame:
        """Build a concise workbook manifest."""
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "cohort_input", "value": str(self.config.cohort_xlsx_name)},
            {"key": "data_summary_input", "value": str(self.config.data_summary_xlsx_name)},
            {"key": "cohort_sheet_name", "value": str(self.config.cohort_sheet_name)},
            {"key": "figure1a_output", "value": "figure1a_taxon_composition"},
            {"key": "figure1b_output", "value": "figure1b_function_composition"},
            {"key": "figure1c_output", "value": "figure1c_length_distribution"},
            {"key": "supplementary_figure1a_output", "value": "supplementary_figure1a_hero_negative_snr_vs_p10"},
            {"key": "supplementary_figure1a_protein", "value": str(self.config.supplementary_figure1a_protein_name)},
            {"key": "supplementary_figure1b_output", "value": "supplementary_figure1b_hero_positive_snr_vs_p10"},
            {"key": "supplementary_figure1b_protein", "value": str(self.config.supplementary_figure1b_protein_name)},
            {"key": "supplementary_figure1c_output", "value": "supplementary_figure1c_corr_of_corr"},
            {"key": "supplementary_figure1d_output", "value": "supplementary_figure1d_p10_vs_bits_site_min"},
            {"key": "supplementary_figure1e_output", "value": "supplementary_figure1e_bits_site_min_violin"},
            {"key": "bits_boundary_threshold", "value": str(float(self.config.bits_boundary_threshold))},
            {"key": "condition_logic", "value": "Matched controls inherit below or above threshold status from the targeted MSA partner."},
        ]
        return pd.DataFrame(rows)


@dataclass
class Figure1Plotter:
    """Render Figure 1 and Supplementary Figure 1 plots."""

    config: Figure1Config
    paths: Figure1Paths
    style: ConditionStyle

    def apply_publication_style(self) -> None:
        """Apply the Figure 1 publication plotting style."""
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        mpl.rcParams["font.family"] = "Arial"
        mpl.rcParams["font.size"] = 9
        mpl.rcParams["axes.linewidth"] = 0.8
        mpl.rcParams["axes.grid"] = False
        mpl.rcParams["xtick.major.width"] = 0.8
        mpl.rcParams["ytick.major.width"] = 0.8
        mpl.rcParams["xtick.major.size"] = 3.0
        mpl.rcParams["ytick.major.size"] = 3.0
        mpl.rcParams["legend.frameon"] = False
        sns.set_theme(style="ticks", context="paper")

    def apply_axis_style(self, ax: plt.Axes) -> None:
        """Apply the Figure 1 axis style."""
        ax.grid(False)
        sns.despine(ax=ax, top=True, right=True)

    def save_figure(self, fig: plt.Figure, stem: str) -> None:
        """Save SVG and PNG copies of a figure."""
        save_figure_pair(fig, self.paths.output_dir, stem, dpi=300, bbox_inches="tight")

    def taxon_palette(self) -> Dict[str, str]:
        """Return the Figure 1 taxon palette."""
        return {
            "Bacterium": "#0072B2",
            "Animal": "#E69F00",
            "Fungus": "#009E73",
            "Archaea": "#CC79A7",
            "Plant": "#56B4E9",
            "Virus/Phage": "#D55E00",
            "Other": "#999999",
            "Unknown": "#666666",
        }

    def function_palette(self) -> Dict[str, str]:
        """Return the Figure 1 function palette."""
        return {
            "Enzyme": "#0072B2",
            "Binding/carrier/structural": "#E69F00",
            "Membrane protein": "#009E73",
            "Regulatory/signaling": "#CC79A7",
            "Chaperone/proteostasis": "#56B4E9",
            "Reporter": "#D55E00",
            "Unknown": "#666666",
        }

    def build_legend_handles(self, labels: Sequence[str], palette: Dict[str, str], present_labels: Sequence[str]) -> List[Patch]:
        """Build legend patches without counts."""
        present = {str(label) for label in present_labels}
        handles: List[Patch] = []
        for label in labels:
            if str(label) not in present:
                continue
            handles.append(Patch(facecolor=palette[str(label)], edgecolor="none", label=str(label)))
        return handles

    def save_standalone_legend(self, handles: Sequence[Any], stem: str, ncol: int = 1) -> None:
        """Save a standalone legend figure."""
        if not handles:
            return
        fig = plt.figure(figsize=(4.0, 0.45 + 0.32 * math.ceil(len(handles) / max(1, int(ncol)))))
        fig.legend(handles=handles, loc="center", frameon=False, ncol=int(ncol))
        self.save_figure(fig, stem)

    def plot_donut(self, count_table: pd.DataFrame, category_column: str, palette: Dict[str, str], stem: str, center_text: str) -> None:
        """Plot a donut chart without an embedded legend."""
        if count_table.empty:
            return
        labels = count_table[category_column].astype(str).tolist()
        colors = [palette[str(label)] for label in labels]
        values = count_table["count"].astype(float).tolist()
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.pie(
            values,
            labels=None,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 0.8},
        )
        ax.text(0.0, 0.0, str(center_text), ha="center", va="center")
        ax.set_aspect("equal")
        self.save_figure(fig, stem)

    def plot_length_distribution(self, length_distribution: pd.DataFrame, stem: str) -> None:
        """Plot the protein length distribution as a sorted lollipop plot."""
        plot_table = length_distribution.copy()
        x_values = plot_table["rank_by_length"].to_numpy(dtype=float)
        y_values = plot_table["residues"].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        ax.vlines(x_values, ymin=0.0, ymax=y_values, color="#4D4D4D", linewidth=0.8)
        ax.scatter(x_values, y_values, color="#4D4D4D", s=16, zorder=3)
        ax.set_xlabel("Proteins sorted by length")
        ax.set_ylabel("Protein length (residues)")
        ax.set_xticks([])
        self.apply_axis_style(ax)
        self.save_figure(fig, stem)

    def save_length_legend(self, stem: str) -> None:
        """Save a standalone legend for the length lollipop plot."""
        handles = [Line2D([0], [0], color="#4D4D4D", marker="o", linestyle="-", linewidth=0.8, markersize=4.0, label="One protein")]
        self.save_standalone_legend(handles, stem, ncol=1)

    def plot_hero_scatter(self, point_table: pd.DataFrame, protein_name: str, stem: str) -> None:
        """Plot one fixed-protein MSA-SNR versus p10 scatter panel."""
        if point_table.empty:
            return
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.scatter(point_table["snr_used"], point_table["p10_used"], color=self.config.si_scatter_color, s=18, alpha=0.8, edgecolors="none")
        ax.set_xlabel("MSA-SNR")
        ax.set_ylabel("bits/site (p10)")
        ax.set_title(str(protein_name))
        self.apply_axis_style(ax)
        self.save_figure(fig, stem)

    def plot_corr_of_corr_scatter(self, corr_table: pd.DataFrame, stem: str) -> None:
        """Plot Supplementary Figure 1c correlation-of-correlations scatter."""
        if corr_table.empty:
            return
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        ax.scatter(corr_table["rho_snr_p10"], corr_table["rho_pid_p10"], color=self.config.si_scatter_color, s=24, alpha=0.8, edgecolors="none")
        ax.axline((0.0, 0.0), slope=1.0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(r"$\rho$(MSA-SNR, bits/site (p10))")
        ax.set_ylabel(r"$\rho$(PID median, bits/site (p10))")
        self.apply_axis_style(ax)
        self.save_figure(fig, stem)

    def plot_p10_vs_bits_site_min(self, point_table: pd.DataFrame, stem: str) -> None:
        """Plot Supplementary Figure 1d p10 versus minimum bits/site."""
        if point_table.empty:
            return
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        ax.scatter(point_table["p10_used"], point_table["bits_site_min"], color=self.config.si_scatter_color, s=18, alpha=0.75, edgecolors="none")
        ax.set_xlabel("bits/site (p10)")
        ax.set_ylabel("bits/site (min)")
        self.apply_axis_style(ax)
        self.save_figure(fig, stem)

    def plot_bits_site_min_violin(self, point_table: pd.DataFrame, stem: str) -> None:
        """Plot Supplementary Figure 1e bits/site minimum by condition."""
        if point_table.empty:
            return
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        sns.violinplot(
            ax=ax,
            data=point_table,
            x="condition4",
            y="bits_site_min",
            hue="condition4",
            order=self.style.get_order(),
            palette=self.style.get_palette(),
            cut=0,
            inner="box",
            dodge=False,
            linewidth=0.8,
            legend=False,
        )
        ax.set_xlabel("")
        ax.set_ylabel("bits/site (min)")
        ax.tick_params(axis="x", rotation=20)
        if ax.legend_ is not None:
            ax.legend_.remove()
        self.apply_axis_style(ax)
        self.save_figure(fig, stem)

    def save_condition_legend(self, point_table: pd.DataFrame, stem: str) -> None:
        """Save a standalone legend for four condition groups."""
        present_labels = point_table["condition4"].astype(str).tolist() if not point_table.empty else []
        handles = self.build_legend_handles(self.style.get_order(), self.style.get_palette(), present_labels)
        self.save_standalone_legend(handles, stem, ncol=1)


@dataclass
class Figure1Runner:
    """Coordinate Figure 1 analysis, plotting, and source-data export."""

    config: Figure1Config

    def run(self) -> None:
        """Run the Figure 1 and Supplementary Figure 1 build."""
        paths = Figure1Paths.from_root_dir(Path(".").resolve(), self.config)
        paths.ensure_dirs()
        style = ConditionStyle()
        analyzer = Figure1Analyzer(self.config)
        plotter = Figure1Plotter(self.config, paths, style)
        plotter.apply_publication_style()
        summary = analyzer.read_data_summary(paths.data_summary_xlsx_path)
        cohort = analyzer.read_cohort_by_sample(paths.cohort_xlsx_path, self.config.cohort_sheet_name)
        cohort_condition = add_threshold_condition_columns(cohort, threshold=float(self.config.bits_boundary_threshold))
        taxon_counts = analyzer.build_taxon_counts(summary)
        function_counts = analyzer.build_function_counts(summary)
        length_distribution = analyzer.build_length_distribution(summary)
        length_summary = analyzer.build_length_summary(summary)
        targeted_msa = analyzer.build_targeted_msa_subset(cohort_condition)
        supplementary_figure1a_points = analyzer.build_hero_point_table(targeted_msa, self.config.supplementary_figure1a_protein_name)
        supplementary_figure1b_points = analyzer.build_hero_point_table(targeted_msa, self.config.supplementary_figure1b_protein_name)
        supplementary_figure1ab_stats = analyzer.build_hero_stats_table(supplementary_figure1a_points, supplementary_figure1b_points)
        supplementary_figure1c_points, supplementary_figure1c_stats = analyzer.compute_corr_of_corr(cohort_condition)
        supplementary_figure1d_points, supplementary_figure1d_stats = analyzer.compute_p10_vs_bits_site_min(cohort_condition)
        supplementary_figure1e_points, supplementary_figure1e_summary = analyzer.build_bits_site_min_violin_tables(cohort_condition, style)
        plotter.plot_donut(taxon_counts, "taxon_group", plotter.taxon_palette(), "figure1a_taxon_composition", "N=45")
        plotter.plot_donut(function_counts, "functional_group", plotter.function_palette(), "figure1b_function_composition", "N=45")
        plotter.plot_length_distribution(length_distribution, "figure1c_length_distribution")
        taxon_handles = plotter.build_legend_handles(self.config.taxon_order, plotter.taxon_palette(), taxon_counts["taxon_group"].astype(str).tolist())
        function_handles = plotter.build_legend_handles(self.config.functional_order, plotter.function_palette(), function_counts["functional_group"].astype(str).tolist())
        plotter.save_standalone_legend(taxon_handles, "legend_figure1a_taxon_groups")
        plotter.save_standalone_legend(function_handles, "legend_figure1b_function_groups")
        plotter.save_length_legend("legend_figure1c_length_distribution")
        plotter.plot_hero_scatter(supplementary_figure1a_points, self.config.supplementary_figure1a_protein_name, "supplementary_figure1a_hero_negative_snr_vs_p10")
        plotter.plot_hero_scatter(supplementary_figure1b_points, self.config.supplementary_figure1b_protein_name, "supplementary_figure1b_hero_positive_snr_vs_p10")
        plotter.plot_corr_of_corr_scatter(supplementary_figure1c_points, "supplementary_figure1c_corr_of_corr")
        plotter.plot_p10_vs_bits_site_min(supplementary_figure1d_points, "supplementary_figure1d_p10_vs_bits_site_min")
        plotter.plot_bits_site_min_violin(supplementary_figure1e_points, "supplementary_figure1e_bits_site_min_violin")
        plotter.save_condition_legend(supplementary_figure1e_points, "legend_supplementary_figure1e_condition_groups")
        tables = {
            "readme": analyzer.build_readme_table(),
            "figure1a_taxon_composition": taxon_counts,
            "figure1b_function_composition": function_counts,
            "figure1c_length_distribution": length_distribution,
            "figure1c_length_summary": length_summary,
            "supplementary_figure1a_points": supplementary_figure1a_points,
            "supplementary_figure1b_points": supplementary_figure1b_points,
            "supplementary_figure1ab_stats": supplementary_figure1ab_stats,
            "supplementary_figure1c_corr_of_corr": supplementary_figure1c_points,
            "supplementary_figure1c_stats": supplementary_figure1c_stats,
            "supplementary_figure1d_points": supplementary_figure1d_points,
            "supplementary_figure1d_stats": supplementary_figure1d_stats,
            "supplementary_figure1e_points": supplementary_figure1e_points,
            "supplementary_figure1e_summary": supplementary_figure1e_summary,
        }
        writer = ExcelNotebookWriter(paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        print(f"Wrote Figure 1 outputs to {paths.output_dir}")


def main() -> None:
    """Run the Figure 1 and Supplementary Figure 1 generator."""
    runner = Figure1Runner(Figure1Config())
    runner.run()


if __name__ == "__main__":
    main()
