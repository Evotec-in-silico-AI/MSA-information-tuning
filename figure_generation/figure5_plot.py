import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

from figure_common import ConditionStyle
from figure_common import operational_bits_site_threshold
from figure_common import ExcelNotebookWriter
from figure_common import add_threshold_condition_columns
from figure_common import convert_archived_columns
from figure_common import require_columns
from figure_common import save_figure_pair


@dataclass(frozen=True)
class Figure5Config:
    """Configure Figure 5 and Supplementary Figure 3 outputs."""

    cohort_xlsx_name: str = "cohort.xlsx"
    cohort_sheet_name: str = "by_sample"
    compiled_final_dir: str = "compiled/final"
    final_per_site_subdir: str = "per_site"
    output_dir_name: str = "figure5_and_supplementary_figure3_outputs"
    output_xlsx_name: str = "figure5_and_supplementary_figure3_source_data.xlsx"
    excel_chunk_rows: int = 1000000
    bps_min_threshold: float = operational_bits_site_threshold
    bits_axis_min: float = 0.0
    decode_steps_expected: int = 12
    plddt_thresholds: Tuple[float, float] = (0.7, 0.9)
    calibration_bins: int = 10
    novelty_quantile_bins: int = 10
    uncertainty_quantile_bins: int = 10
    msa_bits_bin_edges: Tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.25,
        4.321928,
    )


@dataclass(frozen=True)
class Figure5Paths:
    """Resolve input and output paths."""

    root_dir: Path
    cohort_xlsx_path: Path
    final_per_site_root: Path
    output_dir: Path
    output_xlsx_path: Path

    @classmethod
    def from_config(cls, config: Figure5Config) -> "Figure5Paths":
        """Build paths from the current working directory."""
        root_dir = Path(".").resolve()
        output_dir = root_dir / str(config.output_dir_name)
        final_per_site_root = root_dir / str(config.compiled_final_dir) / str(config.final_per_site_subdir)
        return cls(
            root_dir=root_dir,
            cohort_xlsx_path=root_dir / str(config.cohort_xlsx_name),
            final_per_site_root=final_per_site_root,
            output_dir=output_dir,
            output_xlsx_path=output_dir / str(config.output_xlsx_name),
        )

    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Figure5DataLoader:
    """Load cohort metadata and per-site partitions."""

    config: Figure5Config

    def read_cohort_by_sample(self, cohort_xlsx_path: Path) -> pd.DataFrame:
        """Read and validate the cohort workbook."""
        required_columns = [
            "sample_id",
            "protein_name",
            "outer_label",
            "control_type",
            "bits_site_min",
            "max_protein_id",
            "decode_steps",
            "corruption_rate_init",
        ]
        if not Path(cohort_xlsx_path).exists():
            raise FileNotFoundError(f"Missing cohort workbook: {cohort_xlsx_path}")
        cohort = pd.read_excel(cohort_xlsx_path, sheet_name=str(self.config.cohort_sheet_name), engine="openpyxl")
        require_columns(cohort, required_columns, "cohort workbook")
        output = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type"]:
            output[column_name] = output[column_name].astype(str)
        for column_name in ["bits_site_min", "max_protein_id", "decode_steps", "corruption_rate_init"]:
            output[column_name] = pd.to_numeric(output[column_name], errors="coerce")
        if output["sample_id"].astype(str).nunique() != int(output.shape[0]):
            raise ValueError("The cohort by-sample sheet has non-unique sample_id values")
        return convert_archived_columns(output, information_columns=("bits_site_min",))

    def add_condition_and_pair_columns(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Add condition labels and pair identifiers."""
        output = add_threshold_condition_columns(cohort, threshold=float(self.config.bps_min_threshold))
        pair_id_columns = self.select_pair_id_columns(output)
        output["pair_id_columns"] = ",".join(pair_id_columns)
        output["pair_id"] = output[pair_id_columns].astype(str).agg("::".join, axis=1)
        return output

    def select_pair_id_columns(self, cohort: pd.DataFrame) -> List[str]:
        """Select columns used to identify paired targeted-control rows."""
        candidate_columns = [
            "protein_name",
            "outer_label_base",
            "corruption_rate_init",
            "seed",
            "temperature",
            "decode_steps",
            "control_rep",
        ]
        existing_columns = [column_name for column_name in candidate_columns if column_name in cohort.columns]
        if "protein_name" not in existing_columns or "outer_label_base" not in existing_columns:
            raise ValueError("Pairing requires protein_name and outer_label_base")
        return existing_columns

    def list_protein_dirs(self, final_per_site_root: Path) -> List[Path]:
        """List protein partition directories."""
        root_path = Path(final_per_site_root)
        if not root_path.exists():
            return []
        return sorted([path for path in root_path.glob("protein_name=*") if path.is_dir()])

    def protein_name_from_dir(self, protein_dir: Path) -> str:
        """Extract protein name from a partition directory."""
        directory_name = Path(protein_dir).name
        return directory_name.split("=", 1)[1] if directory_name.startswith("protein_name=") else directory_name

    def read_per_site_for_protein(self, per_site_path: Path, columns: Sequence[str]) -> pd.DataFrame:
        """Read one per-site parquet table."""
        table = pq.read_table(Path(per_site_path), columns=list(columns))
        frame = table.to_pandas(self_destruct=True)
        return convert_archived_columns(frame, information_columns=("msa_bits_per_site",))


@dataclass
class Figure5Analyzer:
    """Compute Figure 5 and Supplementary Figure 3 source data."""

    config: Figure5Config

    def required_per_site_columns(self) -> List[str]:
        """Return required per-site columns."""
        return [
            "sample_id",
            "protein_name",
            "site_index",
            "msa_bits_per_site",
            "msa_occupancy",
            "mask_initial",
            "commit_step_index",
            "is_mutation",
            "plddt_mean",
            "pae_mean",
            "nll_cond_all",
            "entropy_cond_all_nats",
            "nll_uncond_all",
            "entropy_uncond_all_nats",
            "jsd_cond_uncond_all_nats",
        ]

    def add_derived_per_site(self, per_site: pd.DataFrame) -> pd.DataFrame:
        """Add derived per-site columns."""
        output = per_site.copy()
        output["sample_id"] = output["sample_id"].astype(str)
        output["protein_name"] = output["protein_name"].astype(str)
        output["site_index"] = pd.to_numeric(output["site_index"], errors="coerce").astype("Int64")
        output["is_mutation"] = pd.to_numeric(output["is_mutation"], errors="coerce").fillna(0).astype(np.int8)
        output["mask_initial"] = pd.to_numeric(output["mask_initial"], errors="coerce").fillna(0).astype(np.int8)
        output["commit_step_index"] = pd.to_numeric(output["commit_step_index"], errors="coerce").fillna(-1).astype(np.int16)
        numeric_columns = [
            "msa_bits_per_site",
            "msa_occupancy",
            "entropy_cond_all_nats",
            "nll_cond_all",
            "entropy_uncond_all_nats",
            "nll_uncond_all",
            "jsd_cond_uncond_all_nats",
            "plddt_mean",
            "pae_mean",
        ]
        for column_name in numeric_columns:
            output[column_name] = pd.to_numeric(output[column_name], errors="coerce")
        output["delta_nll_all"] = output["nll_uncond_all"] - output["nll_cond_all"]
        output["delta_entropy_all"] = output["entropy_uncond_all_nats"] - output["entropy_cond_all_nats"]
        plddt70, plddt90 = self.config.plddt_thresholds
        output["fold_plddt70"] = (output["plddt_mean"] >= float(plddt70)).astype(np.int8)
        output["fold_plddt90"] = (output["plddt_mean"] >= float(plddt90)).astype(np.int8)
        output["plddt_mean_0_100"] = output["plddt_mean"] * 100.0
        return output

    def compute_sample_aggregates(self, per_site: pd.DataFrame, cohort_meta: pd.DataFrame) -> pd.DataFrame:
        """Compute per-sample aggregate metrics."""
        group_columns = [
            "sample_id",
            "protein_name",
            "condition4",
            "control_type",
            "outer_label",
            "outer_label_base",
            "condition_match_key",
            "pair_id",
            "pair_id_columns",
            "target_bits_site_min",
            "target_stratum",
            "max_protein_id",
            "corruption_rate_init",
            "decode_steps",
        ]
        available_group_columns = [column_name for column_name in group_columns if column_name in cohort_meta.columns]
        base = per_site.merge(cohort_meta[available_group_columns], on=["sample_id", "protein_name"], how="left", validate="many_to_one")
        base = base[base["condition4"].notna()].copy()
        records: List[Dict[str, Any]] = []
        for sample_key, sample_group in base.groupby(["sample_id", "protein_name"], sort=False):
            first_row = sample_group.iloc[0]
            record: Dict[str, Any] = {column_name: first_row[column_name] for column_name in available_group_columns}
            record["n_sites"] = int(sample_group.shape[0])
            record["edited_fraction"] = float(np.mean(sample_group["is_mutation"].to_numpy(dtype=np.int8, copy=False)))
            record["mean_entropy"] = float(np.nanmean(sample_group["entropy_cond_all_nats"].to_numpy(dtype=float, copy=False)))
            record["mean_nll"] = float(np.nanmean(sample_group["nll_cond_all"].to_numpy(dtype=float, copy=False)))
            record["mean_jsd"] = float(np.nanmean(sample_group["jsd_cond_uncond_all_nats"].to_numpy(dtype=float, copy=False)))
            record["mean_plddt_0_100"] = float(np.nanmean(sample_group["plddt_mean_0_100"].to_numpy(dtype=float, copy=False)))
            record["fold_rate_plddt70"] = float(np.nanmean(sample_group["fold_plddt70"].to_numpy(dtype=float, copy=False)))
            record["fold_rate_plddt90"] = float(np.nanmean(sample_group["fold_plddt90"].to_numpy(dtype=float, copy=False)))
            record["mean_pae"] = float(np.nanmean(sample_group["pae_mean"].to_numpy(dtype=float, copy=False)))
            record["novelty_max_protein_id"] = 100.0 - float(first_row["max_protein_id"])
            records.append(record)
        return pd.DataFrame(records)

    def compute_protein_level_pair_summaries(self, sample_summary: pd.DataFrame) -> pd.DataFrame:
        """Compute protein-level condition means for Figure 5f deltas."""
        columns = [
            "protein_name",
            "condition4",
            "edited_fraction",
            "mean_entropy",
            "mean_nll",
            "mean_jsd",
            "mean_plddt_0_100",
            "fold_rate_plddt70",
            "fold_rate_plddt90",
            "mean_pae",
        ]
        base = sample_summary[[column_name for column_name in columns if column_name in sample_summary.columns]].copy()
        numeric_columns = [column_name for column_name in base.columns if column_name not in ["protein_name", "condition4"]]
        summary = base.groupby(["protein_name", "condition4"], sort=False)[numeric_columns].mean(numeric_only=True).reset_index()
        return summary.rename(columns={"protein_name": "pair_id"})

    def compute_bits_binned_curves(self, per_site: pd.DataFrame, cohort_meta: pd.DataFrame) -> pd.DataFrame:
        """Compute protein-median curves versus MSA bits per site."""
        edges = np.array(self.config.msa_bits_bin_edges, dtype=float)
        base = per_site.merge(cohort_meta[["sample_id", "protein_name", "condition4"]], on=["sample_id", "protein_name"], how="left", validate="many_to_one")
        base = base[base["condition4"].notna()].copy()
        base = base[np.isfinite(base["msa_bits_per_site"].to_numpy(dtype=float, copy=False))].copy()
        base = base[base["msa_bits_per_site"] >= float(self.config.bits_axis_min)].copy()
        base["bits_bin"] = pd.cut(base["msa_bits_per_site"], bins=edges, include_lowest=True)
        records: List[Dict[str, Any]] = []
        for group_key, group in base.groupby(["protein_name", "condition4", "bits_bin"], sort=False):
            protein_name, condition4, bits_bin = group_key
            if pd.isna(bits_bin) or not isinstance(bits_bin, pd.Interval):
                continue
            records.append(
                {
                    "protein_name": str(protein_name),
                    "condition4": str(condition4),
                    "bits_bin": str(bits_bin),
                    "n": int(group.shape[0]),
                    "bits_bin_mid": float(bits_bin.mid),
                    "mutation_rate": float(np.mean(group["is_mutation"].to_numpy(dtype=np.int8, copy=False))),
                    "mean_jsd": float(np.nanmean(group["jsd_cond_uncond_all_nats"].to_numpy(dtype=float, copy=False))),
                    "mean_delta_nll": float(np.nanmean(group["delta_nll_all"].to_numpy(dtype=float, copy=False))),
                }
            )
        protein_bin = pd.DataFrame(records)
        if protein_bin.empty:
            return protein_bin
        return (
            protein_bin.groupby(["condition4", "bits_bin_mid"], sort=False)
            .agg(
                protein_median_mutation_rate=("mutation_rate", "median"),
                protein_median_jsd=("mean_jsd", "median"),
                protein_median_delta_nll=("mean_delta_nll", "median"),
                proteins=("protein_name", "nunique"),
            )
            .reset_index()
        )

    def compute_commit_step_event_profiles(self, per_site: pd.DataFrame, cohort_meta: pd.DataFrame) -> pd.DataFrame:
        """Compute profiles for residues committed at each decode step."""
        base = per_site.merge(cohort_meta[["sample_id", "protein_name", "condition4"]], on=["sample_id", "protein_name"], how="left", validate="many_to_one")
        base = base[base["condition4"].notna()].copy()
        committed = base[(base["is_mutation"] == 1) & (base["commit_step_index"] >= 1)].copy()
        committed = committed[committed["commit_step_index"] <= int(self.config.decode_steps_expected)].copy()
        records: List[Dict[str, Any]] = []
        for group_key, group in committed.groupby(["protein_name", "condition4", "commit_step_index"], sort=False):
            protein_name, condition4, step_index = group_key
            records.append(
                {
                    "protein_name": str(protein_name),
                    "condition4": str(condition4),
                    "step_index": int(step_index),
                    "n_commits": int(group.shape[0]),
                    "mean_msa_bits": float(np.nanmean(group["msa_bits_per_site"].to_numpy(dtype=float, copy=False))),
                    "mean_entropy": float(np.nanmean(group["entropy_cond_all_nats"].to_numpy(dtype=float, copy=False))),
                    "mean_plddt_0_100": float(np.nanmean(group["plddt_mean_0_100"].to_numpy(dtype=float, copy=False))),
                    "fold_rate_plddt70": float(np.nanmean(group["fold_plddt70"].to_numpy(dtype=float, copy=False))),
                }
            )
        protein_step = pd.DataFrame(records)
        if protein_step.empty:
            return protein_step
        return (
            protein_step.groupby(["condition4", "step_index"], sort=False)
            .agg(
                protein_median_msa_bits=("mean_msa_bits", "median"),
                protein_median_entropy=("mean_entropy", "median"),
                protein_median_plddt_0_100=("mean_plddt_0_100", "median"),
                protein_median_fold_rate=("fold_rate_plddt70", "median"),
                proteins=("protein_name", "nunique"),
                n_commits_total=("n_commits", "sum"),
            )
            .reset_index()
        )

    def compute_commit_step_cumulative_dynamics(self, per_site: pd.DataFrame, cohort_meta: pd.DataFrame) -> pd.DataFrame:
        """Compute cumulative pLDDT profiles for edits committed by each step."""
        base = per_site.merge(cohort_meta[["sample_id", "protein_name", "condition4"]], on=["sample_id", "protein_name"], how="left", validate="many_to_one")
        base = base[base["condition4"].notna()].copy()
        committed = base[(base["is_mutation"] == 1) & (base["commit_step_index"] >= 1)].copy()
        committed = committed[committed["commit_step_index"] <= int(self.config.decode_steps_expected)].copy()
        records: List[Dict[str, Any]] = []
        for group_key, group in committed.groupby(["protein_name", "condition4"], sort=False):
            protein_name, condition4 = group_key
            step_values = group["commit_step_index"].to_numpy(dtype=int, copy=False)
            denominator = int(step_values.shape[0])
            if denominator == 0:
                continue
            for step_index in range(1, int(self.config.decode_steps_expected) + 1):
                committed_mask = step_values <= step_index
                committed_count = int(np.sum(committed_mask))
                sub = group.iloc[np.where(committed_mask)[0]] if committed_count else group.iloc[[]]
                records.append(
                    {
                        "protein_name": str(protein_name),
                        "condition4": str(condition4),
                        "step_index": int(step_index),
                        "fraction_committed_by_step": float(committed_count / float(denominator)),
                        "mean_plddt_0_100_committed_by_step": float(np.nanmean(sub["plddt_mean_0_100"].to_numpy(dtype=float, copy=False))) if committed_count else float("nan"),
                    }
                )
        protein_step = pd.DataFrame(records)
        if protein_step.empty:
            return protein_step
        return (
            protein_step.groupby(["condition4", "step_index"], sort=False)
            .agg(
                protein_median_fraction_committed=("fraction_committed_by_step", "median"),
                protein_median_plddt_0_100=("mean_plddt_0_100_committed_by_step", "median"),
                proteins=("protein_name", "nunique"),
            )
            .reset_index()
        )

    def compute_paired_effect_size_summary(self, pair_summary: pd.DataFrame) -> pd.DataFrame:
        """Build Figure 5f targeted-minus-control delta rows."""
        metric_specs = [
            ("edited_fraction", "Edited fraction per sample"),
            ("mean_entropy", "Mean conditional entropy (nats)"),
            ("mean_plddt_0_100", "Mean pLDDT (0–100)"),
        ]
        stratum_specs = [
            ("below", "below targeted", "below matched control"),
            ("above", "above targeted", "above matched control"),
        ]
        records: List[Dict[str, Any]] = []
        if pair_summary.empty:
            return pd.DataFrame(records)
        for metric_column, metric_label in metric_specs:
            if metric_column not in pair_summary.columns:
                continue
            wide = pair_summary.pivot_table(index="pair_id", columns="condition4", values=metric_column, aggfunc="mean")
            for stratum_label, targeted_condition, control_condition in stratum_specs:
                if targeted_condition not in wide.columns or control_condition not in wide.columns:
                    continue
                paired = wide[[targeted_condition, control_condition]].dropna()
                deltas = paired[targeted_condition] - paired[control_condition]
                for pair_id_value, delta_value in deltas.items():
                    records.append(
                        {
                            "pair_id": str(pair_id_value),
                            "stratum": str(stratum_label),
                            "targeted_condition": str(targeted_condition),
                            "control_condition": str(control_condition),
                            "metric": str(metric_column),
                            "metric_label": str(metric_label),
                            "delta_targeted_minus_control": float(delta_value),
                        }
                    )
        return pd.DataFrame(records)

    def compute_calibration_tables(self, per_site: pd.DataFrame, cohort_meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute pLDDT70 calibration rows for entropy and NLL scores."""
        base = per_site.merge(cohort_meta[["sample_id", "protein_name", "condition4"]], on=["sample_id", "protein_name"], how="left", validate="many_to_one")
        base = base[base["condition4"].notna()].copy()
        records: List[Dict[str, Any]] = []
        for group_key, group in base.groupby(["protein_name", "condition4"], sort=False):
            protein_name, condition4 = group_key
            scores_by_name = {
                "entropy": -pd.to_numeric(group["entropy_cond_all_nats"], errors="coerce").to_numpy(dtype=float),
                "nll": -pd.to_numeric(group["nll_cond_all"], errors="coerce").to_numpy(dtype=float),
            }
            labels = pd.to_numeric(group["fold_plddt70"], errors="coerce").to_numpy(dtype=int)
            for score_name, scores in scores_by_name.items():
                record = self.compute_binary_calibration(scores, labels, score_name)
                record["protein_name"] = str(protein_name)
                record["condition4"] = str(condition4)
                records.append(record)
        per_protein = pd.DataFrame(records)
        if per_protein.empty:
            return per_protein, per_protein
        per_condition = (
            per_protein.groupby(["condition4", "score_name", "target"], sort=False)
            .agg(auc_median=("auc", "median"), brier_median=("brier", "median"), ece_median=("ece", "median"), proteins=("protein_name", "nunique"))
            .reset_index()
        )
        return per_protein, per_condition

    def compute_binary_calibration(self, scores: np.ndarray, labels: np.ndarray, score_name: str) -> Dict[str, Any]:
        """Compute rank-based AUC, Brier score, and calibration error."""
        finite_mask = np.isfinite(scores) & np.isfinite(labels.astype(float))
        finite_scores = scores[finite_mask]
        finite_labels = labels[finite_mask].astype(int)
        if finite_scores.size < 200 or int(np.sum(finite_labels == 1)) == 0 or int(np.sum(finite_labels == 0)) == 0:
            return {"score_name": score_name, "target": "plddt70", "n": int(finite_scores.size), "auc": float("nan"), "brier": float("nan"), "ece": float("nan")}
        auc_value = self.auc_rank(finite_scores, finite_labels)
        probability = self.rank_probability(finite_scores)
        brier_value = float(np.mean((probability - finite_labels.astype(float)) ** 2))
        ece_value = self.ece_binned(probability, finite_labels.astype(float), bins=int(self.config.calibration_bins))
        return {"score_name": score_name, "target": "plddt70", "n": int(finite_scores.size), "auc": float(auc_value), "brier": float(brier_value), "ece": float(ece_value)}

    def auc_rank(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute ROC AUC from rank sums."""
        label_array = labels.astype(np.int32, copy=False)
        positive_mask = label_array == 1
        negative_mask = label_array == 0
        positive_count = int(np.sum(positive_mask))
        negative_count = int(np.sum(negative_mask))
        if positive_count == 0 or negative_count == 0:
            return float("nan")
        ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64, copy=False)
        positive_rank_sum = float(np.sum(ranks[positive_mask]))
        return float((positive_rank_sum - (positive_count * (positive_count + 1) / 2.0)) / float(positive_count * negative_count))

    def rank_probability(self, scores: np.ndarray) -> np.ndarray:
        """Map scores onto rank-normalized probabilities."""
        if scores.size == 0:
            return np.zeros((0,), dtype=np.float64)
        ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64, copy=False)
        if scores.size == 1:
            return np.array([0.5], dtype=np.float64)
        return (ranks - 1.0) / float(scores.size - 1)

    def ece_binned(self, probability: np.ndarray, label: np.ndarray, bins: int) -> float:
        """Compute expected calibration error with quantile bins."""
        if probability.size < 5:
            return float("nan")
        probability_array = probability.astype(np.float64, copy=False)
        label_array = label.astype(np.float64, copy=False)
        edges = np.quantile(probability_array, np.linspace(0.0, 1.0, int(bins) + 1))
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)
        if edges.size < 2:
            return float("nan")
        error_value = 0.0
        count_value = float(probability_array.size)
        for edge_index in range(edges.size - 1):
            low_value = float(edges[edge_index])
            high_value = float(edges[edge_index + 1])
            if edge_index == 0:
                bin_mask = (probability_array >= low_value) & (probability_array <= high_value)
            else:
                bin_mask = (probability_array > low_value) & (probability_array <= high_value)
            if not np.any(bin_mask):
                continue
            accuracy = float(np.mean(label_array[bin_mask]))
            confidence = float(np.mean(probability_array[bin_mask]))
            proportion = float(np.sum(bin_mask) / count_value)
            error_value += abs(accuracy - confidence) * proportion
        return float(error_value)

    def compute_phase_diagram_table(self, sample_summary: pd.DataFrame) -> pd.DataFrame:
        """Compute the Supplementary Figure 3e foldability surface grid."""
        frame = sample_summary.copy()
        frame["edited_fraction"] = pd.to_numeric(frame["edited_fraction"], errors="coerce")
        frame["mean_entropy"] = pd.to_numeric(frame["mean_entropy"], errors="coerce")
        finite_mask = np.isfinite(frame["edited_fraction"].to_numpy(dtype=float)) & np.isfinite(frame["mean_entropy"].to_numpy(dtype=float))
        frame = frame.loc[finite_mask].copy()
        frame = self.assign_groupwise_qcut(frame, "condition4", "edited_fraction", int(self.config.novelty_quantile_bins), "novelty_bin")
        frame = self.assign_groupwise_qcut(frame, "condition4", "mean_entropy", int(self.config.uncertainty_quantile_bins), "uncertainty_bin")
        frame = frame[frame["novelty_bin"].notna() & frame["uncertainty_bin"].notna()].copy()
        if frame.empty:
            return pd.DataFrame()
        protein_grid = frame.groupby(["protein_name", "condition4", "novelty_bin", "uncertainty_bin"], sort=False).agg(mean_plddt_0_100=("mean_plddt_0_100", "mean"), n_source_samples=("sample_id", "nunique")).reset_index()
        records: List[Dict[str, Any]] = []
        for group_key, group in protein_grid.groupby(["condition4", "novelty_bin", "uncertainty_bin"], sort=False):
            condition4, novelty_bin, uncertainty_bin = group_key
            records.append(
                {
                    "condition4": str(condition4),
                    "novelty_axis": "edited_fraction",
                    "uncertainty_axis": "mean_entropy",
                    "novelty_bin": str(novelty_bin),
                    "uncertainty_bin": str(uncertainty_bin),
                    "n_proteins": int(group["protein_name"].nunique()),
                    "mean_plddt_0_100": float(np.nanmean(group["mean_plddt_0_100"].to_numpy(dtype=float, copy=False))),
                }
            )
        return pd.DataFrame(records)

    def assign_groupwise_qcut(self, frame: pd.DataFrame, group_column: str, value_column: str, bins: int, output_column: str) -> pd.DataFrame:
        """Assign quantile bins within condition groups."""
        output = frame.copy()
        output[output_column] = None
        for group_value, group_frame in output.groupby(group_column, sort=False):
            values = pd.to_numeric(group_frame[value_column], errors="coerce")
            finite_mask = np.isfinite(values.to_numpy(dtype=float, copy=False))
            if int(finite_mask.sum()) < max(40, int(bins) * 4):
                continue
            try:
                quantile_bins = pd.qcut(values[finite_mask], q=int(bins), duplicates="drop")
            except ValueError:
                continue
            output.loc[group_frame.index[finite_mask], output_column] = quantile_bins.astype(str).to_numpy()
        return output


@dataclass
class Figure5Plotter:
    """Render Figure 5 and Supplementary Figure 3 panels."""

    config: Figure5Config
    paths: Figure5Paths
    style: ConditionStyle

    def apply_publication_style(self) -> None:
        """Apply publication style defaults."""
        sns.set_style("white")
        mpl.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "lines.linewidth": 2.0, "lines.markersize": 4.0, "savefig.dpi": 300, "figure.dpi": 120, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})

    def apply_axis_style(self, axis: plt.Axes) -> None:
        """Apply per-axis styling."""
        axis.grid(False)
        axis.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True)
        axis.xaxis.set_ticks_position("bottom")
        axis.yaxis.set_ticks_position("left")
        sns.despine(ax=axis, top=True, right=True)

    def save_condition_legend(self) -> None:
        """Save the standalone condition legend."""
        handles: List[Any] = []
        labels: List[str] = []
        for condition_label in self.style.get_order():
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=self.style.get_color(condition_label), markeredgecolor="none", markersize=9))
            labels.append(condition_label)
        fig = plt.figure(figsize=(4.4, 1.2))
        legend = fig.legend(handles, labels, loc="center", frameon=False, ncol=2, handlelength=1.0, handletextpad=0.6, borderaxespad=0.2, columnspacing=1.2)
        fig.savefig(self.paths.output_dir / "figure5_legend_conditions.svg", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend])
        fig.savefig(self.paths.output_dir / "figure5_legend_conditions.png", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend])
        plt.close(fig)

    def save_condition_series_panel(self, frame: pd.DataFrame, x_column: str, y_column: str, xlabel: str, ylabel: str, output_stem: str, xlim: Optional[Tuple[float, float]] = None, xticks: Optional[Sequence[int]] = None) -> None:
        """Save one condition-colored line panel."""
        source = frame[frame["condition4"].isin(self.style.get_order())].copy()
        fig, axis = plt.subplots(figsize=(5.6, 3.4))
        for condition in self.style.get_order():
            group = source[source["condition4"] == condition].sort_values(x_column)
            axis.plot(pd.to_numeric(group[x_column], errors="coerce"), pd.to_numeric(group[y_column], errors="coerce"), color=self.style.get_color(condition))
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        if xlim is not None:
            axis.set_xlim(*xlim)
        if xticks is not None:
            axis.set_xticks(list(xticks))
        axis.legend([], [], frameon=False)
        self.apply_axis_style(axis)
        fig.tight_layout()
        save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)

    def plot_edit_probability_vs_msa_bits(self, bits_curves: pd.DataFrame) -> None:
        """Plot Figure 5a."""
        self.save_condition_series_panel(bits_curves, "bits_bin_mid", "protein_median_mutation_rate", "MSA information (bits/site)", "P(edit) per residue", "figure5a_edit_probability_vs_msa_bits", xlim=(float(self.config.bits_axis_min), max(self.config.msa_bits_bin_edges)))

    def plot_conditioning_lift_vs_msa_bits(self, bits_curves: pd.DataFrame) -> None:
        """Plot Supplementary Figure 3a-b."""
        xlim = (float(self.config.bits_axis_min), max(self.config.msa_bits_bin_edges))
        self.save_condition_series_panel(bits_curves, "bits_bin_mid", "protein_median_jsd", "MSA information (bits/site)", "JSD(cond, uncond) [nats]", "suppfig3a_conditioning_lift_vs_msa_bits_panel_jsd", xlim=xlim)
        self.save_condition_series_panel(bits_curves, "bits_bin_mid", "protein_median_delta_nll", "MSA information (bits/site)", "ΔNLL = NLL_uncond − NLL_cond", "suppfig3b_conditioning_lift_vs_msa_bits_panel_delta_nll", xlim=xlim)

    def plot_commit_step_event_profiles(self, event_profiles: pd.DataFrame) -> None:
        """Plot Figure 5b-d."""
        source = event_profiles.copy()
        source["step_index"] = pd.to_numeric(source["step_index"], errors="coerce").astype("Int64")
        xticks = range(1, int(self.config.decode_steps_expected) + 1)
        xlim = (1.0, float(self.config.decode_steps_expected))
        self.save_condition_series_panel(source, "step_index", "protein_median_msa_bits", "Commit step (1–12)", "Median MSA bits/site of residues committed at step", "figure5b_commit_step_event_profiles_panel_msa_bits", xlim=xlim, xticks=xticks)
        self.save_condition_series_panel(source, "step_index", "protein_median_entropy", "Commit step (1–12)", "Median entropy [nats] of residues committed at step", "figure5c_commit_step_event_profiles_panel_entropy", xlim=xlim, xticks=xticks)
        self.save_condition_series_panel(source, "step_index", "protein_median_plddt_0_100", "Commit step (1–12)", "Median pLDDT (0–100) of residues committed at step", "figure5d_commit_step_event_profiles_panel_plddt", xlim=xlim, xticks=xticks)

    def plot_commit_step_cumulative_dynamics(self, cumulative_dynamics: pd.DataFrame) -> None:
        """Plot Figure 5e."""
        source = cumulative_dynamics.copy()
        source["step_index"] = pd.to_numeric(source["step_index"], errors="coerce").astype("Int64")
        xticks = range(1, int(self.config.decode_steps_expected) + 1)
        self.save_condition_series_panel(source, "step_index", "protein_median_plddt_0_100", "Decode step (1–12)", "Median pLDDT (0–100) among committed edits", "figure5e_decode_step_cumulative_commit_dynamics_panel_plddt", xlim=(1.0, float(self.config.decode_steps_expected)), xticks=xticks)

    def plot_paired_effect_size_summary(self, effect_summary: pd.DataFrame) -> None:
        """Plot Figure 5f split effect-size panels."""
        if effect_summary.empty:
            return
        metric_stems = {
            "Edited fraction per sample": "figure5f_left_effect_sizes_panel_Edited_fraction_per_sample",
            "Mean conditional entropy (nats)": "figure5f_center_effect_sizes_panel_Mean_conditional_entropy_(nats)",
            "Mean pLDDT (0–100)": "figure5f_right_effect_sizes_panel_Mean_pLDDT_(0_100)",
        }
        grey_palette = {"below": "#C7C7C7", "above": "#7A7A7A"}
        for metric_label, output_stem in metric_stems.items():
            subset = effect_summary[effect_summary["metric_label"] == metric_label].copy()
            if subset.empty:
                continue
            fig, axis = plt.subplots(figsize=(4.2, 3.5))
            sns.violinplot(ax=axis, data=subset, x="stratum", y="delta_targeted_minus_control", hue="stratum", order=["below", "above"], hue_order=["below", "above"], palette=grey_palette, dodge=False, cut=0, inner="box", linewidth=0.8, saturation=1.0, legend=False)
            axis.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.9)
            axis.set_title(metric_label)
            axis.set_xlabel("")
            axis.set_ylabel("Paired delta (targeted − control)")
            self.apply_axis_style(axis)
            fig.tight_layout()
            save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)

    def plot_calibration_violins(self, calibration_per_protein: pd.DataFrame) -> None:
        """Plot Supplementary Figure 3c-d with entropy and NLL rows pooled."""
        source = calibration_per_protein.copy()
        source = source[(source["condition4"].isin(self.style.get_order())) & (source["target"] == "plddt70")].copy()
        panel_specs = [("brier", "Brier score (pLDDT≥70)", "suppfig3c_calibration_brier_plddt70"), ("auc", "AUC (pLDDT≥70)", "suppfig3d_calibration_auc_plddt70")]
        for metric_key, y_label, output_stem in panel_specs:
            if source.empty:
                continue
            fig, axis = plt.subplots(figsize=(4.8, 3.2))
            sns.violinplot(ax=axis, data=source, x="condition4", y=metric_key, hue="condition4", order=self.style.get_order(), palette=self.style.get_palette(), cut=0, inner=None, linewidth=0.8, dodge=False, legend=False)
            axis.set_xlabel("")
            axis.set_ylabel(y_label)
            axis.tick_params(axis="x", rotation=20)
            self.apply_axis_style(axis)
            fig.tight_layout()
            save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)

    def plot_foldability_phase_panels(self, phase_grid: pd.DataFrame) -> None:
        """Plot Supplementary Figure 3e split heatmap panels."""
        if phase_grid.empty:
            return
        values = pd.to_numeric(phase_grid["mean_plddt_0_100"], errors="coerce").to_numpy(dtype=float, copy=False)
        finite_values = values[np.isfinite(values)]
        vmin = float(np.min(finite_values)) if finite_values.size else 0.0
        vmax = float(np.max(finite_values)) if finite_values.size else 1.0
        cmap_name = "viridis"
        cmap = mpl.colormaps.get_cmap(cmap_name).copy()
        cmap.set_bad(color="white")
        output_stem_by_condition = {
            "above targeted": "suppfig3e_top_left_foldability_surface_mean_plddt_novelty_edited_fraction_uncert_mean_entropy_panel_above_targeted",
            "above matched control": "suppfig3e_top_right_foldability_surface_mean_plddt_novelty_edited_fraction_uncert_mean_entropy_panel_above_matched_control",
            "below targeted": "suppfig3e_bottom_left_foldability_surface_mean_plddt_novelty_edited_fraction_uncert_mean_entropy_panel_below_targeted",
            "below matched control": "suppfig3e_bottom_right_foldability_surface_mean_plddt_novelty_edited_fraction_uncert_mean_entropy_panel_below_matched_control",
        }
        for condition in ["above targeted", "above matched control", "below targeted", "below matched control"]:
            subset = phase_grid[phase_grid["condition4"] == condition].copy()
            self.save_foldability_phase_panel(subset, condition, vmin, vmax, cmap, output_stem_by_condition[condition])
        self.save_heatmap_colorbar(vmin, vmax, cmap_name, "Mean pLDDT (0–100)", "suppfig3e_foldability_surface_mean_plddt_novelty_edited_fraction_uncert_mean_entropy_colorbar")

    def save_foldability_phase_panel(self, phase_grid: pd.DataFrame, condition: str, vmin: float, vmax: float, cmap: mpl.colors.Colormap, output_stem: str) -> None:
        """Save one foldability surface heatmap panel."""
        fig, axis = plt.subplots(figsize=(4.0, 3.4))
        if phase_grid.empty:
            axis.set_title(str(condition))
            axis.axis("off")
            save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)
            return
        pivot = phase_grid.pivot_table(index="uncertainty_bin", columns="novelty_bin", values="mean_plddt_0_100", aggfunc="mean")
        matrix = pivot.to_numpy(dtype=float, copy=True)
        axis.imshow(np.ma.masked_invalid(matrix), origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(str(condition))
        axis.set_xlabel("Novelty deciles (low → high)")
        axis.set_ylabel("Uncertainty deciles (low → high)")
        axis.set_xticks(range(pivot.shape[1]))
        axis.set_yticks(range(pivot.shape[0]))
        axis.set_xticklabels([str(index + 1) for index in range(pivot.shape[1])])
        axis.set_yticklabels([str(index + 1) for index in range(pivot.shape[0])])
        axis.grid(False)
        sns.despine(ax=axis, top=True, right=True, left=False, bottom=False)
        fig.tight_layout()
        save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)

    def save_heatmap_colorbar(self, vmin: float, vmax: float, cmap_name: str, label: str, output_stem: str) -> None:
        """Save one standalone heatmap colorbar."""
        fig, axis = plt.subplots(figsize=(1.2, 3.0))
        fig.subplots_adjust(left=0.55, right=0.85)
        cmap = mpl.colormaps.get_cmap(cmap_name).copy()
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar_mappable.set_array([])
        colorbar = fig.colorbar(scalar_mappable, cax=axis)
        colorbar.set_label(label)
        axis.grid(False)
        fig.savefig(self.paths.output_dir / f"{output_stem}.svg", dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(self.paths.output_dir / f"{output_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


@dataclass
class Figure5Runner:
    """Coordinate Figure 5 generation."""

    config: Figure5Config

    def run(self) -> None:
        """Run the Figure 5 and Supplementary Figure 3 pipeline."""
        paths = Figure5Paths.from_config(self.config)
        paths.ensure_dirs()
        style = ConditionStyle()
        loader = Figure5DataLoader(self.config)
        analyzer = Figure5Analyzer(self.config)
        plotter = Figure5Plotter(self.config, paths, style)
        plotter.apply_publication_style()
        start_time = time.time()
        cohort_raw = loader.read_cohort_by_sample(paths.cohort_xlsx_path)
        cohort_meta = loader.add_condition_and_pair_columns(cohort_raw)
        protein_dirs = loader.list_protein_dirs(paths.final_per_site_root)
        if not protein_dirs:
            raise FileNotFoundError(f"No protein partitions found under {paths.final_per_site_root}")
        per_site_tables: List[pd.DataFrame] = []
        per_site_columns = analyzer.required_per_site_columns()
        for protein_dir in protein_dirs:
            protein_name = loader.protein_name_from_dir(protein_dir)
            per_site_path = protein_dir / "per_site.parquet"
            if not per_site_path.exists():
                raise FileNotFoundError(f"Missing per_site.parquet for protein_name={protein_name}: {per_site_path}")
            protein_per_site = loader.read_per_site_for_protein(per_site_path, per_site_columns)
            protein_per_site = analyzer.add_derived_per_site(protein_per_site)
            per_site_tables.append(protein_per_site)
        per_site_all = pd.concat(per_site_tables, ignore_index=True)
        sample_summary = analyzer.compute_sample_aggregates(per_site_all, cohort_meta)
        paired_summary = analyzer.compute_protein_level_pair_summaries(sample_summary)
        bits_curves = analyzer.compute_bits_binned_curves(per_site_all, cohort_meta)
        commit_event_profiles = analyzer.compute_commit_step_event_profiles(per_site_all, cohort_meta)
        commit_cumulative_dynamics = analyzer.compute_commit_step_cumulative_dynamics(per_site_all, cohort_meta)
        paired_effect_size_summary = analyzer.compute_paired_effect_size_summary(paired_summary)
        calibration_per_protein, calibration_per_condition = analyzer.compute_calibration_tables(per_site_all, cohort_meta)
        phase_grid = analyzer.compute_phase_diagram_table(sample_summary)
        plotter.save_condition_legend()
        plotter.plot_edit_probability_vs_msa_bits(bits_curves)
        plotter.plot_conditioning_lift_vs_msa_bits(bits_curves)
        plotter.plot_commit_step_event_profiles(commit_event_profiles)
        plotter.plot_commit_step_cumulative_dynamics(commit_cumulative_dynamics)
        plotter.plot_paired_effect_size_summary(paired_effect_size_summary)
        plotter.plot_calibration_violins(calibration_per_protein)
        plotter.plot_foldability_phase_panels(phase_grid)
        tables = {
            "readme": self.build_readme_table(cohort_meta, per_site_all, sample_summary),
            "figure5a_edit_probability": bits_curves,
            "figure5b_d_commit_events": commit_event_profiles,
            "figure5e_cumulative_plddt": commit_cumulative_dynamics,
            "figure5f_effect_sizes": paired_effect_size_summary,
            "suppfig3a_b_conditioning_lift": bits_curves,
            "suppfig3c_d_calibration_values": calibration_per_protein,
            "suppfig3c_d_calibration_summary": calibration_per_condition,
            "suppfig3e_foldability_surface": phase_grid,
            "sample_summary": sample_summary,
        }
        writer = ExcelNotebookWriter(paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        elapsed = float(time.time() - start_time)
        print(f"Wrote Figure 5 outputs to {paths.output_dir} in {elapsed:.1f}s")

    def build_readme_table(self, cohort_meta: pd.DataFrame, per_site_all: pd.DataFrame, sample_summary: pd.DataFrame) -> pd.DataFrame:
        """Build the source-data workbook readme."""
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "cohort_input", "value": str(self.config.cohort_xlsx_name)},
            {"key": "cohort_sheet_name", "value": str(self.config.cohort_sheet_name)},
            {"key": "per_site_input", "value": f"{self.config.compiled_final_dir}/{self.config.final_per_site_subdir}/protein_name=*/per_site.parquet"},
            {"key": "threshold_bits_site_min", "value": str(float(self.config.bps_min_threshold))},
            {"key": "n_cohort_rows", "value": str(int(cohort_meta.shape[0]))},
            {"key": "n_per_site_rows", "value": str(int(per_site_all.shape[0]))},
            {"key": "n_sample_summary_rows", "value": str(int(sample_summary.shape[0]))},
            {"key": "condition_logic", "value": "Matched controls inherit below or above threshold status from the targeted MSA partner."},
            {"key": "calibration_panels", "value": "Brier and AUC panels use pLDDT70 rows pooled across entropy and NLL scores."},
        ]
        return pd.DataFrame(rows)


def main() -> None:
    """Generate Figure 5 and Supplementary Figure 3 outputs."""
    runner = Figure5Runner(Figure5Config())
    runner.run()


if __name__ == "__main__":
    main()
