import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy import stats

from figure_common import ConditionStyle
from figure_common import operational_bits_site_threshold
from figure_common import ExcelNotebookWriter
from figure_common import add_threshold_condition_columns
from figure_common import convert_archived_columns
from figure_common import require_columns
from figure_common import save_figure_pair


@dataclass(frozen=True)
class Figure6Config:
    """Configure Figure 6 and Supplementary Figure 4 outputs."""

    cohort_xlsx_name: str = "cohort.xlsx"
    cohort_sheet_name: str = "by_sample"
    compiled_final_dir: str = "compiled/final"
    final_per_site_subdir: str = "per_site"
    output_dir_name: str = "figure6_and_supplementary_figure4_outputs"
    output_xlsx_name: str = "figure6_source_data.xlsx"
    excel_chunk_rows: int = 1000000
    bits_site_min_threshold: float = operational_bits_site_threshold
    min_anchor_length: int = 100
    min_sample_plddt_mean: float = 70.0
    smoothing_window: int = 9
    mutation_rug_ymin_axes: float = 1.01
    mutation_rug_ymax_axes: float = 1.045
    title_pad_points: float = 12.0
    bootstrap_replicates: int = 5000
    bootstrap_seed: int = 42
    local_flow_min_finite_residues: int = 25
    local_flow_protein_aggregator: str = "median"
    bitsite_distribution_bin_width: float = 0.05
    bitsite_distribution_plddt_band_edges: Tuple[float, float, float] = (50.0, 70.0, 90.0)
    bitsite_distribution_pae_band_edges: Tuple[float, float, float] = (5.0, 10.0, 20.0)
    bitsite_distribution_grouping_mode: str = "condition4"
    bitsite_distribution_weighting_mode: str = "protein_balanced"
    bitsite_distribution_min_bin_count: int = 25
    expected_below_sample_id: str = "P07740__msa_p10_sweep__p10_2p35__rate_0p950000__cand3__seed42"
    expected_above_sample_id: str = "P07740__msa_p10_sweep__p10_3p10__rate_0p750000__cand8__seed42"


@dataclass(frozen=True)
class Figure6Paths:
    """Resolve Figure 6 input and output paths."""

    root_dir: Path
    cohort_xlsx_path: Path
    final_per_site_root: Path
    output_dir: Path
    output_xlsx_path: Path

    @classmethod
    def from_config(cls, config: Figure6Config) -> "Figure6Paths":
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


@dataclass(frozen=True)
class HeroSelectionResult:
    """Store selected MSA pair and hero samples."""

    protein_name: str
    anchor_sequence: str
    below_outer_label: str
    above_outer_label: str
    below_sample_id: str
    above_sample_id: str
    selection_metric_name: str
    selection_metric_diff: float


@dataclass
class Figure6DataLoader:
    """Load cohort metadata and per-site traces."""

    config: Figure6Config
    paths: Figure6Paths

    def read_cohort_by_sample(self) -> pd.DataFrame:
        """Read and validate the cohort by-sample sheet."""
        required_columns = [
            "sample_id",
            "protein_name",
            "outer_label",
            "control_type",
            "anchor_sequence",
            "seq_length",
            "bits_site_min",
            "max_protein_id",
            "plddt_mean",
        ]
        if not self.paths.cohort_xlsx_path.exists():
            raise FileNotFoundError(f"Missing cohort workbook: {self.paths.cohort_xlsx_path}")
        cohort = pd.read_excel(self.paths.cohort_xlsx_path, sheet_name=str(self.config.cohort_sheet_name), engine="openpyxl")
        require_columns(cohort, required_columns, "cohort workbook")
        if "bits_site_mean" not in cohort.columns and "msa_bits_per_site_mean" not in cohort.columns:
            raise ValueError("Cohort workbook must contain bits_site_mean or msa_bits_per_site_mean")
        output = cohort.copy()
        for column_name in ["sample_id", "protein_name", "outer_label", "control_type", "anchor_sequence"]:
            output[column_name] = output[column_name].astype(str)
        output["control_type"] = output["control_type"].str.lower()
        numeric_columns = [
            "seq_length",
            "bits_site_min",
            "max_protein_id",
            "plddt_mean",
            "candidate_index",
            "seed",
            "candidate_seed",
            "corruption_rate_init",
            "temperature",
            "decode_steps",
            "bits_site_mean",
            "msa_bits_per_site_mean",
        ]
        for column_name in numeric_columns:
            if column_name in output.columns:
                output[column_name] = pd.to_numeric(output[column_name], errors="coerce")
        if output["sample_id"].astype(str).nunique() != int(output.shape[0]):
            raise ValueError("The cohort by-sample sheet has non-unique sample_id values")
        output["plddt_mean_display"] = self.scale_plddt_series_to_display(output["plddt_mean"])
        output["novelty"] = 100.0 - pd.to_numeric(output["max_protein_id"], errors="coerce")
        information_columns = tuple(column_name for column_name in ("bits_site_min", "bits_site_mean", "msa_bits_per_site_mean") if column_name in output.columns)
        output = convert_archived_columns(output, information_columns=information_columns)
        output["selection_metric_value"] = self.select_msa_mean_metric(output)
        return output

    def select_msa_mean_metric(self, cohort: pd.DataFrame) -> pd.Series:
        """Return the preferred unweighted mean bits/site metric."""
        if "bits_site_mean" in cohort.columns:
            metric = pd.to_numeric(cohort["bits_site_mean"], errors="coerce")
            if np.isfinite(metric.to_numpy(dtype=float, copy=False)).any():
                return metric
        if "msa_bits_per_site_mean" in cohort.columns:
            metric = pd.to_numeric(cohort["msa_bits_per_site_mean"], errors="coerce")
            if np.isfinite(metric.to_numpy(dtype=float, copy=False)).any():
                return metric
        raise ValueError("No finite unweighted mean bits/site column found in cohort workbook")

    def scale_plddt_series_to_display(self, series: pd.Series) -> pd.Series:
        """Scale pLDDT to 0-100 if source values appear to be 0-1."""
        values = pd.to_numeric(series, errors="coerce")
        finite_values = values[np.isfinite(values.to_numpy(dtype=float, copy=False))]
        if finite_values.empty:
            return values
        quantile_95 = float(np.nanquantile(finite_values.to_numpy(dtype=float, copy=False), 0.95))
        return values * 100.0 if quantile_95 <= 1.0 else values

    def build_per_site_path(self, protein_name: str) -> Path:
        """Return a per-site parquet path for one protein."""
        return self.paths.final_per_site_root / f"protein_name={protein_name}" / "per_site.parquet"

    def read_per_site_for_protein(self, protein_name: str, columns: Sequence[str]) -> pd.DataFrame:
        """Read per-site parquet for one protein."""
        per_site_path = self.build_per_site_path(protein_name)
        if not per_site_path.exists():
            raise FileNotFoundError(f"Missing per_site parquet for protein={protein_name}: {per_site_path}")
        table = pq.read_table(per_site_path, columns=list(columns))
        frame = table.to_pandas(self_destruct=True)
        return convert_archived_columns(frame, information_columns=("msa_bits_per_site",))

    def read_per_site_for_proteins(self, protein_names: Sequence[str], columns: Sequence[str]) -> pd.DataFrame:
        """Read per-site parquet files for multiple proteins."""
        frames: List[pd.DataFrame] = []
        unique_names = pd.Index([str(protein_name) for protein_name in protein_names]).dropna().unique().tolist()
        for protein_name in unique_names:
            frames.append(self.read_per_site_for_protein(protein_name, columns))
        if not frames:
            return pd.DataFrame(columns=list(columns))
        return pd.concat(frames, axis=0, ignore_index=True)


@dataclass
class Figure6Analyzer:
    """Analyze hero traces and Figure 6 population summaries."""

    config: Figure6Config

    def required_per_site_columns(self) -> List[str]:
        """Return required per-site columns."""
        return [
            "sample_id",
            "protein_name",
            "site_index",
            "is_mutation",
            "msa_bits_per_site",
            "entropy_cond_all_nats",
            "plddt_mean",
            "pae_mean",
        ]

    def scale_plddt_series_to_display(self, series: pd.Series) -> pd.Series:
        """Scale pLDDT to 0-100 if source values appear to be 0-1."""
        values = pd.to_numeric(series, errors="coerce")
        finite_values = values[np.isfinite(values.to_numpy(dtype=float, copy=False))]
        if finite_values.empty:
            return values
        quantile_95 = float(np.nanquantile(finite_values.to_numpy(dtype=float, copy=False), 0.95))
        return values * 100.0 if quantile_95 <= 1.0 else values

    def build_targeted_candidate_table(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Filter cohort rows to hero-selection-eligible targeted candidates."""
        output = cohort[cohort["control_type"] == "targeted"].copy()
        output = output[output["seq_length"] >= int(self.config.min_anchor_length)].copy()
        required_numeric = ["bits_site_min", "selection_metric_value", "plddt_mean_display", "novelty"]
        mask = np.ones((output.shape[0],), dtype=bool)
        for column_name in required_numeric:
            values = pd.to_numeric(output[column_name], errors="coerce").to_numpy(dtype=float, copy=False)
            mask &= np.isfinite(values)
        return output.loc[mask].copy()

    def compute_msa_summary_table(self, candidate: pd.DataFrame) -> pd.DataFrame:
        """Compute one row per targeted MSA for hero selection."""
        records: List[Dict[str, Any]] = []
        for group_key, group in candidate.groupby(["protein_name", "outer_label"], sort=False):
            protein_name, outer_label = group_key
            first = group.iloc[0]
            anchor_unique = group["anchor_sequence"].astype(str).nunique()
            if anchor_unique != 1:
                raise ValueError(f"MSA {protein_name} / {outer_label} has multiple anchor_sequence values")
            plddt_values = pd.to_numeric(group["plddt_mean_display"], errors="coerce").to_numpy(dtype=float, copy=False)
            min_plddt = float(np.nanmin(plddt_values)) if plddt_values.size else float("nan")
            valid_msa = bool(np.isfinite(min_plddt) and min_plddt >= float(self.config.min_sample_plddt_mean))
            bits_site_min_value = float(first["bits_site_min"])
            side_value = "below" if bits_site_min_value < float(self.config.bits_site_min_threshold) else "above"
            novelty_values = pd.to_numeric(group["novelty"], errors="coerce").to_numpy(dtype=float, copy=False)
            records.append(
                {
                    "protein_name": str(protein_name),
                    "outer_label": str(outer_label),
                    "anchor_sequence": str(first["anchor_sequence"]),
                    "seq_length": int(first["seq_length"]),
                    "bits_site_min": bits_site_min_value,
                    "selection_metric_value": float(first["selection_metric_value"]),
                    "side": side_value,
                    "n_samples": int(group.shape[0]),
                    "min_plddt_mean_display": min_plddt,
                    "max_plddt_mean_display": float(np.nanmax(plddt_values)) if plddt_values.size else float("nan"),
                    "valid_msa": valid_msa,
                    "invalid_reason": "" if valid_msa else f"min_plddt_mean_display<{self.config.min_sample_plddt_mean}",
                    "max_novelty": float(np.nanmax(novelty_values)) if novelty_values.size else float("nan"),
                }
            )
        return pd.DataFrame(records)

    def compute_pair_candidates(self, msa_summary: pd.DataFrame) -> pd.DataFrame:
        """Enumerate valid below-above targeted MSA pairs within anchor contexts."""
        valid = msa_summary[msa_summary["valid_msa"]].copy()
        records: List[Dict[str, Any]] = []
        for group_key, protein_group in valid.groupby(["protein_name", "anchor_sequence"], sort=False):
            protein_name, anchor_sequence = group_key
            below_group = protein_group[protein_group["side"] == "below"].copy()
            above_group = protein_group[protein_group["side"] == "above"].copy()
            if below_group.empty or above_group.empty:
                continue
            for below_index, below_row in below_group.iterrows():
                for above_index, above_row in above_group.iterrows():
                    diff_value = abs(float(above_row["selection_metric_value"]) - float(below_row["selection_metric_value"]))
                    records.append(
                        {
                            "protein_name": str(protein_name),
                            "anchor_sequence": str(anchor_sequence),
                            "below_outer_label": str(below_row["outer_label"]),
                            "above_outer_label": str(above_row["outer_label"]),
                            "below_bits_site_min": float(below_row["bits_site_min"]),
                            "above_bits_site_min": float(above_row["bits_site_min"]),
                            "below_selection_metric_value": float(below_row["selection_metric_value"]),
                            "above_selection_metric_value": float(above_row["selection_metric_value"]),
                            "selection_metric_diff": diff_value,
                            "seq_length": int(below_row["seq_length"]),
                        }
                    )
        return pd.DataFrame(records)

    def select_best_msa_pair(self, pair_candidates: pd.DataFrame) -> pd.Series:
        """Return the best MSA pair by maximal unweighted mean bits/site difference."""
        if pair_candidates.empty:
            raise ValueError("No targeted MSA pairs survive hero-selection filters")
        sort_columns = ["selection_metric_diff", "protein_name", "below_outer_label", "above_outer_label"]
        ordered = pair_candidates.sort_values(sort_columns, ascending=[False, True, True, True], kind="mergesort")
        return ordered.iloc[0]

    def select_best_sample_for_msa(self, candidate: pd.DataFrame, protein_name: str, outer_label: str) -> pd.Series:
        """Choose the highest-novelty sample from one targeted MSA."""
        subset = candidate[(candidate["protein_name"] == str(protein_name)) & (candidate["outer_label"] == str(outer_label))].copy()
        if subset.empty:
            raise ValueError(f"No candidates found for protein={protein_name}, outer_label={outer_label}")
        ordered = subset.sort_values(["novelty", "plddt_mean_display", "sample_id"], ascending=[False, False, True], kind="mergesort")
        return ordered.iloc[0]

    def select_heroes(self, candidate: pd.DataFrame, msa_summary: pd.DataFrame) -> Tuple[HeroSelectionResult, pd.DataFrame, pd.DataFrame]:
        """Select the winning MSA pair and one hero sample from each side."""
        pair_candidates = self.compute_pair_candidates(msa_summary)
        best_pair = self.select_best_msa_pair(pair_candidates)
        below_sample = self.select_best_sample_for_msa(candidate, str(best_pair["protein_name"]), str(best_pair["below_outer_label"]))
        above_sample = self.select_best_sample_for_msa(candidate, str(best_pair["protein_name"]), str(best_pair["above_outer_label"]))
        if str(below_sample["anchor_sequence"]) != str(above_sample["anchor_sequence"]):
            raise ValueError("Selected below and above hero samples do not share the same anchor_sequence")
        result = HeroSelectionResult(
            protein_name=str(best_pair["protein_name"]),
            anchor_sequence=str(best_pair["anchor_sequence"]),
            below_outer_label=str(best_pair["below_outer_label"]),
            above_outer_label=str(best_pair["above_outer_label"]),
            below_sample_id=str(below_sample["sample_id"]),
            above_sample_id=str(above_sample["sample_id"]),
            selection_metric_name="unweighted_mean_bits_site",
            selection_metric_diff=float(best_pair["selection_metric_diff"]),
        )
        selected = pd.DataFrame(
            [
                self.build_selected_sample_record("below", result, below_sample),
                self.build_selected_sample_record("above", result, above_sample),
            ]
        )
        return result, pair_candidates, selected

    def build_selected_sample_record(self, side: str, selection: HeroSelectionResult, sample: pd.Series) -> Dict[str, Any]:
        """Build a selected-sample provenance row."""
        outer_label = selection.below_outer_label if side == "below" else selection.above_outer_label
        sample_id = selection.below_sample_id if side == "below" else selection.above_sample_id
        return {
            "side": str(side),
            "protein_name": str(selection.protein_name),
            "outer_label": str(outer_label),
            "sample_id": str(sample_id),
            "novelty": float(sample["novelty"]),
            "bits_site_min": float(sample["bits_site_min"]),
            "selection_metric_value": float(sample["selection_metric_value"]),
            "plddt_mean_display": float(sample["plddt_mean_display"]),
            "seq_length": int(sample["seq_length"]),
            "anchor_sequence": str(sample["anchor_sequence"]),
        }

    def validate_expected_heroes(self, selection: HeroSelectionResult) -> None:
        """Fail loudly if dynamic hero selection does not match manuscript heroes."""
        if selection.below_sample_id != str(self.config.expected_below_sample_id):
            raise ValueError(f"Dynamic below hero selection changed: expected {self.config.expected_below_sample_id}, observed {selection.below_sample_id}")
        if selection.above_sample_id != str(self.config.expected_above_sample_id):
            raise ValueError(f"Dynamic above hero selection changed: expected {self.config.expected_above_sample_id}, observed {selection.above_sample_id}")

    def build_trace_tables(self, per_site: pd.DataFrame, selection: HeroSelectionResult) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Build below and above hero trace tables."""
        below_trace = self.build_one_trace_table(per_site, selection.below_sample_id, "below", selection.protein_name, selection.below_outer_label)
        above_trace = self.build_one_trace_table(per_site, selection.above_sample_id, "above", selection.protein_name, selection.above_outer_label)
        return below_trace, above_trace

    def build_one_trace_table(self, per_site: pd.DataFrame, sample_id: str, side: str, protein_name: str, outer_label: str) -> pd.DataFrame:
        """Build one raw and smoothed hero trace table."""
        output = per_site[per_site["sample_id"] == str(sample_id)].copy()
        if output.empty:
            raise ValueError(f"Per-site rows missing for sample_id={sample_id}")
        output = output.sort_values("site_index", kind="mergesort").reset_index(drop=True)
        output["protein_name"] = str(protein_name)
        output["outer_label"] = str(outer_label)
        output["side"] = str(side)
        output["site_index"] = pd.to_numeric(output["site_index"], errors="coerce").astype(int)
        for column_name in ["msa_bits_per_site", "entropy_cond_all_nats", "plddt_mean", "pae_mean", "is_mutation"]:
            output[column_name] = pd.to_numeric(output[column_name], errors="coerce")
        output["plddt_mean_display"] = self.scale_plddt_series_to_display(output["plddt_mean"])
        output["is_mutation"] = output["is_mutation"].fillna(0).astype(np.int8)
        output["msa_bits_per_site_smooth"] = self.compute_centered_rolling_mean(output["msa_bits_per_site"])
        output["entropy_cond_all_nats_smooth"] = self.compute_centered_rolling_mean(output["entropy_cond_all_nats"])
        output["plddt_mean_display_smooth"] = self.compute_centered_rolling_mean(output["plddt_mean_display"])
        output["pae_mean_smooth"] = self.compute_centered_rolling_mean(output["pae_mean"])
        return output[
            [
                "sample_id",
                "protein_name",
                "outer_label",
                "side",
                "site_index",
                "is_mutation",
                "msa_bits_per_site",
                "msa_bits_per_site_smooth",
                "entropy_cond_all_nats",
                "entropy_cond_all_nats_smooth",
                "plddt_mean_display",
                "plddt_mean_display_smooth",
                "pae_mean",
                "pae_mean_smooth",
            ]
        ].copy()

    def compute_centered_rolling_mean(self, series: pd.Series) -> pd.Series:
        """Return a centered rolling mean."""
        return pd.to_numeric(series, errors="coerce").rolling(window=int(self.config.smoothing_window), center=True, min_periods=1).mean()

    def build_delta_trace_table(self, below_trace: pd.DataFrame, above_trace: pd.DataFrame) -> pd.DataFrame:
        """Build above-minus-below delta trace table."""
        columns = [
            "msa_bits_per_site",
            "msa_bits_per_site_smooth",
            "entropy_cond_all_nats",
            "entropy_cond_all_nats_smooth",
            "plddt_mean_display",
            "plddt_mean_display_smooth",
            "pae_mean",
            "pae_mean_smooth",
        ]
        left = below_trace[["site_index"] + columns].copy()
        right = above_trace[["site_index"] + columns].copy()
        merged = left.merge(right, on="site_index", how="inner", suffixes=("_below", "_above"), validate="one_to_one")
        output = pd.DataFrame({"site_index": merged["site_index"].astype(int)})
        for column_name in columns:
            output[f"delta_{column_name}"] = merged[f"{column_name}_above"] - merged[f"{column_name}_below"]
        return output

    def build_panel_spearman_table(self, trace: pd.DataFrame) -> pd.DataFrame:
        """Compute Spearman correlations across hero panel combinations."""
        panel_map = [
            ("msa_bits_per_site", "MSA bits/site"),
            ("entropy_cond_all_nats", "Model entropy [nats]"),
            ("plddt_mean_display", "Chai-1 pLDDT (0-100)"),
            ("pae_mean", "Chai-1 PAE"),
        ]
        records: List[Dict[str, Any]] = []
        sample_id = str(trace["sample_id"].iloc[0])
        side = str(trace["side"].iloc[0])
        for left_index in range(len(panel_map)):
            left_column, left_label = panel_map[left_index]
            for right_index in range(left_index + 1, len(panel_map)):
                right_column, right_label = panel_map[right_index]
                left_values = pd.to_numeric(trace[left_column], errors="coerce").to_numpy(dtype=float, copy=False)
                right_values = pd.to_numeric(trace[right_column], errors="coerce").to_numpy(dtype=float, copy=False)
                finite_mask = np.isfinite(left_values) & np.isfinite(right_values)
                n_pairs = int(finite_mask.sum())
                if n_pairs >= 3:
                    rho_value, p_value = stats.spearmanr(left_values[finite_mask], right_values[finite_mask])
                else:
                    rho_value, p_value = float("nan"), float("nan")
                records.append(
                    {
                        "sample_id": sample_id,
                        "side": side,
                        "panel_a_column": left_column,
                        "panel_a_label": left_label,
                        "panel_b_column": right_column,
                        "panel_b_label": right_label,
                        "n": n_pairs,
                        "spearman_rho": float(rho_value),
                        "p_value": float(p_value),
                    }
                )
        return pd.DataFrame(records)

    def add_condition_columns(self, cohort: pd.DataFrame) -> pd.DataFrame:
        """Add threshold-based condition columns with shared utility semantics."""
        output = add_threshold_condition_columns(cohort, threshold=float(self.config.bits_site_min_threshold))
        pair_columns = self.select_pair_id_columns(output)
        output["pair_id_columns"] = ",".join(pair_columns)
        output["pair_id"] = output[pair_columns].astype(str).agg("__".join, axis=1)
        return output

    def select_pair_id_columns(self, cohort: pd.DataFrame) -> List[str]:
        """Select matched-control pair identifier columns."""
        candidate_columns = ["protein_name", "outer_label_base", "corruption_rate_init", "seed", "temperature", "decode_steps", "control_rep"]
        existing_columns = [column_name for column_name in candidate_columns if column_name in cohort.columns]
        if "protein_name" not in existing_columns or "outer_label_base" not in existing_columns:
            raise ValueError("Pairing requires protein_name and outer_label_base")
        return existing_columns

    def compute_local_information_flow_by_sample(self, per_site: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        """Compute per-sample local coupling between residue MSA bits and outcomes."""
        meta_columns = ["sample_id", "protein_name", "outer_label", "control_type", "target_bits_site_min", "target_stratum", "condition4", "pair_id"]
        available_meta_columns = [column_name for column_name in meta_columns if column_name in cohort.columns]
        meta = cohort[available_meta_columns].drop_duplicates(subset=["sample_id"]).copy()
        overlapping_columns = [column_name for column_name in meta.columns if column_name != "sample_id" and column_name in per_site.columns]
        merge_columns = [column_name for column_name in meta.columns if column_name == "sample_id" or column_name not in overlapping_columns]
        merged = per_site.merge(meta[merge_columns], on="sample_id", how="inner", validate="many_to_one")
        for column_name in ["msa_bits_per_site", "entropy_cond_all_nats", "plddt_mean", "pae_mean"]:
            merged[column_name] = pd.to_numeric(merged[column_name], errors="coerce")
        merged["plddt_mean_display"] = self.scale_plddt_series_to_display(merged["plddt_mean"])
        outcome_map = {
            "entropy": ("entropy_cond_all_nats", "Local entropy [nats]"),
            "plddt": ("plddt_mean_display", "Local Chai-1 pLDDT (0-100)"),
            "pae": ("pae_mean", "Local Chai-1 PAE"),
        }
        records: List[Dict[str, Any]] = []
        for sample_id, sample_group in merged.groupby("sample_id", sort=False):
            first = sample_group.iloc[0]
            exposure_values = pd.to_numeric(sample_group["msa_bits_per_site"], errors="coerce").to_numpy(dtype=float, copy=False)
            for outcome_key, outcome_spec in outcome_map.items():
                outcome_column, outcome_label = outcome_spec
                outcome_values = pd.to_numeric(sample_group[outcome_column], errors="coerce").to_numpy(dtype=float, copy=False)
                finite_mask = np.isfinite(exposure_values) & np.isfinite(outcome_values)
                n_residues = int(finite_mask.sum())
                if n_residues >= int(self.config.local_flow_min_finite_residues):
                    rho_value, p_value = stats.spearmanr(exposure_values[finite_mask], outcome_values[finite_mask])
                else:
                    rho_value, p_value = float("nan"), float("nan")
                rho_float = float(rho_value) if np.isfinite(rho_value) else float("nan")
                rho_clip = float(np.clip(rho_float, -0.999999, 0.999999)) if np.isfinite(rho_float) else float("nan")
                z_fisher = float(np.arctanh(rho_clip)) if np.isfinite(rho_clip) else float("nan")
                records.append(
                    {
                        "sample_id": str(sample_id),
                        "protein_name": str(first["protein_name"]),
                        "outer_label": str(first["outer_label"]) if "outer_label" in sample_group.columns else "",
                        "control_type": str(first["control_type"]) if "control_type" in sample_group.columns else "",
                        "target_bits_site_min": float(first["target_bits_site_min"]) if "target_bits_site_min" in sample_group.columns else float("nan"),
                        "target_stratum": str(first["target_stratum"]) if "target_stratum" in sample_group.columns else "",
                        "condition4": str(first["condition4"]) if "condition4" in sample_group.columns else "",
                        "pair_id": str(first["pair_id"]) if "pair_id" in sample_group.columns else "",
                        "outcome": str(outcome_key),
                        "outcome_label": str(outcome_label),
                        "n_residues": n_residues,
                        "rho_spearman": rho_float,
                        "p_value": float(p_value) if np.isfinite(p_value) else float("nan"),
                        "z_fisher": z_fisher,
                    }
                )
        return pd.DataFrame(records)

    def aggregate_local_information_flow_by_pair(self, local_flow: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sample-level local-flow statistics by matched pair."""
        required_columns = ["pair_id", "condition4", "outcome", "outcome_label", "rho_spearman"]
        require_columns(local_flow, required_columns, "local flow sample table")
        working = local_flow.copy()
        working["rho_spearman"] = pd.to_numeric(working["rho_spearman"], errors="coerce")
        grouped = working.groupby(["pair_id", "condition4", "outcome", "outcome_label"], sort=False)
        output = grouped.agg(pair_rho_spearman=("rho_spearman", "mean"), n_samples=("sample_id", "nunique")).reset_index()
        return output[["pair_id", "condition4", "outcome", "outcome_label", "pair_rho_spearman", "n_samples"]]

    def aggregate_local_information_flow_by_protein(self, local_flow: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sample-level local-flow statistics by protein."""
        required_columns = ["protein_name", "condition4", "outcome", "outcome_label", "rho_spearman"]
        require_columns(local_flow, required_columns, "local flow sample table")
        working = local_flow.copy()
        working["rho_spearman"] = pd.to_numeric(working["rho_spearman"], errors="coerce")
        aggregator = str(self.config.local_flow_protein_aggregator)
        grouped = working.groupby(["protein_name", "condition4", "outcome", "outcome_label"], sort=False)
        if aggregator == "median":
            output = grouped.agg(protein_rho_spearman=("rho_spearman", "median"), n_samples=("sample_id", "nunique")).reset_index()
        elif aggregator == "mean":
            output = grouped.agg(protein_rho_spearman=("rho_spearman", "mean"), n_samples=("sample_id", "nunique")).reset_index()
        else:
            raise ValueError("local_flow_protein_aggregator must be median or mean")
        return output[["protein_name", "condition4", "outcome", "outcome_label", "protein_rho_spearman", "n_samples"]]

    def bootstrap_local_information_flow_protein_deltas(self, local_flow_by_protein: pd.DataFrame) -> pd.DataFrame:
        """Bootstrap below-minus-above deltas from protein-level local flow."""
        comparison_specs = [
            ("targeted", "below targeted", "above targeted"),
            ("matched_control", "below matched control", "above matched control"),
        ]
        rng = np.random.default_rng(int(self.config.bootstrap_seed))
        records: List[Dict[str, Any]] = []
        for outcome_value, outcome_group in local_flow_by_protein.groupby("outcome", sort=False):
            outcome_label = str(outcome_group["outcome_label"].iloc[0])
            for comparison_key, below_condition, above_condition in comparison_specs:
                below_values = outcome_group.loc[outcome_group["condition4"] == below_condition, "protein_rho_spearman"].to_numpy(dtype=float, copy=False)
                above_values = outcome_group.loc[outcome_group["condition4"] == above_condition, "protein_rho_spearman"].to_numpy(dtype=float, copy=False)
                below_values = below_values[np.isfinite(below_values)]
                above_values = above_values[np.isfinite(above_values)]
                if below_values.size == 0 or above_values.size == 0:
                    continue
                observed_delta = float(np.mean(below_values) - np.mean(above_values))
                for replicate_index in range(int(self.config.bootstrap_replicates)):
                    below_sample = rng.choice(below_values, size=below_values.size, replace=True)
                    above_sample = rng.choice(above_values, size=above_values.size, replace=True)
                    records.append(
                        {
                            "comparison": str(comparison_key),
                            "below_condition": str(below_condition),
                            "above_condition": str(above_condition),
                            "outcome": str(outcome_value),
                            "outcome_label": outcome_label,
                            "replicate_index": int(replicate_index),
                            "delta_rho": float(np.mean(below_sample) - np.mean(above_sample)),
                            "observed_delta_rho": observed_delta,
                        }
                    )
        return pd.DataFrame(records)

    def summarize_bootstrap_local_information_flow_protein(self, local_flow_by_protein: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
        """Summarize protein-level local-flow bootstrap deltas."""
        records: List[Dict[str, Any]] = []
        if bootstrap.empty:
            return pd.DataFrame(records)
        for group_key, group in bootstrap.groupby(["comparison", "below_condition", "above_condition", "outcome", "outcome_label"], sort=False):
            comparison_key, below_condition, above_condition, outcome_value, outcome_label = group_key
            delta_values = pd.to_numeric(group["delta_rho"], errors="coerce").to_numpy(dtype=float, copy=False)
            delta_values = delta_values[np.isfinite(delta_values)]
            observed_delta = float(group["observed_delta_rho"].iloc[0])
            below_group = local_flow_by_protein[(local_flow_by_protein["condition4"] == below_condition) & (local_flow_by_protein["outcome"] == outcome_value)].copy()
            above_group = local_flow_by_protein[(local_flow_by_protein["condition4"] == above_condition) & (local_flow_by_protein["outcome"] == outcome_value)].copy()
            below_rho = pd.to_numeric(below_group["protein_rho_spearman"], errors="coerce")
            above_rho = pd.to_numeric(above_group["protein_rho_spearman"], errors="coerce")
            if delta_values.size:
                ci_lo = float(np.quantile(delta_values, 0.025))
                ci_hi = float(np.quantile(delta_values, 0.975))
                prob_nonpos = float(np.mean(delta_values <= 0.0))
                prob_nonneg = float(np.mean(delta_values >= 0.0))
                p_boot = float(min(1.0, 2.0 * min(prob_nonpos, prob_nonneg)))
                median_delta = float(np.nanmedian(delta_values))
            else:
                ci_lo = float("nan")
                ci_hi = float("nan")
                p_boot = float("nan")
                median_delta = float("nan")
            records.append(
                {
                    "comparison": str(comparison_key),
                    "below_condition": str(below_condition),
                    "above_condition": str(above_condition),
                    "outcome": str(outcome_value),
                    "outcome_label": str(outcome_label),
                    "n_below_proteins": int(below_group.shape[0]),
                    "n_above_proteins": int(above_group.shape[0]),
                    "observed_mean_rho_below": float(np.nanmean(below_rho.to_numpy(dtype=float, copy=False))),
                    "observed_mean_rho_above": float(np.nanmean(above_rho.to_numpy(dtype=float, copy=False))),
                    "observed_delta_rho": observed_delta,
                    "bootstrap_median_delta_rho": median_delta,
                    "bootstrap_ci_lo": ci_lo,
                    "bootstrap_ci_hi": ci_hi,
                    "bootstrap_p_value": p_boot,
                    "bootstrap_significant": bool(np.isfinite(ci_lo) and np.isfinite(ci_hi) and ((ci_lo > 0.0) or (ci_hi < 0.0))),
                }
            )
        return pd.DataFrame(records)

    def build_bitsite_distribution_residue_table(self, per_site: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
        """Build merged residue table for bits/site confidence distributions."""
        output_columns = [
            "sample_id",
            "protein_name",
            "site_index",
            "is_mutation",
            "control_type",
            "target_bits_site_min",
            "target_stratum",
            "condition4",
            "pair_id",
            "msa_bits_per_site",
            "plddt_mean",
            "plddt_mean_display",
            "pae_mean",
        ]
        if per_site.empty or cohort.empty:
            return pd.DataFrame(columns=output_columns)
        meta_columns = ["sample_id", "protein_name", "control_type", "target_bits_site_min", "target_stratum", "condition4", "pair_id"]
        available_meta_columns = [column_name for column_name in meta_columns if column_name in cohort.columns]
        meta = cohort[available_meta_columns].drop_duplicates(subset=["sample_id"]).copy()
        overlapping_columns = [column_name for column_name in meta.columns if column_name != "sample_id" and column_name in per_site.columns]
        merge_columns = [column_name for column_name in meta.columns if column_name == "sample_id" or column_name not in overlapping_columns]
        merged = per_site.merge(meta[merge_columns], on="sample_id", how="inner", validate="many_to_one")
        for column_name in ["sample_id", "protein_name"]:
            merged[column_name] = merged[column_name].astype(str)
        for column_name in ["site_index", "is_mutation", "msa_bits_per_site", "plddt_mean", "pae_mean"]:
            merged[column_name] = pd.to_numeric(merged[column_name], errors="coerce")
        merged["plddt_mean_display"] = self.scale_plddt_series_to_display(merged["plddt_mean"])
        merged["is_mutation"] = merged["is_mutation"].fillna(0).astype(np.int8)
        for column_name in ["control_type", "target_stratum", "condition4", "pair_id"]:
            if column_name in merged.columns:
                merged[column_name] = merged[column_name].astype(str)
        finite_mask = np.isfinite(merged["site_index"].to_numpy(dtype=float, copy=False))
        finite_mask &= np.isfinite(merged["msa_bits_per_site"].to_numpy(dtype=float, copy=False))
        output = merged.loc[finite_mask].copy()
        output["site_index"] = output["site_index"].astype(int)
        for column_name in output_columns:
            if column_name not in output.columns:
                output[column_name] = float("nan") if column_name in ["target_bits_site_min", "msa_bits_per_site", "plddt_mean", "plddt_mean_display", "pae_mean"] else ""
        return output[output_columns].reset_index(drop=True)

    def compute_bitsite_bin_edges(self, residue: pd.DataFrame) -> np.ndarray:
        """Compute fixed-width global bits/site bin edges."""
        values = pd.to_numeric(residue["msa_bits_per_site"], errors="coerce").to_numpy(dtype=float, copy=False)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.array([], dtype=float)
        bin_width = float(self.config.bitsite_distribution_bin_width)
        if not np.isfinite(bin_width) or bin_width <= 0.0:
            raise ValueError("bitsite_distribution_bin_width must be positive")
        min_value = float(np.nanmin(values))
        max_value = float(np.nanmax(values))
        start_value = float(np.floor(min_value / bin_width) * bin_width)
        stop_value = float(np.ceil(max_value / bin_width) * bin_width)
        if not np.isfinite(stop_value) or stop_value <= max_value or np.isclose(stop_value, max_value):
            stop_value += bin_width
        edges = np.arange(start_value, stop_value + (0.5 * bin_width), bin_width, dtype=float)
        if edges.size < 2:
            edges = np.array([start_value, start_value + bin_width], dtype=float)
        return np.round(edges, 10)

    def assign_bitsite_bins(self, residue: pd.DataFrame, bin_edges: np.ndarray) -> pd.DataFrame:
        """Assign residue-level bits/site values to bins."""
        output = residue.copy()
        if bin_edges.size < 2:
            output["bitsite_bin_index"] = pd.Series(dtype="Int64")
            output["bitsite_bin_left"] = float("nan")
            output["bitsite_bin_right"] = float("nan")
            output["bitsite_bin_label"] = ""
            output["bitsite_bin_midpoint"] = float("nan")
            return output
        values = pd.to_numeric(output["msa_bits_per_site"], errors="coerce")
        bin_codes = pd.cut(values, bins=bin_edges, labels=False, right=False, include_lowest=True)
        output["bitsite_bin_index"] = pd.Series(bin_codes, index=output.index, dtype="Int64")
        left_map = {int(index): float(bin_edges[index]) for index in range(len(bin_edges) - 1)}
        right_map = {int(index): float(bin_edges[index + 1]) for index in range(len(bin_edges) - 1)}
        output["bitsite_bin_left"] = output["bitsite_bin_index"].map(left_map)
        output["bitsite_bin_right"] = output["bitsite_bin_index"].map(right_map)
        output["bitsite_bin_midpoint"] = (pd.to_numeric(output["bitsite_bin_left"], errors="coerce") + pd.to_numeric(output["bitsite_bin_right"], errors="coerce")) / 2.0
        output["bitsite_bin_label"] = [f"[{float(left):.2f}, {float(right):.2f})" if np.isfinite(left) and np.isfinite(right) else "" for left, right in zip(output["bitsite_bin_left"], output["bitsite_bin_right"])]
        assigned_mask = output["bitsite_bin_index"].notna().to_numpy(dtype=bool, copy=False)
        finite_mask = np.isfinite(pd.to_numeric(output["msa_bits_per_site"], errors="coerce").to_numpy(dtype=float, copy=False))
        if np.any(finite_mask & ~assigned_mask):
            raise ValueError("Some finite msa_bits_per_site values were not assigned to a bits/site bin")
        return output

    def get_plddt_band_order(self) -> List[str]:
        """Return canonical pLDDT band order."""
        low_edge, mid_edge, high_edge = [float(value) for value in self.config.bitsite_distribution_plddt_band_edges]
        return [f"<{int(low_edge)}", f"{int(low_edge)}–{int(mid_edge)}", f"{int(mid_edge)}–{int(high_edge)}", f"≥{int(high_edge)}"]

    def get_pae_band_order(self) -> List[str]:
        """Return canonical PAE band order."""
        low_edge, mid_edge, high_edge = [float(value) for value in self.config.bitsite_distribution_pae_band_edges]
        return [f"≤{int(low_edge)}", f"{int(low_edge)}–{int(mid_edge)}", f"{int(mid_edge)}–{int(high_edge)}", f">{int(high_edge)}"]

    def assign_plddt_bands(self, residue: pd.DataFrame) -> pd.DataFrame:
        """Assign pLDDT values to bands."""
        output = residue.copy()
        values = pd.to_numeric(output["plddt_mean_display"], errors="coerce")
        low_edge, mid_edge, high_edge = [float(value) for value in self.config.bitsite_distribution_plddt_band_edges]
        conditions = [values < low_edge, (values >= low_edge) & (values < mid_edge), (values >= mid_edge) & (values < high_edge), values >= high_edge]
        output["plddt_band"] = np.select(conditions, self.get_plddt_band_order(), default="")
        return output

    def assign_pae_bands(self, residue: pd.DataFrame) -> pd.DataFrame:
        """Assign PAE values to bands."""
        output = residue.copy()
        values = pd.to_numeric(output["pae_mean"], errors="coerce")
        low_edge, mid_edge, high_edge = [float(value) for value in self.config.bitsite_distribution_pae_band_edges]
        conditions = [values <= low_edge, (values > low_edge) & (values <= mid_edge), (values > mid_edge) & (values <= high_edge), values > high_edge]
        output["pae_band"] = np.select(conditions, self.get_pae_band_order(), default="")
        return output

    def filter_bitsite_residue_subset(self, residue: pd.DataFrame, edit_subset: str) -> pd.DataFrame:
        """Return a residue subset for bits/site distribution analysis."""
        subset_value = str(edit_subset)
        if subset_value == "all":
            return residue.copy()
        mutation_values = pd.to_numeric(residue["is_mutation"], errors="coerce").fillna(0).astype(np.int8)
        if subset_value == "edited_only":
            return residue.loc[mutation_values == 1].copy()
        if subset_value == "unedited_only":
            return residue.loc[mutation_values == 0].copy()
        raise ValueError(f"Unsupported bits/site distribution edit_subset: {subset_value}")

    def build_band_grid(self, key_frame: pd.DataFrame, band_column: str, band_order: Sequence[str]) -> pd.DataFrame:
        """Build complete key-by-band grid."""
        if key_frame.empty:
            return pd.DataFrame(columns=list(key_frame.columns) + [str(band_column)])
        left = key_frame.drop_duplicates().copy()
        left["grid_join_key"] = 1
        right = pd.DataFrame({str(band_column): [str(value) for value in band_order]})
        right["grid_join_key"] = 1
        output = left.merge(right, on="grid_join_key", how="inner", validate="many_to_many")
        return output.drop(columns=["grid_join_key"])

    def summarize_bitsite_band_distribution_for_grouping(self, residue: pd.DataFrame, grouping_column: str, grouping_type: str, edit_subset: str, band_column: str, band_order: Sequence[str], output_columns: Sequence[str]) -> pd.DataFrame:
        """Summarize band probabilities across bits/site bins."""
        if residue.empty:
            return pd.DataFrame(columns=list(output_columns))
        required_columns = [grouping_column, "protein_name", "sample_id", "bitsite_bin_index", "bitsite_bin_left", "bitsite_bin_right", "bitsite_bin_label", "bitsite_bin_midpoint", band_column]
        require_columns(residue, required_columns, "bitsite residue summary input")
        working = residue[required_columns].copy()
        for column_name in [grouping_column, "protein_name", "sample_id"]:
            working[column_name] = working[column_name].astype(str)
        working = working[(working[grouping_column] != "") & (working[band_column] != "") & working["bitsite_bin_index"].notna()].copy()
        if working.empty:
            return pd.DataFrame(columns=list(output_columns))
        working["bitsite_bin_index"] = working["bitsite_bin_index"].astype(int)
        raw_band_counts = working.groupby([grouping_column, "bitsite_bin_index", band_column], sort=False).size().rename("n_residues_band").reset_index()
        raw_bin_counts = working.groupby([grouping_column, "bitsite_bin_index"], sort=False).size().rename("n_residues_total").reset_index()
        sample_counts = working.groupby([grouping_column, "bitsite_bin_index"], sort=False)["sample_id"].nunique().rename("n_samples").reset_index()
        protein_counts = working.groupby([grouping_column, "bitsite_bin_index"], sort=False)["protein_name"].nunique().rename("n_proteins").reset_index()
        probability_summary = self.compute_bitsite_probability_summary(working, grouping_column, band_column, band_order, raw_band_counts, raw_bin_counts)
        bin_metadata = working[[grouping_column, "bitsite_bin_index", "bitsite_bin_left", "bitsite_bin_right", "bitsite_bin_label", "bitsite_bin_midpoint"]].drop_duplicates(subset=[grouping_column, "bitsite_bin_index"]).copy()
        supported_bins = raw_bin_counts[raw_bin_counts["n_residues_total"] >= int(self.config.bitsite_distribution_min_bin_count)].copy()
        if supported_bins.empty:
            return pd.DataFrame(columns=list(output_columns))
        grid_records: List[Dict[str, Any]] = []
        for row_index, row in supported_bins.iterrows():
            for band_value in band_order:
                grid_records.append({grouping_column: str(row[grouping_column]), "bitsite_bin_index": int(row["bitsite_bin_index"]), band_column: str(band_value)})
        output = pd.DataFrame(grid_records)
        output = output.merge(bin_metadata, on=[grouping_column, "bitsite_bin_index"], how="left", validate="many_to_one")
        output = output.merge(raw_bin_counts, on=[grouping_column, "bitsite_bin_index"], how="left", validate="many_to_one")
        output = output.merge(sample_counts, on=[grouping_column, "bitsite_bin_index"], how="left", validate="many_to_one")
        output = output.merge(protein_counts, on=[grouping_column, "bitsite_bin_index"], how="left", validate="many_to_one")
        output = output.merge(raw_band_counts, on=[grouping_column, "bitsite_bin_index", band_column], how="left", validate="one_to_one")
        output = output.merge(probability_summary, on=[grouping_column, "bitsite_bin_index", band_column], how="left", validate="one_to_one")
        for column_name in ["n_residues_band", "n_residues_total", "n_samples", "n_proteins"]:
            output[column_name] = pd.to_numeric(output[column_name], errors="coerce").fillna(0).astype(int)
        output["probability"] = pd.to_numeric(output["probability"], errors="coerce").fillna(0.0)
        output["grouping_type"] = str(grouping_type)
        output["group_label"] = output[grouping_column].astype(str)
        output["edit_subset"] = str(edit_subset)
        output = output[list(output_columns)].copy()
        output = output.sort_values(["grouping_type", "group_label", "edit_subset", "bitsite_bin_index", band_column], ascending=[True, True, True, True, True], kind="mergesort").reset_index(drop=True)
        self.validate_bitsite_distribution_summary(output, band_column)
        return output

    def compute_bitsite_probability_summary(self, working: pd.DataFrame, grouping_column: str, band_column: str, band_order: Sequence[str], raw_band_counts: pd.DataFrame, raw_bin_counts: pd.DataFrame) -> pd.DataFrame:
        """Compute band probability summary under configured weighting."""
        weighting_mode = str(self.config.bitsite_distribution_weighting_mode)
        if weighting_mode == "raw_residue":
            probability_summary = raw_band_counts.merge(raw_bin_counts, on=[grouping_column, "bitsite_bin_index"], how="left", validate="many_to_one")
            probability_summary["probability"] = probability_summary["n_residues_band"] / probability_summary["n_residues_total"]
            return probability_summary[[grouping_column, "bitsite_bin_index", band_column, "probability"]].copy()
        sample_band_counts = working.groupby([grouping_column, "protein_name", "sample_id", "bitsite_bin_index", band_column], sort=False).size().rename("sample_band_count").reset_index()
        sample_bin_counts = working.groupby([grouping_column, "protein_name", "sample_id", "bitsite_bin_index"], sort=False).size().rename("sample_bin_total").reset_index()
        sample_grid = self.build_band_grid(sample_bin_counts[[grouping_column, "protein_name", "sample_id", "bitsite_bin_index"]], band_column, band_order)
        sample_grid = sample_grid.merge(sample_bin_counts, on=[grouping_column, "protein_name", "sample_id", "bitsite_bin_index"], how="left", validate="many_to_one")
        sample_grid = sample_grid.merge(sample_band_counts, on=[grouping_column, "protein_name", "sample_id", "bitsite_bin_index", band_column], how="left", validate="one_to_one")
        sample_grid["sample_band_count"] = pd.to_numeric(sample_grid["sample_band_count"], errors="coerce").fillna(0.0)
        sample_grid["sample_probability"] = sample_grid["sample_band_count"] / sample_grid["sample_bin_total"]
        if weighting_mode == "sample_balanced":
            return sample_grid.groupby([grouping_column, "bitsite_bin_index", band_column], sort=False)["sample_probability"].mean().rename("probability").reset_index()
        if weighting_mode == "protein_balanced":
            protein_probabilities = sample_grid.groupby([grouping_column, "protein_name", "bitsite_bin_index", band_column], sort=False)["sample_probability"].mean().rename("protein_probability").reset_index()
            protein_grid = self.build_band_grid(sample_grid[[grouping_column, "protein_name", "bitsite_bin_index"]].drop_duplicates(), band_column, band_order)
            protein_grid = protein_grid.merge(protein_probabilities, on=[grouping_column, "protein_name", "bitsite_bin_index", band_column], how="left", validate="one_to_one")
            protein_grid["protein_probability"] = pd.to_numeric(protein_grid["protein_probability"], errors="coerce").fillna(0.0)
            return protein_grid.groupby([grouping_column, "bitsite_bin_index", band_column], sort=False)["protein_probability"].mean().rename("probability").reset_index()
        raise ValueError("bitsite_distribution_weighting_mode must be raw_residue, sample_balanced, or protein_balanced")

    def validate_bitsite_distribution_summary(self, summary: pd.DataFrame, band_column: str) -> None:
        """Validate bits/site distribution summary probabilities."""
        if summary.empty:
            return
        duplicate_columns = ["grouping_type", "group_label", "edit_subset", "bitsite_bin_index", band_column]
        if summary.duplicated(subset=duplicate_columns).any():
            raise ValueError(f"Duplicate rows detected in bits/site distribution summary for {band_column}")
        probability_values = pd.to_numeric(summary["probability"], errors="coerce").to_numpy(dtype=float, copy=False)
        if not np.all(np.isfinite(probability_values)):
            raise ValueError(f"Non-finite probabilities detected in bits/site distribution summary for {band_column}")
        if np.any(probability_values < -1.0e-10) or np.any(probability_values > 1.0 + 1.0e-10):
            raise ValueError(f"Out-of-range probabilities detected in bits/site distribution summary for {band_column}")
        probability_sums = summary.groupby(["grouping_type", "group_label", "edit_subset", "bitsite_bin_index"], sort=False)["probability"].sum().reset_index(name="probability_sum")
        probability_sum_values = pd.to_numeric(probability_sums["probability_sum"], errors="coerce").to_numpy(dtype=float, copy=False)
        if not np.allclose(probability_sum_values, 1.0, atol=1.0e-6):
            invalid_rows = probability_sums.loc[~np.isclose(probability_sum_values, 1.0, atol=1.0e-6)].head(10)
            raise ValueError(f"Band probabilities do not sum to 1 for {band_column}. Examples: {invalid_rows.to_dict(orient='records')}")

    def build_bitsite_plddt_distribution_summary(self, residue: pd.DataFrame) -> pd.DataFrame:
        """Build pLDDT distribution summaries for all required residue subsets."""
        output_columns = self.get_plddt_distribution_summary_columns()
        if residue.empty:
            return pd.DataFrame(columns=output_columns)
        binned = self.assign_bitsite_bins(residue, self.compute_bitsite_bin_edges(residue))
        binned = self.assign_plddt_bands(binned)
        return self.build_bitsite_distribution_summary(binned, "plddt_band", self.get_plddt_band_order(), output_columns)

    def build_bitsite_pae_distribution_summary(self, residue: pd.DataFrame) -> pd.DataFrame:
        """Build PAE distribution summaries for all required residue subsets."""
        output_columns = self.get_pae_distribution_summary_columns()
        if residue.empty:
            return pd.DataFrame(columns=output_columns)
        binned = self.assign_bitsite_bins(residue, self.compute_bitsite_bin_edges(residue))
        binned = self.assign_pae_bands(binned)
        return self.build_bitsite_distribution_summary(binned, "pae_band", self.get_pae_band_order(), output_columns)

    def build_bitsite_distribution_summary(self, binned: pd.DataFrame, band_column: str, band_order: Sequence[str], output_columns: Sequence[str]) -> pd.DataFrame:
        """Build overall and condition-level distribution summaries."""
        grouping_column = str(self.config.bitsite_distribution_grouping_mode)
        if grouping_column not in binned.columns:
            raise ValueError(f"Grouping column for bits/site distribution is missing: {grouping_column}")
        summary_frames: List[pd.DataFrame] = []
        for edit_subset in ["all", "edited_only", "unedited_only"]:
            subset = self.filter_bitsite_residue_subset(binned, edit_subset)
            if subset.empty:
                continue
            grouped_summary = self.summarize_bitsite_band_distribution_for_grouping(subset, grouping_column, grouping_column, edit_subset, band_column, band_order, output_columns)
            overall_subset = subset.copy()
            overall_subset["analysis_group"] = "all_samples"
            overall_summary = self.summarize_bitsite_band_distribution_for_grouping(overall_subset, "analysis_group", "overall", edit_subset, band_column, band_order, output_columns)
            if not grouped_summary.empty:
                summary_frames.append(grouped_summary)
            if not overall_summary.empty:
                summary_frames.append(overall_summary)
        if not summary_frames:
            return pd.DataFrame(columns=output_columns)
        output = pd.concat(summary_frames, axis=0, ignore_index=True)
        self.validate_bitsite_distribution_summary(output, band_column)
        return output[list(output_columns)]

    def get_plddt_distribution_summary_columns(self) -> List[str]:
        """Return pLDDT distribution summary columns."""
        return ["grouping_type", "group_label", "edit_subset", "bitsite_bin_index", "bitsite_bin_left", "bitsite_bin_right", "bitsite_bin_label", "bitsite_bin_midpoint", "plddt_band", "n_residues_band", "n_residues_total", "n_samples", "n_proteins", "probability"]

    def get_pae_distribution_summary_columns(self) -> List[str]:
        """Return PAE distribution summary columns."""
        return ["grouping_type", "group_label", "edit_subset", "bitsite_bin_index", "bitsite_bin_left", "bitsite_bin_right", "bitsite_bin_label", "bitsite_bin_midpoint", "pae_band", "n_residues_band", "n_residues_total", "n_samples", "n_proteins", "probability"]

    def build_trace_dictionary(self) -> pd.DataFrame:
        """Describe hero trace columns."""
        rows = [
            {"column": "sample_id", "description": "Selected de novo protein sample identifier."},
            {"column": "site_index", "description": "Residue index used for plotting."},
            {"column": "is_mutation", "description": "Generated residue differs from anchor at this site."},
            {"column": "msa_bits_per_site", "description": "Per-column unweighted MSA bits/site trace."},
            {"column": "entropy_cond_all_nats", "description": "Per-residue conditioned model entropy in nats."},
            {"column": "plddt_mean_display", "description": "Per-residue Chai-1 pLDDT on the 0-100 scale."},
            {"column": "pae_mean", "description": "Per-residue flattened Chai-1 PAE mean."},
            {"column": "smooth columns", "description": "Centered rolling means used for display."},
        ]
        return pd.DataFrame(rows)

    def build_local_flow_dictionary(self) -> pd.DataFrame:
        """Describe local information-flow outputs."""
        rows = [
            {"column": "rho_spearman", "description": "Per-sample Spearman correlation across residues between local MSA bits/site and local outcome."},
            {"column": "pair_rho_spearman", "description": "Mean sample-level rho_spearman within one matched pair, condition, and outcome."},
            {"column": "protein_rho_spearman", "description": "Protein-level local-flow statistic for one protein, condition, and outcome."},
            {"column": "delta_rho", "description": "Bootstrap replicate below-minus-above difference in mean protein-level local coupling."},
            {"column": "bootstrap_ci_lo/bootstrap_ci_hi", "description": "Empirical 95% bootstrap confidence interval for delta_rho."},
        ]
        return pd.DataFrame(rows)

    def build_bitsite_dictionary(self, metric_name: str) -> pd.DataFrame:
        """Describe bits/site distribution outputs."""
        rows = [
            {"column": "msa_bits_per_site", "description": "Per-residue unweighted MSA bits/site value."},
            {"column": "edit_subset", "description": "Residue subset used for summary: all, edited_only, or unedited_only."},
            {"column": "bitsite_bin fields", "description": "Global fixed-width residue-level bits/site bins."},
            {"column": f"{metric_name}_band", "description": "Per-residue confidence band used for probability summaries."},
            {"column": "probability", "description": "Within-bin probability of each band under configured weighting."},
            {"column": "group_label", "description": "all_samples or one of the four condition4 groups."},
        ]
        return pd.DataFrame(rows)


@dataclass
class Figure6Plotter:
    """Render Figure 6 and Supplementary Figure 4 panels."""

    config: Figure6Config
    paths: Figure6Paths
    style: ConditionStyle

    def apply_publication_style(self) -> None:
        """Apply publication plotting defaults."""
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
        """Apply per-axis styling."""
        axis.grid(False)
        axis.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False)
        axis.xaxis.set_ticks_position("bottom")
        axis.yaxis.set_ticks_position("left")
        sns.despine(ax=axis, top=True, right=True)

    def plot_hero_stacked(self, trace: pd.DataFrame, title_text: str, output_stem: str, color_key: str) -> None:
        """Render one vertically stacked hero trace plot."""
        fig, axes = plt.subplots(4, 1, figsize=(10.8, 7.6), sharex=True)
        self.draw_stacked_into_axes(axes, trace, title_text, color_key)
        axes[-1].set_xlabel("Residue index")
        fig.tight_layout(h_pad=0.35)
        save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)

    def draw_stacked_into_axes(self, axes: Sequence[plt.Axes], trace: pd.DataFrame, title_text: str, color_key: str) -> None:
        """Draw stacked hero traces into existing axes."""
        color_value = self.style.get_color(color_key)
        accent_color = self.style.blend_hex_with_white(color_value, 0.45)
        x_values = trace["site_index"].to_numpy(dtype=int, copy=False)
        panel_specs = [
            ("msa_bits_per_site", "msa_bits_per_site_smooth", "MSA bits/site"),
            ("entropy_cond_all_nats", "entropy_cond_all_nats_smooth", "Model entropy [nats]"),
            ("plddt_mean_display", "plddt_mean_display_smooth", "Chai-1 pLDDT (0-100)"),
            ("pae_mean", "pae_mean_smooth", "Chai-1 PAE"),
        ]
        for axis, panel in zip(axes, panel_specs):
            raw_column, smooth_column, y_label = panel
            raw_values = pd.to_numeric(trace[raw_column], errors="coerce").to_numpy(dtype=float, copy=False)
            smooth_values = pd.to_numeric(trace[smooth_column], errors="coerce").to_numpy(dtype=float, copy=False)
            axis.plot(x_values, raw_values, color=accent_color, linewidth=0.9, alpha=0.55)
            axis.plot(x_values, smooth_values, color=color_value, linewidth=2.2, alpha=0.95)
            axis.set_ylabel(y_label)
            if raw_column == "plddt_mean_display":
                axis.set_ylim(bottom=0.0, top=100.0)
            self.apply_axis_style(axis)
        self.draw_mutation_rug(axes[0], trace)
        axes[0].set_title(title_text, pad=float(self.config.title_pad_points))

    def draw_mutation_rug(self, axis: plt.Axes, trace: pd.DataFrame) -> None:
        """Draw mutation rug above the first hero panel."""
        mutation_mask = pd.to_numeric(trace["is_mutation"], errors="coerce").fillna(0).astype(np.int8).to_numpy(dtype=np.int8, copy=False)
        if not np.any(mutation_mask == 1):
            return
        mutation_sites = trace.loc[mutation_mask == 1, "site_index"].to_numpy(dtype=int, copy=False)
        ymin = float(self.config.mutation_rug_ymin_axes)
        ymax = float(self.config.mutation_rug_ymax_axes)
        if ymax <= ymin:
            raise ValueError("mutation_rug_ymax_axes must be greater than mutation_rug_ymin_axes")
        axis.vlines(mutation_sites, ymin, ymax, color="#111111", linewidth=0.6, alpha=0.75, transform=axis.get_xaxis_transform(), clip_on=False)

    def get_targeted_condition_order(self) -> List[str]:
        """Return targeted-only condition order for Figure 6 condition panels."""
        return ["below targeted", "above targeted"]

    def get_targeted_condition_palette(self) -> Dict[str, str]:
        """Return targeted-only condition colors."""
        return {condition: self.style.get_color(condition) for condition in self.get_targeted_condition_order()}

    def plot_local_information_flow_violins_combined(self, local_flow_by_protein: pd.DataFrame) -> None:
        """Plot Figure 6b protein-level local information-flow violins."""
        outcome_order = ["entropy", "plddt", "pae"]
        condition_order = self.get_targeted_condition_order()
        condition_palette = self.get_targeted_condition_palette()
        outcome_labels = {"entropy": "Local entropy [nats]", "plddt": "Local Chai-1 pLDDT (0-100)", "pae": "Local Chai-1 PAE"}
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 3.9), sharey=True)
        for axis, outcome_value in zip(axes, outcome_order):
            plot_frame = local_flow_by_protein[local_flow_by_protein["outcome"] == str(outcome_value)].copy()
            plot_frame = plot_frame[plot_frame["condition4"].isin(condition_order)].copy()
            plot_frame["protein_rho_spearman"] = pd.to_numeric(plot_frame["protein_rho_spearman"], errors="coerce")
            plot_frame = plot_frame[np.isfinite(plot_frame["protein_rho_spearman"].to_numpy(dtype=float, copy=False))].copy()
            if plot_frame.empty:
                axis.set_title(str(outcome_value))
                axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
                self.apply_axis_style(axis)
                continue
            sns.violinplot(ax=axis, data=plot_frame, x="condition4", y="protein_rho_spearman", hue="condition4", order=condition_order, palette=condition_palette, cut=0, inner=None, linewidth=0.8, legend=False)
            axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.85)
            axis.set_xlabel("")
            axis.set_ylabel("Local Spearman ρ" if axis is axes[0] else "")
            axis.set_title(outcome_labels.get(str(outcome_value), str(outcome_value)))
            axis.tick_params(axis="x", rotation=20)
            self.apply_axis_style(axis)
        fig.tight_layout(w_pad=0.6)
        save_figure_pair(fig, self.paths.output_dir, "figure6b_local_information_flow_violins_combined_1x3_vertical", dpi=300)

    def get_plddt_band_order(self) -> List[str]:
        """Return pLDDT band order."""
        low_edge, mid_edge, high_edge = [float(value) for value in self.config.bitsite_distribution_plddt_band_edges]
        return [f"<{int(low_edge)}", f"{int(low_edge)}–{int(mid_edge)}", f"{int(mid_edge)}–{int(high_edge)}", f"≥{int(high_edge)}"]

    def get_pae_band_order(self) -> List[str]:
        """Return PAE band order."""
        low_edge, mid_edge, high_edge = [float(value) for value in self.config.bitsite_distribution_pae_band_edges]
        return [f"≤{int(low_edge)}", f"{int(low_edge)}–{int(mid_edge)}", f"{int(mid_edge)}–{int(high_edge)}", f">{int(high_edge)}"]

    def get_plddt_band_palette(self) -> Dict[str, str]:
        """Return pLDDT band palette."""
        return {label: color for label, color in zip(self.get_plddt_band_order(), ["#F1E6FA", "#D2B7EC", "#A77AD8", "#5B2A86"])}

    def get_pae_band_palette(self) -> Dict[str, str]:
        """Return PAE band palette."""
        return {label: color for label, color in zip(self.get_pae_band_order(), ["#7F2F5C", "#C978A6", "#E7B6CF", "#F8E6EF"])}

    def save_bitsite_band_legend(self, band_order: Sequence[str], band_palette: Dict[str, str], output_stem: str) -> None:
        """Save standalone confidence-band legend."""
        handles: List[Any] = []
        labels: List[str] = []
        for band_label in band_order:
            handles.append(mpl.patches.Patch(facecolor=band_palette[str(band_label)], edgecolor=band_palette[str(band_label)], linewidth=0.8))
            labels.append(str(band_label))
        fig = plt.figure(figsize=(4.8, 1.2))
        legend = fig.legend(handles, labels, loc="center", frameon=False, ncol=len(labels), handlelength=1.0, handletextpad=0.6, borderaxespad=0.2, columnspacing=1.1)
        fig.savefig(self.paths.output_dir / f"{output_stem}.svg", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend])
        fig.savefig(self.paths.output_dir / f"{output_stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend])
        plt.close(fig)

    def build_bitsite_probability_pivot(self, summary: pd.DataFrame, band_column: str, band_order: Sequence[str]) -> pd.DataFrame:
        """Pivot one bits/site summary slice for plotting."""
        if summary.empty:
            return pd.DataFrame()
        pivot = summary.pivot_table(index=["bitsite_bin_index", "bitsite_bin_left", "bitsite_bin_right", "bitsite_bin_midpoint", "bitsite_bin_label"], columns=str(band_column), values="probability", aggfunc="first", fill_value=0.0).reset_index()
        for band_label in band_order:
            if band_label not in pivot.columns:
                pivot[band_label] = 0.0
        return pivot.sort_values(["bitsite_bin_index"], ascending=[True], kind="mergesort").reset_index(drop=True)

    def plot_bitsite_distribution_combined(self, summary: pd.DataFrame, band_column: str, band_order: Sequence[str], band_palette: Dict[str, str], metric_label: str, edit_subset: str, output_stem: str) -> None:
        """Plot one pooled bits/site band distribution."""
        overall = summary[(summary["grouping_type"] == "overall") & (summary["edit_subset"] == str(edit_subset))].copy()
        if overall.empty:
            return
        pivot = self.build_bitsite_probability_pivot(overall, band_column, band_order)
        if pivot.empty:
            return
        x_values = pivot["bitsite_bin_midpoint"].to_numpy(dtype=float, copy=False)
        x_left = float(np.nanmin(pd.to_numeric(pivot["bitsite_bin_left"], errors="coerce").to_numpy(dtype=float, copy=False)))
        x_right = float(np.nanmax(pd.to_numeric(pivot["bitsite_bin_right"], errors="coerce").to_numpy(dtype=float, copy=False)))
        fig, axis = plt.subplots(figsize=(11.2, 4.8))
        cumulative = np.zeros((pivot.shape[0],), dtype=float)
        for band_label in band_order:
            probability_values = pd.to_numeric(pivot[str(band_label)], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
            next_cumulative = cumulative + probability_values
            axis.fill_between(x_values, cumulative, next_cumulative, step="mid", color=band_palette[str(band_label)], alpha=0.92)
            cumulative = next_cumulative
        axis.set_xlim(x_left, x_right)
        axis.set_xlabel("Per-residue MSA bits/site")
        axis.set_ylabel("Probability")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"Whole-dataset local {metric_label} distribution by residue-level MSA bits/site\n{self.build_edit_subset_title(edit_subset)}", pad=float(self.config.title_pad_points))
        self.apply_axis_style(axis)
        fig.tight_layout()
        save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)

    def plot_bitsite_distribution_by_condition(self, summary: pd.DataFrame, band_column: str, band_order: Sequence[str], band_palette: Dict[str, str], metric_label: str, edit_subset: str, output_stem: str) -> None:
        """Plot one targeted-only condition bits/site band distribution."""
        condition_order = self.get_targeted_condition_order()
        condition_frame = summary[(summary["grouping_type"] == str(self.config.bitsite_distribution_grouping_mode)) & (summary["edit_subset"] == str(edit_subset))].copy()
        condition_frame = condition_frame[condition_frame["group_label"].isin(condition_order)].copy()
        if condition_frame.empty:
            return
        x_left_values = pd.to_numeric(condition_frame["bitsite_bin_left"], errors="coerce").to_numpy(dtype=float, copy=False)
        x_right_values = pd.to_numeric(condition_frame["bitsite_bin_right"], errors="coerce").to_numpy(dtype=float, copy=False)
        x_left_limit = float(np.nanmin(x_left_values)) if np.isfinite(x_left_values).any() else 0.0
        x_right_limit = float(np.nanmax(x_right_values)) if np.isfinite(x_right_values).any() else 1.0
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4), sharex=True, sharey=True)
        axes_flat = list(axes.flatten())
        for axis, condition_label in zip(axes_flat, condition_order):
            subset = condition_frame[condition_frame["group_label"] == str(condition_label)].copy()
            if subset.empty:
                axis.set_title(str(condition_label), pad=float(self.config.title_pad_points))
                axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
                axis.set_xlim(x_left_limit, x_right_limit)
                axis.set_ylim(0.0, 1.0)
                self.apply_axis_style(axis)
                continue
            pivot = self.build_bitsite_probability_pivot(subset, band_column, band_order)
            x_values = pivot["bitsite_bin_midpoint"].to_numpy(dtype=float, copy=False)
            cumulative = np.zeros((pivot.shape[0],), dtype=float)
            for band_label in band_order:
                probability_values = pd.to_numeric(pivot[str(band_label)], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
                next_cumulative = cumulative + probability_values
                axis.fill_between(x_values, cumulative, next_cumulative, step="mid", color=band_palette[str(band_label)], alpha=0.92)
                cumulative = next_cumulative
            axis.set_title(str(condition_label), pad=float(self.config.title_pad_points))
            axis.set_xlim(x_left_limit, x_right_limit)
            axis.set_ylim(0.0, 1.0)
            self.apply_axis_style(axis)
        for axis in axes_flat:
            axis.set_xlabel("Per-residue MSA bits/site")
        axes_flat[0].set_ylabel("Probability")
        fig.suptitle(f"Residue-level local {metric_label} distributions by MSA bits/site\n{self.build_edit_subset_title(edit_subset)}", y=0.98)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        save_figure_pair(fig, self.paths.output_dir, output_stem, dpi=300)

    def build_edit_subset_title(self, edit_subset: str) -> str:
        """Return display title for a residue subset."""
        title_map = {"all": "All residues", "edited_only": "Edited residues only", "unedited_only": "Unedited residues only"}
        return title_map.get(str(edit_subset), str(edit_subset))


@dataclass
class Figure6Runner:
    """Coordinate Figure 6 analysis, plotting, and export."""

    config: Figure6Config

    def run(self) -> None:
        """Run Figure 6 and Supplementary Figure 4 generator."""
        paths = Figure6Paths.from_config(self.config)
        paths.ensure_dirs()
        style = ConditionStyle()
        loader = Figure6DataLoader(self.config, paths)
        analyzer = Figure6Analyzer(self.config)
        plotter = Figure6Plotter(self.config, paths, style)
        plotter.apply_publication_style()
        start_time = time.time()
        if not paths.final_per_site_root.exists():
            raise FileNotFoundError(f"Missing compiled final per-site root: {paths.final_per_site_root}")
        cohort = loader.read_cohort_by_sample()
        targeted_candidates = analyzer.build_targeted_candidate_table(cohort)
        targeted_msa_summary = analyzer.compute_msa_summary_table(targeted_candidates)
        selection, pair_candidates, selected_samples = analyzer.select_heroes(targeted_candidates, targeted_msa_summary)
        analyzer.validate_expected_heroes(selection)
        per_site_hero = loader.read_per_site_for_protein(selection.protein_name, analyzer.required_per_site_columns())
        below_trace, above_trace = analyzer.build_trace_tables(per_site_hero, selection)
        below_spearman = analyzer.build_panel_spearman_table(below_trace)
        above_spearman = analyzer.build_panel_spearman_table(above_trace)
        panel_spearman = pd.concat([below_spearman, above_spearman], axis=0, ignore_index=True)
        delta_trace = analyzer.build_delta_trace_table(below_trace, above_trace)
        cohort_condition = analyzer.add_condition_columns(cohort)
        all_per_site = loader.read_per_site_for_proteins(cohort_condition["protein_name"].astype(str).unique().tolist(), analyzer.required_per_site_columns())
        local_flow = analyzer.compute_local_information_flow_by_sample(all_per_site, cohort_condition)
        local_flow_by_pair = analyzer.aggregate_local_information_flow_by_pair(local_flow)
        local_flow_by_protein = analyzer.aggregate_local_information_flow_by_protein(local_flow)
        bootstrap_local_flow = analyzer.bootstrap_local_information_flow_protein_deltas(local_flow_by_protein)
        bootstrap_local_flow_summary = analyzer.summarize_bootstrap_local_information_flow_protein(local_flow_by_protein, bootstrap_local_flow)
        local_flow_plot_source = local_flow_by_protein[local_flow_by_protein["condition4"].isin(plotter.get_targeted_condition_order())].copy()
        bitsite_base = analyzer.build_bitsite_distribution_residue_table(all_per_site, cohort_condition)
        plddt_distribution_summary = analyzer.build_bitsite_plddt_distribution_summary(bitsite_base)
        pae_distribution_summary = analyzer.build_bitsite_pae_distribution_summary(bitsite_base)
        plotter.save_bitsite_band_legend(plotter.get_plddt_band_order(), plotter.get_plddt_band_palette(), "figure6_legend_plddt_bands")
        plotter.save_bitsite_band_legend(plotter.get_pae_band_order(), plotter.get_pae_band_palette(), "figure6_legend_pae_bands")
        plotter.plot_hero_stacked(below_trace, f"Hero below operational threshold (~1 bit/site)\n{selection.below_sample_id}", "figure6a_left_hero_below_targeted_P07740_msa_p10_sweep_p10_2p35_rate_0p950000_cand3_seed42", "below targeted")
        plotter.plot_hero_stacked(above_trace, f"Hero at or above operational threshold (~1 bit/site)\n{selection.above_sample_id}", "figure6a_right_hero_above_targeted_P07740_msa_p10_sweep_p10_3p10_rate_0p750000_cand8_seed42", "above targeted")
        plotter.plot_local_information_flow_violins_combined(local_flow_by_protein)
        plotter.plot_bitsite_distribution_combined(plddt_distribution_summary, "plddt_band", plotter.get_plddt_band_order(), plotter.get_plddt_band_palette(), "pLDDT", "all", "figure6c_bitsite_plddt_distribution_all_samples_all")
        plotter.plot_bitsite_distribution_combined(pae_distribution_summary, "pae_band", plotter.get_pae_band_order(), plotter.get_pae_band_palette(), "PAE", "all", "figure6d_bitsite_pae_distribution_all_samples_all")
        plotter.plot_bitsite_distribution_by_condition(plddt_distribution_summary, "plddt_band", plotter.get_plddt_band_order(), plotter.get_plddt_band_palette(), "pLDDT", "edited_only", "suppfig4b_bitsite_plddt_distribution_condition4_edited_only")
        plotter.plot_bitsite_distribution_by_condition(pae_distribution_summary, "pae_band", plotter.get_pae_band_order(), plotter.get_pae_band_palette(), "PAE", "edited_only", "suppfig4c_bitsite_pae_distribution_condition4_edited_only")
        tables = {
            "readme": self.build_readme_table(selection, selected_samples, cohort_condition, all_per_site),
            "figure6a_hero_selected_samples": selected_samples,
            "figure6a_hero_selection_provenance": self.build_selection_provenance(selection),
            "figure6a_hero_trace_below": below_trace,
            "figure6a_hero_trace_above": above_trace,
            "figure6a_hero_panel_spearman": panel_spearman,
            "figure6a_hero_delta_trace": delta_trace,
            "figure6b_local_flow_by_sample": local_flow,
            "figure6b_local_flow_by_pair": local_flow_by_pair,
            "figure6b_local_flow_by_protein": local_flow_by_protein,
            "figure6b_local_flow_bootstrap": bootstrap_local_flow,
            "figure6b_local_flow_summary": bootstrap_local_flow_summary,
            "figure6b_local_flow_plot_source": local_flow_plot_source,
            "figure6c_suppfig4b_plddt_dist": plddt_distribution_summary,
            "figure6d_suppfig4c_pae_dist": pae_distribution_summary,
            "suppfig4b_plddt_plot_source": plddt_distribution_summary[(plddt_distribution_summary["grouping_type"] == str(self.config.bitsite_distribution_grouping_mode)) & (plddt_distribution_summary["edit_subset"] == "edited_only") & (plddt_distribution_summary["group_label"].isin(plotter.get_targeted_condition_order()))].copy(),
            "suppfig4c_pae_plot_source": pae_distribution_summary[(pae_distribution_summary["grouping_type"] == str(self.config.bitsite_distribution_grouping_mode)) & (pae_distribution_summary["edit_subset"] == "edited_only") & (pae_distribution_summary["group_label"].isin(plotter.get_targeted_condition_order()))].copy(),
            "hero_msa_summary": targeted_msa_summary,
            "hero_msa_pair_candidates": pair_candidates,
            "hero_trace_dictionary": analyzer.build_trace_dictionary(),
            "local_flow_dictionary": analyzer.build_local_flow_dictionary(),
            "bitsite_plddt_dictionary": analyzer.build_bitsite_dictionary("plddt"),
            "bitsite_pae_dictionary": analyzer.build_bitsite_dictionary("pae"),
        }
        writer = ExcelNotebookWriter(paths.output_xlsx_path, chunk_rows=int(self.config.excel_chunk_rows), include_inventory=False)
        writer.write(tables)
        elapsed = float(time.time() - start_time)
        print(f"Wrote Figure 6 outputs to {paths.output_dir} in {elapsed:.1f}s")

    def build_selection_provenance(self, selection: HeroSelectionResult) -> pd.DataFrame:
        """Build hero-selection provenance table."""
        return pd.DataFrame(
            [
                {
                    "protein_name": selection.protein_name,
                    "anchor_sequence": selection.anchor_sequence,
                    "below_outer_label": selection.below_outer_label,
                    "above_outer_label": selection.above_outer_label,
                    "below_sample_id": selection.below_sample_id,
                    "above_sample_id": selection.above_sample_id,
                    "selection_metric_name": selection.selection_metric_name,
                    "selection_metric_diff": float(selection.selection_metric_diff),
                    "bits_site_min_threshold": float(self.config.bits_site_min_threshold),
                    "min_anchor_length": int(self.config.min_anchor_length),
                    "min_sample_plddt_mean": float(self.config.min_sample_plddt_mean),
                }
            ]
        )

    def build_readme_table(self, selection: HeroSelectionResult, selected_samples: pd.DataFrame, cohort_condition: pd.DataFrame, all_per_site: pd.DataFrame) -> pd.DataFrame:
        """Build source-data workbook readme."""
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "cohort_input", "value": str(self.config.cohort_xlsx_name)},
            {"key": "cohort_sheet_name", "value": str(self.config.cohort_sheet_name)},
            {"key": "per_site_input", "value": f"{self.config.compiled_final_dir}/{self.config.final_per_site_subdir}/protein_name=*/per_site.parquet"},
            {"key": "output_dir", "value": str(self.config.output_dir_name)},
            {"key": "threshold_bits_site_min", "value": str(float(self.config.bits_site_min_threshold))},
            {"key": "hero_selection_story", "value": "Targeted-only below/at-or-above operational-threshold (~1 bit/site) hero pair selected dynamically by the manuscript methods."},
            {"key": "msa_pair_selection_metric", "value": "Largest absolute difference in unweighted mean bits/site across below/above targeted MSAs within one anchor/reference protein."},
            {"key": "hero_sample_selection_rule", "value": "Highest novelty, defined as 100 - max_protein_id, within each selected targeted MSA."},
            {"key": "selected_below_sample_id", "value": selection.below_sample_id},
            {"key": "selected_above_sample_id", "value": selection.above_sample_id},
            {"key": "selected_pair_metric_diff", "value": str(float(selection.selection_metric_diff))},
            {"key": "smoothing_window", "value": str(int(self.config.smoothing_window))},
            {"key": "n_cohort_rows", "value": str(int(cohort_condition.shape[0]))},
            {"key": "n_per_site_rows", "value": str(int(all_per_site.shape[0]))},
            {"key": "local_information_flow_primary_analysis_unit", "value": "protein_name × condition4 × outcome"},
            {"key": "local_information_flow_protein_aggregator", "value": str(self.config.local_flow_protein_aggregator)},
            {"key": "local_information_flow_plot_groups", "value": "below targeted; above targeted"},
            {"key": "local_information_flow_bootstrap_replicates", "value": str(int(self.config.bootstrap_replicates))},
            {"key": "local_information_flow_bootstrap_seed", "value": str(int(self.config.bootstrap_seed))},
            {"key": "bitsite_distribution_bin_width", "value": str(float(self.config.bitsite_distribution_bin_width))},
            {"key": "bitsite_distribution_weighting", "value": str(self.config.bitsite_distribution_weighting_mode)},
            {"key": "bitsite_distribution_min_bin_count", "value": str(int(self.config.bitsite_distribution_min_bin_count))},
            {"key": "figure_outputs", "value": "Figure 6a-d and Supplementary Figure 4b-c only; Supplementary Figure 4a mediation panels are generated by figure6_mediation_plot.py."},
            {"key": "selected_rows_json", "value": selected_samples.to_json(orient="records")},
        ]
        return pd.DataFrame(rows)


def main() -> None:
    """Generate Figure 6 and Supplementary Figure 4 outputs."""
    runner = Figure6Runner(Figure6Config())
    runner.run()


if __name__ == "__main__":
    main()
