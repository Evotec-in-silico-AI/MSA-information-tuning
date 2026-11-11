
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import MMCIFParser
from prody import calcShannonEntropy


@dataclass(frozen=True)
class Figure7MetaConfig:
    """Configure Figure 7 local structural meta-analysis."""

    results_dir_name: str = "results"
    experimental_dir_name: str = "experimental_structures"
    jackhmmer_msa_dir_name: str = "msas"
    jackhmmer_models_dir_name: str = "chai1_models_jackhmmer"
    tuned_folder_prefix: str = "below_folding_budget_"
    tuned_msa_subdir_name: str = "msas"
    tuned_models_subdir_name: str = "chai1_models"
    output_dir_name: str = "figure7_meta_outputs"
    output_xlsx_name: str = "figure7_meta_notebook.xlsx"
    logs_dir_name: str = "figure7_meta_logs"
    text_log_name: str = "figure7_meta.log"
    jsonl_log_name: str = "figure7_meta.jsonl"
    excel_chunk_rows: int = 1_000_000
    target_fractions: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30)
    target_range_definitions: Tuple[Tuple[str, str, float, float], ...] = (("target_0_10", "0–10% targets", 0.0, 10.0), ("target_15_20", "15–20% targets", 15.0, 20.0), ("target_25_30", "25–30% targets", 25.0, 30.0))
    restrict_to_csv_valid_common_set: bool = True
    common_set_required_metric_columns: Tuple[str, ...] = ("mean_tm_score", "mean_rmsd", "mean_ptm")
    require_positive_common_set_tm_score: bool = True
    enforce_complete_local_target_set: bool = True
    bitsite_threshold: float = 2.0
    bitsite_bin_width: float = 0.05
    heatmap_bitsite_bin_width: float = 0.05
    sequence_window_radius: int = 5
    min_residues_per_pair: int = 25
    min_pdbs_per_bitsite_bin: int = 5
    min_residues_per_bitsite_bin: int = 100
    min_pdbs_per_heatmap_cell: int = 3
    min_residues_per_heatmap_cell: int = 100
    min_residues_per_transition_group: int = 25
    heatmap_delta_clip_angstrom: float = 5.0
    parallel_enabled: bool = True
    parallel_max_workers: int = 32
    parallel_chunksize: int = 1
    improvement_threshold_angstrom: float = 1.0
    absolute_error_band_edges: Tuple[float, float, float, float] = (1.0, 2.0, 4.0, 8.0)
    delta_error_band_edges: Tuple[float, float, float, float] = (-5.0, -1.0, 1.0, 5.0)
    gate_thresholds: Tuple[Tuple[str, Optional[float]], ...] = (
        ("all_pairs", None),
        ("delta_ptm_gt_0", 0.0),
        ("delta_ptm_gt_0p03", 0.03),
    )
    absolute_error_band_order: Tuple[str, ...] = ("≤1 Å", "1–2 Å", "2–4 Å", "4–8 Å", ">8 Å")
    absolute_error_band_colors: Tuple[str, ...] = ("#00441B", "#238B45", "#74C476", "#C7E9C0", "#F0F9E8")
    delta_error_band_order: Tuple[str, ...] = (">5 Å improvement", "1–5 Å improvement", "−1–1 Å little change", "−5–−1 Å worsening", "<−5 Å worsening")
    delta_error_band_colors: Tuple[str, ...] = ("#00441B", "#238B45", "#74C476", "#C7E9C0", "#F0F9E8")
    transition_class_order: Tuple[str, ...] = ("above_to_above", "above_to_below", "below_to_below", "below_to_above")
    transition_class_labels: Tuple[str, ...] = ("above→above", "above→below", "below→below", "below→above")
    heatmap_cmap_colors: Tuple[str, ...] = ("#F0F9E8", "#C7E9C0", "#74C476", "#238B45", "#00441B")
    figure_width: float = 9.8
    figure_height: float = 4.8
    heatmap_width: float = 6.4
    heatmap_height: float = 5.8
    title_pad_points: float = 12.0
    progress_log_every_pdb: int = 1
    flush_logs_on_progress: bool = True
    plot_file_formats: Tuple[str, ...] = ("svg", "png")
    export_raw_residue_table: bool = False


@dataclass(frozen=True)
class Figure7MetaPaths:
    """Resolve input and output paths."""

    root_dir: Path
    results_dir: Path
    experimental_dir: Path
    jackhmmer_msa_dir: Path
    jackhmmer_models_dir: Path
    output_dir: Path
    output_xlsx_path: Path
    logs_dir: Path
    text_log_path: Path
    jsonl_log_path: Path


@dataclass(frozen=True)
class SelectedModel:
    """Store one selected model."""

    model_index: int
    model_path: Path
    score_path: Path
    ptm: float


class JsonlLogger:
    """Write JSONL records with flush."""

    def __init__(self, path: Path) -> None:
        """Initialize logger."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def write(self, event: str, payload: Dict[str, Any]) -> None:
        """Write one JSONL record."""
        record = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": str(event)}
        record.update(payload)
        self.handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self.handle.flush()

    def close(self) -> None:
        """Close logger."""
        self.handle.close()


class ExcelNotebookWriter:
    """Write named tables to XLSX with chunking."""

    def __init__(self, path: Path, chunk_rows: int) -> None:
        """Initialize writer."""
        self.path = Path(path)
        self.chunk_rows = int(chunk_rows)

    def write(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Write workbook."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(self.path, engine="openpyxl") as writer:
            used_names: Dict[str, int] = {}
            for table_name, table_df in tables.items():
                df = table_df.copy() if table_df is not None else pd.DataFrame()
                if df.shape[0] <= self.chunk_rows:
                    df.to_excel(writer, sheet_name=self.build_sheet_name(str(table_name), used_names), index=False)
                else:
                    for chunk_index, start_index in enumerate(range(0, df.shape[0], self.chunk_rows), start=1):
                        chunk = df.iloc[start_index:start_index + self.chunk_rows].copy()
                        chunk.to_excel(writer, sheet_name=self.build_sheet_name(f"{table_name}_{chunk_index}", used_names), index=False)

    def build_sheet_name(self, requested_name: str, used_names: Dict[str, int]) -> str:
        """Return an Excel-safe sheet name."""
        cleaned = requested_name.replace("/", "_").replace("\\", "_").replace("?", "_").replace("*", "_")
        cleaned = cleaned.replace("[", "_").replace("]", "_").replace(":", "_")
        base = cleaned[:31] if cleaned else "sheet"
        if base not in used_names:
            used_names[base] = 1
            return base
        used_names[base] += 1
        suffix = f"_{used_names[base]}"
        return f"{base[:31 - len(suffix)]}{suffix}"


class Figure7MetaDataLoader:
    """Load Figure 7 files."""

    def __init__(self, config: Figure7MetaConfig, paths: Figure7MetaPaths, logger: logging.Logger) -> None:
        """Initialize loader."""
        self.config = config
        self.paths = paths
        self.logger = logger
        self.parser = MMCIFParser(QUIET=True)
        self.experimental_coord_cache: Dict[str, np.ndarray] = {}
        self.jackhmmer_coord_cache: Dict[str, np.ndarray] = {}
        self.jackhmmer_bits_cache: Dict[str, np.ndarray] = {}
        self.jackhmmer_model_cache: Dict[str, SelectedModel] = {}

    def target_token(self, target_fraction: float) -> str:
        """Format target fraction token."""
        target_string = f"{float(target_fraction):.3f}".rstrip("0").rstrip(".")
        return target_string.replace(".", "p")

    def target_folder(self, target_fraction: float) -> Path:
        """Return tuned target folder."""
        return self.paths.root_dir / f"{self.config.tuned_folder_prefix}{self.target_token(target_fraction)}"

    def tuned_msa_dir(self, target_fraction: float) -> Path:
        """Return tuned MSA directory."""
        return self.target_folder(target_fraction) / str(self.config.tuned_msa_subdir_name)

    def tuned_models_dir(self, target_fraction: float) -> Path:
        """Return tuned model directory."""
        return self.target_folder(target_fraction) / str(self.config.tuned_models_subdir_name)

    def comparison_csv_path(self, mode: str, target_fraction: Optional[float] = None) -> Path:
        """Return comparison CSV path."""
        if mode == "jackhmmer":
            return self.paths.results_dir / "chai1_structural_comparison_jackhmmer.csv"
        if target_fraction is None:
            raise ValueError("target_fraction is required for tuned CSVs")
        return self.paths.results_dir / f"chai1_structural_comparison_below_folding_budget_{self.target_token(float(target_fraction))}.csv"

    def read_comparison_csv(self, path: Path) -> pd.DataFrame:
        """Read comparison CSV."""
        if not path.exists():
            raise FileNotFoundError(f"Missing comparison CSV: {path}")
        df = pd.read_csv(path)
        if "pdb_id" not in df.columns:
            raise ValueError(f"Comparison CSV lacks pdb_id: {path}")
        out = df.copy()
        out["pdb_id"] = out["pdb_id"].astype(str)
        for column_name in ["mean_ptm", "mean_tm_score", "mean_rmsd"]:
            if column_name in out.columns:
                out[column_name] = pd.to_numeric(out[column_name], errors="coerce")
        return out

    def find_first_a3m(self, msa_dir: Path, pdb_id: str) -> Optional[Path]:
        """Return first matching A3M."""
        if not msa_dir.is_dir():
            return None
        matches = sorted(msa_dir.glob(f"{pdb_id}*.a3m"))
        return matches[0] if matches else None

    def load_a3m_sequences(self, path: Path) -> List[str]:
        """Load A3M sequences."""
        sequences: List[str] = []
        current_lines: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                if line.startswith(">"):
                    if current_lines:
                        sequences.append("".join(current_lines))
                    current_lines = []
                else:
                    current_lines.append(line)
        if current_lines:
            sequences.append("".join(current_lines))
        if not sequences:
            raise ValueError(f"No sequences found in A3M: {path}")
        return sequences

    def sanitize_a3m_sequences(self, sequences: Sequence[str]) -> List[str]:
        """Sanitize A3M rows."""
        cleaned: List[str] = []
        for sequence in sequences:
            sanitized = sequence.replace(".", "")
            sanitized = "".join(char for char in sanitized if (("A" <= char <= "Z") or char == "-"))
            cleaned.append(sanitized)
        lengths = {len(sequence) for sequence in cleaned}
        if len(lengths) != 1:
            raise ValueError(f"Sanitized A3M rows have unequal lengths: {lengths}")
        return cleaned

    def project_to_anchor_match_columns(self, sequences: Sequence[str]) -> List[str]:
        """Project rows to first-row non-gap columns."""
        if not sequences:
            return []
        keep_indices = [index for index, char in enumerate(str(sequences[0])) if char != "-"]
        return ["".join(str(sequence)[index] for index in keep_indices) for sequence in sequences]

    def compute_bitsite_vector_from_a3m(self, path: Path) -> np.ndarray:
        """Compute ProDy bits/site vector from A3M."""
        projected = self.project_to_anchor_match_columns(self.sanitize_a3m_sequences(self.load_a3m_sequences(path)))
        if not projected:
            raise ValueError(f"No projected A3M sequences: {path}")
        char_array = np.empty((len(projected), len(projected[0])), dtype="S1")
        for row_index, sequence in enumerate(projected):
            char_array[row_index, :] = np.frombuffer(sequence.encode("ascii", "replace"), dtype="S1")
        entropy_bits = np.asarray(calcShannonEntropy(char_array), dtype=np.float64)
        return np.log2(20.0) - entropy_bits

    def get_jackhmmer_bitsite_vector(self, pdb_id: str) -> np.ndarray:
        """Return cached JackHMMER bits/site vector."""
        if pdb_id not in self.jackhmmer_bits_cache:
            a3m_path = self.find_first_a3m(self.paths.jackhmmer_msa_dir, pdb_id)
            if a3m_path is None:
                raise FileNotFoundError(f"Missing JackHMMER A3M for {pdb_id}")
            self.jackhmmer_bits_cache[pdb_id] = self.compute_bitsite_vector_from_a3m(a3m_path)
        return self.jackhmmer_bits_cache[pdb_id]

    def get_tuned_bitsite_vector(self, pdb_id: str, target_fraction: float) -> np.ndarray:
        """Return tuned bits/site vector."""
        a3m_path = self.find_first_a3m(self.tuned_msa_dir(target_fraction), pdb_id)
        if a3m_path is None:
            raise FileNotFoundError(f"Missing tuned A3M for {pdb_id} target={target_fraction}")
        return self.compute_bitsite_vector_from_a3m(a3m_path)

    def load_ptm(self, path: Path) -> float:
        """Load pTM from NPZ."""
        data = np.load(path)
        if "ptm" not in data.files:
            raise ValueError(f"Missing ptm array in {path}")
        values = np.asarray(data["ptm"]).flatten()
        if values.size == 0:
            raise ValueError(f"Empty ptm array in {path}")
        return float(values[0])

    def select_highest_ptm_model(self, models_dir: Path) -> SelectedModel:
        """Select highest-pTM model."""
        if not models_dir.is_dir():
            raise FileNotFoundError(f"Missing models directory: {models_dir}")
        candidates: List[SelectedModel] = []
        for model_index in range(5):
            model_path = models_dir / f"pred.model_idx_{model_index}.cif"
            score_path = models_dir / f"scores.model_idx_{model_index}.npz"
            if model_path.exists() and score_path.exists():
                candidates.append(SelectedModel(model_index, model_path, score_path, self.load_ptm(score_path)))
        if not candidates:
            raise FileNotFoundError(f"No complete model candidates in {models_dir}")
        candidates.sort(key=lambda candidate: (-float(candidate.ptm), int(candidate.model_index)))
        return candidates[0]

    def get_jackhmmer_selected_model(self, pdb_id: str) -> SelectedModel:
        """Return cached JackHMMER selected model."""
        if pdb_id not in self.jackhmmer_model_cache:
            self.jackhmmer_model_cache[pdb_id] = self.select_highest_ptm_model(self.paths.jackhmmer_models_dir / str(pdb_id))
        return self.jackhmmer_model_cache[pdb_id]

    def get_tuned_selected_model(self, pdb_id: str, target_fraction: float) -> SelectedModel:
        """Return tuned selected model."""
        return self.select_highest_ptm_model(self.tuned_models_dir(target_fraction) / str(pdb_id))

    def load_ca_coordinates(self, cif_path: Path, chain_rule: str) -> Tuple[np.ndarray, str, int]:
        """Load Cα coordinates from CIF."""
        if not cif_path.exists():
            raise FileNotFoundError(f"Missing CIF: {cif_path}")
        structure = self.parser.get_structure(cif_path.stem, str(cif_path))
        chains = list(structure.get_chains())
        if not chains:
            raise ValueError(f"No chains found in CIF: {cif_path}")
        selected_chain = None
        if chain_rule == "experimental":
            for chain in chains:
                if str(chain.id) == "A" and self.chain_has_ca(chain):
                    selected_chain = chain
                    break
        if selected_chain is None:
            for chain in chains:
                if self.chain_has_ca(chain):
                    selected_chain = chain
                    break
        if selected_chain is None:
            raise ValueError(f"No Cα-containing chain found in CIF: {cif_path}")
        coords: List[np.ndarray] = []
        for residue in selected_chain.get_residues():
            if "CA" in residue:
                coords.append(np.asarray(residue["CA"].get_coord(), dtype=np.float64))
        if not coords:
            raise ValueError(f"Selected chain has no Cα atoms: {cif_path}")
        return np.vstack(coords), str(selected_chain.id), len(chains)

    def chain_has_ca(self, chain: Any) -> bool:
        """Return whether chain has Cα atoms."""
        for residue in chain.get_residues():
            if "CA" in residue:
                return True
        return False

    def get_experimental_coordinates(self, pdb_id: str) -> np.ndarray:
        """Return cached experimental coordinates."""
        if pdb_id not in self.experimental_coord_cache:
            coords, chain_id, chain_count = self.load_ca_coordinates(self.paths.experimental_dir / f"{pdb_id}.cif", "experimental")
            if chain_id != "A":
                self.logger.info("Experimental chain fallback for %s: selected chain=%s among %s chains", pdb_id, chain_id, chain_count)
            self.experimental_coord_cache[pdb_id] = coords
        return self.experimental_coord_cache[pdb_id]

    def get_jackhmmer_coordinates(self, pdb_id: str) -> np.ndarray:
        """Return cached JackHMMER coordinates."""
        if pdb_id not in self.jackhmmer_coord_cache:
            model = self.get_jackhmmer_selected_model(pdb_id)
            coords, _, _ = self.load_ca_coordinates(model.model_path, "predicted")
            self.jackhmmer_coord_cache[pdb_id] = coords
        return self.jackhmmer_coord_cache[pdb_id]


class Figure7MetaAnalyzer:
    """Analyze local structural errors."""

    def __init__(self, config: Figure7MetaConfig, loader: Figure7MetaDataLoader, logger: logging.Logger, jsonl_logger: JsonlLogger) -> None:
        """Initialize analyzer."""
        self.config = config
        self.loader = loader
        self.logger = logger
        self.jsonl_logger = jsonl_logger

    def build_common_target_inventory(self, comparison_tables: Dict[float, pd.DataFrame], jackhmmer_df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """Build common PDB IDs across valid CSV rows."""
        jackhmmer_valid_ids, jackhmmer_record = self.get_valid_comparison_pdb_ids(jackhmmer_df, "jackhmmer", float("nan"))
        common_ids = set(jackhmmer_valid_ids)
        records: List[Dict[str, Any]] = [jackhmmer_record]
        for target_fraction, table in comparison_tables.items():
            target_ids, record = self.get_valid_comparison_pdb_ids(table, "tuned", float(target_fraction))
            common_ids &= set(target_ids)
            records.append(record)
        common_sorted = sorted(common_ids)
        for record in records:
            record["n_common_pdb_ids_after_intersection"] = int(len(common_sorted))
            record["common_set_restricted_to_valid_metrics"] = bool(self.config.restrict_to_csv_valid_common_set)
        return common_sorted, pd.DataFrame(records)

    def get_valid_comparison_pdb_ids(self, comparison_df: pd.DataFrame, comparison_type: str, target_fraction: float) -> Tuple[List[str], Dict[str, Any]]:
        """Return valid PDB IDs from comparison CSV."""
        working = comparison_df.copy()
        working["pdb_id"] = working["pdb_id"].astype(str)
        all_ids = set(working["pdb_id"].tolist())
        if bool(self.config.restrict_to_csv_valid_common_set):
            valid_mask = np.ones((working.shape[0],), dtype=bool)
            for column_name in self.config.common_set_required_metric_columns:
                if column_name not in working.columns:
                    raise ValueError(f"Comparison table lacks required common-set metric column: {column_name}")
                values = pd.to_numeric(working[column_name], errors="coerce").to_numpy(dtype=float, copy=False)
                valid_mask &= np.isfinite(values)
            if bool(self.config.require_positive_common_set_tm_score):
                tm_values = pd.to_numeric(working["mean_tm_score"], errors="coerce").to_numpy(dtype=float, copy=False)
                valid_mask &= tm_values > 0.0
            valid_ids = set(working.loc[valid_mask, "pdb_id"].tolist())
        else:
            valid_ids = set(all_ids)
        record = {
            "comparison_type": str(comparison_type),
            "target_fraction": float(target_fraction) if np.isfinite(target_fraction) else float("nan"),
            "target_percent": float(100.0 * target_fraction) if np.isfinite(target_fraction) else float("nan"),
            "n_rows": int(working.shape[0]),
            "n_unique_pdb_ids": int(len(all_ids)),
            "n_valid_unique_pdb_ids": int(len(valid_ids)),
        }
        return sorted(valid_ids), record

    def kabsch_align(self, moving: np.ndarray, fixed: np.ndarray) -> np.ndarray:
        """Align moving coordinates onto fixed coordinates."""
        if moving.shape != fixed.shape:
            raise ValueError(f"Kabsch coordinate shape mismatch: moving={moving.shape}, fixed={fixed.shape}")

        moving_centroid = np.mean(moving, axis=0)
        fixed_centroid = np.mean(fixed, axis=0)
        moving_centered = moving - moving_centroid
        fixed_centered = fixed - fixed_centroid
        covariance = moving_centered.T @ fixed_centered
        left, _, right_t = np.linalg.svd(covariance)

        correction = np.eye(3)
        determinant = np.linalg.det(left @ right_t)
        correction[2, 2] = -1.0 if determinant < 0.0 else 1.0

        rotation = left @ correction @ right_t

        return moving_centered @ rotation + fixed_centroid

    def compute_ca_error(self, predicted: np.ndarray, experimental: np.ndarray) -> np.ndarray:
        """Compute per-residue Cα error."""
        return np.linalg.norm(self.kabsch_align(predicted, experimental) - experimental, axis=1)

    def compute_aligned_ca_error(self, predicted: np.ndarray, experimental: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute aligned Cα coordinates and per-residue error."""
        aligned = self.kabsch_align(predicted, experimental)
        error = np.linalg.norm(aligned - experimental, axis=1)
        return aligned, error

    def validate_sequence_window_config(self) -> None:
        """Validate sequence-window settings."""
        radius_value = int(self.config.sequence_window_radius)
        if radius_value < 0:
            raise ValueError(f"sequence_window_radius must be non-negative: {radius_value}")

    def compute_centered_sequence_window_mean(self, values: np.ndarray, window_radius: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute centered sequence-window means."""
        radius_value = int(window_radius)
        if radius_value < 0:
            raise ValueError(f"window_radius must be non-negative: {radius_value}")
        values_array = np.asarray(values, dtype=float)
        if values_array.ndim != 1:
            raise ValueError(f"Sequence-window values must be one-dimensional: shape={values_array.shape}")
        residue_count = int(values_array.size)
        means = np.empty((residue_count,), dtype=float)
        counts = np.empty((residue_count,), dtype=int)
        for site_index in range(residue_count):
            start_index = max(0, site_index - radius_value)
            end_index = min(residue_count, site_index + radius_value + 1)
            window_values = values_array[start_index:end_index]
            means[site_index] = float(np.mean(window_values))
            counts[site_index] = int(window_values.size)
        return means, counts

    def assign_transition_class(self, jackhmmer_bits: np.ndarray, tuned_bits: np.ndarray) -> np.ndarray:
        """Assign information transition classes."""
        threshold = float(self.config.bitsite_threshold)
        jack_below = jackhmmer_bits < threshold
        tuned_below = tuned_bits < threshold
        out = np.full(jackhmmer_bits.shape, "", dtype=object)
        out[(~jack_below) & (~tuned_below)] = "above_to_above"
        out[(~jack_below) & tuned_below] = "above_to_below"
        out[jack_below & tuned_below] = "below_to_below"
        out[jack_below & (~tuned_below)] = "below_to_above"
        return out

    def assign_absolute_error_bands(self, values: np.ndarray) -> np.ndarray:
        """Assign absolute error bands."""
        edge_1, edge_2, edge_3, edge_4 = [float(value) for value in self.config.absolute_error_band_edges]
        bands = np.full(values.shape, "", dtype=object)
        bands[values <= edge_1] = self.config.absolute_error_band_order[0]
        bands[(values > edge_1) & (values <= edge_2)] = self.config.absolute_error_band_order[1]
        bands[(values > edge_2) & (values <= edge_3)] = self.config.absolute_error_band_order[2]
        bands[(values > edge_3) & (values <= edge_4)] = self.config.absolute_error_band_order[3]
        bands[values > edge_4] = self.config.absolute_error_band_order[4]
        return bands

    def assign_delta_error_bands(self, values: np.ndarray) -> np.ndarray:
        """Assign delta error bands."""
        low_edge, mild_low_edge, mild_high_edge, high_edge = [float(value) for value in self.config.delta_error_band_edges]
        bands = np.full(values.shape, "", dtype=object)
        bands[values > high_edge] = self.config.delta_error_band_order[0]
        bands[(values > mild_high_edge) & (values <= high_edge)] = self.config.delta_error_band_order[1]
        bands[(values >= mild_low_edge) & (values <= mild_high_edge)] = self.config.delta_error_band_order[2]
        bands[(values >= low_edge) & (values < mild_low_edge)] = self.config.delta_error_band_order[3]
        bands[values < low_edge] = self.config.delta_error_band_order[4]
        return bands

    def process_all_pairs(self, common_pdb_ids: Sequence[str], jackhmmer_df: pd.DataFrame, tuned_tables: Dict[float, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process all common PDB-target pairs."""
        self.validate_sequence_window_config()
        if bool(self.config.parallel_enabled) and int(self.config.parallel_max_workers) > 1:
            return self.process_all_pairs_parallel(common_pdb_ids, jackhmmer_df, tuned_tables)
        return self.process_all_pairs_serial(common_pdb_ids, jackhmmer_df, tuned_tables)

    def process_all_pairs_serial(self, common_pdb_ids: Sequence[str], jackhmmer_df: pd.DataFrame, tuned_tables: Dict[float, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process common PDB-target pairs serially."""
        residue_frames: List[pd.DataFrame] = []
        pair_records: List[Dict[str, Any]] = []
        exclusion_records: List[Dict[str, Any]] = []
        jackhmmer_lookup = jackhmmer_df.drop_duplicates(subset=["pdb_id"]).set_index("pdb_id")
        tuned_lookups = {target: table.drop_duplicates(subset=["pdb_id"]).set_index("pdb_id") for target, table in tuned_tables.items()}
        pdb_total = len(common_pdb_ids)
        for pdb_index, pdb_id in enumerate(common_pdb_ids, start=1):
            retained_pairs = 0
            excluded_pairs = 0
            for target_fraction in self.config.target_fractions:
                try:
                    residue_df, pair_record = self.process_one_pair(str(pdb_id), float(target_fraction), jackhmmer_lookup, tuned_lookups[float(target_fraction)])
                    residue_frames.append(residue_df)
                    pair_records.append(pair_record)
                    retained_pairs += 1
                except Exception as exc:
                    exclusion_records.append({"pdb_id": str(pdb_id), "target_fraction": float(target_fraction), "target_percent": float(100.0 * target_fraction), "reason": type(exc).__name__, "message": str(exc)})
                    excluded_pairs += 1
            self.log_pdb_progress(pdb_index, pdb_total, str(pdb_id), retained_pairs, excluded_pairs)
        residue_table = pd.concat(residue_frames, axis=0, ignore_index=True) if residue_frames else pd.DataFrame()
        return residue_table, pd.DataFrame(pair_records), pd.DataFrame(exclusion_records)

    def process_all_pairs_parallel(self, common_pdb_ids: Sequence[str], jackhmmer_df: pd.DataFrame, tuned_tables: Dict[float, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process common PDB-target pairs with process-level parallelism."""
        residue_frames: List[pd.DataFrame] = []
        pair_records: List[Dict[str, Any]] = []
        exclusion_records: List[Dict[str, Any]] = []
        pdb_total = len(common_pdb_ids)
        max_workers = max(1, int(self.config.parallel_max_workers))
        self.logger.info("Processing %s PDB IDs with %s parallel workers", pdb_total, max_workers)
        self.flush_logger_handlers()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=initialize_figure7_worker, initargs=(self.config, self.loader.paths, jackhmmer_df, tuned_tables)) as executor:
            future_to_pdb = {executor.submit(process_figure7_pdb_worker, str(pdb_id)): str(pdb_id) for pdb_id in common_pdb_ids}
            for completed_index, future in enumerate(concurrent.futures.as_completed(future_to_pdb), start=1):
                pdb_id = future_to_pdb[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"pdb_id": str(pdb_id), "residue_df": pd.DataFrame(), "pair_records": [], "exclusion_records": [{"pdb_id": str(pdb_id), "target_fraction": float("nan"), "target_percent": float("nan"), "reason": type(exc).__name__, "message": str(exc)}], "retained_pairs": 0, "excluded_pairs": len(self.config.target_fractions)}
                if not result["residue_df"].empty:
                    residue_frames.append(result["residue_df"])
                pair_records.extend(result["pair_records"])
                exclusion_records.extend(result["exclusion_records"])
                self.log_pdb_progress(completed_index, pdb_total, str(result["pdb_id"]), int(result["retained_pairs"]), int(result["excluded_pairs"]))
        residue_table = pd.concat(residue_frames, axis=0, ignore_index=True) if residue_frames else pd.DataFrame()
        return residue_table, pd.DataFrame(pair_records), pd.DataFrame(exclusion_records)

    def log_pdb_progress(self, pdb_index: int, pdb_total: int, pdb_id: str, retained_pairs: int, excluded_pairs: int) -> None:
        """Log one PDB progress update."""
        if (int(pdb_index) % int(self.config.progress_log_every_pdb)) == 0 or int(pdb_index) == int(pdb_total):
            self.logger.info("%s out of %s PDBs processed: %s; retained_pairs=%s; excluded_pairs=%s", pdb_index, pdb_total, pdb_id, retained_pairs, excluded_pairs)
            self.flush_logger_handlers()
            self.jsonl_logger.write("pdb_done", {"pdb_index": int(pdb_index), "pdb_total": int(pdb_total), "pdb_id": str(pdb_id), "retained_pairs": int(retained_pairs), "excluded_pairs": int(excluded_pairs)})

    def process_one_pair(self, pdb_id: str, target_fraction: float, jackhmmer_lookup: pd.DataFrame, tuned_lookup: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process one PDB-target pair."""
        experimental_coords = self.loader.get_experimental_coordinates(pdb_id)
        jackhmmer_coords = self.loader.get_jackhmmer_coordinates(pdb_id)
        tuned_model = self.loader.get_tuned_selected_model(pdb_id, target_fraction)
        tuned_coords, _, _ = self.loader.load_ca_coordinates(tuned_model.model_path, "predicted")
        jackhmmer_bits = self.loader.get_jackhmmer_bitsite_vector(pdb_id)
        tuned_bits = self.loader.get_tuned_bitsite_vector(pdb_id, target_fraction)
        lengths = {"experimental_ca": int(experimental_coords.shape[0]), "jackhmmer_ca": int(jackhmmer_coords.shape[0]), "tuned_ca": int(tuned_coords.shape[0]), "jackhmmer_bits": int(jackhmmer_bits.size), "tuned_bits": int(tuned_bits.size)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Length mismatch for {pdb_id} target={target_fraction}: {lengths}")
        residue_count = int(experimental_coords.shape[0])
        if residue_count < int(self.config.min_residues_per_pair):
            raise ValueError(f"Insufficient residues for {pdb_id} target={target_fraction}: {residue_count}")
        _, jackhmmer_error = self.compute_aligned_ca_error(jackhmmer_coords, experimental_coords)
        _, tuned_error = self.compute_aligned_ca_error(tuned_coords, experimental_coords)
        delta_error = jackhmmer_error - tuned_error
        window_radius = int(self.config.sequence_window_radius)
        windowed_jackhmmer_bits, sequence_window_residue_count = self.compute_centered_sequence_window_mean(jackhmmer_bits, window_radius)
        windowed_tuned_bits, _ = self.compute_centered_sequence_window_mean(tuned_bits, window_radius)
        windowed_delta_bits, _ = self.compute_centered_sequence_window_mean(tuned_bits.astype(float) - jackhmmer_bits.astype(float), window_radius)
        windowed_jackhmmer_error, _ = self.compute_centered_sequence_window_mean(jackhmmer_error, window_radius)
        windowed_tuned_error, _ = self.compute_centered_sequence_window_mean(tuned_error, window_radius)
        windowed_delta_error, _ = self.compute_centered_sequence_window_mean(delta_error, window_radius)
        absolute_jackhmmer_error = windowed_jackhmmer_error
        absolute_tuned_error = windowed_tuned_error
        absolute_delta_error = windowed_delta_error
        heatmap_delta_error = windowed_delta_error
        local_jackhmmer_error = windowed_jackhmmer_error
        local_tuned_error = windowed_tuned_error
        local_delta_error = windowed_delta_error
        selected_jackhmmer_model = self.loader.get_jackhmmer_selected_model(pdb_id)
        jackhmmer_row = jackhmmer_lookup.loc[pdb_id]
        tuned_row = tuned_lookup.loc[pdb_id]
        jackhmmer_mean_ptm = float(pd.to_numeric(pd.Series([jackhmmer_row.get("mean_ptm", np.nan)]), errors="coerce").iloc[0])
        tuned_mean_ptm = float(pd.to_numeric(pd.Series([tuned_row.get("mean_ptm", np.nan)]), errors="coerce").iloc[0])
        if np.isfinite(jackhmmer_mean_ptm) and np.isfinite(tuned_mean_ptm):
            delta_ptm_for_gate = tuned_mean_ptm - jackhmmer_mean_ptm
            ptm_gate_source = "comparison_mean_ptm"
        else:
            delta_ptm_for_gate = float(tuned_model.ptm - selected_jackhmmer_model.ptm)
            ptm_gate_source = "selected_model_ptm"
        residue_df = pd.DataFrame({"pdb_id": str(pdb_id), "target_fraction": float(target_fraction), "target_percent": float(100.0 * target_fraction), "site_index": np.arange(residue_count, dtype=int), "sequence_window_radius": window_radius, "sequence_window_residue_count": sequence_window_residue_count, "pointwise_jackhmmer_bits_site": jackhmmer_bits.astype(float), "pointwise_tuned_bits_site": tuned_bits.astype(float), "pointwise_delta_bits_site": tuned_bits.astype(float) - jackhmmer_bits.astype(float), "jackhmmer_bits_site": windowed_jackhmmer_bits, "tuned_bits_site": windowed_tuned_bits, "delta_bits_site": windowed_delta_bits, "transition_class": self.assign_transition_class(windowed_jackhmmer_bits, windowed_tuned_bits), "pointwise_jackhmmer_ca_error": jackhmmer_error, "pointwise_tuned_ca_error": tuned_error, "pointwise_delta_ca_error": delta_error, "jackhmmer_ca_error": windowed_jackhmmer_error, "tuned_ca_error": windowed_tuned_error, "delta_ca_error": windowed_delta_error, "absolute_jackhmmer_ca_error": absolute_jackhmmer_error, "absolute_tuned_ca_error": absolute_tuned_error, "absolute_delta_ca_error": absolute_delta_error, "heatmap_delta_ca_error": heatmap_delta_error, "local_jackhmmer_ca_error": local_jackhmmer_error, "local_tuned_ca_error": local_tuned_error, "local_delta_ca_error": local_delta_error, "absolute_error_band": self.assign_absolute_error_bands(absolute_tuned_error), "delta_error_band": self.assign_delta_error_bands(local_delta_error), "is_local_improvement": local_delta_error > float(self.config.improvement_threshold_angstrom)})
        pair_record = {"pdb_id": str(pdb_id), "target_fraction": float(target_fraction), "target_percent": float(100.0 * target_fraction), "n_residues": int(residue_count), "jackhmmer_model_index": int(selected_jackhmmer_model.model_index), "jackhmmer_selected_ptm": float(selected_jackhmmer_model.ptm), "tuned_model_index": int(tuned_model.model_index), "tuned_selected_ptm": float(tuned_model.ptm), "jackhmmer_mean_ptm": jackhmmer_mean_ptm, "tuned_mean_ptm": tuned_mean_ptm, "delta_ptm_for_gate": float(delta_ptm_for_gate), "ptm_gate_source": str(ptm_gate_source), "mean_pointwise_jackhmmer_ca_error": float(np.mean(jackhmmer_error)), "mean_pointwise_tuned_ca_error": float(np.mean(tuned_error)), "mean_pointwise_delta_ca_error": float(np.mean(delta_error)), "mean_jackhmmer_ca_error": float(np.mean(windowed_jackhmmer_error)), "mean_tuned_ca_error": float(np.mean(windowed_tuned_error)), "mean_delta_ca_error": float(np.mean(windowed_delta_error)), "probability_local_improvement": float(np.mean(windowed_delta_error > float(self.config.improvement_threshold_angstrom))), "sequence_window_radius": int(window_radius), "mean_sequence_window_residue_count": float(np.mean(sequence_window_residue_count))}
        return residue_df, pair_record

    def filter_to_complete_local_target_set(self, residue_df: pd.DataFrame, pair_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Filter to PDB IDs retained for every configured target."""
        if pair_df.empty:
            return residue_df.copy(), pair_df.copy(), pd.DataFrame(columns=["pdb_id", "n_retained_targets", "is_complete_local_target"])
        target_set = {float(value) for value in self.config.target_fractions}
        records: List[Dict[str, Any]] = []
        complete_ids: List[str] = []
        for pdb_id, group_df in pair_df.groupby("pdb_id", sort=False):
            retained_targets = {float(value) for value in pd.to_numeric(group_df["target_fraction"], errors="coerce").dropna().tolist()}
            is_complete = retained_targets == target_set
            if is_complete:
                complete_ids.append(str(pdb_id))
            records.append({"pdb_id": str(pdb_id), "n_retained_targets": int(len(retained_targets)), "n_expected_targets": int(len(target_set)), "is_complete_local_target": bool(is_complete), "retained_targets_json": json.dumps(sorted(retained_targets)), "missing_targets_json": json.dumps(sorted(target_set - retained_targets))})
        complete_set = set(complete_ids)
        return residue_df[residue_df["pdb_id"].astype(str).isin(complete_set)].copy(), pair_df[pair_df["pdb_id"].astype(str).isin(complete_set)].copy(), pd.DataFrame(records)

    def flush_logger_handlers(self) -> None:
        """Flush logger handlers."""
        if bool(self.config.flush_logs_on_progress):
            for handler in self.logger.handlers:
                handler.flush()

    def build_bitsite_bin_edges(self, values: np.ndarray, bin_width: float) -> np.ndarray:
        """Build fixed-width bits/site bin edges."""
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return np.array([], dtype=float)
        bin_width_value = float(bin_width)
        if not np.isfinite(bin_width_value) or bin_width_value <= 0.0:
            raise ValueError(f"Bits/site bin width must be positive: {bin_width}")
        start = float(np.floor(float(np.min(finite_values)) / bin_width_value) * bin_width_value)
        stop = float(np.ceil(float(np.max(finite_values)) / bin_width_value) * bin_width_value)
        if stop <= float(np.max(finite_values)) or np.isclose(stop, float(np.max(finite_values))):
            stop += bin_width_value
        return np.round(np.arange(start, stop + (0.5 * bin_width_value), bin_width_value, dtype=float), 10)

    def assign_bitsite_bins(self, df: pd.DataFrame, value_column: str, prefix: str, edges: np.ndarray) -> pd.DataFrame:
        """Assign bits/site bins."""
        out = df.copy()
        values = pd.to_numeric(out[value_column], errors="coerce")
        codes = pd.cut(values, bins=edges, labels=False, right=False, include_lowest=True)
        out[f"{prefix}_bin_index"] = pd.Series(codes, index=out.index, dtype="Int64")
        left_map = {index: float(edges[index]) for index in range(len(edges) - 1)}
        right_map = {index: float(edges[index + 1]) for index in range(len(edges) - 1)}
        out[f"{prefix}_bin_left"] = out[f"{prefix}_bin_index"].map(left_map)
        out[f"{prefix}_bin_right"] = out[f"{prefix}_bin_index"].map(right_map)
        out[f"{prefix}_bin_midpoint"] = (pd.to_numeric(out[f"{prefix}_bin_left"], errors="coerce") + pd.to_numeric(out[f"{prefix}_bin_right"], errors="coerce")) / 2.0
        out[f"{prefix}_bin_label"] = out.apply(lambda row: self.format_bin_label(row[f"{prefix}_bin_left"], row[f"{prefix}_bin_right"]), axis=1)
        return out

    def format_bin_label(self, left_value: Any, right_value: Any) -> str:
        """Format bin label."""
        left_numeric = pd.to_numeric(pd.Series([left_value]), errors="coerce").iloc[0]
        right_numeric = pd.to_numeric(pd.Series([right_value]), errors="coerce").iloc[0]
        if np.isfinite(left_numeric) and np.isfinite(right_numeric):
            return f"[{float(left_numeric):.2f}, {float(right_numeric):.2f})"
        return ""

    def build_gate_membership(self, pair_df: pd.DataFrame) -> pd.DataFrame:
        """Expand pairs across gates."""
        records: List[Dict[str, Any]] = []
        for _, row in pair_df.iterrows():
            delta_ptm = float(row["delta_ptm_for_gate"])
            for gate_name, gate_threshold in self.config.gate_thresholds:
                if gate_threshold is None or delta_ptm > float(gate_threshold):
                    record = row.to_dict()
                    record["gate"] = str(gate_name)
                    record["gate_threshold"] = float("nan") if gate_threshold is None else float(gate_threshold)
                    records.append(record)
        return pd.DataFrame(records)
    def build_target_range_membership(self, gate_pair_df: pd.DataFrame) -> pd.DataFrame:
        """Expand gated pairs across combined and configured target ranges."""
        records: List[Dict[str, Any]] = []
        for _, row in gate_pair_df.iterrows():
            target_percent = float(row["target_percent"])
            combined_record = row.to_dict()
            combined_record["target_range"] = "all_targets"
            combined_record["target_range_label"] = "all targets"
            records.append(combined_record)
            for range_name, range_label, lower_percent, upper_percent in self.config.target_range_definitions:
                lower_value = float(lower_percent)
                upper_value = float(upper_percent)
                if target_percent >= lower_value and target_percent <= upper_value:
                    range_record = row.to_dict()
                    range_record["target_range"] = str(range_name)
                    range_record["target_range_label"] = str(range_label)
                    records.append(range_record)
        return pd.DataFrame(records)


    def build_absolute_summary_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build absolute-error summary input columns."""
        out = df.copy()
        out["jackhmmer_ca_error"] = pd.to_numeric(out["absolute_jackhmmer_ca_error"], errors="coerce")
        out["tuned_ca_error"] = pd.to_numeric(out["absolute_tuned_ca_error"], errors="coerce")
        out["delta_ca_error"] = pd.to_numeric(out["absolute_delta_ca_error"], errors="coerce")
        out["absolute_error_band"] = self.assign_absolute_error_bands(out["tuned_ca_error"].to_numpy(dtype=float, copy=False))
        out["sequence_window_residue_count"] = pd.to_numeric(out["sequence_window_residue_count"], errors="coerce")
        return out

    def build_local_improvement_summary_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build local-improvement summary input columns."""
        out = df.copy()
        out["jackhmmer_ca_error"] = pd.to_numeric(out["local_jackhmmer_ca_error"], errors="coerce")
        out["tuned_ca_error"] = pd.to_numeric(out["local_tuned_ca_error"], errors="coerce")
        out["delta_ca_error"] = pd.to_numeric(out["local_delta_ca_error"], errors="coerce")
        out["delta_error_band"] = self.assign_delta_error_bands(out["delta_ca_error"].to_numpy(dtype=float, copy=False))
        out["is_local_improvement"] = out["delta_ca_error"] > float(self.config.improvement_threshold_angstrom)
        out["sequence_window_residue_count"] = pd.to_numeric(out["sequence_window_residue_count"], errors="coerce")
        return out

    def build_heatmap_summary_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build heatmap summary input columns."""
        out = df.copy()
        out["delta_ca_error"] = pd.to_numeric(out["heatmap_delta_ca_error"], errors="coerce")
        out["sequence_window_residue_count"] = pd.to_numeric(out["sequence_window_residue_count"], errors="coerce")
        return out

    def build_all_summaries(self, residue_df: pd.DataFrame, pair_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Build plot summaries."""
        if residue_df.empty or pair_df.empty:
            return {"absolute_error_bitsite_summary": pd.DataFrame(), "delta_error_transition_summary": pd.DataFrame(), "delta_error_transition_by_target": pd.DataFrame(), "heatmap_delta_error_summary": pd.DataFrame(), "transition_mean_delta_ca_error_summary": pd.DataFrame(), "local_delta_by_information_state": pd.DataFrame(), "newly_below_delta_summary": pd.DataFrame(), "newly_below_rescue_enrichment": pd.DataFrame(), "local_residue_error_summary": pd.DataFrame()}
        all_bits = np.concatenate([pd.to_numeric(residue_df["jackhmmer_bits_site"], errors="coerce").to_numpy(dtype=float, copy=False), pd.to_numeric(residue_df["tuned_bits_site"], errors="coerce").to_numpy(dtype=float, copy=False)])
        gate_pair_df = self.build_gate_membership(pair_df)
        ranged_gate_pair_df = self.build_target_range_membership(gate_pair_df)
        gate_key_df = ranged_gate_pair_df[["pdb_id", "target_fraction", "gate", "target_range", "target_range_label"]].drop_duplicates().copy()
        gated = residue_df.merge(gate_key_df, on=["pdb_id", "target_fraction"], how="inner", validate="many_to_many")
        baseline_bits_df = residue_df[pd.to_numeric(residue_df["target_fraction"], errors="coerce") == 0.0][["pdb_id", "site_index", "tuned_bits_site"]].drop_duplicates(subset=["pdb_id", "site_index"]).rename(columns={"tuned_bits_site": "baseline_zero_target_bits_site"})
        absolute_df = self.build_absolute_summary_input(gated)
        improvement_df = self.build_local_improvement_summary_input(gated)
        heatmap_df = self.build_heatmap_summary_input(gated)
        bitsite_edges = self.build_bitsite_bin_edges(all_bits, float(self.config.bitsite_bin_width))
        heatmap_edges = self.build_bitsite_bin_edges(all_bits, float(self.config.heatmap_bitsite_bin_width))
        bitsite_binned = self.assign_bitsite_bins(absolute_df, "tuned_bits_site", "tuned_bitsite", bitsite_edges)
        heatmap_binned = self.assign_bitsite_bins(heatmap_df, "tuned_bits_site", "tuned_bitsite", heatmap_edges)
        heatmap_binned = self.assign_bitsite_bins(heatmap_binned, "jackhmmer_bits_site", "jackhmmer_bitsite", heatmap_edges)
        return {"absolute_error_bitsite_summary": self.summarize_absolute_error_by_bitsite(bitsite_binned), "delta_error_transition_summary": self.summarize_delta_error_by_transition(improvement_df), "delta_error_transition_by_target": self.summarize_delta_error_by_transition_and_target(improvement_df), "heatmap_delta_error_summary": self.summarize_heatmap_delta_error(heatmap_binned), "transition_mean_delta_ca_error_summary": self.summarize_transition_mean_delta_ca_error(improvement_df), "local_delta_by_information_state": self.summarize_delta_by_tuned_information_state(improvement_df), "newly_below_delta_summary": self.summarize_newly_below_delta(improvement_df, baseline_bits_df), "newly_below_rescue_enrichment": self.summarize_newly_below_rescue_enrichment(improvement_df, baseline_bits_df), "local_residue_error_summary": self.summarize_local_error_overview(improvement_df)}

    def summarize_absolute_error_by_bitsite(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize absolute error by tuned bits/site."""
        if df.empty:
            return pd.DataFrame()
        working = df[df["tuned_bitsite_bin_index"].notna()].copy()
        working["tuned_bitsite_bin_index"] = working["tuned_bitsite_bin_index"].astype(int)
        pair_prob = self.compute_pair_band_probabilities(working, ["gate", "target_range", "target_range_label", "pdb_id", "target_fraction", "tuned_bitsite_bin_index"], "absolute_error_band", self.config.absolute_error_band_order)
        metadata = working[["gate", "target_range", "target_range_label", "tuned_bitsite_bin_index", "tuned_bitsite_bin_left", "tuned_bitsite_bin_right", "tuned_bitsite_bin_midpoint", "tuned_bitsite_bin_label"]].drop_duplicates()
        return self.add_sequence_window_metadata(self.aggregate_pair_probabilities(pair_prob, metadata, ["gate", "target_range", "target_range_label", "tuned_bitsite_bin_index"], "absolute_error_band", self.config.absolute_error_band_order, ["tuned_bitsite_bin_left", "tuned_bitsite_bin_right", "tuned_bitsite_bin_midpoint", "tuned_bitsite_bin_label"], int(self.config.min_pdbs_per_bitsite_bin), int(self.config.min_residues_per_bitsite_bin)), float(self.config.sequence_window_radius))

    def summarize_delta_error_by_transition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize delta error by transition."""
        if df.empty:
            return pd.DataFrame()
        working = df[df["transition_class"].isin(self.config.transition_class_order)].copy()
        pair_prob = self.compute_pair_band_probabilities(working, ["gate", "target_range", "target_range_label", "pdb_id", "target_fraction", "transition_class"], "delta_error_band", self.config.delta_error_band_order)
        metadata = pd.DataFrame({"transition_class": list(self.config.transition_class_order), "transition_label": list(self.config.transition_class_labels)})
        gates = working[["gate", "target_range", "target_range_label"]].drop_duplicates().copy()
        metadata = gates.assign(join_key=1).merge(metadata.assign(join_key=1), on="join_key", how="inner").drop(columns=["join_key"])
        return self.add_sequence_window_metadata(self.aggregate_pair_probabilities(pair_prob, metadata, ["gate", "target_range", "target_range_label", "transition_class"], "delta_error_band", self.config.delta_error_band_order, ["transition_label"], 1, int(self.config.min_residues_per_transition_group)), float(self.config.sequence_window_radius))

    def summarize_delta_error_by_transition_and_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize delta error by transition and target."""
        if df.empty:
            return pd.DataFrame()
        working = df[df["transition_class"].isin(self.config.transition_class_order)].copy()
        pair_prob = self.compute_pair_band_probabilities(working, ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "pdb_id", "transition_class"], "delta_error_band", self.config.delta_error_band_order)
        metadata = working[["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "transition_class"]].drop_duplicates().copy()
        metadata["transition_label"] = metadata["transition_class"].map({key: value for key, value in zip(self.config.transition_class_order, self.config.transition_class_labels)})
        return self.add_sequence_window_metadata(self.aggregate_pair_probabilities(pair_prob, metadata, ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "transition_class"], "delta_error_band", self.config.delta_error_band_order, ["transition_label"], 1, int(self.config.min_residues_per_transition_group)), float(self.config.sequence_window_radius))

    def summarize_heatmap_delta_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize mean delta Cα error heatmap."""
        if df.empty:
            return pd.DataFrame()
        working = df[df["jackhmmer_bitsite_bin_index"].notna() & df["tuned_bitsite_bin_index"].notna()].copy()
        working["jackhmmer_bitsite_bin_index"] = working["jackhmmer_bitsite_bin_index"].astype(int)
        working["tuned_bitsite_bin_index"] = working["tuned_bitsite_bin_index"].astype(int)
        metric_columns = ["delta_ca_error", "jackhmmer_ca_error", "tuned_ca_error", "delta_bits_site", "jackhmmer_bits_site", "tuned_bits_site", "pointwise_delta_ca_error"]
        for metric_column in metric_columns:
            if metric_column in working.columns:
                working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce")
        keys = ["gate", "target_range", "target_range_label", "pdb_id", "target_fraction", "jackhmmer_bitsite_bin_index", "tuned_bitsite_bin_index"]
        aggregation = {metric_column: "mean" for metric_column in metric_columns if metric_column in working.columns}
        pair_values = working.groupby(keys, sort=False).agg(aggregation).reset_index()
        pair_values = pair_values.rename(columns={metric_column: f"pair_mean_{metric_column}" for metric_column in aggregation})
        pdb_group_columns = ["gate", "target_range", "target_range_label", "pdb_id", "jackhmmer_bitsite_bin_index", "tuned_bitsite_bin_index"]
        pair_mean_columns = [column for column in pair_values.columns if column.startswith("pair_mean_")]
        pdb_values = pair_values.groupby(pdb_group_columns, sort=False)[pair_mean_columns].mean().reset_index()
        summary_group_columns = ["gate", "target_range", "target_range_label", "jackhmmer_bitsite_bin_index", "tuned_bitsite_bin_index"]
        summary = pdb_values.groupby(summary_group_columns, sort=False)[pair_mean_columns].mean().reset_index()
        rename_map = {"pair_mean_delta_ca_error": "mean_delta_ca_error", "pair_mean_jackhmmer_ca_error": "mean_jackhmmer_ca_error", "pair_mean_tuned_ca_error": "mean_tuned_ca_error", "pair_mean_delta_bits_site": "mean_delta_bits_site", "pair_mean_jackhmmer_bits_site": "mean_jackhmmer_bits_site", "pair_mean_tuned_bits_site": "mean_tuned_bits_site", "pair_mean_pointwise_delta_ca_error": "mean_pointwise_delta_ca_error"}
        summary = summary.rename(columns={key: value for key, value in rename_map.items() if key in summary.columns})
        n_pdbs = pdb_values.groupby(summary_group_columns, sort=False)["pdb_id"].nunique().rename("n_pdbs").reset_index()
        n_pairs = pair_values.groupby(summary_group_columns, sort=False).size().rename("n_pairs").reset_index()
        raw_counts = working.groupby(summary_group_columns, sort=False).size().rename("n_residues").reset_index()
        window_counts = working.groupby(summary_group_columns, sort=False)["sequence_window_residue_count"].agg(mean_sequence_window_residue_count="mean", median_sequence_window_residue_count="median").reset_index()
        metadata = working[["gate", "target_range", "target_range_label", "jackhmmer_bitsite_bin_index", "jackhmmer_bitsite_bin_left", "jackhmmer_bitsite_bin_right", "jackhmmer_bitsite_bin_midpoint", "jackhmmer_bitsite_bin_label", "tuned_bitsite_bin_index", "tuned_bitsite_bin_left", "tuned_bitsite_bin_right", "tuned_bitsite_bin_midpoint", "tuned_bitsite_bin_label"]].drop_duplicates()
        out = summary.merge(n_pdbs, on=summary_group_columns, how="left", validate="one_to_one")
        out = out.merge(n_pairs, on=summary_group_columns, how="left", validate="one_to_one")
        out = out.merge(raw_counts, on=summary_group_columns, how="left", validate="one_to_one")
        out = out.merge(window_counts, on=summary_group_columns, how="left", validate="one_to_one")
        out = out.merge(metadata, on=summary_group_columns, how="left", validate="many_to_one")
        out = out[(out["n_residues"] >= int(self.config.min_residues_per_heatmap_cell)) & (out["n_pdbs"] >= int(self.config.min_pdbs_per_heatmap_cell))].copy()
        out["sequence_window_radius"] = int(self.config.sequence_window_radius)
        out["sequence_window_width"] = int((2 * int(self.config.sequence_window_radius)) + 1)
        out["sequence_window_boundary_rule"] = "truncate"
        return out.sort_values(["gate", "target_range", "jackhmmer_bitsite_bin_index", "tuned_bitsite_bin_index"], kind="mergesort").reset_index(drop=True)

    def summarize_transition_mean_delta_ca_error(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize mean local errors by information-transition zone."""
        if df.empty:
            return pd.DataFrame()
        working = df[df["transition_class"].isin(self.config.transition_class_order)].copy()
        metric_columns = ["jackhmmer_ca_error", "tuned_ca_error", "delta_ca_error", "is_local_improvement"]
        for metric_column in metric_columns:
            working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce")
        pair_group_columns = ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "pdb_id", "transition_class"]
        pair_values = working.groupby(pair_group_columns, sort=False).agg(pair_mean_jackhmmer_ca_error=("jackhmmer_ca_error", "mean"), pair_mean_tuned_ca_error=("tuned_ca_error", "mean"), pair_mean_delta_ca_error=("delta_ca_error", "mean"), pair_probability_local_improvement=("is_local_improvement", "mean"), n_residues=("delta_ca_error", "size")).reset_index()
        pdb_group_columns = ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "pdb_id", "transition_class"]
        pdb_values = pair_values.groupby(pdb_group_columns, sort=False).agg(pdb_mean_jackhmmer_ca_error=("pair_mean_jackhmmer_ca_error", "mean"), pdb_mean_tuned_ca_error=("pair_mean_tuned_ca_error", "mean"), pdb_mean_delta_ca_error=("pair_mean_delta_ca_error", "mean"), pdb_probability_local_improvement=("pair_probability_local_improvement", "mean"), n_residues=("n_residues", "sum")).reset_index()
        summary_group_columns = ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "transition_class"]
        summary = pdb_values.groupby(summary_group_columns, sort=False).agg(mean_jackhmmer_ca_error=("pdb_mean_jackhmmer_ca_error", "mean"), mean_tuned_ca_error=("pdb_mean_tuned_ca_error", "mean"), mean_delta_ca_error=("pdb_mean_delta_ca_error", "mean"), probability_local_improvement=("pdb_probability_local_improvement", "mean"), n_pdbs=("pdb_id", "nunique"), n_residues=("n_residues", "sum")).reset_index()
        summary["transition_label"] = summary["transition_class"].map({key: value for key, value in zip(self.config.transition_class_order, self.config.transition_class_labels)})
        summary = summary[summary["n_residues"] >= int(self.config.min_residues_per_transition_group)].copy()
        return self.add_sequence_window_metadata(summary.sort_values(["gate", "target_range", "target_fraction", "transition_class"], kind="mergesort").reset_index(drop=True), float(self.config.sequence_window_radius))

    def summarize_local_error_overview(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize local error overview."""
        if df.empty:
            return pd.DataFrame()
        records: List[Dict[str, Any]] = []
        for (gate, target_range, target_range_label), gate_df in df.groupby(["gate", "target_range", "target_range_label"], sort=False):
            records.append({"gate": str(gate), "target_range": str(target_range), "target_range_label": str(target_range_label), "n_pdbs": int(gate_df["pdb_id"].nunique()), "n_pairs": int(gate_df[["pdb_id", "target_fraction"]].drop_duplicates().shape[0]), "n_residues": int(gate_df.shape[0]), "mean_jackhmmer_ca_error": float(pd.to_numeric(gate_df["jackhmmer_ca_error"], errors="coerce").mean()), "mean_tuned_ca_error": float(pd.to_numeric(gate_df["tuned_ca_error"], errors="coerce").mean()), "mean_delta_ca_error": float(pd.to_numeric(gate_df["delta_ca_error"], errors="coerce").mean()), "probability_local_improvement": float(pd.to_numeric(gate_df["is_local_improvement"], errors="coerce").mean())})
        return pd.DataFrame(records)

    def compute_pair_band_probabilities(self, df: pd.DataFrame, grouping_columns: Sequence[str], band_column: str, band_order: Sequence[str]) -> pd.DataFrame:
        """Compute pair-level band probabilities."""
        counts = df.groupby(list(grouping_columns) + [str(band_column)], sort=False).size().rename("n_residues_band").reset_index()
        totals = df.groupby(list(grouping_columns), sort=False).size().rename("n_residues_total").reset_index()
        grid = totals[list(grouping_columns)].drop_duplicates().assign(join_key=1).merge(pd.DataFrame({str(band_column): list(band_order)}).assign(join_key=1), on="join_key", how="inner").drop(columns=["join_key"])
        out = grid.merge(totals, on=list(grouping_columns), how="left", validate="many_to_one")
        out = out.merge(counts, on=list(grouping_columns) + [str(band_column)], how="left", validate="one_to_one")
        out["n_residues_band"] = pd.to_numeric(out["n_residues_band"], errors="coerce").fillna(0).astype(int)
        out["probability"] = out["n_residues_band"] / out["n_residues_total"]
        return out

    def aggregate_pair_probabilities(self, pair_prob: pd.DataFrame, metadata: pd.DataFrame, merge_columns: Sequence[str], band_column: str, band_order: Sequence[str], metadata_columns: Sequence[str], min_pdbs: int, min_residues: int) -> pd.DataFrame:
        """Aggregate pair probabilities to PDB-balanced summaries."""
        if pair_prob.empty:
            return pd.DataFrame()
        pdb_group_columns = list(merge_columns) + ["pdb_id", str(band_column)]
        pdb_prob = pair_prob.groupby(pdb_group_columns, sort=False)["probability"].mean().rename("pdb_probability").reset_index()
        summary = pdb_prob.groupby(list(merge_columns) + [str(band_column)], sort=False)["pdb_probability"].mean().rename("probability").reset_index()
        n_pdbs = pdb_prob.groupby(list(merge_columns) + [str(band_column)], sort=False)["pdb_id"].nunique().rename("n_pdbs").reset_index()
        n_pairs = pair_prob.groupby(list(merge_columns) + [str(band_column)], sort=False).size().rename("n_pairs_band_grid").reset_index()
        raw_counts = pair_prob.groupby(list(merge_columns) + [str(band_column)], sort=False)["n_residues_band"].sum().rename("n_residues_band").reset_index()
        raw_totals = pair_prob.groupby(list(merge_columns) + [str(band_column)], sort=False)["n_residues_total"].sum().rename("n_residues_total_band_grid").reset_index()
        out = summary.merge(n_pdbs, on=list(merge_columns) + [str(band_column)], how="left", validate="one_to_one")
        out = out.merge(n_pairs, on=list(merge_columns) + [str(band_column)], how="left", validate="one_to_one")
        out = out.merge(raw_counts, on=list(merge_columns) + [str(band_column)], how="left", validate="one_to_one")
        out = out.merge(raw_totals, on=list(merge_columns) + [str(band_column)], how="left", validate="one_to_one")
        out = out.merge(metadata[list(merge_columns) + list(metadata_columns)].drop_duplicates(), on=list(merge_columns), how="left", validate="many_to_one")
        out["band_order_index"] = out[str(band_column)].map({str(value): index for index, value in enumerate(band_order)}).astype(int)
        out = out[(out["n_residues_total_band_grid"] >= int(min_residues)) & (out["n_pdbs"] >= int(min_pdbs))].copy()
        out = out.sort_values(list(merge_columns) + ["band_order_index"], kind="mergesort").reset_index(drop=True)
        self.validate_probability_summary(out, list(merge_columns), band_column)
        return out

    def validate_probability_summary(self, df: pd.DataFrame, group_columns: Sequence[str], band_column: str) -> None:
        """Validate probability sums."""
        if df.empty:
            return
        sums = df.groupby(list(group_columns), sort=False)["probability"].sum().reset_index(name="probability_sum")
        invalid = sums[~np.isclose(pd.to_numeric(sums["probability_sum"], errors="coerce"), 1.0, atol=1.0e-6)]
        if not invalid.empty:
            raise ValueError(f"Band probabilities do not sum to one for {band_column}: {invalid.head(10).to_dict(orient='records')}")

    def add_sequence_window_metadata(self, df: pd.DataFrame, window_radius_value: float) -> pd.DataFrame:
        """Add sequence-window metadata to a summary table."""
        out = df.copy()
        if not out.empty:
            out["sequence_window_radius"] = int(self.config.sequence_window_radius)
            out["sequence_window_width"] = int((2 * int(self.config.sequence_window_radius)) + 1)
            out["sequence_window_boundary_rule"] = "truncate"
        return out

    def summarize_delta_by_tuned_information_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summarize local delta error by tuned information state and target."""
        if df.empty:
            return pd.DataFrame()
        working = df.copy()
        threshold = float(self.config.bitsite_threshold)
        working["tuned_information_state"] = np.where(pd.to_numeric(working["tuned_bits_site"], errors="coerce") < threshold, "tuned_below_2bps", "tuned_at_or_above_2bps")
        pair_prob = self.compute_pair_band_probabilities(working, ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "pdb_id", "tuned_information_state"], "delta_error_band", self.config.delta_error_band_order)
        metadata = working[["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "tuned_information_state"]].drop_duplicates().copy()
        out = self.aggregate_pair_probabilities(pair_prob, metadata, ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "tuned_information_state"], "delta_error_band", self.config.delta_error_band_order, [], 1, int(self.config.min_residues_per_transition_group))
        return self.add_sequence_window_metadata(out, float(self.config.sequence_window_radius))

    def summarize_newly_below_delta(self, df: pd.DataFrame, baseline_bits_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize local delta error for positions newly below threshold relative to 0 percent target."""
        if df.empty or baseline_bits_df.empty:
            return pd.DataFrame()
        working = df.merge(baseline_bits_df, on=["pdb_id", "site_index"], how="left", validate="many_to_one")
        threshold = float(self.config.bitsite_threshold)
        current_below = pd.to_numeric(working["tuned_bits_site"], errors="coerce") < threshold
        baseline_above = pd.to_numeric(working["baseline_zero_target_bits_site"], errors="coerce") >= threshold
        is_zero_target = np.isclose(pd.to_numeric(working["target_fraction"], errors="coerce").to_numpy(dtype=float, copy=False), 0.0)
        working["newly_below_state"] = np.where(is_zero_target, "zero_target_reference", np.where(current_below & baseline_above, "newly_below_2bps", "not_newly_below_2bps"))
        pair_prob = self.compute_pair_band_probabilities(working, ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "pdb_id", "newly_below_state"], "delta_error_band", self.config.delta_error_band_order)
        metadata = working[["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "newly_below_state"]].drop_duplicates().copy()
        out = self.aggregate_pair_probabilities(pair_prob, metadata, ["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "newly_below_state"], "delta_error_band", self.config.delta_error_band_order, [], 1, int(self.config.min_residues_per_transition_group))
        return self.add_sequence_window_metadata(out, float(self.config.sequence_window_radius))

    def summarize_newly_below_rescue_enrichment(self, df: pd.DataFrame, baseline_bits_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize local rescue enrichment among newly below-threshold positions."""
        if df.empty or baseline_bits_df.empty:
            return pd.DataFrame()
        working = df.merge(baseline_bits_df, on=["pdb_id", "site_index"], how="left", validate="many_to_one")
        threshold = float(self.config.bitsite_threshold)
        current_below = pd.to_numeric(working["tuned_bits_site"], errors="coerce") < threshold
        baseline_above = pd.to_numeric(working["baseline_zero_target_bits_site"], errors="coerce") >= threshold
        is_zero_target = np.isclose(pd.to_numeric(working["target_fraction"], errors="coerce").to_numpy(dtype=float, copy=False), 0.0)
        working = working[~is_zero_target].copy()
        working["newly_below_2bps"] = current_below[~is_zero_target].to_numpy(dtype=bool, copy=False) & baseline_above[~is_zero_target].to_numpy(dtype=bool, copy=False)
        records: List[Dict[str, Any]] = []
        for improvement_threshold in (2.0, 5.0, 10.0):
            working["is_rescued"] = pd.to_numeric(working["delta_ca_error"], errors="coerce") > float(improvement_threshold)
            grouped = working.groupby(["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "pdb_id", "newly_below_2bps"], sort=False)["is_rescued"].mean().rename("pdb_target_rescue_probability").reset_index()
            summary = grouped.groupby(["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "newly_below_2bps"], sort=False)["pdb_target_rescue_probability"].mean().rename("rescue_probability").reset_index()
            counts = grouped.groupby(["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "newly_below_2bps"], sort=False)["pdb_id"].nunique().rename("n_pdbs").reset_index()
            merged = summary.merge(counts, on=["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "newly_below_2bps"], how="left", validate="one_to_one")
            for _, row in merged.iterrows():
                records.append({"gate": str(row["gate"]), "target_range": str(row["target_range"]), "target_range_label": str(row["target_range_label"]), "target_fraction": float(row["target_fraction"]), "target_percent": float(row["target_percent"]), "improvement_threshold_angstrom": float(improvement_threshold), "newly_below_2bps": bool(row["newly_below_2bps"]), "rescue_probability": float(row["rescue_probability"]), "n_pdbs": int(row["n_pdbs"])})
        out = pd.DataFrame(records)
        if out.empty:
            return out
        pivot = out.pivot_table(index=["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "improvement_threshold_angstrom"], columns="newly_below_2bps", values="rescue_probability", aggfunc="first").reset_index()
        pivot = pivot.rename(columns={False: "not_newly_below_rescue_probability", True: "newly_below_rescue_probability"})
        for column_name in ["not_newly_below_rescue_probability", "newly_below_rescue_probability"]:
            if column_name not in pivot.columns:
                pivot[column_name] = np.nan
        epsilon = 1.0e-9
        pivot["newly_below_rescue_odds"] = (pivot["newly_below_rescue_probability"] + epsilon) / (1.0 - pivot["newly_below_rescue_probability"] + epsilon)
        pivot["not_newly_below_rescue_odds"] = (pivot["not_newly_below_rescue_probability"] + epsilon) / (1.0 - pivot["not_newly_below_rescue_probability"] + epsilon)
        pivot["newly_below_rescue_odds_ratio"] = pivot["newly_below_rescue_odds"] / pivot["not_newly_below_rescue_odds"]
        count_pivot = out.pivot_table(index=["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "improvement_threshold_angstrom"], columns="newly_below_2bps", values="n_pdbs", aggfunc="first").reset_index().rename(columns={False: "not_newly_below_n_pdbs", True: "newly_below_n_pdbs"})
        pivot = pivot.merge(count_pivot, on=["gate", "target_range", "target_range_label", "target_fraction", "target_percent", "improvement_threshold_angstrom"], how="left")
        return self.add_sequence_window_metadata(pivot, float(self.config.sequence_window_radius))

    def build_dictionary_tables(self) -> Dict[str, pd.DataFrame]:
        """Build workbook dictionary tables."""
        transition_df = pd.DataFrame({"transition_class": list(self.config.transition_class_order), "transition_label": list(self.config.transition_class_labels), "description": ["JackHMMER bits/site ≥ threshold and tuned bits/site ≥ threshold.", "JackHMMER bits/site ≥ threshold and tuned bits/site < threshold.", "JackHMMER bits/site < threshold and tuned bits/site < threshold.", "JackHMMER bits/site < threshold and tuned bits/site ≥ threshold."]})
        absolute_df = pd.DataFrame({"absolute_error_band": list(self.config.absolute_error_band_order), "color": list(self.config.absolute_error_band_colors), "interpretation": ["best", "better", "intermediate", "worse", "worst"]})
        delta_df = pd.DataFrame({"delta_error_band": list(self.config.delta_error_band_order), "color": list(self.config.delta_error_band_colors), "interpretation": ["best", "better", "little change", "worse", "worst"]})
        return {"information_transition_dictionary": transition_df, "absolute_error_band_dictionary": absolute_df, "delta_error_band_dictionary": delta_df}


class NullJsonlLogger:
    """Ignore worker JSONL events."""

    def write(self, event: str, payload: Dict[str, Any]) -> None:
        """Ignore one JSONL record."""
        return None

    def close(self) -> None:
        """Close no resources."""
        return None


FIGURE7_WORKER_ANALYZER: Optional[Figure7MetaAnalyzer] = None
FIGURE7_WORKER_JACKHMMER_LOOKUP: Optional[pd.DataFrame] = None
FIGURE7_WORKER_TUNED_LOOKUPS: Optional[Dict[float, pd.DataFrame]] = None
FIGURE7_WORKER_CONFIG: Optional[Figure7MetaConfig] = None


def initialize_figure7_worker(config: Figure7MetaConfig, paths: Figure7MetaPaths, jackhmmer_df: pd.DataFrame, tuned_tables: Dict[float, pd.DataFrame]) -> None:
    """Initialize process-local analysis objects."""
    global FIGURE7_WORKER_ANALYZER, FIGURE7_WORKER_JACKHMMER_LOOKUP, FIGURE7_WORKER_TUNED_LOOKUPS, FIGURE7_WORKER_CONFIG
    worker_logger = logging.getLogger(f"figure7_meta_worker_{os.getpid()}")
    worker_logger.handlers = []
    worker_logger.addHandler(logging.NullHandler())
    worker_logger.setLevel(logging.ERROR)
    loader = Figure7MetaDataLoader(config, paths, worker_logger)
    FIGURE7_WORKER_ANALYZER = Figure7MetaAnalyzer(config, loader, worker_logger, NullJsonlLogger())
    FIGURE7_WORKER_ANALYZER.validate_sequence_window_config()
    FIGURE7_WORKER_JACKHMMER_LOOKUP = jackhmmer_df.drop_duplicates(subset=["pdb_id"]).set_index("pdb_id")
    FIGURE7_WORKER_TUNED_LOOKUPS = {float(target): table.drop_duplicates(subset=["pdb_id"]).set_index("pdb_id") for target, table in tuned_tables.items()}
    FIGURE7_WORKER_CONFIG = config


def process_figure7_pdb_worker(pdb_id: str) -> Dict[str, Any]:
    """Process one PDB ID in a worker process."""
    if FIGURE7_WORKER_ANALYZER is None or FIGURE7_WORKER_JACKHMMER_LOOKUP is None or FIGURE7_WORKER_TUNED_LOOKUPS is None or FIGURE7_WORKER_CONFIG is None:
        raise RuntimeError("Figure 7 worker was not initialized")
    residue_frames: List[pd.DataFrame] = []
    pair_records: List[Dict[str, Any]] = []
    exclusion_records: List[Dict[str, Any]] = []
    retained_pairs = 0
    excluded_pairs = 0
    for target_fraction in FIGURE7_WORKER_CONFIG.target_fractions:
        try:
            residue_df, pair_record = FIGURE7_WORKER_ANALYZER.process_one_pair(str(pdb_id), float(target_fraction), FIGURE7_WORKER_JACKHMMER_LOOKUP, FIGURE7_WORKER_TUNED_LOOKUPS[float(target_fraction)])
            residue_frames.append(residue_df)
            pair_records.append(pair_record)
            retained_pairs += 1
        except Exception as exc:
            exclusion_records.append({"pdb_id": str(pdb_id), "target_fraction": float(target_fraction), "target_percent": float(100.0 * target_fraction), "reason": type(exc).__name__, "message": str(exc)})
            excluded_pairs += 1
    residue_df = pd.concat(residue_frames, axis=0, ignore_index=True) if residue_frames else pd.DataFrame()
    return {"pdb_id": str(pdb_id), "residue_df": residue_df, "pair_records": pair_records, "exclusion_records": exclusion_records, "retained_pairs": int(retained_pairs), "excluded_pairs": int(excluded_pairs)}


class Figure7MetaPlotter:
    """Render plots."""

    def __init__(self, config: Figure7MetaConfig, paths: Figure7MetaPaths, logger: logging.Logger) -> None:
        """Initialize plotter."""
        self.config = config
        self.paths = paths
        self.logger = logger

    def apply_publication_style(self) -> None:
        """Apply publication style."""
        sns.set_style("white")
        mpl.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "savefig.dpi": 300, "figure.dpi": 120, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})

    def apply_axis_style(self, ax: plt.Axes) -> None:
        """Apply axis style."""
        ax.grid(False)
        ax.tick_params(axis="both", which="both", bottom=True, top=False, left=True, right=False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        sns.despine(ax=ax, top=True, right=True)

    def save_figure(self, fig: plt.Figure, out_stem: str) -> None:
        """Save figure."""
        for suffix in self.config.plot_file_formats:
            fig.savefig(self.paths.output_dir / f"{out_stem}.{suffix}", dpi=300, bbox_inches="tight")

    def save_band_legend(self, labels: Sequence[str], colors: Sequence[str], out_stem: str) -> None:
        """Save legend."""
        handles = [mpl.patches.Patch(facecolor=color, edgecolor=color, linewidth=0.8) for color in colors]
        fig = plt.figure(figsize=(6.0, 1.2))
        legend_obj = fig.legend(handles, list(labels), loc="center", frameon=False, ncol=min(len(labels), 5), handlelength=1.0, handletextpad=0.6, borderaxespad=0.2, columnspacing=1.0)
        for suffix in self.config.plot_file_formats:
            fig.savefig(self.paths.output_dir / f"{out_stem}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.12, bbox_extra_artists=[legend_obj])
        plt.close(fig)

    def color_map(self, labels: Sequence[str], colors: Sequence[str]) -> Dict[str, str]:
        """Build color map."""
        return {str(label): str(color) for label, color in zip(labels, colors)}

    def sequence_window_token(self) -> str:
        """Return a compact sequence-window token."""
        return f"w{int(self.config.sequence_window_radius)}"

    def build_delta_heatmap_cmap(self) -> mpl.colors.LinearSegmentedColormap:
        """Build the Figure 3-style diverging heatmap colormap."""
        orange = "#E69F00"
        blue = "#0072B2"
        cmap = mpl.colors.LinearSegmentedColormap.from_list("okabe_ito_delta", [orange, "#FFFFFF", blue], N=256)
        cmap.set_bad(color="white")
        return cmap

    def target_range_plot_specs(self) -> List[Tuple[str, str]]:
        """Return target-range plot specifications."""
        specs = [("all_targets", "all targets")]
        specs.extend([(str(name), str(label)) for name, label, _, _ in self.config.target_range_definitions])
        return specs

    def save_heatmap_delta_colorbar(self) -> None:
        """Save standalone heatmap delta-error colorbar."""
        clip_value = float(self.config.heatmap_delta_clip_angstrom)
        norm = mpl.colors.TwoSlopeNorm(vmin=-clip_value, vcenter=0.0, vmax=clip_value)
        scalar_mappable = mpl.cm.ScalarMappable(cmap=self.build_delta_heatmap_cmap(), norm=norm)
        scalar_mappable.set_array([])
        fig, ax = plt.subplots(figsize=(1.2, 3.0))
        fig.subplots_adjust(left=0.55, right=0.85)
        colorbar = fig.colorbar(scalar_mappable, cax=ax)
        colorbar.set_label("Mean JackHMMER error − tuned error (Å)")
        ax.grid(False)
        out_stem = "legend__heatmap_mean_delta_ca_error__colorbar"
        self.save_figure(fig, out_stem)
        plt.close(fig)

    def build_probability_pivot(self, summary_df: pd.DataFrame, band_column: str, band_order: Sequence[str], sort_column: str) -> pd.DataFrame:
        """Pivot probability table."""
        metadata_columns = [column for column in summary_df.columns if column.endswith("_bin_left") or column.endswith("_bin_right") or column.endswith("_bin_midpoint") or column.endswith("_bin_label")]
        pivot = summary_df.pivot_table(index=[str(sort_column)] + metadata_columns, columns=str(band_column), values="probability", aggfunc="first", fill_value=0.0).reset_index()
        for band_label in band_order:
            if str(band_label) not in pivot.columns:
                pivot[str(band_label)] = 0.0
        return pivot.sort_values(str(sort_column), kind="mergesort").reset_index(drop=True)

    def plot_absolute_error_by_bitsite(self, summary_df: pd.DataFrame, gate: str, target_range: str) -> Optional[str]:
        """Plot absolute error by bits/site."""
        sub = summary_df[(summary_df["gate"] == str(gate)) & (summary_df["target_range"] == str(target_range))].copy()
        if sub.empty:
            self.logger.info("Skipping absolute-error plot for gate=%s target_range=%s because no summary rows were available", gate, target_range)
            return None
        pivot = self.build_probability_pivot(sub, "absolute_error_band", self.config.absolute_error_band_order, "tuned_bitsite_bin_index")
        x_values = pivot["tuned_bitsite_bin_midpoint"].to_numpy(dtype=float, copy=False)
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        cumulative = np.zeros((pivot.shape[0],), dtype=float)
        color_map = self.color_map(self.config.absolute_error_band_order, self.config.absolute_error_band_colors)
        for band_label in self.config.absolute_error_band_order:
            values = pd.to_numeric(pivot[str(band_label)], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)
            next_cumulative = cumulative + values
            ax.fill_between(x_values, cumulative, next_cumulative, step="mid", color=color_map[str(band_label)], alpha=0.94)
            cumulative = next_cumulative
        ax.set_xlim(float(np.nanmin(pd.to_numeric(pivot["tuned_bitsite_bin_left"], errors="coerce"))), float(np.nanmax(pd.to_numeric(pivot["tuned_bitsite_bin_right"], errors="coerce"))))
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Tuned centered-window MSA bits/site")
        ax.set_ylabel("PDB-balanced probability")
        target_range_label = str(sub["target_range_label"].iloc[0]) if "target_range_label" in sub.columns and not sub.empty else str(target_range)
        ax.set_title(f"Tuned local Cα error by MSA bits/site; bin={self.config.bitsite_bin_width:g}; centered sequence window radius={self.config.sequence_window_radius:g}\n{gate}; {target_range_label}", pad=self.config.title_pad_points)
        self.apply_axis_style(ax)
        fig.tight_layout()
        window_token = self.sequence_window_token()
        out_stem = f"absolute_tuned_ca_error_bands_by_tuned_bitsite__sequence_window__{window_token}__{target_range}__{gate}"
        self.save_figure(fig, out_stem)
        plt.close(fig)
        return out_stem

    def plot_delta_error_by_transition(self, summary_df: pd.DataFrame, gate: str, target_range: str) -> Optional[str]:
        """Plot delta error by transition."""
        sub = summary_df[(summary_df["gate"] == str(gate)) & (summary_df["target_range"] == str(target_range))].copy()
        if sub.empty:
            self.logger.info("Skipping transition plot for gate=%s target_range=%s because no summary rows were available", gate, target_range)
            return None
        label_map = {key: value for key, value in zip(self.config.transition_class_order, self.config.transition_class_labels)}
        color_map = self.color_map(self.config.delta_error_band_order, self.config.delta_error_band_colors)
        fig, ax = plt.subplots(figsize=(self.config.figure_width, self.config.figure_height))
        x_positions = np.arange(len(self.config.transition_class_order), dtype=float)
        bottoms = np.zeros((len(self.config.transition_class_order),), dtype=float)
        for band_label in reversed(self.config.delta_error_band_order):
            heights = []
            for transition_class in self.config.transition_class_order:
                row = sub[(sub["transition_class"] == str(transition_class)) & (sub["delta_error_band"] == str(band_label))]
                heights.append(float(row["probability"].iloc[0]) if not row.empty else 0.0)
            height_array = np.asarray(heights, dtype=float)
            ax.bar(x_positions, height_array, bottom=bottoms, color=color_map[str(band_label)], edgecolor="white", linewidth=0.4, width=0.72)
            bottoms += height_array
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label_map[str(value)] for value in self.config.transition_class_order])
        ax.set_ylabel("PDB-balanced probability")
        target_range_label = str(sub["target_range_label"].iloc[0]) if "target_range_label" in sub.columns and not sub.empty else str(target_range)
        ax.set_title(f"Local improvement by information-transition class; centered sequence window radius={self.config.sequence_window_radius:g}\n{gate}; {target_range_label}", pad=self.config.title_pad_points)
        self.apply_axis_style(ax)
        fig.tight_layout()
        window_token = self.sequence_window_token()
        out_stem = f"local_improvement_bands_by_information_transition__sequence_window__{window_token}__{target_range}__{gate}"
        self.save_figure(fig, out_stem)
        plt.close(fig)
        return out_stem

    def plot_heatmap_delta_error(self, summary_df: pd.DataFrame, gate: str, target_range: str) -> Optional[str]:
        """Plot mean delta Cα error heatmap without embedded colorbar."""
        sub = summary_df[(summary_df["gate"] == str(gate)) & (summary_df["target_range"] == str(target_range))].copy()
        if sub.empty:
            self.logger.info("Skipping delta-error heatmap for gate=%s target_range=%s because no summary rows were available", gate, target_range)
            return None
        pivot = sub.pivot_table(index="tuned_bitsite_bin_index", columns="jackhmmer_bitsite_bin_index", values="mean_delta_ca_error", aggfunc="first")
        if pivot.empty:
            return None
        y_indices = sorted(pivot.index.tolist())
        x_indices = sorted(pivot.columns.tolist())
        pivot = pivot.loc[y_indices, x_indices]
        x_meta = sub[["jackhmmer_bitsite_bin_index", "jackhmmer_bitsite_bin_midpoint"]].drop_duplicates().set_index("jackhmmer_bitsite_bin_index")
        y_meta = sub[["tuned_bitsite_bin_index", "tuned_bitsite_bin_midpoint"]].drop_duplicates().set_index("tuned_bitsite_bin_index")
        x_labels = [f"{float(x_meta.loc[index, 'jackhmmer_bitsite_bin_midpoint']):.2f}" if index in x_meta.index else str(index) for index in x_indices]
        y_labels = [f"{float(y_meta.loc[index, 'tuned_bitsite_bin_midpoint']):.2f}" if index in y_meta.index else str(index) for index in y_indices]
        clip_value = float(self.config.heatmap_delta_clip_angstrom)
        fig, ax = plt.subplots(figsize=(self.config.heatmap_width, self.config.heatmap_height))
        norm = mpl.colors.TwoSlopeNorm(vmin=-clip_value, vcenter=0.0, vmax=clip_value)
        ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=self.build_delta_heatmap_cmap(), norm=norm)
        ax.set_xlabel("JackHMMER centered-window MSA bits/site")
        ax.set_ylabel("Tuned centered-window MSA bits/site")
        target_range_label = str(sub["target_range_label"].iloc[0]) if "target_range_label" in sub.columns and not sub.empty else str(target_range)
        ax.set_title(f"PDB-balanced mean ΔCα error; positive = tuned improves; bin={self.config.heatmap_bitsite_bin_width:g}; centered sequence window radius={self.config.sequence_window_radius:g}\n{gate}; {target_range_label}", pad=self.config.title_pad_points)
        self.set_heatmap_ticks(ax, x_labels, y_labels)
        self.apply_axis_style(ax)
        fig.tight_layout()
        window_token = self.sequence_window_token()
        out_stem = f"heatmap_mean_delta_ca_error__sequence_window__{window_token}__{target_range}__jackhmmer_bitsite_vs_tuned_bitsite__{gate}"
        self.save_figure(fig, out_stem)
        plt.close(fig)
        return out_stem

    def set_heatmap_ticks(self, ax: plt.Axes, x_labels: Sequence[str], y_labels: Sequence[str]) -> None:
        """Set heatmap ticks."""
        x_step = max(1, int(np.ceil(len(x_labels) / 8)))
        y_step = max(1, int(np.ceil(len(y_labels) / 8)))
        x_positions = np.arange(0, len(x_labels), x_step)
        y_positions = np.arange(0, len(y_labels), y_step)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([x_labels[index] for index in x_positions], rotation=45, ha="right")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([y_labels[index] for index in y_positions])

    def render_all_plots(self, summaries: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Render all plots."""
        records: List[Dict[str, Any]] = []
        self.save_band_legend(self.config.absolute_error_band_order, self.config.absolute_error_band_colors, "legend__absolute_ca_error_bands")
        self.save_band_legend(self.config.delta_error_band_order, self.config.delta_error_band_colors, "legend__delta_ca_error_bands")
        self.save_heatmap_delta_colorbar()
        for target_range, target_range_label in self.target_range_plot_specs():
            for gate_name, _ in self.config.gate_thresholds:
                for plot_type, out_stem in [
                    ("absolute_error_bitsite", self.plot_absolute_error_by_bitsite(summaries["absolute_error_bitsite_summary"], str(gate_name), str(target_range))),
                    ("delta_error_transition", self.plot_delta_error_by_transition(summaries["delta_error_transition_summary"], str(gate_name), str(target_range))),
                    ("heatmap_delta_ca_error", self.plot_heatmap_delta_error(summaries["heatmap_delta_error_summary"], str(gate_name), str(target_range))),
                ]:
                    if out_stem is not None:
                        records.append({"gate": str(gate_name), "target_range": str(target_range), "target_range_label": str(target_range_label), "plot_type": str(plot_type), "output_stem": str(out_stem)})
        return pd.DataFrame(records)


class Figure7MetaRunner:
    """Coordinate Figure 7 meta-analysis."""

    def __init__(self, config: Figure7MetaConfig) -> None:
        """Initialize runner."""
        self.config = config
        self.paths = self.build_paths(config)
        self.logger = self.build_logger(self.paths.text_log_path)
        self.jsonl_logger = JsonlLogger(self.paths.jsonl_log_path)
        self.loader = Figure7MetaDataLoader(config, self.paths, self.logger)
        self.analyzer = Figure7MetaAnalyzer(config, self.loader, self.logger, self.jsonl_logger)
        self.plotter = Figure7MetaPlotter(config, self.paths, self.logger)

    def build_paths(self, config: Figure7MetaConfig) -> Figure7MetaPaths:
        """Resolve paths."""
        root_dir = Path(".").resolve()
        output_dir = root_dir / str(config.output_dir_name)
        logs_dir = root_dir / str(config.logs_dir_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return Figure7MetaPaths(root_dir=root_dir, results_dir=root_dir / str(config.results_dir_name), experimental_dir=root_dir / str(config.experimental_dir_name), jackhmmer_msa_dir=root_dir / str(config.jackhmmer_msa_dir_name), jackhmmer_models_dir=root_dir / str(config.jackhmmer_models_dir_name), output_dir=output_dir, output_xlsx_path=output_dir / str(config.output_xlsx_name), logs_dir=logs_dir, text_log_path=logs_dir / str(config.text_log_name), jsonl_log_path=logs_dir / str(config.jsonl_log_name))

    def build_logger(self, log_path: Path) -> logging.Logger:
        """Build logger."""
        logger = logging.getLogger("figure7_meta")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        return logger

    def validate_inputs(self) -> None:
        """Validate directories."""
        for path in [self.paths.results_dir, self.paths.experimental_dir, self.paths.jackhmmer_msa_dir, self.paths.jackhmmer_models_dir]:
            if not path.is_dir():
                raise FileNotFoundError(f"Missing required directory: {path}")
        for target_fraction in self.config.target_fractions:
            target_folder = self.loader.target_folder(float(target_fraction))
            if not target_folder.is_dir():
                raise FileNotFoundError(f"Missing tuned target folder: {target_folder}")

    def build_readme_table(self, common_pdb_ids: Sequence[str]) -> pd.DataFrame:
        """Build README sheet."""
        rows = [
            {"key": "created_utc", "value": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            {"key": "analysis", "value": "Figure 7 sequence-window local structural meta-analysis using global Cα Kabsch alignment."},
            {"key": "targets", "value": json.dumps([float(value) for value in self.config.target_fractions])},
            {"key": "target_ranges", "value": json.dumps([{"name": name, "label": label, "lower_percent": lower, "upper_percent": upper} for name, label, lower, upper in self.config.target_range_definitions])},
            {"key": "gate_thresholds", "value": json.dumps([{"name": name, "threshold": threshold} for name, threshold in self.config.gate_thresholds])},
            {"key": "n_csv_valid_common_pdb_ids", "value": str(len(common_pdb_ids))},
            {"key": "model_selection", "value": "Highest Chai-1 pTM model selected independently for JackHMMER and tuned-MSA conditions."},
            {"key": "bitsite_calculation", "value": "ProDy Shannon entropy on projected A3Ms; bits/site = log2(20) - entropy."},
            {"key": "alignment", "value": "Global Cα Kabsch alignment to experimental Chain A when available, otherwise first Cα-containing experimental chain."},
            {"key": "delta_ca_error", "value": "jackhmmer_ca_error - tuned_ca_error after centered sequence-window averaging; positive values indicate local improvement by tuned MSA."},
            {"key": "restrict_to_csv_valid_common_set", "value": str(bool(self.config.restrict_to_csv_valid_common_set))},
            {"key": "enforce_complete_local_target_set", "value": str(bool(self.config.enforce_complete_local_target_set))},
            {"key": "bitsite_bin_width", "value": str(float(self.config.bitsite_bin_width))},
            {"key": "heatmap_bitsite_bin_width", "value": str(float(self.config.heatmap_bitsite_bin_width))},
            {"key": "sequence_window_radius", "value": str(int(self.config.sequence_window_radius))},
            {"key": "sequence_window_width", "value": str(int((2 * int(self.config.sequence_window_radius)) + 1))},
            {"key": "sequence_window_boundary_rule", "value": "truncate"},
            {"key": "min_pdbs_per_bitsite_bin", "value": str(int(self.config.min_pdbs_per_bitsite_bin))},
            {"key": "min_residues_per_bitsite_bin", "value": str(int(self.config.min_residues_per_bitsite_bin))},
            {"key": "min_pdbs_per_heatmap_cell", "value": str(int(self.config.min_pdbs_per_heatmap_cell))},
            {"key": "min_residues_per_heatmap_cell", "value": str(int(self.config.min_residues_per_heatmap_cell))},
            {"key": "parallel_enabled", "value": str(bool(self.config.parallel_enabled))},
            {"key": "parallel_max_workers", "value": str(int(self.config.parallel_max_workers))},
            {"key": "export_raw_residue_table", "value": str(bool(self.config.export_raw_residue_table))},
        ]
        return pd.DataFrame(rows)

    def run(self) -> None:
        """Run analysis."""
        self.logger.info("Starting Figure 7 local structural meta-analysis")
        self.jsonl_logger.write("start", {"output_dir": str(self.paths.output_dir)})
        self.validate_inputs()
        self.plotter.apply_publication_style()
        jackhmmer_df = self.loader.read_comparison_csv(self.loader.comparison_csv_path("jackhmmer"))
        tuned_tables = {float(target_fraction): self.loader.read_comparison_csv(self.loader.comparison_csv_path("tuned", float(target_fraction))) for target_fraction in self.config.target_fractions}
        common_pdb_ids, target_inventory_df = self.analyzer.build_common_target_inventory(tuned_tables, jackhmmer_df)
        self.logger.info("Built CSV-valid common target set with %s PDB IDs across %s target fractions", len(common_pdb_ids), len(self.config.target_fractions))
        self.analyzer.flush_logger_handlers()
        self.jsonl_logger.write("common_targets_done", {"n_common_pdb_ids": int(len(common_pdb_ids))})
        residue_df, pair_df, exclusion_df = self.analyzer.process_all_pairs(common_pdb_ids, jackhmmer_df, tuned_tables)
        self.logger.info("Local-error processing complete before complete-target filtering: retained_pairs=%s excluded_pairs=%s residue_rows_in_memory=%s", pair_df.shape[0], exclusion_df.shape[0], residue_df.shape[0])
        self.analyzer.flush_logger_handlers()
        local_common_inventory_df = pd.DataFrame()
        if bool(self.config.enforce_complete_local_target_set):
            residue_df, pair_df, local_common_inventory_df = self.analyzer.filter_to_complete_local_target_set(residue_df, pair_df)
            n_complete = int(local_common_inventory_df["is_complete_local_target"].sum()) if not local_common_inventory_df.empty else 0
            self.logger.info("Complete local target filtering retained %s PDB IDs, %s PDB-target pairs, and %s residue rows", n_complete, pair_df.shape[0], residue_df.shape[0])
            self.analyzer.flush_logger_handlers()
            self.jsonl_logger.write("complete_local_target_filter_done", {"n_complete_pdb_ids": n_complete, "n_retained_pairs": int(pair_df.shape[0]), "n_residue_rows": int(residue_df.shape[0])})
        summaries = self.analyzer.build_all_summaries(residue_df, pair_df)
        dictionary_tables = self.analyzer.build_dictionary_tables()
        plot_manifest_df = self.plotter.render_all_plots(summaries)
        notebook_tables: Dict[str, pd.DataFrame] = {"readme": self.build_readme_table(common_pdb_ids), "target_fraction_inventory": target_inventory_df, "csv_valid_common_target_inventory": pd.DataFrame({"pdb_id": list(common_pdb_ids)}), "local_common_inventory": local_common_inventory_df, "model_selection_table": pair_df, "pair_exclusion_log": exclusion_df, "plot_manifest": plot_manifest_df}
        notebook_tables.update(dictionary_tables)
        notebook_tables.update(summaries)
        if bool(self.config.export_raw_residue_table):
            notebook_tables["raw_residue_table"] = residue_df
        ExcelNotebookWriter(self.paths.output_xlsx_path, int(self.config.excel_chunk_rows)).write(notebook_tables)
        self.logger.info("Wrote Figure 7 meta-analysis workbook: %s", self.paths.output_xlsx_path)
        self.jsonl_logger.write("done", {"xlsx": str(self.paths.output_xlsx_path), "n_plots": int(plot_manifest_df.shape[0])})
        self.jsonl_logger.close()


def main() -> None:
    """Generate Figure 7 local structural meta-analysis plots and workbook."""
    Figure7MetaRunner(Figure7MetaConfig()).run()


if __name__ == "__main__":
    main()
