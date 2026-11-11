import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


operational_bits_site_threshold = 0.9720939469655248


@dataclass
class SheetNamePolicy:
    """Create safe, unique Excel sheet names."""

    max_len: int = 31

    def make_safe(self, proposed_name: str) -> str:
        """Return a valid unique Excel sheet name."""
        if not hasattr(self, "used_names"):
            self.used_names: Dict[str, int] = {}
        cleaned_name = re.sub(r"[:\\/?*\[\]]", "_", str(proposed_name)).strip()
        if not cleaned_name:
            cleaned_name = "Sheet"
        cleaned_name = cleaned_name[: int(self.max_len)].strip("'")
        if not cleaned_name:
            cleaned_name = "Sheet"
        base_name = cleaned_name
        if base_name not in self.used_names:
            self.used_names[base_name] = 1
            return base_name
        counter_value = self.used_names[base_name]
        while True:
            suffix = f"_{counter_value}"
            keep_length = max(1, int(self.max_len) - len(suffix))
            candidate_name = (base_name[:keep_length] + suffix)[: int(self.max_len)]
            if candidate_name not in self.used_names:
                self.used_names[base_name] = counter_value + 1
                self.used_names[candidate_name] = 1
                return candidate_name
            counter_value += 1


@dataclass
class ExcelNotebookWriter:
    """Write a compact multi-sheet Excel source-data workbook."""

    xlsx_path: Path
    chunk_rows: int = 1000000
    include_inventory: bool = False

    def write(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Write all non-empty tables to workbook sheets."""
        output_path = Path(self.xlsx_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet_policy = SheetNamePolicy(max_len=31)
        inventory_rows: List[Dict[str, Any]] = []
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for table_name, table_value in tables.items():
                frame = table_value.copy() if isinstance(table_value, pd.DataFrame) else pd.DataFrame(table_value)
                if frame.shape[0] == 0:
                    continue
                self.write_frame(writer, sheet_policy, str(table_name), frame, inventory_rows)
            if bool(self.include_inventory) and inventory_rows:
                self.write_frame(writer, sheet_policy, "export_inventory", pd.DataFrame(inventory_rows), [])

    def write_frame(self, writer: pd.ExcelWriter, sheet_policy: SheetNamePolicy, table_name: str, frame: pd.DataFrame, inventory_rows: List[Dict[str, Any]]) -> None:
        """Write one dataframe, splitting it into parts when needed."""
        row_count = int(frame.shape[0])
        if row_count <= int(self.chunk_rows):
            sheet_name = sheet_policy.make_safe(table_name)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            inventory_rows.append({"sheet_base": str(table_name), "rows": row_count, "parts": 1})
            return
        part_count = int(math.ceil(row_count / float(self.chunk_rows)))
        for part_index in range(part_count):
            low_index = int(part_index * int(self.chunk_rows))
            high_index = int(min((part_index + 1) * int(self.chunk_rows), row_count))
            part_frame = frame.iloc[low_index:high_index].copy()
            sheet_name = sheet_policy.make_safe(f"{table_name}_p{part_index + 1}")
            part_frame.to_excel(writer, sheet_name=sheet_name, index=False)
        inventory_rows.append({"sheet_base": str(table_name), "rows": row_count, "parts": part_count})


@dataclass(frozen=True)
class ConditionStyle:
    """Provide fixed colors and ordering for manuscript condition labels."""

    def blend_hex_with_white(self, hex_color: str, blend_factor: float) -> str:
        """Lighten a hex color by blending it with white."""
        color_text = str(hex_color).lstrip("#")
        red_value = int(color_text[0:2], 16)
        green_value = int(color_text[2:4], 16)
        blue_value = int(color_text[4:6], 16)
        red_out = int(round(red_value + (255 - red_value) * float(blend_factor)))
        green_out = int(round(green_value + (255 - green_value) * float(blend_factor)))
        blue_out = int(round(blue_value + (255 - blue_value) * float(blend_factor)))
        return "#{:02X}{:02X}{:02X}".format(red_out, green_out, blue_out)

    def get_order(self) -> List[str]:
        """Return the canonical condition order."""
        return ["below targeted", "below matched control", "above targeted", "above matched control"]

    def get_palette(self) -> Dict[str, str]:
        """Return the canonical condition palette."""
        orange_dark = "#E69F00"
        return {
            "below targeted": "#56B4E9",
            "below matched control": self.blend_hex_with_white(orange_dark, 0.35),
            "above targeted": "#0072B2",
            "above matched control": orange_dark,
        }

    def get_color(self, condition: str) -> str:
        """Return a condition color."""
        return self.get_palette().get(str(condition), "#444444")


def require_columns(frame: pd.DataFrame, required_columns: Sequence[str], context: str) -> None:
    """Fail when required columns are absent."""
    missing_columns = [str(column_name) for column_name in required_columns if str(column_name) not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {context}: {missing_columns}")



def convert_stored_information_to_bits(values: pd.Series) -> pd.Series:
    """Convert archived ProDy-derived information coordinates to bits/site."""
    numeric_values = pd.to_numeric(values, errors="raise")
    maximum_bits = math.log2(20.0)
    converted = maximum_bits - (maximum_bits - numeric_values) / math.log(2.0)
    finite_values = converted[np.isfinite(converted)]
    tolerance = 1e-6
    if ((finite_values < -tolerance) | (finite_values > maximum_bits + tolerance)).any():
        raise ValueError("Corrected information values fall outside the 20-state range")
    return converted


def convert_stored_effective_alphabet(values: pd.Series) -> pd.Series:
    """Convert archived effective-alphabet values derived from entropy in nats."""
    numeric_values = pd.to_numeric(values, errors="raise")
    converted = np.power(numeric_values, 1.0 / math.log(2.0))
    finite_values = converted[np.isfinite(converted)]
    tolerance = 1e-6
    if ((finite_values < 1.0 - tolerance) | (finite_values > 20.0 + tolerance)).any():
        raise ValueError("Corrected effective-alphabet values fall outside [1, 20]")
    return converted


def convert_archived_columns(frame: pd.DataFrame, information_columns: Sequence[str] = (), effective_alphabet_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Convert explicitly named archived columns for manuscript-facing analysis."""
    required_columns = list(information_columns) + list(effective_alphabet_columns)
    require_columns(frame, required_columns, "archived analysis table")
    out = frame.copy()
    for column_name in information_columns:
        out[column_name] = convert_stored_information_to_bits(out[column_name])
    for column_name in effective_alphabet_columns:
        out[column_name] = convert_stored_effective_alphabet(out[column_name])
    return out

def strip_null_outer_label_suffix(outer_label: str) -> str:
    """Strip a null-control suffix from an outer label."""
    delimiter = "_" * 2 + "null_"
    label_text = str(outer_label)
    if delimiter in label_text:
        return label_text.split(delimiter, 1)[0]
    return label_text


def add_threshold_condition_columns(frame: pd.DataFrame, threshold: float = operational_bits_site_threshold) -> pd.DataFrame:
    """Add targeted-regime condition labels used across manuscript figures."""
    required_columns = ["protein_name", "outer_label", "control_type", "bits_site_min"]
    require_columns(frame, required_columns, "condition-label input")
    out = frame.copy()
    out["protein_name"] = out["protein_name"].astype(str)
    out["outer_label"] = out["outer_label"].astype(str)
    out["control_type"] = out["control_type"].astype(str).str.lower()
    out["bits_site_min"] = pd.to_numeric(out["bits_site_min"], errors="coerce")
    out["outer_label_base"] = out["outer_label"].map(strip_null_outer_label_suffix)
    match_columns = ["protein_name", "outer_label_base"]
    targeted = out[out["control_type"] == "targeted"].copy()
    if targeted.empty:
        raise ValueError("No targeted rows are available for condition-label assignment")
    targeted_bits = targeted.groupby(match_columns, sort=False)["bits_site_min"].agg(["first", "nunique"]).reset_index()
    inconsistent = targeted_bits[pd.to_numeric(targeted_bits["nunique"], errors="coerce").fillna(0).astype(int) > 1]
    if not inconsistent.empty:
        examples = inconsistent[match_columns].head(10).to_dict(orient="records")
        raise ValueError(f"Targeted rows disagree on bits_site_min within matched groups: {examples}")
    targeted_bits = targeted_bits.rename(columns={"first": "target_bits_site_min"})
    out = out.merge(targeted_bits[match_columns + ["target_bits_site_min"]], on=match_columns, how="left", validate="many_to_one")
    if out["target_bits_site_min"].isna().any():
        missing_keys = out.loc[out["target_bits_site_min"].isna(), match_columns].drop_duplicates().head(10)
        raise ValueError(f"Missing targeted match for condition labels. Example keys: {missing_keys.to_dict(orient='records')}")
    out["target_stratum"] = np.where(pd.to_numeric(out["target_bits_site_min"], errors="coerce") < float(threshold), "below", "above")
    targeted_mask = out["control_type"] == "targeted"
    out["condition4"] = np.where(targeted_mask, out["target_stratum"] + " targeted", out["target_stratum"] + " matched control")
    out["condition_match_key"] = out["protein_name"].astype(str) + "::" + out["outer_label_base"].astype(str)
    return out


def safe_mean(values: Sequence[float]) -> float:
    """Return finite mean or NaN."""
    value_array = np.asarray(values, dtype=float)
    finite_values = value_array[np.isfinite(value_array)]
    return float(np.mean(finite_values)) if finite_values.size else float("nan")


def safe_sample_sd(values: Sequence[float]) -> float:
    """Return finite sample standard deviation or NaN."""
    value_array = np.asarray(values, dtype=float)
    finite_values = value_array[np.isfinite(value_array)]
    return float(np.std(finite_values, ddof=1)) if finite_values.size >= 2 else float("nan")



def compute_spearman_with_finite_filter(x_values: Sequence[float], y_values: Sequence[float], minimum_count: int = 3) -> Tuple[float, float, int]:
    """Compute Spearman correlation after finite-value filtering."""
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(x_array) & np.isfinite(y_array)
    pair_count = int(finite_mask.sum())
    if pair_count < int(minimum_count):
        return float("nan"), float("nan"), pair_count
    rho_value, p_value = stats.spearmanr(x_array[finite_mask], y_array[finite_mask])
    return float(rho_value), float(p_value), pair_count

def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """Compute Benjamini-Hochberg q-values while preserving NaNs."""
    p_array = np.asarray(p_values, dtype=float)
    q_array = np.full(p_array.shape, np.nan, dtype=float)
    finite_indices = np.where(np.isfinite(p_array))[0]
    if finite_indices.size == 0:
        return q_array.tolist()
    finite_p = p_array[finite_indices]
    order = np.argsort(finite_p)
    sorted_p = finite_p[order]
    sorted_q = np.empty_like(sorted_p, dtype=float)
    running_minimum = 1.0
    total_count = float(sorted_p.size)
    for reverse_index in range(sorted_p.size - 1, -1, -1):
        rank = float(reverse_index + 1)
        adjusted_value = float(sorted_p[reverse_index] * total_count / rank)
        running_minimum = min(running_minimum, adjusted_value)
        sorted_q[reverse_index] = min(running_minimum, 1.0)
    restored_q = np.empty_like(sorted_q, dtype=float)
    restored_q[order] = sorted_q
    q_array[finite_indices] = restored_q
    return q_array.tolist()


def save_figure_pair(fig: plt.Figure, output_dir: Path, stem: str, dpi: int = 300, bbox_inches: str = "tight") -> None:
    """Save SVG and PNG copies of a Matplotlib figure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / f"{stem}.svg", dpi=int(dpi), bbox_inches=bbox_inches)
    fig.savefig(output_path / f"{stem}.png", dpi=int(dpi), bbox_inches=bbox_inches)
    plt.close(fig)
