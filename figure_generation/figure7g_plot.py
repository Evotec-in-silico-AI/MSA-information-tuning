#!/usr/bin/env python3
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pymol2
from Bio.PDB import MMCIFParser
from prody import calcMSAOccupancy, calcShannonEntropy


PDB_IDS = ["7UUJ"]
INPUT_DIR = Path("7UUJ_example")
OUTPUT_DIR = Path("figure7_and_supplementary_figure5_outputs")
TARGET_LABELS = ["0", "10", "20", "30"]
EXPERIMENTAL_CHAIN_ID = "A"
PREDICTED_CHAIN_ID = None
FOLDING_BUDGET_BPS = 0.9720939469655248
RENDER_MODE = "overlay"
ALIGNMENT_METHOD = "super"
ROTATE_X_DEGREES = 0.0
ROTATE_Y_DEGREES = 120.0
ROTATE_Z_DEGREES = 0.0
PNG_WIDTH = 1800
PNG_HEIGHT = 1400
PNG_DPI = 300
RAY_TRACE = True
BACKGROUND_IS_TRANSPARENT = True
EXPERIMENTAL_TRANSPARENCY = 0.55
EXPERIMENTAL_COLOR_NAME = "figure7g_experimental_gray"
EXPERIMENTAL_COLOR_RGB = (0.72, 0.72, 0.72)
PREDICTED_ABOVE_COLOR_NAME = "figure7g_predicted_above_blue"
PREDICTED_ABOVE_COLOR_RGB = (0, 114, 178)
PREDICTED_BELOW_COLOR_NAME = "figure7g_predicted_below_pink"
PREDICTED_BELOW_COLOR_RGB = (204, 121, 167)
LOG_LEVEL = "INFO"
FAIL_IF_TARGET_FILE_MISSING = True


@dataclass(frozen=True)
class TargetSpec:
    """Represent one tuned-MSA target."""
    label: str
    percent: float
    token: str


@dataclass(frozen=True)
class ResidueRecord:
    """Represent one CA-containing residue in a structure chain."""
    chain_id: str
    residue_number: int
    residue_name: str


@dataclass(frozen=True)
class TargetPaths:
    """Store paths for one target condition."""
    predicted_cif: Path
    a3m_path: Path
    overlay_png: Path
    experimental_png: Path
    predicted_png: Path


def configure_logging() -> None:
    """Configure logging for the render run."""
    logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def build_target_specs() -> List[TargetSpec]:
    """Return all supported Figure 7 target specifications."""
    return [
        TargetSpec(label="0", percent=0.0, token="0"),
        TargetSpec(label="2", percent=2.0, token="0p02"),
        TargetSpec(label="5", percent=5.0, token="0p05"),
        TargetSpec(label="7.5", percent=7.5, token="0p075"),
        TargetSpec(label="10", percent=10.0, token="0p1"),
        TargetSpec(label="12.5", percent=12.5, token="0p125"),
        TargetSpec(label="15", percent=15.0, token="0p15"),
        TargetSpec(label="20", percent=20.0, token="0p2"),
        TargetSpec(label="25", percent=25.0, token="0p25"),
        TargetSpec(label="30", percent=30.0, token="0p3"),
    ]


def select_target_specs() -> List[TargetSpec]:
    """Return target specifications requested in the top-level config."""
    targets_by_label = {target.label: target for target in build_target_specs()}
    selected_targets: List[TargetSpec] = []
    for label in TARGET_LABELS:
        if label not in targets_by_label:
            raise ValueError(f"Unsupported target label in TARGET_LABELS: {label}")
        selected_targets.append(targets_by_label[label])
    if not selected_targets:
        raise ValueError("TARGET_LABELS cannot be empty.")
    return selected_targets


def load_a3m(path: Path) -> Tuple[List[str], List[str]]:
    """Load an A3M file into headers and sequences."""
    headers: List[str] = []
    sequences: List[str] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []
    with path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences.append("".join(current_lines))
                current_header = line[1:].strip()
                headers.append(current_header)
                current_lines = []
            else:
                current_lines.append(line)
    if current_header is not None:
        sequences.append("".join(current_lines))
    if not headers or not sequences or len(headers) != len(sequences):
        raise ValueError(f"Malformed A3M: {path}")
    return headers, sequences


def sanitize_a3m_sequences(sequences: Sequence[str]) -> List[str]:
    """Sanitize A3M rows and enforce equal-length rows."""
    cleaned_sequences: List[str] = []
    for sequence in sequences:
        sanitized_sequence = "".join(character for character in sequence if (("A" <= character <= "Z") or character == "-"))
        cleaned_sequences.append(sanitized_sequence)
    lengths = {len(sequence) for sequence in cleaned_sequences}
    if len(lengths) != 1:
        raise ValueError("Sanitization did not yield equal-length sequences.")
    return cleaned_sequences


def project_to_anchor_match_columns(sequences: Sequence[str]) -> List[str]:
    """Keep only columns where the first row is non-gap."""
    if not sequences:
        raise ValueError("Cannot project an empty alignment.")
    anchor_sequence = sequences[0]
    keep_indices = [index for index, character in enumerate(anchor_sequence) if character != "-"]
    if not keep_indices:
        raise ValueError("Anchor row contains no non-gap columns.")
    return ["".join(sequence[index] for index in keep_indices) for sequence in sequences]


def sequences_to_char_array(sequences: Sequence[str]) -> np.ndarray:
    """Convert projected alignment rows to a ProDy-compatible character array."""
    if not sequences:
        raise ValueError("Cannot convert an empty alignment to a character array.")
    row_count = len(sequences)
    column_count = len(sequences[0])
    char_array = np.empty((row_count, column_count), dtype="S1")
    for row_index, sequence in enumerate(sequences):
        char_array[row_index, :] = np.frombuffer(sequence.encode("ascii", "replace"), dtype="S1")
    return char_array


def compute_prody_bits_per_site(a3m_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ProDy bits/site and below-budget labels for one A3M."""
    headers, sequences = load_a3m(a3m_path)
    del headers
    sanitized_sequences = sanitize_a3m_sequences(sequences)
    projected_sequences = project_to_anchor_match_columns(sanitized_sequences)
    char_array = sequences_to_char_array(projected_sequences)
    entropy_nats = np.asarray(calcShannonEntropy(char_array), dtype=np.float64)
    bits_per_site = np.log2(20.0) - entropy_nats / np.log(2.0)
    occupancy = np.asarray(calcMSAOccupancy(char_array, occ="res"), dtype=np.float64)
    below_mask = bits_per_site < float(FOLDING_BUDGET_BPS)
    return bits_per_site, below_mask, occupancy


def parse_chain_residues(cif_path: Path, requested_chain_id: Optional[str]) -> Tuple[str, List[ResidueRecord]]:
    """Parse CA-containing polymer residues from one CIF chain."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(cif_path.stem, str(cif_path))
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if not chains:
        raise ValueError(f"No chains were found in {cif_path}")
    if requested_chain_id is None:
        chain = chains[0]
    else:
        matching_chains = [candidate for candidate in chains if candidate.id == requested_chain_id]
        if not matching_chains:
            available_chain_ids = [candidate.id for candidate in chains]
            raise ValueError(f"Requested chain {requested_chain_id} was not found in {cif_path}. Available chains: {available_chain_ids}")
        chain = matching_chains[0]
    residues: List[ResidueRecord] = []
    seen_residue_numbers = set()
    for residue in chain.get_residues():
        residue_id = residue.id
        hetero_flag = str(residue_id[0])
        residue_number = int(residue_id[1])
        insertion_code = str(residue_id[2]).strip()
        if hetero_flag != " ":
            continue
        if insertion_code:
            raise ValueError(f"Insertion codes are not supported for residue coloring: {cif_path}, chain {chain.id}, residue {residue_id}")
        if not residue.has_id("CA"):
            continue
        if residue_number in seen_residue_numbers:
            raise ValueError(f"Duplicate residue number {residue_number} in {cif_path}, chain {chain.id}")
        seen_residue_numbers.add(residue_number)
        residues.append(ResidueRecord(chain_id=str(chain.id), residue_number=residue_number, residue_name=str(residue.get_resname())))
    if not residues:
        raise ValueError(f"No CA-containing residues were found in {cif_path}, chain {chain.id}")
    return str(chain.id), residues


def validate_coloring_mapping(pdb_id: str, target: TargetSpec, bits_per_site: np.ndarray, predicted_residues: Sequence[ResidueRecord]) -> None:
    """Ensure projected MSA columns map one-to-one to predicted residues."""
    column_count = int(bits_per_site.size)
    residue_count = int(len(predicted_residues))
    if column_count != residue_count:
        raise ValueError(f"{pdb_id} target {target.label} has {column_count} projected A3M columns but {residue_count} predicted CA residues.")


def build_target_paths(pdb_id: str, target: TargetSpec) -> TargetPaths:
    """Build input and output paths for one target."""
    output_dir = OUTPUT_DIR / pdb_id
    return TargetPaths(
        predicted_cif=INPUT_DIR / f"{pdb_id}_target_{target.token}.cif",
        a3m_path=INPUT_DIR / f"{pdb_id}_target_{target.token}.a3m",
        overlay_png=output_dir / f"{pdb_id}_target_{target.token}_overlay.png",
        experimental_png=output_dir / f"{pdb_id}_target_{target.token}_experimental.png",
        predicted_png=output_dir / f"{pdb_id}_target_{target.token}_predicted.png",
    )


def validate_target_files(target_paths: TargetPaths) -> bool:
    """Validate target input files and return whether rendering should proceed."""
    missing_paths = [path for path in [target_paths.predicted_cif, target_paths.a3m_path] if not path.is_file()]
    if not missing_paths:
        return True
    message = f"Missing required target files: {[str(path) for path in missing_paths]}"
    if FAIL_IF_TARGET_FILE_MISSING:
        raise FileNotFoundError(message)
    logging.warning(message)
    return False


def build_residue_selection_terms(residue_numbers: Sequence[int]) -> List[str]:
    """Build PyMOL residue selection terms in manageable chunks."""
    chunk_size = 80
    terms: List[str] = []
    for start_index in range(0, len(residue_numbers), chunk_size):
        chunk = residue_numbers[start_index:start_index + chunk_size]
        if chunk:
            terms.append("+".join(str(number) for number in chunk))
    return terms


def set_pymol_colors(cmd) -> None:
    """Define figure-specific colors in PyMOL."""
    cmd.set_color(EXPERIMENTAL_COLOR_NAME, list(EXPERIMENTAL_COLOR_RGB))
    cmd.set_color(PREDICTED_ABOVE_COLOR_NAME, list(PREDICTED_ABOVE_COLOR_RGB))
    cmd.set_color(PREDICTED_BELOW_COLOR_NAME, list(PREDICTED_BELOW_COLOR_RGB))


def configure_pymol_scene(cmd) -> None:
    """Configure PyMOL rendering defaults."""
    cmd.set("opaque_background", 0 if BACKGROUND_IS_TRANSPARENT else 1)
    cmd.set("ray_opaque_background", 0 if BACKGROUND_IS_TRANSPARENT else 1)
    cmd.set("antialias", 2)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("orthoscopic", 1)
    cmd.bg_color("white")


def apply_experimental_view(cmd) -> Tuple[float, ...]:
    """Orient the camera from the experimental chain and return the view matrix."""
    cmd.orient("experimental_chain")
    if ROTATE_X_DEGREES:
        cmd.turn("x", float(ROTATE_X_DEGREES))
    if ROTATE_Y_DEGREES:
        cmd.turn("y", float(ROTATE_Y_DEGREES))
    if ROTATE_Z_DEGREES:
        cmd.turn("z", float(ROTATE_Z_DEGREES))
    cmd.zoom("experimental_chain", 8.0)
    return tuple(cmd.get_view())


def align_predicted_to_experiment(cmd) -> None:
    """Align predicted structure to the experimental structure."""
    method = ALIGNMENT_METHOD.lower().strip()
    if method == "super":
        cmd.super("predicted_chain", "experimental_chain")
    elif method == "align":
        cmd.align("predicted_chain", "experimental_chain")
    elif method == "cealign":
        cmd.cealign("experimental_chain", "predicted_chain")
    else:
        raise ValueError(f"Unsupported ALIGNMENT_METHOD: {ALIGNMENT_METHOD}")


def color_predicted_by_bits(cmd, predicted_chain_id: str, predicted_residues: Sequence[ResidueRecord], below_mask: np.ndarray) -> None:
    """Color predicted residues by below-budget status."""
    cmd.color(PREDICTED_ABOVE_COLOR_NAME, "predicted_chain")
    below_residue_numbers = [residue.residue_number for residue, is_below in zip(predicted_residues, below_mask) if bool(is_below)]
    for residue_term in build_residue_selection_terms(below_residue_numbers):
        selection = f"predicted_chain and chain {predicted_chain_id} and resi {residue_term}"
        cmd.color(PREDICTED_BELOW_COLOR_NAME, selection)


def prepare_loaded_objects(cmd, experimental_chain_id: str, predicted_chain_id: str) -> None:
    """Create named selections and base representations."""
    cmd.select("experimental_chain", f"experimental and chain {experimental_chain_id} and polymer.protein")
    cmd.select("predicted_chain", f"predicted and chain {predicted_chain_id} and polymer.protein")
    if cmd.count_atoms("experimental_chain and name CA") == 0:
        raise ValueError("PyMOL found no CA atoms in experimental_chain.")
    if cmd.count_atoms("predicted_chain and name CA") == 0:
        raise ValueError("PyMOL found no CA atoms in predicted_chain.")
    cmd.hide("everything")
    cmd.show("cartoon", "experimental_chain")
    cmd.show("cartoon", "predicted_chain")
    cmd.color(EXPERIMENTAL_COLOR_NAME, "experimental_chain")
    cmd.set("cartoon_transparency", float(EXPERIMENTAL_TRANSPARENCY), "experimental_chain")


def render_png(cmd, output_path: Path) -> None:
    """Render the current PyMOL scene to PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd.png(str(output_path), width=int(PNG_WIDTH), height=int(PNG_HEIGHT), dpi=int(PNG_DPI), ray=1 if RAY_TRACE else 0)


def render_outputs(cmd, target_paths: TargetPaths, view: Tuple[float, ...]) -> None:
    """Render configured output modes for the current target."""
    if RENDER_MODE in {"overlay", "both"}:
        cmd.hide("everything")
        cmd.show("cartoon", "experimental_chain")
        cmd.show("cartoon", "predicted_chain")
        cmd.set_view(view)
        render_png(cmd, target_paths.overlay_png)
    if RENDER_MODE in {"separate", "both"}:
        cmd.hide("everything")
        cmd.show("cartoon", "experimental_chain")
        cmd.set_view(view)
        render_png(cmd, target_paths.experimental_png)
        cmd.hide("everything")
        cmd.show("cartoon", "predicted_chain")
        cmd.set_view(view)
        render_png(cmd, target_paths.predicted_png)
    if RENDER_MODE not in {"overlay", "separate", "both"}:
        raise ValueError(f"Unsupported RENDER_MODE: {RENDER_MODE}")


def build_metadata_row(pdb_id: str, target: TargetSpec, target_paths: TargetPaths, predicted_chain_id: str, experimental_residues: Sequence[ResidueRecord], predicted_residues: Sequence[ResidueRecord], bits_per_site: np.ndarray, below_mask: np.ndarray, occupancy: np.ndarray) -> Dict[str, object]:
    """Build one metadata row for a rendered target."""
    below_count = int(np.sum(below_mask))
    projected_columns = int(bits_per_site.size)
    return {
        "pdb_id": pdb_id,
        "target_label": target.label,
        "target_percent": target.percent,
        "target_token": target.token,
        "predicted_cif": str(target_paths.predicted_cif),
        "a3m_path": str(target_paths.a3m_path),
        "experimental_chain_id": EXPERIMENTAL_CHAIN_ID,
        "predicted_chain_id": predicted_chain_id,
        "experimental_ca_residues": int(len(experimental_residues)),
        "predicted_ca_residues": int(len(predicted_residues)),
        "projected_columns": projected_columns,
        "columns_below_folding_budget": below_count,
        "fraction_columns_below_folding_budget": float(below_count / projected_columns) if projected_columns > 0 else float("nan"),
        "bps_min": float(np.min(bits_per_site)),
        "bps_p1": float(np.quantile(bits_per_site, 0.01)),
        "bps_p10": float(np.quantile(bits_per_site, 0.10)),
        "bps_median": float(np.median(bits_per_site)),
        "bps_mean": float(np.mean(bits_per_site)),
        "gap_fraction": float(np.mean(1.0 - occupancy)),
        "overlay_png": str(target_paths.overlay_png) if RENDER_MODE in {"overlay", "both"} else "",
        "experimental_png": str(target_paths.experimental_png) if RENDER_MODE in {"separate", "both"} else "",
        "predicted_png": str(target_paths.predicted_png) if RENDER_MODE in {"separate", "both"} else "",
    }


def write_metadata(pdb_id: str, metadata_rows: Sequence[Dict[str, object]]) -> None:
    """Write per-PDB render metadata."""
    if not metadata_rows:
        return
    metadata_path = OUTPUT_DIR / pdb_id / f"{pdb_id}_render_metadata.csv"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)
    logging.info(f"Wrote metadata: {metadata_path.resolve()}")


def render_one_target(cmd, pdb_id: str, target: TargetSpec, experimental_cif: Path, experimental_residues: Sequence[ResidueRecord], base_view: Tuple[float, ...]) -> Dict[str, object]:
    """Render one target condition and return metadata."""
    target_paths = build_target_paths(pdb_id, target)
    if not validate_target_files(target_paths):
        return {}
    bits_per_site, below_mask, occupancy = compute_prody_bits_per_site(target_paths.a3m_path)
    predicted_chain_id, predicted_residues = parse_chain_residues(target_paths.predicted_cif, PREDICTED_CHAIN_ID)
    validate_coloring_mapping(pdb_id, target, bits_per_site, predicted_residues)
    cmd.delete("predicted")
    cmd.delete("predicted_chain")
    cmd.load(str(target_paths.predicted_cif), "predicted")
    prepare_loaded_objects(cmd, EXPERIMENTAL_CHAIN_ID, predicted_chain_id)
    align_predicted_to_experiment(cmd)
    color_predicted_by_bits(cmd, predicted_chain_id, predicted_residues, below_mask)
    cmd.set_view(base_view)
    render_outputs(cmd, target_paths, base_view)
    logging.info(f"Rendered {pdb_id} target {target.label}")
    return build_metadata_row(pdb_id, target, target_paths, predicted_chain_id, experimental_residues, predicted_residues, bits_per_site, below_mask, occupancy)


def render_one_pdb(pymol_instance, pdb_id: str, targets: Sequence[TargetSpec]) -> None:
    """Render all configured targets for one PDB ID."""
    experimental_cif = INPUT_DIR / f"{pdb_id}_exp.cif"
    if not experimental_cif.is_file():
        raise FileNotFoundError(f"Missing experimental CIF: {experimental_cif}")
    experimental_chain_id, experimental_residues = parse_chain_residues(experimental_cif, EXPERIMENTAL_CHAIN_ID)
    if experimental_chain_id != EXPERIMENTAL_CHAIN_ID:
        raise ValueError(f"Experimental chain mismatch for {pdb_id}: expected {EXPERIMENTAL_CHAIN_ID}, got {experimental_chain_id}")
    cmd = pymol_instance.cmd
    cmd.reinitialize()
    set_pymol_colors(cmd)
    configure_pymol_scene(cmd)
    cmd.load(str(experimental_cif), "experimental")
    cmd.select("experimental_chain", f"experimental and chain {EXPERIMENTAL_CHAIN_ID} and polymer.protein")
    cmd.hide("everything")
    cmd.show("cartoon", "experimental_chain")
    cmd.color(EXPERIMENTAL_COLOR_NAME, "experimental_chain")
    cmd.set("cartoon_transparency", float(EXPERIMENTAL_TRANSPARENCY), "experimental_chain")
    base_view = apply_experimental_view(cmd)
    metadata_rows: List[Dict[str, object]] = []
    for target in targets:
        metadata_row = render_one_target(cmd, pdb_id, target, experimental_cif, experimental_residues, base_view)
        if metadata_row:
            metadata_rows.append(metadata_row)
    write_metadata(pdb_id, metadata_rows)


def validate_top_level_config() -> None:
    """Validate top-level configuration values."""
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Missing INPUT_DIR: {INPUT_DIR}")
    if not PDB_IDS:
        raise ValueError("PDB_IDS cannot be empty.")
    if RENDER_MODE not in {"overlay", "separate", "both"}:
        raise ValueError("RENDER_MODE must be overlay, separate, or both.")
    if not 0.0 <= float(EXPERIMENTAL_TRANSPARENCY) <= 1.0:
        raise ValueError("EXPERIMENTAL_TRANSPARENCY must be between 0 and 1.")
    if int(PNG_WIDTH) <= 0 or int(PNG_HEIGHT) <= 0 or int(PNG_DPI) <= 0:
        raise ValueError("PNG_WIDTH, PNG_HEIGHT, and PNG_DPI must be positive.")


def main() -> None:
    """Run the full PyMOL rendering workflow."""
    configure_logging()
    validate_top_level_config()
    targets = select_target_specs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with pymol2.PyMOL() as pymol_instance:
        for pdb_id in PDB_IDS:
            render_one_pdb(pymol_instance, pdb_id, targets)


if __name__ == "__main__":
    main()
