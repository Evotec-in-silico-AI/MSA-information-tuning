#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from prody import calcMSAOccupancy, calcShannonEntropy

JACKHMMER_FOLDING_BUDGET_BPS = 2.0


def configure_logging(level: str) -> None:
    """Configure root logging with the given level string."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')


def resolve_paths(mode: str, below_folding_budget_val: Optional[str]) -> Tuple[Path, Path, Path, Optional[Path]]:
    """Return (experimental_dir, results_dir, models_root, msa_dir) for mode and value."""
    root = Path('.').resolve()
    experimental_dir = root / 'experimental_structures'
    results_dir = root / 'results'
    if mode == 'no_msa':
        models_root = root / 'chai1_models_no_msa'
        msa_dir = None
    elif mode == 'jackhmmer':
        models_root = root / 'chai1_models_jackhmmer'
        msa_dir = root / 'msas'
    elif mode == 'below_folding_budget':
        if not below_folding_budget_val:
            raise ValueError('below_folding_budget mode requires --below_folding_budget_val')
        tuned = root / f'below_folding_budget_{below_folding_budget_val}'
        models_root = tuned / 'chai1_models'
        msa_dir = tuned / 'msas'
    else:
        raise ValueError(f'Unsupported mode: {mode}')
    return experimental_dir, results_dir, models_root, msa_dir


def list_experimental(experimental_dir: Path) -> List[Path]:
    """List experimental CIF files under the directory."""
    if not experimental_dir.is_dir():
        logging.error(f'Missing experimental structures directory: {experimental_dir}')
        return []
    return sorted(experimental_dir.glob('*.cif'))


def model_structure_path(models_dir: Path, index: int) -> Optional[Path]:
    """Return CIF or PDB path for a given model index if present."""
    cif = models_dir / f'pred.model_idx_{index}.cif'
    pdb = models_dir / f'pred.model_idx_{index}.pdb'
    if cif.is_file():
        return cif
    if pdb.is_file():
        return pdb
    return None


def scores_npz_path(models_dir: Path, index: int) -> Optional[Path]:
    """Return scores.model_idx_{index}.npz path if present."""
    path = models_dir / f'scores.model_idx_{index}.npz'
    return path if path.is_file() else None


def normalize_ws(line: str) -> List[str]:
    """Split a text line by whitespace while collapsing repeats."""
    return [token for token in re.split(r'\s+', line.strip()) if token]


def parse_outfmt2(stdout: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse TMscore -outfmt 2 output and return (TM2, RMSD) when available."""
    header_cols: List[str] = []
    tm2_index: Optional[int] = None
    rmsd_index: Optional[int] = None
    lines = stdout.splitlines()
    for line in lines:
        if line.lstrip().startswith('#'):
            header_cols = normalize_ws(line.lstrip('#'))
            break
    if header_cols:
        name_to_index = {name: i for i, name in enumerate(header_cols)}
        tm2_index = name_to_index.get('TM2')
        rmsd_index = name_to_index.get('RMSD')
    if tm2_index is not None and rmsd_index is not None:
        for line in lines:
            if line.lstrip().startswith('#') or not line.strip():
                continue
            cols = normalize_ws(line)
            if len(cols) > max(tm2_index, rmsd_index):
                try:
                    tm2 = float(cols[tm2_index])
                    rmsd = float(cols[rmsd_index])
                    if 0.0 <= tm2 <= 1.0:
                        return tm2, rmsd
                except Exception:
                    continue
    match_tm = re.search(r'TM-score\s*=\s*([0-9]*\.?[0-9]+)', stdout)
    match_rmsd = re.search(r'RMSD\s*=\s*([0-9]*\.?[0-9]+)', stdout)
    tm_val = float(match_tm.group(1)) if match_tm else None
    rmsd_val = float(match_rmsd.group(1)) if match_rmsd else None
    return tm_val, rmsd_val


def run_tmscore(exec_path: str, query_model: Path, experimental_cif: Path) -> Tuple[Optional[float], Optional[float]]:
    """Execute TMscore and parse its output as (TM2, RMSD)."""
    command = [exec_path, str(query_model), str(experimental_cif), '-outfmt', '2']
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    except Exception as exc:
        logging.error(f'Failed TM-score run: {exc}')
        return None, None
    tm, rmsd = parse_outfmt2(result.stdout or '')
    return tm, rmsd


def load_ptm(npz_path: Path) -> Optional[float]:
    """Load the pTM scalar from a scores.model_idx_{k}.npz file."""
    try:
        data = np.load(npz_path)
        if 'ptm' in data.files:
            vec = np.asarray(data['ptm']).flatten()
            return float(vec[0]) if vec.size else None
        return None
    except Exception as exc:
        logging.info(f'Failed to load pTM from {npz_path}: {exc}')
        return None


def count_a3m_sequences(msa_dir: Optional[Path], pdb_id: str) -> Optional[int]:
    """Count sequences in the first {pdb_id}*.a3m match under msa_dir."""
    if msa_dir is None or not msa_dir.is_dir():
        return None
    matches = sorted(msa_dir.glob(f'{pdb_id}*.a3m'))
    if not matches:
        return None
    path = matches[0]
    try:
        count = 0
        with path.open('r') as handle:
            for line in handle:
                if line.startswith('>'):
                    count += 1
        return count
    except Exception:
        return None


def find_first_a3m_path(msa_dir: Optional[Path], pdb_id: str) -> Optional[Path]:
    """Return the first matching A3M path for a pdb_id if present."""
    if msa_dir is None or not msa_dir.is_dir():
        return None
    matches = sorted(msa_dir.glob(f'{pdb_id}*.a3m'))
    if not matches:
        return None
    return matches[0]


def load_a3m(path: Path) -> Tuple[List[str], List[str]]:
    """Load an A3M file into headers and sequences while preserving case and gaps."""
    headers: List[str] = []
    sequences: List[str] = []
    current_header: Optional[str] = None
    current_lines: List[str] = []
    with path.open('r') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    sequences.append(''.join(current_lines))
                    current_lines = []
                current_header = line[1:].strip()
                headers.append(current_header)
            else:
                current_lines.append(line)
    if current_header is not None:
        sequences.append(''.join(current_lines))
    if not headers or not sequences or len(headers) != len(sequences):
        raise ValueError(f'Malformed A3M: {path}')
    return headers, sequences


def sanitize_a3m_sequences(sequences: List[str]) -> List[str]:
    """Sanitize A3M sequences and enforce equal-length rows."""
    cleaned_sequences: List[str] = []
    for sequence in sequences:
        sanitized_sequence = ''.join(char for char in sequence if (('A' <= char <= 'Z') or char == '-'))
        cleaned_sequences.append(sanitized_sequence)
    lengths = {len(sequence) for sequence in cleaned_sequences}
    if len(lengths) != 1:
        raise ValueError('Sanitization did not yield equal-length sequences.')
    return cleaned_sequences


def project_to_anchor_match_columns(sequences: List[str]) -> List[str]:
    """Keep only columns where the first sequence is not a gap."""
    if not sequences:
        return []
    anchor_sequence = sequences[0]
    keep_indices = [index for index, char in enumerate(anchor_sequence) if char != '-']
    return [''.join(sequence[index] for index in keep_indices) for sequence in sequences]


def sequences_to_char_array(sequences: List[str]) -> np.ndarray:
    """Return a 2D byte-character array for ProDy metrics."""
    if not sequences:
        return np.zeros((0, 0), dtype='S1')
    row_count = len(sequences)
    column_count = len(sequences[0])
    out = np.empty((row_count, column_count), dtype='S1')
    for row_index, sequence in enumerate(sequences):
        out[row_index, :] = np.frombuffer(sequence.encode('ascii', 'replace'), dtype='S1')
    return out


def compute_unweighted_bits_metrics_prody(sequences: List[str], folding_budget_bps: float) -> Dict[str, float]:
    """Compute unweighted bits-per-site metrics after projection."""
    char_array = sequences_to_char_array(sequences)
    entropy_bits = np.asarray(calcShannonEntropy(char_array), dtype=np.float64)
    bits_per_site = np.log2(20.0) - entropy_bits
    occupancy = np.asarray(calcMSAOccupancy(char_array, occ='res'), dtype=np.float64)
    columns_total = int(bits_per_site.size)
    columns_below_budget = int(np.sum(bits_per_site < float(folding_budget_bps))) if bits_per_site.size else 0
    fraction_below_budget = float(columns_below_budget / columns_total) if columns_total > 0 else float('nan')
    gap_fraction = float(np.mean(1.0 - occupancy)) if occupancy.size else float('nan')
    return {
        'bps_min': float(np.min(bits_per_site)) if bits_per_site.size else float('nan'),
        'bps_p1': float(np.quantile(bits_per_site, 0.01)) if bits_per_site.size else float('nan'),
        'bps_p10': float(np.quantile(bits_per_site, 0.10)) if bits_per_site.size else float('nan'),
        'bps_median': float(np.median(bits_per_site)) if bits_per_site.size else float('nan'),
        'bps_mean': float(np.mean(bits_per_site)) if bits_per_site.size else float('nan'),
        'gap_fraction': gap_fraction,
        'projected_columns': float(columns_total),
        'columns_below_folding_budget': float(columns_below_budget),
        'fraction_columns_below_folding_budget': fraction_below_budget,
    }


def compute_jackhmmer_msa_metrics(msa_dir: Optional[Path], pdb_id: str) -> Dict[str, object]:
    """Compute projected bits-per-site metrics for a JackHMMER A3M."""
    msa_path = find_first_a3m_path(msa_dir, pdb_id)
    if msa_path is None:
        return {}
    try:
        headers, sequences = load_a3m(msa_path)
        sanitized_sequences = sanitize_a3m_sequences(sequences)
        projected_sequences = project_to_anchor_match_columns(sanitized_sequences)
        if len(projected_sequences) == 0:
            return {}
        bits_metrics = compute_unweighted_bits_metrics_prody(projected_sequences, folding_budget_bps=JACKHMMER_FOLDING_BUDGET_BPS)
        return bits_metrics
    except Exception as exc:
        logging.info(f'Failed JackHMMER MSA metric calculation for {pdb_id}: {exc}')
        return {}


def read_below_folding_budget_json_if_any(mode: str, below_folding_budget_val: Optional[str], pdb_id: str) -> Dict[str, object]:
    """Read the below_folding_budget JSON sidecar for the pdb_id when in below_folding_budget mode."""
    if mode != 'below_folding_budget' or not below_folding_budget_val:
        return {}
    path = Path(f'./below_folding_budget_{below_folding_budget_val}/json/{pdb_id}.json')
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def build_output_csv_path(results_dir: Path, mode: str, below_folding_budget_val: Optional[str]) -> Path:
    """Return the results CSV path for the mode, adding the value for below_folding_budget."""
    if mode == 'no_msa':
        return results_dir / 'chai1_structural_comparison_no_msa.csv'
    if mode == 'jackhmmer':
        return results_dir / 'chai1_structural_comparison_jackhmmer.csv'
    return results_dir / f'chai1_structural_comparison_below_folding_budget_{below_folding_budget_val}.csv'


def process_one_target(pdb_id: str, models_root: Path, experimental_cif: Path, msa_dir: Optional[Path]) -> Dict[str, str]:
    """Compute per-model metrics and aggregates for one pdb_id and return a CSV row dict."""
    row: Dict[str, str] = {'pdb_id': pdb_id}
    tm_values: List[float] = []
    rmsd_values: List[float] = []
    ptm_values: List[float] = []
    models_dir = models_root / pdb_id
    for i in range(5):
        tm_col = f'tm_score_for_model_{i + 1}'
        rmsd_col = f'rmsd_for_model_{i + 1}'
        ptm_col = f'ptm_for_model_{i + 1}'
        structure_path = model_structure_path(models_dir, i)
        if structure_path is None:
            row[tm_col] = ''
            row[rmsd_col] = ''
        else:
            tm, rmsd = run_tmscore(os.environ.get('TMSCORE_EXEC', '/home/grichardson/TMscore'), structure_path, experimental_cif)
            row[tm_col] = '' if tm is None else f'{tm:.6f}'
            row[rmsd_col] = '' if rmsd is None else f'{rmsd:.6f}'
            if tm is not None:
                tm_values.append(tm)
            if rmsd is not None:
                rmsd_values.append(rmsd)
        npz_path = scores_npz_path(models_dir, i)
        if npz_path is None:
            row[ptm_col] = ''
        else:
            ptm = load_ptm(npz_path)
            row[ptm_col] = '' if ptm is None else f'{ptm:.6f}'
            if ptm is not None:
                ptm_values.append(ptm)
    row['num_sequences_in_alignment'] = '' if msa_dir is None else str(count_a3m_sequences(msa_dir, pdb_id) or '')
    row['mean_tm_score'] = '' if not tm_values else f"{float(np.mean(tm_values)):.6f}"
    row['std_dev_tm_score'] = '' if not tm_values else f"{float(np.std(tm_values)):.6f}"
    row['mean_rmsd'] = '' if not rmsd_values else f"{float(np.mean(rmsd_values)):.6f}"
    row['std_dev_rmsd'] = '' if not rmsd_values else f"{float(np.std(rmsd_values)):.6f}"
    row['mean_ptm'] = '' if not ptm_values else f"{float(np.mean(ptm_values)):.6f}"
    row['std_dev_ptm'] = '' if not ptm_values else f"{float(np.std(ptm_values)):.6f}"
    return row


def append_sidecar_fields(row: Dict[str, str], fields: Dict[str, object]) -> Dict[str, str]:
    """Append sidecar or computed fields to a CSV row dict, string-formatting values."""
    if not fields:
        return row
    out = dict(row)
    for key, value in fields.items():
        try:
            if isinstance(value, float):
                out[key] = f'{value:.6f}'
            elif isinstance(value, (int, str)):
                out[key] = str(value)
            else:
                out[key] = json.dumps(value)
        except Exception:
            out[key] = ''
    return out


def main() -> None:
    """Entry point to assemble per-target metrics into a mode-specific CSV."""
    parser = argparse.ArgumentParser(description='Compare Chai-1 models to experimental structures and assemble CSV results.')
    parser.add_argument('--mode', required=True, choices=['no_msa', 'jackhmmer', 'below_folding_budget'], help='Mode of models to compare')
    parser.add_argument('--below_folding_budget_val', type=str, default=None, help='Value like 0p1 mapping to below_folding_budget_0p1 directory')
    parser.add_argument('--tmscore-exec', type=str, default='/home/grichardson/TMscore', help='Path to TMscore binary')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args()
    configure_logging(args.log_level)
    experimental_dir, results_dir, models_root, msa_dir = resolve_paths(args.mode, args.below_folding_budget_val)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = build_output_csv_path(results_dir, args.mode, args.below_folding_budget_val)
    os.environ['TMSCORE_EXEC'] = args.tmscore_exec
    experimental_files = list_experimental(experimental_dir)
    if not experimental_files:
        logging.error('No experimental CIF files found. Exiting.')
        return
    rows: List[Dict[str, str]] = []
    for path in experimental_files:
        pdb_id = path.stem
        row = process_one_target(pdb_id, models_root, path, msa_dir)
        if args.mode == 'below_folding_budget':
            fields = read_below_folding_budget_json_if_any(args.mode, args.below_folding_budget_val, pdb_id)
            row = append_sidecar_fields(row, fields)
        elif args.mode == 'jackhmmer':
            fields = compute_jackhmmer_msa_metrics(msa_dir, pdb_id)
            row = append_sidecar_fields(row, fields)
        rows.append(row)
    if not rows:
        logging.info('No rows to write.')
        return
    all_keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)
    with out_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f'Wrote CSV: {out_csv.resolve()}')


if __name__ == '__main__':
    main()
