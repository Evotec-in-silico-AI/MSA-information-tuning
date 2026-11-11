#!/usr/bin/env python3
import argparse
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from chai_lab.chai1 import run_inference


def configure_logging(level: str) -> None:
    """Configure root logging with the given level string."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(message)s',
    )


def list_fasta_targets(fasta_dir: Path) -> List[str]:
    """Return target ids from all .fasta files in the directory (stem names)."""
    targets = []
    for path in sorted(fasta_dir.glob('*.fasta')):
        if path.is_file():
            targets.append(path.stem)
    return targets


def resolve_roots_for_mode(mode: str, below_folding_budget_val: Optional[str]) -> Tuple[Path, Optional[Path], Path, Path]:
    """Resolve (fasta_dir, msa_dir, models_root, results_dir) for the selected mode."""
    root = Path('.').resolve()
    fasta_dir = root / 'chai1_fasta'
    if mode == 'no_msa':
        msa_dir = None
        models_root = root / 'chai1_models_no_msa'
        results_dir = root / 'results'
        return fasta_dir, msa_dir, models_root, results_dir
    if mode == 'jackhmmer':
        msa_dir = root / 'msas'
        models_root = root / 'chai1_models_jackhmmer'
        results_dir = root / 'results'
        return fasta_dir, msa_dir, models_root, results_dir
    if mode == 'below_folding_budget':
        if not below_folding_budget_val:
            raise ValueError('below_folding_budget mode requires --below_folding_budget_val, e.g. 0p1')
        tuned_root = root / f'below_folding_budget_{below_folding_budget_val}'
        msa_dir = tuned_root / 'msas'
        models_root = tuned_root / 'chai1_models'
        results_dir = root / 'results'
        return fasta_dir, msa_dir, models_root, results_dir
    raise ValueError(f'Unsupported mode: {mode}')


def find_a3m_for_target(msa_dir: Path, pdb_id: str) -> Optional[Path]:
    """Return the first A3M path in msa_dir matching {pdb_id}*.a3m, or None if absent."""
    candidates = sorted(msa_dir.glob(f'{pdb_id}*.a3m'))
    if not candidates:
        return None
    return candidates[0]


def sanitize_a3m_sequence(sequence: str, strip_dots: bool = True, strip_inserts: bool = True) -> str:
    """Normalize an A3M sequence by dropping '.' and lowercase inserts when requested."""
    normalized_sequence = sequence
    if strip_dots:
        normalized_sequence = normalized_sequence.replace('.', '')
    filtered_chars = []
    for char in normalized_sequence:
        if ('A' <= char <= 'Z') or char == '-' or ('a' <= char <= 'z'):
            filtered_chars.append(char)
    cleaned_sequence = ''.join(filtered_chars)
    if strip_inserts:
        cleaned_sequence = ''.join(char for char in cleaned_sequence if not ('a' <= char <= 'z'))
    return cleaned_sequence


def parse_a3m(a3m_path: Path) -> List[Tuple[str, str]]:
    """Parse A3M into a list of (header, sequence) preserving order and case."""
    rows = []
    current_header = None
    sequence_lines: List[str] = []
    with a3m_path.open('r') as handle:
        for raw_line in handle:
            line = raw_line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    rows.append((current_header, ''.join(sequence_lines)))
                    sequence_lines = []
                current_header = line[1:].strip().split()[0]
            else:
                sequence_lines.append(line)
    if current_header is not None:
        rows.append((current_header, ''.join(sequence_lines)))
    if not rows:
        raise ValueError(f'Empty A3M: {a3m_path}')
    return rows


def project_to_query_match_columns(rows: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Project alignment to positions where the query (row 0) has uppercase A–Z."""
    query_sequence = rows[0][1]
    keep_mask = [('A' <= char <= 'Z') for char in query_sequence]
    projected_rows: List[Tuple[str, str]] = []
    for header, sequence in rows:
        if len(sequence) != len(query_sequence):
            raise ValueError('Alignment rows have unequal lengths before projection')
        projected_chars: List[str] = []
        for index, keep_column in enumerate(keep_mask):
            if keep_column:
                char = sequence[index]
                if 'a' <= char <= 'z':
                    projected_chars.append('-')
                else:
                    projected_chars.append(char.upper())
        projected_rows.append((header, ''.join(projected_chars)))
    return projected_rows


def ensure_equal_lengths(rows: List[Tuple[str, str]]) -> None:
    """Raise if the provided alignment rows are not all the same length."""
    lengths = {len(sequence) for _, sequence in rows}
    if len(lengths) != 1:
        raise ValueError(f'Alignment rows have unequal lengths: {lengths}')


def write_aligned_parquet(rows: List[Tuple[str, str]], out_dir: Path, source_database: str, pairing_key: str) -> Path:
    """Write a Parquet with columns sequence, source_database, pairing_key, comment."""
    query_sequence = rows[0][1]
    ungapped_query_sequence = query_sequence.replace('-', '')
    file_hash = hashlib.sha256(ungapped_query_sequence.encode('utf-8')).hexdigest()
    out_path = out_dir / f'{file_hash}.aligned.pqt'
    sequences: List[str] = []
    source_databases: List[str] = []
    pairing_keys: List[str] = []
    comments: List[str] = []
    sequences.append(query_sequence)
    source_databases.append('query')
    pairing_keys.append(pairing_key)
    comments.append(f'header={rows[0][0]}; snr=NA')
    for header, sequence in rows[1:]:
        sequences.append(sequence)
        source_databases.append(source_database)
        pairing_keys.append(pairing_key)
        comments.append(f'header={header}; snr=NA')
    dataframe = pd.DataFrame({
        'sequence': sequences,
        'source_database': source_databases,
        'pairing_key': pairing_keys,
        'comment': comments,
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    dataframe.to_parquet(out_path, index=False)
    return out_path


def prepare_local_msa_parquet(a3m_path: Path, out_msa_dir: Path, strip_dots: bool, integer_ids: bool) -> Path:
    """Build a query-anchored aligned.pqt from an A3M and save it into out_msa_dir."""
    rows = parse_a3m(a3m_path)
    normalized_rows = [(header, sanitize_a3m_sequence(sequence, strip_dots=strip_dots, strip_inserts=True)) for header, sequence in rows]
    if integer_ids:
        query_index = next((index for index, (header, _) in enumerate(normalized_rows) if header == '0'), None)
        if query_index is None:
            raise ValueError('A3M lacks query row with id=0')
        if '-' in normalized_rows[query_index][1]:
            raise ValueError('Query row id=0 contains gaps')
        if query_index != 0:
            normalized_rows = [normalized_rows[query_index]] + [pair for index, pair in enumerate(normalized_rows) if index != query_index]
        ensure_equal_lengths(normalized_rows)
        return write_aligned_parquet(normalized_rows, out_msa_dir, source_database='uniref90', pairing_key='single_chain')
    projected_rows = project_to_query_match_columns(normalized_rows)
    ensure_equal_lengths(projected_rows)
    return write_aligned_parquet(projected_rows, out_msa_dir, source_database='uniref90', pairing_key='single_chain')


def model_products_complete(models_dir: Path, num_samples: int) -> bool:
    """Return True if all expected CIF and NPZ files are present for the target."""
    for index in range(num_samples):
        cif_path = models_dir / f'pred.model_idx_{index}.cif'
        npz_path = models_dir / f'scores.model_idx_{index}.npz'
        if not (cif_path.is_file() and npz_path.is_file()):
            return False
    return True


def fold_one_target(pdb_id: str, fasta_dir: Path, models_root: Path, use_msa: bool, msa_dir: Optional[Path], integer_ids: bool, strip_dots: bool, device: str, num_samples: int, seed: int) -> None:
    """Fold one target if incomplete; delete and rerun when the models folder is partial."""
    fasta_path = fasta_dir / f'{pdb_id}.fasta'
    if not fasta_path.is_file():
        logging.info(f'Skipping {pdb_id}: FASTA not found: {fasta_path}')
        return
    models_dir = models_root / pdb_id
    if models_dir.exists() and not model_products_complete(models_dir, num_samples):
        shutil.rmtree(models_dir, ignore_errors=True)
    if model_products_complete(models_dir, num_samples):
        logging.info(f'Complete models present, skipping: {pdb_id}')
        return
    models_dir.mkdir(parents=True, exist_ok=True)
    inference_kwargs = {
        'fasta_file': fasta_path,
        'output_dir': models_dir,
        'use_esm_embeddings': True,
        'use_msa_server': False,
        'use_templates_server': False,
        'num_diffn_samples': int(num_samples),
        'seed': int(seed),
        'device': device,
        'low_memory': True,
    }
    if use_msa:
        assert msa_dir is not None
        a3m_path = find_a3m_for_target(msa_dir, pdb_id)
        if a3m_path is None:
            logging.info(f'Skipping {pdb_id}: A3M not found in {msa_dir}')
            return
        try:
            prepare_local_msa_parquet(a3m_path=a3m_path, out_msa_dir=msa_dir, strip_dots=strip_dots, integer_ids=integer_ids)
        except Exception as exc:
            logging.error(f'Failed to prepare aligned.pqt for {pdb_id}: {exc}')
            return
        inference_kwargs['msa_directory'] = msa_dir
    else:
        inference_kwargs['msa_directory'] = None
    logging.info(f"Folding {pdb_id} with Chai-1 (seed={seed}, samples={num_samples}, msa={'ON' if use_msa else 'OFF'})")
    run_inference(**inference_kwargs)


def all_targets_complete(pdb_ids: List[str], models_root: Path, num_samples: int) -> bool:
    """Return True if all provided targets have their five CIFs and five NPZs present."""
    for pdb_id in pdb_ids:
        models_dir = models_root / pdb_id
        if not model_products_complete(models_dir, num_samples):
            return False
    return True


def run_comparator(mode: str, below_folding_budget_val: Optional[str], tmscore_exec: str, log_level: str) -> None:
    """Invoke the comparison script with the given mode and options."""
    command = [
        str(Path(__file__).with_name('compare_chai1_to_experimental.py')),
        '--mode', mode,
        '--tmscore-exec', tmscore_exec,
        '--log-level', log_level,
    ]
    if mode == 'below_folding_budget':
        command += ['--below_folding_budget_val', str(below_folding_budget_val)]
    import subprocess
    subprocess.run(command, check=True)


def main() -> None:
    """Entry point to fold in the requested mode and conditionally run comparison."""
    parser = argparse.ArgumentParser(description='Run Chai-1 folding in three modes and optionally compare to experimental structures.')
    parser.add_argument('--mode', required=True, choices=['no_msa', 'jackhmmer', 'below_folding_budget'], help='Folding mode')
    parser.add_argument('--below_folding_budget_val', type=str, default=None, help='Value like 0p1 that maps to below_folding_budget_0p1; required for below_folding_budget mode')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device string')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of models per target')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for Chai-1')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--tmscore-exec', type=str, default='/data/projects/protein-db/TMscore/TMscore', help='Path to TMscore binary')
    args = parser.parse_args()
    configure_logging(args.log_level)
    fasta_dir, msa_dir, models_root, results_dir = resolve_roots_for_mode(args.mode, args.below_folding_budget_val)
    if not fasta_dir.is_dir():
        raise FileNotFoundError(f'Missing FASTA directory: {fasta_dir}')
    if args.mode in ('jackhmmer', 'below_folding_budget'):
        if msa_dir is None or not msa_dir.is_dir():
            raise FileNotFoundError(f'Missing MSAs directory: {msa_dir}')
    results_dir.mkdir(parents=True, exist_ok=True)
    pdb_ids_all = list_fasta_targets(fasta_dir)
    if not pdb_ids_all:
        logging.info('No FASTA targets found. Nothing to do.')
        return
    if args.mode == 'no_msa':
        eligible_ids = pdb_ids_all
        use_msa = False
        integer_ids = False
        strip_dots = True
    else:
        assert msa_dir is not None
        eligible_ids = [pdb_id for pdb_id in pdb_ids_all if find_a3m_for_target(msa_dir, pdb_id) is not None]
        use_msa = True
        integer_ids = False
        strip_dots = True
    any_folded = False
    for pdb_id in eligible_ids:
        was_complete = model_products_complete(models_root / pdb_id, args.num_samples)
        fold_one_target(
            pdb_id=pdb_id,
            fasta_dir=fasta_dir,
            models_root=models_root,
            use_msa=use_msa,
            msa_dir=msa_dir,
            integer_ids=integer_ids,
            strip_dots=strip_dots,
            device=args.device,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        is_complete = model_products_complete(models_root / pdb_id, args.num_samples)
        if not was_complete and is_complete:
            any_folded = True
    if args.mode == 'below_folding_budget':
        csv_path = results_dir / f'chai1_structural_comparison_below_folding_budget_{args.below_folding_budget_val}.csv'
    elif args.mode == 'jackhmmer':
        csv_path = results_dir / 'chai1_structural_comparison_jackhmmer.csv'
    else:
        csv_path = results_dir / 'chai1_structural_comparison_no_msa.csv'
    if (not any_folded) and all_targets_complete(eligible_ids, models_root, args.num_samples) and csv_path.exists():
        logging.info('All models complete and comparison CSV already exists. Exiting.')
        return
    if all_targets_complete(eligible_ids, models_root, args.num_samples):
        run_comparator(mode=args.mode, below_folding_budget_val=args.below_folding_budget_val, tmscore_exec=args.tmscore_exec, log_level=args.log_level)
    else:
        logging.info('Some eligible targets are incomplete; skipping comparison for now.')


if __name__ == '__main__':
    main()
