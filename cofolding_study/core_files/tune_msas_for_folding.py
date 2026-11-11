#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from prody import calcMSAOccupancy, calcShannonEntropy


def setup_logging(level: int) -> None:
    """Initialize the root logger with a console handler."""
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def load_fasta(path: Path) -> Tuple[List[str], List[str]]:
    """Load a FASTA file into headers and sequences."""
    headers, sequences = [], []
    current_header, current_lines = None, []
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
        raise ValueError(f'Malformed FASTA: {path}')
    return headers, sequences


def load_a3m(path: Path) -> Tuple[List[str], List[str]]:
    """Load an A3M file into headers and sequences (preserve case and gaps)."""
    headers, sequences = [], []
    current_header, current_lines = None, []
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


def sanitize_a3m(sequences: List[str], strip_dots: bool) -> List[str]:
    """Clean an A3M sequence list and enforce equal-length rows."""
    cleaned = []
    for sequence in sequences:
        sanitized_sequence = sequence.replace('.', '') if strip_dots else sequence
        sanitized_sequence = ''.join(char for char in sanitized_sequence if (('A' <= char <= 'Z') or char == '-'))
        cleaned.append(sanitized_sequence)
    lengths = {len(sequence) for sequence in cleaned}
    if len(lengths) != 1:
        raise ValueError('Sanitization did not yield equal-length sequences.')
    return cleaned


def project_to_anchor_match_columns(sequences: List[str]) -> List[str]:
    """Keep only columns where the first sequence is not a gap."""
    if not sequences:
        return []
    anchor_sequence = sequences[0]
    keep_indices = [index for index, char in enumerate(anchor_sequence) if char != '-']
    return [''.join(sequence[index] for index in keep_indices) for sequence in sequences]


def parse_numeric_prefix(header: str) -> int:
    """Extract leading integer from a header token."""
    token = (header or '').strip().split()[0]
    digits = []
    for char in token:
        if char.isdigit():
            digits.append(char)
        else:
            break
    value = ''.join(digits) or '0'
    return int(value)


def sort_and_cap_integer_mode(headers: List[str], sequences: List[str], cap: int) -> Tuple[List[str], List[str]]:
    """Sort by leading integer id, pin id==0 to top, and cap rows."""
    items = [(parse_numeric_prefix(header), index) for index, header in enumerate(headers)]
    items.sort(key=lambda item: item[0])
    order = [index for _, index in items]
    query_index = next((index for numeric_id, index in items if numeric_id == 0), None)
    if query_index is None:
        raise ValueError('Reference with id 0 is missing.')
    ordered_headers = [headers[index] for index in order]
    ordered_sequences = [sequences[index] for index in order]
    if order[0] != query_index:
        query_position = order.index(query_index)
        ordered_headers.insert(0, ordered_headers.pop(query_position))
        ordered_sequences.insert(0, ordered_sequences.pop(query_position))
    if len(ordered_headers) > cap:
        ordered_headers = [ordered_headers[0]] + ordered_headers[1:cap]
        ordered_sequences = [ordered_sequences[0]] + ordered_sequences[1:cap]
    return ordered_headers, ordered_sequences


def uniform_downsample_keep_first(headers: List[str], sequences: List[str], cap: int, rng: np.random.Generator) -> Tuple[List[str], List[str]]:
    """Uniformly sample rows to a cap while keeping the first row."""
    row_count = len(headers)
    if row_count <= cap:
        return headers, sequences
    need = cap - 1
    choice = rng.choice(np.arange(1, row_count, dtype=int), size=need, replace=False)
    keep_indices = [0] + sorted(choice.tolist())
    return [headers[index] for index in keep_indices], [sequences[index] for index in keep_indices]


def fnv1a64(data: bytes) -> int:
    """Compute 64-bit FNV-1a hash."""
    hash_value = 0xcbf29ce484222325
    prime = 0x100000001b3
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * prime) & 0xFFFFFFFFFFFFFFFF
    return hash_value


def simhash64(sequence: str, k: int) -> int:
    """Compute a 64-bit SimHash over ungapped k-mers."""
    vector = [0] * 64
    length = len(sequence)
    added = 0
    if length >= k:
        for index in range(length - k + 1):
            token = sequence[index:index + k]
            if '-' in token:
                continue
            hash_value = fnv1a64(token.encode('ascii', errors='ignore'))
            for bit in range(64):
                if hash_value & (1 << bit):
                    vector[bit] += 1
                else:
                    vector[bit] -= 1
            added += 1
    if added == 0:
        ungapped_sequence = sequence.replace('-', '')
        if ungapped_sequence:
            hash_value = fnv1a64(ungapped_sequence.encode('ascii', errors='ignore'))
            for bit in range(64):
                if hash_value & (1 << bit):
                    vector[bit] += 1
                else:
                    vector[bit] -= 1
            added = 1
    if added == 0:
        hash_value = fnv1a64(sequence.encode('ascii', errors='ignore'))
        for bit in range(64):
            if hash_value & (1 << bit):
                vector[bit] += 1
            else:
                vector[bit] -= 1
    out = 0
    for bit in range(63, -1, -1):
        out <<= 1
        out |= 1 if vector[bit] > 0 else 0
    return out


def simhash_dedup_keep_first(headers: List[str], sequences: List[str], bucket_bits: int, kmer: int) -> Tuple[List[str], List[str]]:
    """Perform SimHash deduplication by keeping one representative per bucket, plus the first row."""
    row_count = len(headers)
    if row_count <= 2:
        return headers, sequences
    if bucket_bits <= 0 or bucket_bits > 16:
        bucket_bits = 12
    buckets: Dict[int, int] = {}
    keep_indices: List[int] = [0]
    for index in range(1, row_count):
        signature = simhash64(sequences[index], kmer)
        bucket_id = (signature >> (64 - bucket_bits)) & ((1 << bucket_bits) - 1)
        if bucket_id not in buckets:
            buckets[bucket_id] = index
    keep_indices.extend(sorted(buckets.values()))
    return [headers[index] for index in keep_indices], [sequences[index] for index in keep_indices]


def simhash_downsample_keep_first(headers: List[str], sequences: List[str], cap: int, rng: np.random.Generator, bucket_bits: int, kmer: int) -> Tuple[List[str], List[str]]:
    """Downsample rows via SimHash buckets while keeping the first row."""
    row_count = len(headers)
    if row_count <= cap:
        return headers, sequences
    need = cap - 1
    if bucket_bits <= 0 or bucket_bits > 16:
        bucket_bits = 12
    buckets: Dict[int, List[int]] = {}
    for index in range(1, row_count):
        signature = simhash64(sequences[index], kmer)
        bucket_id = (signature >> (64 - bucket_bits)) & ((1 << bucket_bits) - 1)
        buckets.setdefault(bucket_id, []).append(index)
    chosen: List[int] = []
    total = sum(len(indices) for indices in buckets.values())
    if total == 0:
        return uniform_downsample_keep_first(headers, sequences, cap, rng)
    allocation: Dict[int, int] = {}
    assigned = 0
    for bucket_id, indices in buckets.items():
        quota = int(np.floor(need * (len(indices) / total)))
        quota = min(quota, len(indices))
        if quota > 0:
            allocation[bucket_id] = quota
            assigned += quota
    remainder = need - assigned
    if remainder > 0:
        remaining_caps = sorted(buckets.items(), key=lambda item: len(item[1]) - allocation.get(item[0], 0), reverse=True)
        cursor = 0
        while remainder > 0 and remaining_caps:
            bucket_id, indices = remaining_caps[cursor % len(remaining_caps)]
            cap_left = len(indices) - allocation.get(bucket_id, 0)
            if cap_left > 0:
                allocation[bucket_id] = allocation.get(bucket_id, 0) + 1
                remainder -= 1
            cursor += 1
    for bucket_id, quota in allocation.items():
        indices = buckets[bucket_id]
        if quota >= len(indices):
            chosen.extend(indices)
        elif quota > 0:
            selected = rng.choice(np.array(indices, dtype=int), size=quota, replace=False)
            chosen.extend(selected.tolist())
    if len(chosen) < need:
        pool = sorted(set(range(1, row_count)) - set(chosen))
        fill = min(need - len(chosen), len(pool))
        if fill > 0:
            extra = rng.choice(np.array(pool, dtype=int), size=fill, replace=False)
            chosen.extend(extra.tolist())
    keep_indices = [0] + sorted(chosen[:need])
    return [headers[index] for index in keep_indices], [sequences[index] for index in keep_indices]


def write_a3m(headers: List[str], sequences: List[str], out_path: Path) -> None:
    """Write headers and sequences in FASTA-like A3M format."""
    with out_path.open('w') as handle:
        for header, sequence in zip(headers, sequences):
            handle.write(f'>{header}\n{sequence}\n')


def build_residue_lut() -> np.ndarray:
    """Return a lookup table mapping ASCII to token ids."""
    lookup_table = np.zeros(128, dtype=np.int16)
    invalid_chars = ['-', 'X', 'U', 'O', '?', '.', '*', 'B', 'Z', 'J']
    for char in invalid_chars:
        lookup_table[ord(char)] = 0
    mapping = {
        'A': 1,
        'E': 2,
        'L': 3,
        'M': 4,
        'C': 5,
        'D': 6,
        'F': 7,
        'G': 8,
        'H': 9,
        'K': 10,
        'N': 11,
        'P': 12,
        'Q': 13,
        'R': 14,
        'S': 15,
        'W': 16,
        'Y': 17,
        'T': 18,
        'V': 19,
        'I': 20,
    }
    for char, value in mapping.items():
        lookup_table[ord(char)] = value
    return lookup_table


def encode_char_array(sequences: List[str], lookup_table: np.ndarray) -> np.ndarray:
    """Encode a sequence list to an int16 token matrix."""
    if not sequences:
        return np.zeros((0, 0), dtype=np.int16)
    uppercase_sequences = [sequence.upper() for sequence in sequences]
    flat_bytes = ''.join(uppercase_sequences).encode('ascii', 'replace')
    array = np.frombuffer(flat_bytes, dtype=np.uint8).reshape((len(sequences), len(sequences[0])))
    encoded = lookup_table[array]
    return encoded.astype(np.int16, copy=False)


class UpperTrianglePid:
    """Upper-triangle storage for no-gap PID values and validity."""

    def __init__(self, n: int, values: np.ndarray, valid: np.ndarray):
        """Initialize with size and backing arrays."""
        self.n = int(n)
        self.values = values
        self.valid = valid

    def index(self, i: int, j: int) -> int:
        """Return flat index for the upper triangle at (i, j)."""
        if i >= j:
            i, j = j, i
        return i * self.n - i * (i + 1) // 2 + (j - i - 1)

    def get(self, i: int, j: int) -> Tuple[float, bool]:
        """Return PID value and validity flag for a pair."""
        if i == j:
            return 0.0, False
        if i > j:
            i, j = j, i
        index = self.index(i, j)
        return float(self.values[index]), bool(self.valid[index])


def compute_pair_block_counts(Xi: np.ndarray, Xj: np.ndarray, feat_block: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute blockwise matches and denominators for PID."""
    block_rows_i, length = Xi.shape
    block_rows_j = Xj.shape[0]
    matches = np.zeros((block_rows_i, block_rows_j), dtype=np.uint32)
    denominator = np.zeros((block_rows_i, block_rows_j), dtype=np.uint32)
    for feature_start in range(0, length, feat_block):
        feature_end = min(feature_start + feat_block, length)
        block_i = Xi[:, feature_start:feature_end]
        block_j = Xj[:, feature_start:feature_end]
        equal = block_i[:, None, :] == block_j[None, :, :]
        no_gap = (block_i[:, None, :] != 0) & (block_j[None, :, :] != 0)
        matches += (equal & no_gap).sum(axis=2, dtype=np.uint32)
        denominator += no_gap.sum(axis=2, dtype=np.uint32)
    return matches, denominator


def build_upper_triangle_pid_buffer(tokens: np.ndarray, row_block: int, col_block: int, feat_block: int) -> UpperTrianglePid:
    """Build an upper-triangle PID buffer from a token matrix."""
    row_count, _ = tokens.shape
    size = row_count * (row_count - 1) // 2
    values = np.zeros(size, dtype=np.float32)
    valid = np.zeros(size, dtype=np.uint8)
    for row_start in range(0, row_count, row_block):
        row_end = min(row_count, row_start + row_block)
        rows_i = tokens[row_start:row_end]
        block_rows_i = row_end - row_start
        for col_start in range(row_start, row_count, col_block):
            col_end = min(row_count, col_start + col_block)
            rows_j = tokens[col_start:col_end]
            matches, denominator = compute_pair_block_counts(rows_i, rows_j, feat_block)
            if col_start >= row_end:
                positive = denominator > 0
                if positive.any():
                    indices_i, indices_j = np.nonzero(positive)
                    global_i = indices_i + row_start
                    global_j = indices_j + col_start
                    flat_index = global_i * row_count - global_i * (global_i + 1) // 2 + (global_j - global_i - 1)
                    pid = matches[positive].astype(np.float32) / denominator[positive].astype(np.float32)
                    values[flat_index] = pid
                    valid[flat_index] = 1
            else:
                triangle_mask = np.triu(np.ones((block_rows_i, col_end - col_start), dtype=bool), k=1 + (row_start == col_start))
                positive = triangle_mask & (denominator > 0)
                if positive.any():
                    indices_i, indices_j = np.nonzero(positive)
                    global_i = indices_i + row_start
                    global_j = indices_j + col_start
                    swap_mask = global_i >= global_j
                    if swap_mask.any():
                        temp = global_i.copy()
                        global_i[swap_mask] = global_j[swap_mask]
                        global_j[swap_mask] = temp[swap_mask]
                    flat_index = global_i * row_count - global_i * (global_i + 1) // 2 + (global_j - global_i - 1)
                    pid = matches[positive].astype(np.float32) / denominator[positive].astype(np.float32)
                    values[flat_index] = pid
                    valid[flat_index] = 1
    return UpperTrianglePid(n=row_count, values=values, valid=valid)


def build_row_accumulators(buffer: UpperTrianglePid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    """Build per-row ordered accumulators to support SNR updates."""
    row_count = buffer.n
    sum_first = np.zeros(row_count, dtype=np.float64)
    sum_second = np.zeros(row_count, dtype=np.float64)
    sumsq_first = np.zeros(row_count, dtype=np.float64)
    sumsq_second = np.zeros(row_count, dtype=np.float64)
    count_first = np.zeros(row_count, dtype=np.int64)
    count_second = np.zeros(row_count, dtype=np.int64)
    index = 0
    for row_index in range(row_count - 1):
        row_length = row_count - row_index - 1
        row_slice = slice(index, index + row_length)
        values = buffer.values[row_slice].astype(np.float64, copy=False)
        mask = buffer.valid[row_slice].astype(bool, copy=False)
        valid_values = values[mask]
        if valid_values.size:
            sum_first[row_index] += valid_values.sum()
            sumsq_first[row_index] += (valid_values * valid_values).sum()
            count_first[row_index] += int(valid_values.size)
            neighbor_indices = np.nonzero(mask)[0] + (row_index + 1)
            if neighbor_indices.size:
                np.add.at(sum_second, neighbor_indices, values[mask])
                np.add.at(sumsq_second, neighbor_indices, values[mask] * values[mask])
                np.add.at(count_second, neighbor_indices, 1)
        index += row_length
    sum_ordered = 2.0 * (sum_first + sum_second)
    sumsq_ordered = 2.0 * (sumsq_first + sumsq_second)
    count_ordered = 2 * (count_first + count_second)
    total_sum = float(sum_ordered.sum() / 2.0)
    total_sumsq = float(sumsq_ordered.sum() / 2.0)
    total_pairs = int(count_ordered.sum() // 2)
    return sum_ordered, sumsq_ordered, count_ordered, total_sum, total_sumsq, total_pairs


def snr_from_moments(total_sum: float, total_sumsq: float, total_pairs: int) -> Tuple[float, float, float]:
    """Return SNR, mean% PID, and sd% PID from PID moments."""
    if total_pairs <= 1:
        return float('nan'), float('nan'), float('nan')
    mean_fraction = total_sum / float(total_pairs)
    variance_fraction = max(total_sumsq / float(total_pairs) - mean_fraction * mean_fraction, 0.0)
    mean_pct = 100.0 * mean_fraction
    sd_pct = 100.0 * float(np.sqrt(variance_fraction))
    if sd_pct <= 1e-12 or not np.isfinite(sd_pct):
        return float('inf'), mean_pct, sd_pct
    return mean_pct / sd_pct, mean_pct, sd_pct


def remove_row_in_place(buffer: UpperTrianglePid, sum_ordered: np.ndarray, sumsq_ordered: np.ndarray, count_ordered: np.ndarray, totals: List[float], active_mask: np.ndarray, row_index: int) -> None:
    """Drop one row and update ordered accumulators and totals."""
    totals[0] -= 0.5 * float(sum_ordered[row_index])
    totals[1] -= 0.5 * float(sumsq_ordered[row_index])
    totals[2] -= int(count_ordered[row_index]) // 2
    for other_index in range(buffer.n):
        if other_index == row_index or not active_mask[other_index]:
            continue
        pid_value, is_valid = buffer.get(min(other_index, row_index), max(other_index, row_index))
        if not is_valid:
            continue
        value = float(pid_value)
        sum_ordered[other_index] -= 2.0 * value
        sumsq_ordered[other_index] -= 2.0 * (value * value)
        count_ordered[other_index] -= 2
    sum_ordered[row_index] = 0.0
    sumsq_ordered[row_index] = 0.0
    count_ordered[row_index] = 0
    active_mask[row_index] = False
    totals[0] = float(sum_ordered.sum() / 2.0)
    totals[1] = float(sumsq_ordered.sum() / 2.0)
    totals[2] = int(count_ordered.sum() // 2)


def build_pid_adjacency_lists(buffer: UpperTrianglePid, pid_threshold: float) -> List[np.ndarray]:
    """Build adjacency lists for PID>=threshold using the PID buffer."""
    row_count = buffer.n
    neighbors: List[List[int]] = [[] for _ in range(row_count)]
    index = 0
    for row_index in range(row_count - 1):
        row_length = row_count - row_index - 1
        row_slice = slice(index, index + row_length)
        values = buffer.values[row_slice]
        mask = buffer.valid[row_slice].astype(bool, copy=False)
        if mask.any():
            neighbor_offsets = np.nonzero(mask & (values >= float(pid_threshold)))[0]
            if neighbor_offsets.size:
                neighbor_indices = (neighbor_offsets + (row_index + 1)).tolist()
                neighbors[row_index].extend(neighbor_indices)
                for neighbor_index in neighbor_indices:
                    neighbors[neighbor_index].append(row_index)
        index += row_length
    return [np.array(sorted(set(indices)), dtype=np.int32) if indices else np.zeros((0,), dtype=np.int32) for indices in neighbors]


def initialize_incremental_meff_state(adjacency: List[np.ndarray], active_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Initialize per-row PID-degree and Meff value for the active set."""
    degrees = np.zeros(active_mask.size, dtype=np.int32)
    for row_index in range(active_mask.size):
        if not active_mask[row_index]:
            continue
        if adjacency[row_index].size == 0:
            degrees[row_index] = 0
        else:
            degrees[row_index] = int(np.sum(active_mask[adjacency[row_index]]))
    weights = 1.0 / (1.0 + degrees.astype(np.float64))
    weights = weights * active_mask.astype(np.float64)
    meff_value = float(weights.sum())
    return degrees, meff_value


def update_incremental_meff_on_drop(adjacency: List[np.ndarray], active_mask: np.ndarray, degrees: np.ndarray, meff_value: float, drop_index: int) -> float:
    """Update Meff and degrees after dropping one row."""
    if not active_mask[drop_index]:
        return meff_value
    dropped_weight = 1.0 / (1.0 + float(degrees[drop_index]))
    meff_value -= dropped_weight
    active_mask[drop_index] = False
    if adjacency[drop_index].size:
        for neighbor_index in adjacency[drop_index]:
            if active_mask[neighbor_index]:
                old_weight = 1.0 / (1.0 + float(degrees[neighbor_index]))
                degrees[neighbor_index] = max(0, int(degrees[neighbor_index]) - 1)
                new_weight = 1.0 / (1.0 + float(degrees[neighbor_index]))
                meff_value += (new_weight - old_weight)
    degrees[drop_index] = 0
    return meff_value


def compute_snr_moments_from_state(sum_ordered: np.ndarray, sumsq_ordered: np.ndarray, count_ordered: np.ndarray) -> Tuple[float, float, float, int]:
    """Return SNR, mean% PID, sd% PID, and pair count from ordered accumulators."""
    total_sum = float(sum_ordered.sum() / 2.0)
    total_sumsq = float(sumsq_ordered.sum() / 2.0)
    total_pairs = int(count_ordered.sum() // 2)
    snr, mean_pct, sd_pct = snr_from_moments(total_sum, total_sumsq, total_pairs)
    return float(snr), float(mean_pct), float(sd_pct), int(total_pairs)


def select_row_to_drop_for_direction(buffer: UpperTrianglePid, sum_ordered: np.ndarray, sumsq_ordered: np.ndarray, count_ordered: np.ndarray, totals: List[float], active_mask: np.ndarray, direction_sign: int) -> Tuple[int, float, float]:
    """Pick a single row that best changes SNR in the requested direction."""
    current_snr, _, _ = snr_from_moments(totals[0], totals[1], int(totals[2]))
    best_row = -1
    best_new_snr = current_snr
    if direction_sign > 0:
        best_score = -np.inf
    else:
        best_score = np.inf
    for row_index in range(1, buffer.n):
        if not active_mask[row_index]:
            continue
        remaining_pairs = int(totals[2]) - int(count_ordered[row_index]) // 2
        if remaining_pairs <= 1:
            continue
        new_sum = totals[0] - 0.5 * float(sum_ordered[row_index])
        new_sumsq = totals[1] - 0.5 * float(sumsq_ordered[row_index])
        new_snr, _, _ = snr_from_moments(new_sum, new_sumsq, remaining_pairs)
        if direction_sign > 0:
            if new_snr > best_score:
                best_score = new_snr
                best_row = row_index
                best_new_snr = new_snr
        else:
            if new_snr < best_score:
                best_score = new_snr
                best_row = row_index
                best_new_snr = new_snr
    if best_row == -1:
        choices = [index for index in range(1, buffer.n) if active_mask[index]]
        if choices:
            best_row = choices[0]
    return best_row, float(best_new_snr), float(current_snr)


def perform_greedy_walk(tokens: np.ndarray, buffer: UpperTrianglePid, adjacency: List[np.ndarray], pid_threshold_for_meff: float, direction_sign: int, max_drops_allowed: int, min_rows_kept: int) -> Dict[str, np.ndarray]:
    """Run a monotone greedy SNR walk (increase or decrease) to the row-drop limit."""
    sum_ordered, sumsq_ordered, count_ordered, total_sum, total_sumsq, total_pairs = build_row_accumulators(buffer)
    active_mask = np.ones(tokens.shape[0], dtype=bool)
    totals = [total_sum, total_sumsq, float(total_pairs)]
    degrees, meff_value = initialize_incremental_meff_state(adjacency, active_mask)
    snr_values: List[float] = []
    mean_pid_values: List[float] = []
    sd_pid_values: List[float] = []
    pair_values: List[int] = []
    meff_values: List[float] = []
    n_values: List[int] = []
    masks: List[np.ndarray] = []
    snr_now, mean_pid_now, sd_pid_now, pair_now = compute_snr_moments_from_state(sum_ordered, sumsq_ordered, count_ordered)
    snr_values.append(float(snr_now))
    mean_pid_values.append(float(mean_pid_now))
    sd_pid_values.append(float(sd_pid_now))
    pair_values.append(int(pair_now))
    meff_values.append(float(meff_value))
    n_values.append(int(active_mask.sum()))
    masks.append(active_mask.copy())
    drops_made = 0
    while drops_made < max_drops_allowed and int(active_mask.sum()) - 1 >= min_rows_kept:
        pick, _, _ = select_row_to_drop_for_direction(buffer, sum_ordered, sumsq_ordered, count_ordered, totals, active_mask, direction_sign)
        if pick == -1:
            break
        meff_value = update_incremental_meff_on_drop(adjacency, active_mask, degrees, meff_value, pick)
        remove_row_in_place(buffer, sum_ordered, sumsq_ordered, count_ordered, totals, active_mask, pick)
        drops_made += 1
        snr_now, mean_pid_now, sd_pid_now, pair_now = compute_snr_moments_from_state(sum_ordered, sumsq_ordered, count_ordered)
        snr_values.append(float(snr_now))
        mean_pid_values.append(float(mean_pid_now))
        sd_pid_values.append(float(sd_pid_now))
        pair_values.append(int(pair_now))
        meff_values.append(float(meff_value))
        n_values.append(int(active_mask.sum()))
        masks.append(active_mask.copy())
    return {
        'snr': np.asarray(snr_values, dtype=np.float64),
        'mean_pid_pct': np.asarray(mean_pid_values, dtype=np.float64),
        'sd_pid_pct': np.asarray(sd_pid_values, dtype=np.float64),
        'pair_count': np.asarray(pair_values, dtype=np.int64),
        'meff': np.asarray(meff_values, dtype=np.float64),
        'n': np.asarray(n_values, dtype=np.int64),
        'mask': np.stack(masks, axis=0),
    }


def sequences_to_char_array(sequences: List[str]) -> np.ndarray:
    """Return a 2D byte-character array ('S1') for ProDy metrics."""
    if not sequences:
        return np.zeros((0, 0), dtype='S1')
    row_count = len(sequences)
    column_count = len(sequences[0])
    out = np.empty((row_count, column_count), dtype='S1')
    for row_index, sequence in enumerate(sequences):
        out[row_index, :] = np.frombuffer(sequence.encode('ascii', 'replace'), dtype='S1')
    return out


def compute_unweighted_bits_metrics_prody(sequences: List[str], folding_budget_bps: float) -> Dict[str, float]:
    """Compute unweighted bits/site metrics and below-budget fractions via ProDy."""
    char_array = sequences_to_char_array(sequences)
    entropy_bits = np.asarray(calcShannonEntropy(char_array), dtype=np.float64)
    bits_site = np.log2(20.0) - entropy_bits
    occupancy = np.asarray(calcMSAOccupancy(char_array, occ='res'), dtype=np.float64)
    columns_total = int(bits_site.size)
    columns_below_budget = int(np.sum(bits_site < float(folding_budget_bps))) if bits_site.size else 0
    fraction_below_budget = float(columns_below_budget / columns_total) if columns_total > 0 else float('nan')
    gap_fraction = float(np.mean(1.0 - occupancy)) if occupancy.size else float('nan')
    return {
        'bps_min': float(np.min(bits_site)) if bits_site.size else float('nan'),
        'bps_p1': float(np.quantile(bits_site, 0.01)) if bits_site.size else float('nan'),
        'bps_p10': float(np.quantile(bits_site, 0.10)) if bits_site.size else float('nan'),
        'bps_median': float(np.median(bits_site)) if bits_site.size else float('nan'),
        'bps_mean': float(np.mean(bits_site)) if bits_site.size else float('nan'),
        'gap_fraction': gap_fraction,
        'projected_columns': float(columns_total),
        'columns_below_folding_budget': float(columns_below_budget),
        'fraction_columns_below_folding_budget': fraction_below_budget,
    }


def filter_rows_by_gap_fraction(headers: List[str], sequences: List[str], max_row_gap_frac: float) -> Tuple[List[str], List[str]]:
    """Drop rows whose gap fraction exceeds a threshold after anchor-column projection; always keep the first row."""
    if not sequences:
        return headers, sequences
    column_count = len(sequences[0])
    keep_indices: List[int] = [0]
    for index in range(1, len(sequences)):
        if column_count == 0:
            continue
        gap_fraction = float(sequences[index].count('-')) / float(column_count)
        if gap_fraction <= float(max_row_gap_frac):
            keep_indices.append(index)
    return [headers[index] for index in keep_indices], [sequences[index] for index in keep_indices]


def format_target_folder_name(target_value: float) -> str:
    """Format folder name as below_folding_budget_0p1 with 'p' instead of dot."""
    target_string = f'{target_value:.3f}'.rstrip('0').rstrip('.')
    target_string = target_string.replace('.', 'p')
    return f'below_folding_budget_{target_string}'


def acquire_lock(msa_id: str, locks_dir: Path) -> Optional[Path]:
    """Create a lockfile to avoid duplicate work across workers."""
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = locks_dir / f'{msa_id}.lock'
    try:
        file_descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(file_descriptor)
        return lock_path
    except FileExistsError:
        return None


def release_lock(lock_path: Optional[Path]) -> None:
    """Remove a lockfile if it exists."""
    if lock_path is not None and lock_path.exists():
        try:
            lock_path.unlink()
        except Exception:
            pass


def build_starting_alignment(headers: List[str], sequences: List[str], mode: str, cap_integer: int, cap_jackhmmer: int, strip_dots: bool, downsample_method: str, simhash_bits: int, simhash_k: int, max_row_gap_frac: float, rng: np.random.Generator) -> Tuple[List[str], List[str]]:
    """Build the starting alignment with: anchor projection -> high-gap drop -> SimHash dedup -> cap downsample."""
    sanitized_sequences = sanitize_a3m(sequences, strip_dots=strip_dots)
    if mode == 'integer':
        ordered_headers, ordered_sequences = sort_and_cap_integer_mode(headers, sanitized_sequences, cap_integer)
    else:
        ordered_headers, ordered_sequences = headers, sanitized_sequences
    projected_sequences = project_to_anchor_match_columns(ordered_sequences)
    filtered_headers, filtered_sequences = filter_rows_by_gap_fraction(ordered_headers, projected_sequences, float(max_row_gap_frac))
    dedup_headers, dedup_sequences = simhash_dedup_keep_first(filtered_headers, filtered_sequences, bucket_bits=simhash_bits, kmer=simhash_k)
    if len(dedup_headers) > cap_jackhmmer:
        if downsample_method == 'uniform':
            downsampled_headers, downsampled_sequences = uniform_downsample_keep_first(dedup_headers, dedup_sequences, cap_jackhmmer, rng)
        else:
            downsampled_headers, downsampled_sequences = simhash_downsample_keep_first(dedup_headers, dedup_sequences, cap_jackhmmer, rng, simhash_bits, simhash_k)
        return downsampled_headers, downsampled_sequences
    return dedup_headers, dedup_sequences


def choose_better_snapshot(current_info: Dict[str, float], candidate_metrics: Dict[str, float], candidate_index: int, candidate_rows: int, target_value: float) -> Dict[str, float]:
    """Return the preferred snapshot record for a given target."""
    candidate_delta = abs(candidate_metrics['fraction_columns_below_folding_budget'] - target_value)
    if candidate_delta < current_info['best_delta']:
        return {
            'best_delta': float(candidate_delta),
            'best_index': int(candidate_index),
            'rows': float(candidate_rows),
            **candidate_metrics,
        }
    if abs(candidate_delta - current_info['best_delta']) <= 1e-12 and candidate_rows > int(current_info['rows']):
        return {
            'best_delta': float(candidate_delta),
            'best_index': int(candidate_index),
            'rows': float(candidate_rows),
            **candidate_metrics,
        }
    return current_info


def build_snapshot_payload(payload_base: Dict[str, float], status: str, target_value: float, tolerance: float, selected_info: Dict[str, float], row_count: int, snr_value: float, pid_mean_pct: float, pid_sd_pct: float, pid_pair_count: int, meff_value: float, direction: str, path_index: int) -> Dict[str, object]:
    """Build a sidecar payload for the selected snapshot."""
    payload = dict(payload_base)
    meff_fraction = float(meff_value / max(1, row_count))
    payload.update({
        'status': status,
        'target_fraction_below_folding_budget': float(target_value),
        'tolerance': float(tolerance),
        'selection_delta': float(selected_info['best_delta']),
        'rows': int(row_count),
        'snr': float(snr_value),
        'pid_mean_pct': float(pid_mean_pct),
        'pid_sd_pct': float(pid_sd_pct),
        'pid_pair_count': int(pid_pair_count),
        'bps_min': float(selected_info['bps_min']),
        'bps_p1': float(selected_info['bps_p1']),
        'bps_p10': float(selected_info['bps_p10']),
        'bps_median': float(selected_info['bps_median']),
        'bps_mean': float(selected_info['bps_mean']),
        'gap_fraction': float(selected_info['gap_fraction']),
        'projected_columns': int(selected_info['projected_columns']),
        'columns_below_folding_budget': int(selected_info['columns_below_folding_budget']),
        'fraction_columns_below_folding_budget': float(selected_info['fraction_columns_below_folding_budget']),
        'meff_theta_0p70': float(meff_value),
        'meff_fraction_theta_0p70': meff_fraction,
        'path_index': int(path_index),
        'direction': direction,
    })
    return payload


def build_not_reached_payload(payload_base: Dict[str, float], target_value: float, tolerance: float, selected_info: Dict[str, float], row_count: int, snr_value: float, pid_mean_pct: float, pid_sd_pct: float, pid_pair_count: int, meff_value: float, direction: str, path_index: int) -> Dict[str, object]:
    """Build a sidecar payload for the closest snapshot when the target is not reached."""
    payload = dict(payload_base)
    meff_fraction = float(meff_value / max(1, row_count))
    payload.update({
        'status': 'not_reached',
        'target_fraction_below_folding_budget': float(target_value),
        'tolerance': float(tolerance),
        'closest_selection_delta': float(selected_info['best_delta']),
        'closest_rows': int(row_count),
        'closest_snr': float(snr_value),
        'closest_pid_mean_pct': float(pid_mean_pct),
        'closest_pid_sd_pct': float(pid_sd_pct),
        'closest_pid_pair_count': int(pid_pair_count),
        'closest_bps_min': float(selected_info['bps_min']),
        'closest_bps_p1': float(selected_info['bps_p1']),
        'closest_bps_p10': float(selected_info['bps_p10']),
        'closest_bps_median': float(selected_info['bps_median']),
        'closest_bps_mean': float(selected_info['bps_mean']),
        'closest_gap_fraction': float(selected_info['gap_fraction']),
        'closest_projected_columns': int(selected_info['projected_columns']),
        'closest_columns_below_folding_budget': int(selected_info['columns_below_folding_budget']),
        'closest_fraction_columns_below_folding_budget': float(selected_info['fraction_columns_below_folding_budget']),
        'closest_meff_theta_0p70': float(meff_value),
        'closest_meff_fraction_theta_0p70': meff_fraction,
        'path_index': int(path_index),
        'direction': direction,
    })
    return payload


def evaluate_snapshots_and_emit(msa_id: str, headers_start: List[str], sequences_start: List[str], tokens_start: np.ndarray, walk_down: Dict[str, np.ndarray], walk_up: Dict[str, np.ndarray], targets: List[float], tolerance: float, snr_start: float, row_drop_limit_frac: float, meff_threshold: float, min_rows_kept: int, folding_budget_bps: float) -> None:
    """Evaluate all snapshots against targets, write per-target folders in CWD with A3M and JSON sidecars."""
    masks = np.concatenate([walk_down['mask'], walk_up['mask'][1:]], axis=0)
    snr_series = np.concatenate([walk_down['snr'], walk_up['snr'][1:]], axis=0)
    mean_series = np.concatenate([walk_down['mean_pid_pct'], walk_up['mean_pid_pct'][1:]], axis=0)
    sd_series = np.concatenate([walk_down['sd_pid_pct'], walk_up['sd_pid_pct'][1:]], axis=0)
    pair_series = np.concatenate([walk_down['pair_count'], walk_up['pair_count'][1:]], axis=0)
    meff_series = np.concatenate([walk_down['meff'], walk_up['meff'][1:]], axis=0)
    row_series = np.concatenate([walk_down['n'], walk_up['n'][1:]], axis=0)
    best_by_target: Dict[float, Dict[str, float]] = {}
    for target_value in targets:
        best_by_target[target_value] = {'best_delta': float('inf'), 'best_index': -1, 'rows': -1.0}
    for snapshot_index in range(masks.shape[0]):
        mask = masks[snapshot_index]
        kept_rows = [index for index, keep in enumerate(mask) if keep]
        sequences = [sequences_start[index] for index in kept_rows]
        metrics = compute_unweighted_bits_metrics_prody(sequences, folding_budget_bps=folding_budget_bps)
        for target_value in targets:
            best_by_target[target_value] = choose_better_snapshot(
                current_info=best_by_target[target_value],
                candidate_metrics=metrics,
                candidate_index=snapshot_index,
                candidate_rows=int(row_series[snapshot_index]),
                target_value=target_value,
            )
    for target_value in targets:
        selected_info = best_by_target[target_value]
        path_index = int(selected_info['best_index']) if selected_info['best_index'] is not None else -1
        folder = Path(format_target_folder_name(target_value))
        path_msas = folder / 'msas'
        path_json = folder / 'json'
        path_msas.mkdir(parents=True, exist_ok=True)
        path_json.mkdir(parents=True, exist_ok=True)
        payload_base = {
            'msa_id': msa_id,
            'row_drop_limit_frac': float(row_drop_limit_frac),
            'meff_threshold': float(meff_threshold),
            'pipeline_version': 'below_folding_budget_v1',
            'start_rows': int(tokens_start.shape[0]),
            'snr_start': float(snr_start),
            'min_rows_kept': int(min_rows_kept),
            'folding_budget_bps': float(folding_budget_bps),
        }
        if path_index < 0:
            payload = dict(payload_base)
            payload.update({
                'status': 'not_reached',
                'target_fraction_below_folding_budget': float(target_value),
                'tolerance': float(tolerance),
                'message': 'no snapshots available',
            })
            (path_json / f'{msa_id}.json').write_text(json.dumps(payload, indent=2))
            continue
        delta = float(selected_info['best_delta']) if np.isfinite(selected_info['best_delta']) else float('inf')
        within_tolerance = bool(delta <= float(tolerance))
        mask = masks[path_index]
        kept_rows = [index for index, keep in enumerate(mask) if keep]
        row_count = int(len(kept_rows))
        direction = 'down' if path_index < walk_down['mask'].shape[0] else 'up'
        snr_value = float(snr_series[path_index])
        pid_mean_pct = float(mean_series[path_index])
        pid_sd_pct = float(sd_series[path_index])
        pid_pair_count = int(pair_series[path_index])
        meff_value = float(meff_series[path_index])
        if within_tolerance:
            out_a3m = path_msas / f'{msa_id}.a3m'
            out_headers = [headers_start[index] for index in kept_rows]
            out_sequences = [sequences_start[index] for index in kept_rows]
            write_a3m(out_headers, out_sequences, out_a3m)
            payload = build_snapshot_payload(
                payload_base=payload_base,
                status='ok',
                target_value=target_value,
                tolerance=tolerance,
                selected_info=selected_info,
                row_count=row_count,
                snr_value=snr_value,
                pid_mean_pct=pid_mean_pct,
                pid_sd_pct=pid_sd_pct,
                pid_pair_count=pid_pair_count,
                meff_value=meff_value,
                direction=direction,
                path_index=path_index,
            )
        else:
            payload = build_not_reached_payload(
                payload_base=payload_base,
                target_value=target_value,
                tolerance=tolerance,
                selected_info=selected_info,
                row_count=row_count,
                snr_value=snr_value,
                pid_mean_pct=pid_mean_pct,
                pid_sd_pct=pid_sd_pct,
                pid_pair_count=pid_pair_count,
                meff_value=meff_value,
                direction=direction,
                path_index=path_index,
            )
        (path_json / f'{msa_id}.json').write_text(json.dumps(payload, indent=2))


def collect_available_ids(mode: str) -> List[str]:
    """Return candidate MSA ids from the msas/ directory for the chosen mode."""
    msa_dir = Path('msas')
    if mode == 'integer':
        return sorted([path.stem for path in msa_dir.glob('*.fasta')])
    return sorted([path.stem for path in msa_dir.glob('*.a3m')])


def collect_already_processed_ids(target_folders: List[str]) -> List[str]:
    """Return ids already processed across per-target folders."""
    done = set()
    for folder in target_folders:
        json_dir = Path(folder) / 'json'
        if not json_dir.is_dir():
            continue
        for path in json_dir.glob('*.json'):
            done.add(path.stem)
    return sorted(list(done))


def main() -> None:
    """CLI entry point for tuning MSAs to target below-budget column fractions using SNR sweeps."""
    parser = argparse.ArgumentParser(description='Tune MSAs to target fractions of projected columns below a folding budget using SNR row sweeps and ProDy bits.')
    parser.add_argument('--n-msas', type=int, required=True, help='Number of MSAs to process from msas/ that are not yet processed.')
    parser.add_argument('--mode', choices=['jackhmmer', 'integer'], required=True, help='Input format selector.')
    parser.add_argument('--strip-dots', action='store_true', help="Remove '.' characters during sanitization.")
    parser.add_argument('--cap-integer', type=int, default=5000, help='Row cap for integer mode.')
    parser.add_argument('--cap-jackhmmer', type=int, default=20000, help='Row cap for jackhmmer mode.')
    parser.add_argument('--downsample-method', choices=['simhash', 'uniform'], default='simhash', help='Downsampling method when rows exceed cap.')
    parser.add_argument('--downsample-seed', type=int, default=42, help='Seed for downsampling reproducibility.')
    parser.add_argument('--selection-seed', type=int, default=None, help='Seed for randomizing the order of remaining MSAs (default: system entropy).')
    parser.add_argument('--simhash-buckets-bits', type=int, default=12, help='Number of leading bits for SimHash bucketing.')
    parser.add_argument('--simhash-k', type=int, default=3, help='k-mer size for SimHash.')
    parser.add_argument('--max-row-gap-frac', type=float, default=0.6, help='Drop rows with gap fraction > this after anchor projection (keep anchor).')
    parser.add_argument('--row-drop-limit-frac', type=float, default=1.0, help='Fraction of starting rows to drop (0..1).')
    parser.add_argument('--min-rows-kept', type=int, default=3, help='Exact minimum rows to keep during the SNR sweep; overrides row-drop-limit-frac when set.')
    parser.add_argument('--meff-threshold', type=float, default=0.70, help='PID threshold (no-gap) for Meff sequence reweighting.')
    parser.add_argument('--folding-budget-bps', type=float, default=2.0, help='Bits/site folding budget threshold used to score projected columns.')
    parser.add_argument('--below-folding-budget-targets', type=str, default='0.00,0.02,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.30,0.40,0.50,0.60,0.75', help='Comma-separated list of target fractions of projected columns below the folding budget.')
    parser.add_argument('--below-folding-budget-tolerance', type=float, default=0.01, help='Tolerance around each below-budget fraction target for acceptance.')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level.')
    args = parser.parse_args()
    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    targets = [float(value) for value in str(args.below_folding_budget_targets).split(',') if str(value).strip()]
    available = collect_available_ids(args.mode)
    target_folders = [format_target_folder_name(target_value) for target_value in targets]
    processed = collect_already_processed_ids(target_folders)
    remaining = sorted(list(set(available) - set(processed)))
    if len(remaining) == 0:
        logging.info('Nothing to do. All MSAs appear processed for the configured targets.')
        return
    selection_rng = random.SystemRandom() if args.selection_seed is None else random.Random(int(args.selection_seed))
    order = remaining[:]
    selection_rng.shuffle(order)
    requested = min(args.n_msas, len(order))
    taken = 0
    Path('locks').mkdir(parents=True, exist_ok=True)
    for msa_id in order:
        if taken >= requested:
            break
        lock_path = acquire_lock(msa_id, Path('locks'))
        if lock_path is None:
            continue
        try:
            msa_dir = Path('msas')
            if args.mode == 'integer':
                in_path = msa_dir / f'{msa_id}.fasta'
                if not in_path.is_file():
                    logging.error(f'Missing FASTA: {in_path}')
                    continue
                headers, sequences = load_fasta(in_path)
            else:
                in_path = msa_dir / f'{msa_id}.a3m'
                if not in_path.is_file():
                    logging.error(f'Missing A3M: {in_path}')
                    continue
                headers, sequences = load_a3m(in_path)
            downsample_rng = np.random.default_rng(args.downsample_seed)
            headers_start, sequences_start = build_starting_alignment(
                headers=headers,
                sequences=sequences,
                mode=args.mode,
                cap_integer=args.cap_integer,
                cap_jackhmmer=args.cap_jackhmmer,
                strip_dots=args.strip_dots,
                downsample_method=args.downsample_method,
                simhash_bits=args.simhash_buckets_bits,
                simhash_k=args.simhash_k,
                max_row_gap_frac=args.max_row_gap_frac,
                rng=downsample_rng,
            )
            lookup_table = build_residue_lut()
            tokens = encode_char_array(sequences_start, lookup_table)
            start_rows = tokens.shape[0]
            if start_rows < 3:
                logging.info(f'[{msa_id}] insufficient rows after projection: {start_rows}')
                for target_value in targets:
                    folder = Path(format_target_folder_name(target_value))
                    (folder / 'json').mkdir(parents=True, exist_ok=True)
                    payload = {
                        'msa_id': msa_id,
                        'status': 'not_reached',
                        'target_fraction_below_folding_budget': float(target_value),
                        'tolerance': float(args.below_folding_budget_tolerance),
                        'pipeline_version': 'below_folding_budget_v1',
                        'message': 'insufficient_rows',
                        'start_rows': int(start_rows),
                        'folding_budget_bps': float(args.folding_budget_bps),
                    }
                    out_json = folder / 'json' / f'{msa_id}.json'
                    out_json.write_text(json.dumps(payload, indent=2))
                taken += 1
                continue
            pid_buffer = build_upper_triangle_pid_buffer(tokens, row_block=1024, col_block=1024, feat_block=512)
            adjacency = build_pid_adjacency_lists(pid_buffer, pid_threshold=float(args.meff_threshold))
            if args.min_rows_kept is not None:
                min_rows_kept = max(1, int(args.min_rows_kept))
            else:
                min_rows_kept = max(1, int(np.ceil((1.0 - float(args.row_drop_limit_frac)) * start_rows)))
            max_drops_allowed = max(0, start_rows - min_rows_kept)
            walk_down = perform_greedy_walk(
                tokens=tokens,
                buffer=pid_buffer,
                adjacency=adjacency,
                pid_threshold_for_meff=float(args.meff_threshold),
                direction_sign=-1,
                max_drops_allowed=max_drops_allowed,
                min_rows_kept=min_rows_kept,
            )
            walk_up = perform_greedy_walk(
                tokens=tokens,
                buffer=pid_buffer,
                adjacency=adjacency,
                pid_threshold_for_meff=float(args.meff_threshold),
                direction_sign=+1,
                max_drops_allowed=max_drops_allowed,
                min_rows_kept=min_rows_kept,
            )
            snr_start, _, _, _ = compute_snr_moments_from_state(*build_row_accumulators(pid_buffer)[:3])
            evaluate_snapshots_and_emit(
                msa_id=msa_id,
                headers_start=headers_start,
                sequences_start=sequences_start,
                tokens_start=tokens,
                walk_down=walk_down,
                walk_up=walk_up,
                targets=targets,
                tolerance=float(args.below_folding_budget_tolerance),
                snr_start=float(snr_start),
                row_drop_limit_frac=float(args.row_drop_limit_frac),
                meff_threshold=float(args.meff_threshold),
                min_rows_kept=int(min_rows_kept),
                folding_budget_bps=float(args.folding_budget_bps),
            )
            taken += 1
        except Exception as exc:
            logging.exception(f'[{msa_id}] failed with error: {exc}')
        finally:
            release_lock(lock_path)
    logging.info(f'Processed {taken} / {requested} MSAs in this worker.')


if __name__ == '__main__':
    main()
