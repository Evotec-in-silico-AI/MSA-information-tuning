from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from Bio import AlignIO
from .io_utils import atomic_pickle_dump, atomic_write_json, atomic_write_numpy_array
from .logger import logger
from prody import calcMSAOccupancy, calcShannonEntropy


def build_residue_lut() -> np.ndarray:
    """Build an ASCII lookup table for residue encoding where 0 denotes gap or unknown and 1..20 denote amino acids."""
    lut = np.zeros(128, dtype=np.int8)
    gap_unknown_chars = ["-", "X", "U", "O", "?", ".", "*", "B", "Z", "J"]
    for ch in gap_unknown_chars:
        lut[ord(ch)] = 0
    mapping = {
        "A": 1,
        "E": 2,
        "L": 3,
        "M": 4,
        "C": 5,
        "D": 6,
        "F": 7,
        "G": 8,
        "H": 9,
        "K": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "W": 16,
        "Y": 17,
        "T": 18,
        "V": 19,
        "I": 20,
    }
    for ch, val in mapping.items():
        lut[ord(ch)] = val
    return lut


def load_alignment_as_char_array(fasta_path: str) -> Tuple[List[int], np.ndarray]:
    """Load a FASTA alignment and return numeric IDs and a Unicode character matrix of shape (rows, cols)."""
    alignment = AlignIO.read(fasta_path, "fasta")
    ids: List[int] = []
    seqs: List[List[str]] = []
    for record in alignment:
        try:
            ids.append(int(str(record.id)))
        except ValueError as e:
            raise ValueError(f"Non-integer record.id encountered: {record.id}") from e
        seqs.append(list(str(record.seq)))
    char_array = np.array(seqs, dtype="<U1")
    logger.info(f"Loaded alignment with {len(ids)} rows and {char_array.shape[1]} columns")
    return ids, char_array


def encode_char_array(char_array: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Encode characters to integer residues using the lookup table defined by build_residue_lut."""
    upper = np.char.upper(char_array)
    b = np.char.encode(upper, "ascii", "replace")
    uint8_view = b.view(np.uint8).reshape(upper.shape)
    encoded = lut[uint8_view]
    out = encoded.astype(np.int16, copy=False)
    logger.info(f"Encoded alignment to token matrix with shape {out.shape}")
    return out


@dataclass
class UpperTrianglePID:
    n: int
    values: np.ndarray
    valid: np.ndarray

    @staticmethod
    def index(i: int, j: int, n: int) -> int:
        """Compute flat index for (i, j) with i < j under row-major upper-triangle packing."""
        if i >= j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)

    def get(self, i: int, j: int) -> Tuple[float, bool]:
        """Return (pid, valid) for a pair; diagonal pairs are invalid."""
        if i == j:
            return (0.0, False)
        if i > j:
            i, j = j, i
        k = self.index(i, j, self.n)
        return float(self.values[k]), bool(self.valid[k])


def upper_triangle_mask(block_rows: int, block_cols: int, i0: int, j0: int) -> np.ndarray:
    """Build a boolean mask selecting global pairs with i < j within a block."""
    i_idx = np.arange(int(i0), int(i0) + int(block_rows))[:, None]
    j_idx = np.arange(int(j0), int(j0) + int(block_cols))[None, :]
    return j_idx > i_idx


def compute_pair_block_pid_counts(Xi: np.ndarray, Xj: np.ndarray, feat_block: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute (matches_no_gap, denom_no_gap) for a block pair using the no-gap PID definition."""
    bi, L = Xi.shape
    bj = Xj.shape[0]
    matches = np.zeros((bi, bj), dtype=np.uint32)
    denom = np.zeros((bi, bj), dtype=np.uint32)
    for f0 in range(0, L, feat_block):
        f1 = min(f0 + feat_block, L)
        A = Xi[:, f0:f1]
        B = Xj[:, f0:f1]
        eq = A[:, None, :] == B[None, :, :]
        ng = (A[:, None, :] != 0) & (B[None, :, :] != 0)
        matches += (eq & ng).sum(axis=2, dtype=np.uint32)
        denom += ng.sum(axis=2, dtype=np.uint32)
    return matches, denom


def build_upper_triangle_pid_buffer(
    X: np.ndarray,
    *,
    row_block: int = 1024,
    col_block: int = 1024,
    feat_block: int = 512,
) -> UpperTrianglePID:
    """Build an upper-triangle buffer of pairwise PID (no-gap)."""
    X = np.asarray(X)
    n, L = X.shape
    size = n * (n - 1) // 2
    values = np.zeros(size, dtype=np.float32)
    valid = np.zeros(size, dtype=np.uint8)
    for i0 in range(0, n, row_block):
        i1 = min(n, i0 + row_block)
        Xi = X[i0:i1]
        bi = i1 - i0
        for j0 in range(i0, n, col_block):
            j1 = min(n, j0 + col_block)
            Xj = X[j0:j1]
            bj = j1 - j0
            fully_above = j0 >= i1
            m, d = compute_pair_block_pid_counts(Xi, Xj, feat_block)
            if fully_above:
                pos = d > 0
                if not pos.any():
                    continue
                ii, jj = np.nonzero(pos)
                gi = ii + i0
                gj = jj + j0
                idx = gi * n - gi * (gi + 1) // 2 + (gj - gi - 1)
                pid = m[pos].astype(np.float32) / d[pos].astype(np.float32)
                values[idx] = pid
                valid[idx] = 1
            else:
                tri = upper_triangle_mask(bi, bj, i0, j0)
                if not tri.any():
                    continue
                pos = tri & (d > 0)
                if not pos.any():
                    continue
                ii, jj = np.nonzero(pos)
                gi = ii + i0
                gj = jj + j0
                swap = gi >= gj
                if swap.any():
                    tmp = gi.copy()
                    gi[swap] = gj[swap]
                    gj[swap] = tmp[swap]
                idx = gi * n - gi * (gi + 1) // 2 + (gj - gi - 1)
                pid = m[pos].astype(np.float32) / d[pos].astype(np.float32)
                values[idx] = pid
                valid[idx] = 1
    return UpperTrianglePID(n=n, values=values, valid=valid)


def compute_meff_weights(buf: UpperTrianglePID, pid_threshold: float = 0.80) -> Tuple[np.ndarray, float]:
    """Compute Meff-style row weights from a no-gap PID buffer."""
    n = int(buf.n)
    counts = np.ones((n,), dtype=np.int64)
    idx = 0
    for i in range(n - 1):
        row_len = n - i - 1
        sl = slice(idx, idx + row_len)
        vals = buf.values[sl]
        valid = buf.valid[sl].astype(bool)
        keep = valid & (vals >= float(pid_threshold))
        if np.any(keep):
            js = np.nonzero(keep)[0] + (i + 1)
            counts[i] += int(js.size)
            np.add.at(counts, js.astype(np.int64), 1)
        idx += row_len
    weights = 1.0 / np.maximum(1.0, counts.astype(np.float64))
    meff = float(np.sum(weights))
    return weights.astype(np.float32, copy=False), meff


def per_position_coverage_and_entropy(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column coverage (non-gap fraction) and Shannon entropy (bits) ignoring gaps."""
    R, L = X.shape
    cov = np.zeros((L,), dtype=np.float32)
    ent = np.zeros((L,), dtype=np.float32)
    if R == 0 or L == 0:
        return cov, ent
    non_gap = (X != 0).astype(np.float32)
    cov = non_gap.sum(axis=0, dtype=np.float32) / float(R)
    for j in range(L):
        col = X[:, j]
        aa = col[(col > 0)]
        if aa.size == 0:
            ent[j] = 0.0
            continue
        counts = np.bincount(aa, minlength=21).astype(np.float64)[1:21]
        total = counts.sum()
        if total <= 0:
            ent[j] = 0.0
            continue
        p = counts / total
        ent[j] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    return cov, ent


def compute_bits_for_columns_inplace(
    counts_21xL: np.ndarray,
    cover_L: np.ndarray,
    bits_L: np.ndarray,
    update_cols: np.ndarray,
) -> None:
    """Recompute bits[L] in place for a set of columns whose counts/cover have changed."""
    if update_cols.size == 0:
        return
    counts_pos = counts_21xL[1:21, update_cols].astype(np.float64, copy=False)
    total = np.clip(cover_L[update_cols].astype(np.float64, copy=False), 1.0, None)
    probs = counts_pos / total
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0.0, np.log2(probs), 0.0)
        entropy_bits = -np.sum(probs * logp, axis=0)
    bits_L[update_cols] = np.log2(20.0) - entropy_bits


@dataclass
class RowAccumulators:
    sum_ordered: np.ndarray
    sumsq_ordered: np.ndarray
    cnt_ordered: np.ndarray
    total_sum: float
    total_sumsq: float
    total_pairs: int


def build_row_accumulators(buf: UpperTrianglePID) -> RowAccumulators:
    """Build ordered-pair accumulators and global moments used to compute SNR cheaply per drop."""
    n = buf.n
    sum_first = np.zeros(n, dtype=np.float64)
    sum_second = np.zeros(n, dtype=np.float64)
    sumsq_first = np.zeros(n, dtype=np.float64)
    sumsq_second = np.zeros(n, dtype=np.float64)
    cnt_first = np.zeros(n, dtype=np.int64)
    cnt_second = np.zeros(n, dtype=np.int64)
    idx = 0
    for i in range(n - 1):
        row_len = n - i - 1
        sl = slice(idx, idx + row_len)
        vals = buf.values[sl].astype(np.float64, copy=False)
        msk = buf.valid[sl].astype(bool, copy=False)
        v = vals[msk]
        if v.size:
            sum_first[i] += v.sum()
            sumsq_first[i] += (v * v).sum()
            cnt_first[i] += int(v.size)
            js = np.nonzero(msk)[0] + (i + 1)
            if js.size:
                np.add.at(sum_second, js, v)
                np.add.at(sumsq_second, js, v * v)
                np.add.at(cnt_second, js, 1)
        idx += row_len
    sum_ordered = 2.0 * (sum_first + sum_second)
    sumsq_ordered = 2.0 * (sumsq_first + sumsq_second)
    cnt_ordered = 2 * (cnt_first + cnt_second)
    total_sum = float(sum_ordered.sum() / 2.0)
    total_sumsq = float(sumsq_ordered.sum() / 2.0)
    total_pairs = int(cnt_ordered.sum() // 2)
    return RowAccumulators(
        sum_ordered=sum_ordered,
        sumsq_ordered=sumsq_ordered,
        cnt_ordered=cnt_ordered,
        total_sum=total_sum,
        total_sumsq=total_sumsq,
        total_pairs=total_pairs,
    )


def snr_from_moments(total_sum: float, total_sumsq: float, total_pairs: int) -> Tuple[float, float, float]:
    """Return (snr, mean_percent, sd_percent) for no-gap PID fractions."""
    if total_pairs <= 1:
        return (float("nan"), float("nan"), float("nan"))
    mean_f = total_sum / float(total_pairs)
    var_f = max(total_sumsq / float(total_pairs) - mean_f * mean_f, 0.0)
    mean_pct = 100.0 * mean_f
    sd_pct = 100.0 * float(np.sqrt(var_f))
    if sd_pct <= 1e-12 or not np.isfinite(sd_pct):
        return (float("inf"), mean_pct, sd_pct)
    return (mean_pct / sd_pct, mean_pct, sd_pct)


def remove_row_in_place(buf: UpperTrianglePID, ra: RowAccumulators, active_mask: np.ndarray, row_index: int) -> None:
    """Deactivate a row and update row accumulators and the active mask in place."""
    ra.total_sum -= 0.5 * float(ra.sum_ordered[row_index])
    ra.total_sumsq -= 0.5 * float(ra.sumsq_ordered[row_index])
    ra.total_pairs -= int(ra.cnt_ordered[row_index]) // 2
    n = buf.n
    for k in range(n):
        if k == row_index or not active_mask[k]:
            continue
        pid_value, ok = buf.get(min(k, row_index), max(k, row_index))
        if not ok:
            continue
        v = float(pid_value)
        ra.sum_ordered[k] -= 2.0 * v
        ra.sumsq_ordered[k] -= 2.0 * (v * v)
        ra.cnt_ordered[k] -= 2
    ra.sum_ordered[row_index] = 0.0
    ra.sumsq_ordered[row_index] = 0.0
    ra.cnt_ordered[row_index] = 0
    active_mask[row_index] = False
    ra.total_sum = float(ra.sum_ordered.sum() / 2.0)
    ra.total_sumsq = float(ra.sumsq_ordered.sum() / 2.0)
    ra.total_pairs = int(ra.cnt_ordered.sum() // 2)


def copy_row_accumulators(ra: RowAccumulators) -> RowAccumulators:
    """Create a deep copy of the row accumulators structure."""
    return RowAccumulators(
        sum_ordered=ra.sum_ordered.copy(),
        sumsq_ordered=ra.sumsq_ordered.copy(),
        cnt_ordered=ra.cnt_ordered.copy(),
        total_sum=float(ra.total_sum),
        total_sumsq=float(ra.total_sumsq),
        total_pairs=int(ra.total_pairs),
    )


def compute_p10_prody_effmask(char_array: np.ndarray) -> float:
    """Compute bits/site p10 using ProDy with a 0.5 occupancy mask and log2(20) maximum."""
    if char_array.ndim != 2 or char_array.size == 0:
        return 0.0
    char_bytes = char_array.astype("S1", copy=False)
    occupancy_fraction = np.asarray(calcMSAOccupancy(char_bytes, occ="res", count=False), dtype=float)
    effective_mask = occupancy_fraction >= 0.5
    if not np.any(effective_mask):
        return 0.0
    entropy_bits = np.asarray(calcShannonEntropy(char_bytes, ambiguity=True, omitgaps=True), dtype=float)
    bits_site = np.log2(20.0) - entropy_bits
    p10_value = float(np.quantile(bits_site[effective_mask], 0.10))
    return p10_value


def run_single_direction_search(
    X: np.ndarray,
    char_selected: np.ndarray,
    pid_buf: UpperTrianglePID,
    ra_base: RowAccumulators,
    keep_mask_base: np.ndarray,
    counts_21xL_base: np.ndarray,
    cover_L_base: np.ndarray,
    bits_L_base: np.ndarray,
    row_non_gap_cols: List[np.ndarray],
    row_tokens_at_cols: List[np.ndarray],
    direction: int,
    max_drops: int,
    target_p10: float,
    tolerance: float,
) -> Dict[str, Any]:
    """Run a greedy SNR search in one direction, dropping rows up to max_drops, and track ProDy-based p10 proximity."""
    keep_mask = keep_mask_base.copy()
    active_mask = keep_mask.copy()
    ra = copy_row_accumulators(ra_base)
    counts_21xL = counts_21xL_base.copy()
    cover_L = cover_L_base.copy()
    bits_L = bits_L_base.copy()
    n_rows = X.shape[0]
    n_kept = int(keep_mask.sum())
    snr_init, _, _ = snr_from_moments(ra.total_sum, ra.total_sumsq, ra.total_pairs)
    p10_init = compute_p10_prody_effmask(char_selected[keep_mask])
    best_keep_mask = keep_mask.copy()
    best_proximity = abs(float(p10_init) - float(target_p10))
    tolerance_hit = best_proximity <= float(tolerance)
    drops = 0
    while drops < int(max_drops) and n_kept > 1:
        best_row_index = -1
        best_snr_value = -float("inf") if direction > 0 else float("inf")
        for r in range(1, n_rows):
            if not keep_mask[r] or not active_mask[r]:
                continue
            remaining_pairs = ra.total_pairs - int(ra.cnt_ordered[r]) // 2
            if remaining_pairs <= 1:
                continue
            sum_pairs = ra.total_sum - 0.5 * float(ra.sum_ordered[r])
            sumsq_pairs = ra.total_sumsq - 0.5 * float(ra.sumsq_ordered[r])
            snr_try, _, _ = snr_from_moments(sum_pairs, sumsq_pairs, remaining_pairs)
            if direction > 0:
                if snr_try > best_snr_value:
                    best_snr_value = snr_try
                    best_row_index = int(r)
            else:
                if snr_try < best_snr_value:
                    best_snr_value = snr_try
                    best_row_index = int(r)
        if best_row_index < 0:
            break
        commit_cols = row_non_gap_cols[best_row_index]
        commit_tokens = row_tokens_at_cols[best_row_index]
        if commit_cols.size:
            np.add.at(counts_21xL, (commit_tokens, commit_cols), -1)
            cover_L[commit_cols] -= 1
            compute_bits_for_columns_inplace(counts_21xL, cover_L, bits_L, commit_cols)
        keep_mask[best_row_index] = False
        remove_row_in_place(pid_buf, ra, active_mask, best_row_index)
        n_kept -= 1
        p10_now = compute_p10_prody_effmask(char_selected[keep_mask])
        proximity_now = abs(float(p10_now) - float(target_p10))
        if proximity_now + 1e-12 < best_proximity:
            best_proximity = proximity_now
            best_keep_mask = keep_mask.copy()
        if proximity_now <= float(tolerance):
            tolerance_hit = True
        drops += 1
    snr_final, _, _ = snr_from_moments(ra.total_sum, ra.total_sumsq, ra.total_pairs)
    p10_final_path = compute_p10_prody_effmask(char_selected[keep_mask])
    best_p10 = compute_p10_prody_effmask(char_selected[best_keep_mask])
    info = {
        "initial_snr": float(snr_init),
        "final_snr": float(snr_final),
        "initial_p10": float(p10_init),
        "final_p10_path_end": float(p10_final_path),
        "best_p10": float(best_p10),
        "best_keep_mask": best_keep_mask,
        "tolerance_hit": bool(tolerance_hit),
        "drops_attempted": int(drops),
    }
    return info


def greedy_tune_rows_with_snr(
    X: np.ndarray,
    char_selected: np.ndarray,
    target_p10: float,
    tolerance: float,
    max_drop_fraction: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute two greedy SNR searches, up and down, using ProDy-based p10 for selection."""
    n_rows, _ = X.shape
    keep_mask_base = np.ones(n_rows, dtype=bool)
    keep_mask_base[0] = True
    max_drops = int(max(0, int(np.floor(float(max_drop_fraction) * n_rows))))
    logger.info(f"SNR loop: max_drop_fraction={float(max_drop_fraction):.4f} max_drops={int(max_drops)} start_rows={int(n_rows)}")
    row_non_gap_cols, row_tokens_at_cols = precompute_row_non_gap_indices_and_tokens(X)
    counts_21xL_base, cover_L_base, bits_L_base = initialize_column_stats_from_kept(X, keep_mask_base)
    p10_cur = compute_p10_prody_effmask(char_selected[keep_mask_base])
    if abs(p10_cur - target_p10) <= tolerance:
        kept_idx = np.nonzero(keep_mask_base)[0]
        return X[keep_mask_base].copy(), {"method": "snr", "final_p10": float(p10_cur), "kept_idx": kept_idx, "search_info": {}}
    pid_buf = build_upper_triangle_pid_buffer(X)
    ra_base = build_row_accumulators(pid_buf)
    direction_first = +1 if float(target_p10) >= float(p10_cur) else -1
    info_first = run_single_direction_search(
        X=X,
        char_selected=char_selected,
        pid_buf=pid_buf,
        ra_base=ra_base,
        keep_mask_base=keep_mask_base,
        counts_21xL_base=counts_21xL_base,
        cover_L_base=cover_L_base,
        bits_L_base=bits_L_base,
        row_non_gap_cols=row_non_gap_cols,
        row_tokens_at_cols=row_tokens_at_cols,
        direction=int(direction_first),
        max_drops=int(max_drops),
        target_p10=float(target_p10),
        tolerance=float(tolerance),
    )
    logger.info(
        f"SNR search {'up' if direction_first > 0 else 'down'}: "
        f"snr_init={info_first['initial_snr']:.4f} snr_final={info_first['final_snr']:.4f} "
        f"p10_init={info_first['initial_p10']:.4f} p10_best={info_first['best_p10']:.4f} "
        f"p10_end={info_first['final_p10_path_end']:.4f} drops={info_first['drops_attempted']} "
        f"tol_hit={info_first['tolerance_hit']}"
    )
    if info_first["tolerance_hit"]:
        best_mask = info_first["best_keep_mask"]
        best_array = X[best_mask].copy()
        kept_idx = np.nonzero(best_mask)[0]
        best_p10 = compute_p10_prody_effmask(char_selected[best_mask])
        buf_best = build_upper_triangle_pid_buffer(best_array)
        ra_best = build_row_accumulators(buf_best)
        snr_used_value, _, _ = snr_from_moments(ra_best.total_sum, ra_best.total_sumsq, ra_best.total_pairs)
        return best_array, {
            "method": "snr",
            "final_p10": float(best_p10),
            "kept_idx": kept_idx,
            "search_info": {"first": info_first},
            "snr_initial": float(info_first["initial_snr"]),
            "snr_used": float(snr_used_value),
        }
    direction_second = -direction_first
    info_second = run_single_direction_search(
        X=X,
        char_selected=char_selected,
        pid_buf=pid_buf,
        ra_base=ra_base,
        keep_mask_base=keep_mask_base,
        counts_21xL_base=counts_21xL_base,
        cover_L_base=cover_L_base,
        bits_L_base=bits_L_base,
        row_non_gap_cols=row_non_gap_cols,
        row_tokens_at_cols=row_tokens_at_cols,
        direction=int(direction_second),
        max_drops=int(max_drops),
        target_p10=float(target_p10),
        tolerance=float(tolerance),
    )
    logger.info(
        f"SNR search {'up' if direction_second > 0 else 'down'}: "
        f"snr_init={info_second['initial_snr']:.4f} snr_final={info_second['final_snr']:.4f} "
        f"p10_init={info_second['initial_p10']:.4f} p10_best={info_second['best_p10']:.4f} "
        f"p10_end={info_second['final_p10_path_end']:.4f} drops={info_second['drops_attempted']} "
        f"tol_hit={info_second['tolerance_hit']}"
    )
    prox_first = abs(float(info_first["best_p10"]) - float(target_p10))
    prox_second = abs(float(info_second["best_p10"]) - float(target_p10))
    chosen = "first" if prox_first <= prox_second else "second"
    chosen_info = info_first if chosen == "first" else info_second
    best_mask = chosen_info["best_keep_mask"]
    best_array = X[best_mask].copy()
    kept_idx = np.nonzero(best_mask)[0]
    final_p10 = compute_p10_prody_effmask(char_selected[best_mask])
    buf_best = build_upper_triangle_pid_buffer(best_array)
    ra_best = build_row_accumulators(buf_best)
    snr_used_value, _, _ = snr_from_moments(ra_best.total_sum, ra_best.total_sumsq, ra_best.total_pairs)
    return best_array, {
        "method": "snr",
        "final_p10": float(final_p10),
        "kept_idx": kept_idx,
        "search_info": {"first": info_first, "second": info_second, "chosen": chosen},
        "snr_initial": float(info_first["initial_snr"]),
        "snr_used": float(snr_used_value),
    }


def write_sidecars_and_meta(
    output_path: str,
    optimized_array: np.ndarray,
    selected_cols: List[int],
    selected_ids_final: List[int],
    selected_rows_original: List[int],
    p10_target: float,
    p10_achieved: float,
    p10_tolerance: float,
    row_weights_theta070: np.ndarray,
    row_weights_theta080: np.ndarray,
    meff_theta070: float,
    meff_theta080: float,
    snr_initial: float,
    snr_used: float,
    p10_initial: float,
    p10_used: float,
) -> str:
    """Write array, coverage, entropy, row weights (θ=0.70/0.80), SNR and JSON metadata; return JSON path."""
    atomic_pickle_dump(output_path, optimized_array)
    logger.info(f"Wrote compressed alignment to {output_path}")
    weights_path_070 = Path(str(Path(output_path).with_suffix("")) + "_row_weights_theta0p70.npy")
    weights_path_080 = Path(str(Path(output_path).with_suffix("")) + "_row_weights_theta0p80.npy")
    atomic_write_numpy_array(weights_path_070, row_weights_theta070.astype(np.float32, copy=False))
    atomic_write_numpy_array(weights_path_080, row_weights_theta080.astype(np.float32, copy=False))
    bits_site, coverage, entropy_bits, eff_mask = compute_bits_site_and_masks(optimized_array)
    cov_path = Path(str(Path(output_path).with_suffix("")) + "_coverage.npy")
    ent_path = Path(str(Path(output_path).with_suffix("")) + "_entropy_bits.npy")
    atomic_write_numpy_array(cov_path, coverage.astype(np.float32, copy=False))
    atomic_write_numpy_array(ent_path, entropy_bits.astype(np.float32, copy=False))
    rounded = round(float(p10_target), 2)
    json_path = str(Path(output_path).with_suffix("")) + f"_p10_{rounded}.json"
    meta = {
        "bits_site_p10_target": float(p10_target),
        "bits_site_p10_initial": float(p10_initial),
        "bits_site_p10_used": float(p10_used),
        "bits_site_p10_achieved": float(p10_achieved),
        "bits_site_p10_tolerance": float(p10_tolerance),
        "n_selected_rows": int(optimized_array.shape[0]),
        "n_selected_cols": int(optimized_array.shape[1]),
        "selected_cols": [int(c) for c in selected_cols],
        "selected_ids": [int(v) for v in selected_ids_final],
        "selected_rows": [int(v) for v in selected_rows_original],
        "array_path": str(output_path),
        "meff_theta0p70": float(meff_theta070),
        "meff_fraction_theta0p70": float(meff_theta070 / max(1, int(optimized_array.shape[0]))),
        "meff_theta0p80": float(meff_theta080),
        "meff_fraction_theta0p80": float(meff_theta080 / max(1, int(optimized_array.shape[0]))),
        "row_weights_theta0p70_path": str(weights_path_070),
        "row_weights_theta0p80_path": str(weights_path_080),
        "coverage_path": str(cov_path),
        "entropy_bits_path": str(ent_path),
        "snr_initial": float(snr_initial),
        "snr_used": float(snr_used),
        "compression_version": "prody_cov_entropy+snr_v1",
    }
    atomic_write_json(json_path, meta)
    logger.info(f"Wrote compression JSON to {json_path}")
    return json_path


def tune_to_p10(
    paths: Dict[str, Any],
    *,
    p10_target: float,
    p10_tolerance: float,
    apply_row_pruning: bool,
    max_row_drops_fraction: float,
    do_swap_refinement: bool,
    swap_budget: int,
    row_shuffle_seed: int,
) -> Tuple[int, int, float, str, float, float, float, float]:
    """Tune alignment to a target p10 using ProDy-based p10. Return sizes, achieved p10, json path, and SNR/p10 provenance."""
    alignment_path = paths["alignment"]
    output_path = paths["compressed_alignment"]
    ids, char_array = load_alignment_as_char_array(alignment_path)
    try:
        target_idx = ids.index(0)
    except ValueError:
        raise ValueError("Protein with id=0 not found in alignment IDs")
    lut = build_residue_lut()
    X_tokens = encode_char_array(char_array, lut)
    X_shuffled, row_ids_shuffled, row_indices_shuffled = shuffle_rows_with_anchor_top(
        X_tokens, ids, target_idx, seed=int(row_shuffle_seed)
    )
    char_shuffled = shuffle_rows_with_anchor_top_chars(char_array, ids, target_idx, seed=int(row_shuffle_seed))
    X_selected, selected_cols = select_anchor_kept_columns(X_shuffled)
    char_selected, _ = select_anchor_kept_columns_chars(char_shuffled)
    p10_pre = compute_p10_prody_effmask(char_selected)
    logger.info(f"Pre-prune: rows={X_selected.shape[0]} cols={X_selected.shape[1]} p10_bits_prody_effmask={p10_pre:.4f}")
    if apply_row_pruning:
        optimized_array, details = greedy_tune_rows_with_snr(
            X=X_selected,
            char_selected=char_selected,
            target_p10=float(p10_target),
            tolerance=float(p10_tolerance),
            max_drop_fraction=float(max_row_drops_fraction),
        )
        kept_idx_after = np.array(details["kept_idx"], dtype=np.int64)
        final_p10 = float(details["final_p10"])
        snr_initial_value = float(details.get("snr_initial", float("nan")))
        snr_used_value = float(details.get("snr_used", float("nan")))
    else:
        optimized_array = X_selected
        kept_idx_after = np.arange(optimized_array.shape[0], dtype=np.int64)
        final_p10 = compute_p10_prody_effmask(char_selected[kept_idx_after])
        buf_full = build_upper_triangle_pid_buffer(optimized_array)
        ra_full = build_row_accumulators(buf_full)
        snr_val, _, _ = snr_from_moments(ra_full.total_sum, ra_full.total_sumsq, ra_full.total_pairs)
        snr_initial_value = float(snr_val)
        snr_used_value = float(snr_val)
    selected_ids_final = [int(row_ids_shuffled[i]) for i in kept_idx_after.tolist()]
    selected_rows_original = [int(row_indices_shuffled[i]) for i in kept_idx_after.tolist()]
    logger.info(f"Post-prune: rows={optimized_array.shape[0]} cols={optimized_array.shape[1]} p10_bits_prody_effmask={final_p10:.4f}")
    pid_buf_final = build_upper_triangle_pid_buffer(optimized_array)
    row_weights_070, meff_070 = compute_meff_weights(pid_buf_final, pid_threshold=0.70)
    row_weights_080, meff_080 = compute_meff_weights(pid_buf_final, pid_threshold=0.80)
    json_path = write_sidecars_and_meta(
        output_path=str(output_path),
        optimized_array=optimized_array,
        selected_cols=selected_cols,
        selected_ids_final=selected_ids_final,
        selected_rows_original=selected_rows_original,
        p10_target=float(p10_target),
        p10_achieved=float(final_p10),
        p10_tolerance=float(p10_tolerance),
        row_weights_theta070=row_weights_070,
        row_weights_theta080=row_weights_080,
        meff_theta070=float(meff_070),
        meff_theta080=float(meff_080),
        snr_initial=float(snr_initial_value),
        snr_used=float(snr_used_value),
        p10_initial=float(p10_pre),
        p10_used=float(final_p10),
    )
    return int(optimized_array.shape[0]), int(optimized_array.shape[1]), float(final_p10), str(json_path), float(snr_initial_value), float(snr_used_value), float(p10_pre), float(final_p10)


def tokens_to_char_array(encoded_tokens: np.ndarray) -> np.ndarray:
    """Return a 2D byte-character array ('S1') for ProDy metrics."""
    lut = np.array([b"-", b"A", b"E", b"L", b"M", b"C", b"D", b"F", b"G", b"H",
                    b"K", b"N", b"P", b"Q", b"R", b"S", b"W", b"Y", b"T", b"V", b"I"], dtype="S1")
    flat = encoded_tokens.ravel().astype(np.int64)
    flat_clipped = np.clip(flat, 0, 20)
    char_flat = lut[flat_clipped]
    return char_flat.reshape(encoded_tokens.shape)


def compute_bits_site_and_masks(encoded_tokens: np.ndarray, effective_occupancy: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute bits/site, coverage, entropy_bits, and effective-column mask via ProDy."""
    char_array = tokens_to_char_array(encoded_tokens)
    occupancy = np.asarray(calcMSAOccupancy(char_array, occ="res", count=False), dtype=np.float32)
    entropy_bits = np.asarray(calcShannonEntropy(char_array, ambiguity=True, omitgaps=True), dtype=np.float32)
    bits_site = (np.log2(20.0) - entropy_bits.astype(np.float64)).astype(np.float64)
    eff_mask = occupancy >= float(effective_occupancy)
    return bits_site, occupancy.astype(np.float32), entropy_bits.astype(np.float32), eff_mask


def select_anchor_kept_columns(X: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Select columns where the anchor (row 0) is non-gap for the numeric array."""
    if X.shape[0] == 0 or X.shape[1] == 0:
        logger.info("Selected 0 of 0 columns (anchor non-gap only)")
        return X, []
    total_cols = int(X.shape[1])
    mask = X[0, :] != 0
    kept = int(mask.sum())
    cols = np.nonzero(mask)[0].tolist()
    logger.info(f"Selected {kept} of {total_cols} columns (anchor non-gap only)")
    return X[:, mask], cols


def shuffle_rows_with_anchor_top(X: np.ndarray, ids: List[int], target_idx: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place the anchor row on top and shuffle the rest deterministically for the numeric array."""
    N = X.shape[0]
    top_row = X[target_idx: target_idx + 1, :]
    rest_positions = [i for i in range(N) if i != target_idx]
    rest_rows = X[rest_positions, :]
    rest_ids = np.array([ids[i] for i in rest_positions], dtype=np.int64)
    rest_orig_idx = np.array(rest_positions, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(rest_rows.shape[0])
    shuffled_rest_rows = rest_rows[perm]
    shuffled_rest_ids = rest_ids[perm]
    shuffled_rest_orig_idx = rest_orig_idx[perm]
    shuffled_array = np.vstack([top_row, shuffled_rest_rows]).astype(np.int16, copy=False)
    row_ids_shuffled = np.concatenate(([ids[target_idx]], shuffled_rest_ids))
    row_indices_shuffled = np.concatenate(([target_idx], shuffled_rest_orig_idx))
    logger.info(f"Shuffled rows with anchor on top; shape now {shuffled_array.shape}")
    return shuffled_array, row_ids_shuffled, row_indices_shuffled


def shuffle_rows_with_anchor_top_chars(char_array: np.ndarray, ids: List[int], target_idx: int, seed: int) -> np.ndarray:
    """Place the anchor row on top and shuffle the rest deterministically for the character array."""
    n_rows = char_array.shape[0]
    top_row = char_array[target_idx:target_idx + 1, :]
    rest_positions = [i for i in range(n_rows) if i != target_idx]
    rest_rows = char_array[rest_positions, :]
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(rest_rows.shape[0])
    shuffled_rest_rows = rest_rows[perm]
    shuffled = np.vstack([top_row, shuffled_rest_rows]).astype(char_array.dtype, copy=False)
    return shuffled


def select_anchor_kept_columns_chars(char_array: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Select columns where the anchor (row 0) is non-gap for the character array."""
    if char_array.shape[0] == 0 or char_array.shape[1] == 0:
        return char_array, []
    mask = char_array[0, :] != "-"
    cols = np.nonzero(mask)[0].tolist()
    selected = char_array[:, mask]
    return selected, cols


def precompute_row_non_gap_indices_and_tokens(X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Precompute per-row lists of non-gap column indices and their token values for fast updates."""
    rows, cols = np.nonzero(X)
    tokens = X[rows, cols]
    n_rows = X.shape[0]
    idx_lists: List[np.ndarray] = [np.empty((0,), dtype=np.int64) for _ in range(n_rows)]
    tok_lists: List[np.ndarray] = [np.empty((0,), dtype=np.int16) for _ in range(n_rows)]
    if rows.size == 0:
        return idx_lists, tok_lists
    breaks = np.where(np.diff(rows) != 0)[0] + 1
    splits = np.split(np.arange(rows.size), breaks)
    for block in splits:
        r = int(rows[block[0]])
        idx_lists[r] = cols[block].astype(np.int64, copy=False)
        tok_lists[r] = tokens[block].astype(np.int16, copy=False)
    return idx_lists, tok_lists


def initialize_column_stats_from_kept(X: np.ndarray, keep_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build counts[21,L], cover[L], and bits[L] from the current kept mask for numeric updates."""
    kept = X[keep_mask]
    if kept.size == 0:
        L = X.shape[1]
        return np.zeros((21, L), dtype=np.int32), np.zeros((L,), dtype=np.int32), np.zeros((L,), dtype=np.float64)
    rows_idx, cols_idx = np.nonzero(kept)
    token_vals = kept[rows_idx, cols_idx].astype(np.int32, copy=False)
    L = kept.shape[1]
    flat_index = token_vals + 21 * cols_idx
    counts_flat = np.bincount(flat_index, minlength=21 * L)
    counts_21xL = counts_flat.reshape(21, L)
    cover_L = np.bincount(cols_idx, minlength=L).astype(np.int32, copy=False)
    counts_pos = counts_21xL[1:21, :].astype(np.float64, copy=False)
    total = np.clip(cover_L.astype(np.float64, copy=False), 1.0, None)
    probs = counts_pos / total
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(probs > 0.0, np.log2(probs), 0.0)
        entropy_bits = -np.sum(probs * logp, axis=0)
    bits_L = (np.log2(20.0) - entropy_bits).astype(np.float64, copy=False)
    return counts_21xL.astype(np.int32, copy=False), cover_L, bits_L
