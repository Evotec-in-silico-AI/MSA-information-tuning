from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_comm
from prody import (
    calcMSAOccupancy,
    calcShannonEntropy,
    buildMutinfoMatrix,
    applyMutinfoCorr,
)
from scipy.sparse.linalg import eigsh
from modules import alignment_compression as ac


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for compressed-MSA statistics.
    """
    parser = argparse.ArgumentParser(description="MSA statistics on compressed alignments.")
    parser.add_argument("--root", type=str, default="runs", help="Root directory containing run folders.")
    parser.add_argument(
        "--msa_tag",
        type=str,
        default="compressed_msa",
        help="Tag for output folder naming under run_root/msa_statistics/.",
    )
    parser.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=["msa_p10_sweep", "msa_p10_sweep_null"],
        help="Experiment families to include.",
    )
    parser.add_argument(
        "--group_by",
        type=str,
        nargs="+",
        default=["shard_group", "p10_target", "family", "control_rep"],
        help="Grouping columns present in shard manifests and selected rows.",
    )
    parser.add_argument("--mi_top_quantile", type=float, default=0.99, help="Quantile for MI(APC) summaries.")
    parser.add_argument(
        "--effective_col_min_occupancy",
        type=float,
        default=0.50,
        help="Minimum occupancy to treat a column as effective for bits/site proportions.",
    )
    parser.add_argument(
        "--pid_threshold",
        type=float,
        default=0.70,
        help="No-gap PID threshold used for MSA similarity graph.",
    )
    parser.add_argument(
        "--graph_core_cuts",
        type=str,
        default="1,2,3,5",
        help="Comma-separated k-core thresholds to summarize (% nodes with core>=k).",
    )
    parser.add_argument("--write_csv", action="store_true", default=True, help="Write CSV outputs.")
    parser.add_argument("--write_xlsx", action="store_true", default=True, help="Write XLSX outputs.")
    parser.add_argument("--write_parquet", action="store_true", default=True, help="Write Parquet outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes for group computations.",
    )
    parser.add_argument(
        "--meff_thetas",
        type=str,
        default="0.85,0.80,0.75,0.70,0.65,0.62",
        help="Comma-separated list of PID identity cutoffs used to compute Meff sweeps.",
    )
    return parser.parse_args(argv)


def set_global_seed(seed_value: int) -> None:
    """
    Seed NumPy RNG for reproducibility.
    """
    np.random.seed(int(seed_value))


def find_run_roots(root_path: Path) -> List[Path]:
    """
    Find run roots that contain both 'experiments' and 'merged' subdirectories.
    """
    run_roots: List[Path] = []
    if not root_path.exists():
        return run_roots
    for experiments_dir in root_path.rglob("experiments"):
        candidate_root = experiments_dir.parent
        if (candidate_root / "merged").exists():
            run_roots.append(candidate_root)
    deduped: List[Path] = []
    seen: set = set()
    for p in sorted(run_roots):
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def discover_shard_manifests(run_root: Path, families: List[str]) -> List[Path]:
    """
    Locate shard manifest CSV files for requested families under a run root.
    """
    paths: List[Path] = []
    for family in families:
        base = run_root / "experiments" / family
        if base.exists():
            paths.extend(sorted(base.rglob("shard_manifest.csv")))
    return paths


def read_manifest_csvs(manifest_paths: List[Path]) -> pd.DataFrame:
    """
    Read and combine shard manifest CSVs into a single DataFrame with 'family' and 'manifest_path'.
    """
    frames: List[pd.DataFrame] = []
    for path in manifest_paths:
        try:
            df = pd.read_csv(path)
            df["manifest_path"] = str(path)
            if "experiments" in path.parts:
                fam = path.parts[path.parts.index("experiments") + 1]
            else:
                fam = ""
            df["family"] = fam
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def find_fold_queue_merged(run_root: Path, protein_name: Optional[str]) -> pd.DataFrame:
    """
    Load the run-level merged fold queue CSV that includes structure metrics.
    """
    merged_dir = run_root / "merged"
    if not merged_dir.exists():
        return pd.DataFrame()
    candidate = None
    if protein_name:
        p = merged_dir / f"{protein_name}_fold_queue_merged.csv"
        candidate = p if p.exists() else None
    if candidate is None:
        candidates = sorted(merged_dir.glob("*_fold_queue_merged.csv"))
        if candidates:
            candidate = candidates[0]
    if candidate is None or not candidate.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(candidate)
    except Exception:
        return pd.DataFrame()


def load_compressed_alignment_matrix(pkl_path: Path) -> np.ndarray:
    """
    Load the compressed integer token matrix from pickle.
    """
    import pickle
    with open(str(pkl_path), "rb") as handle:
        arr = np.asarray(pickle.load(handle))
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"Compressed alignment must be non-empty 2D; got shape {arr.shape}")
        return arr.astype(np.int16, copy=False)


def build_reverse_residue_lut() -> Dict[int, str]:
    """
    Build reverse LUT mapping token integers to residue characters.
    """
    return {
        0: "-",
        1: "A",
        2: "E",
        3: "L",
        4: "M",
        5: "C",
        6: "D",
        7: "F",
        8: "G",
        9: "H",
        10: "K",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "W",
        17: "Y",
        18: "T",
        19: "V",
        20: "I",
    }


def int_matrix_to_char_array(encoded: np.ndarray, reverse_lut: Dict[int, str]) -> np.ndarray:
    """
    Return a 2D byte-character array (dtype='S1') for entropy and MI computations.
    """
    lut_bytes: Dict[int, bytes] = {k: v.encode("ascii") for k, v in reverse_lut.items()}
    n_rows, n_cols = encoded.shape
    flat = encoded.ravel()
    out_flat = np.fromiter((lut_bytes.get(int(t), b"-") for t in flat), dtype="S1", count=flat.size)
    return out_flat.reshape(n_rows, n_cols)


def compute_entropy_and_coverage(char_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-column Shannon entropy (bits, gaps excluded) and occupancy.
    """
    entropy_bits = np.asarray(calcShannonEntropy(char_array))
    occupancy = np.asarray(calcMSAOccupancy(char_array, occ="res"))
    return entropy_bits.astype(np.float64), occupancy.astype(np.float64)


def compute_bits_site_from_entropy(entropy_bits: np.ndarray) -> np.ndarray:
    """
    Compute bits/site as log2(20) - entropy_bits.
    """
    max_bits = math.log2(20.0)
    return max_bits - np.asarray(entropy_bits, dtype=np.float64)


def compute_weighted_bits_site_theta070(encoded: np.ndarray, occupancy: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute Meff-weighted per-column bits/site using θ=0.70 row weights and effective-column mean.
    """
    pid_buffer = ac.build_upper_triangle_pid_buffer(encoded)
    row_weights, _ = ac.compute_meff_weights(pid_buffer, pid_threshold=0.70)
    w = np.asarray(row_weights, dtype=np.float64).reshape(-1, 1)
    w /= (w.sum() + 1e-12)
    n_rows, n_cols = encoded.shape
    bits_site_weighted = np.full(n_cols, np.nan, dtype=np.float64)
    for j in range(n_cols):
        col = encoded[:, j]
        nongap = (col != 0)
        if nongap.sum() == 0:
            continue
        w_eff = w[nongap, 0]
        toks = col[nongap]
        hist = np.bincount(toks, weights=w_eff, minlength=21)[1:21]
        p = hist / (hist.sum() + 1e-12)
        p = p[p > 0]
        hj = -np.sum(p * (np.log(p) / np.log(2.0)))
        bits_site_weighted[j] = math.log2(20.0) - hj
    eff_mask = occupancy >= 0.5
    mean_eff = float(np.nanmean(bits_site_weighted[eff_mask])) if np.any(eff_mask) else float("nan")
    return bits_site_weighted, mean_eff


def compute_anchor_pid_summaries(encoded: np.ndarray) -> Dict[str, float]:
    """
    Compute percent identity summaries to anchor (row 0) using the no-gap definition.
    """
    if encoded.shape[0] < 2:
        return {"anchor_pid_mean": np.nan, "anchor_pid_median": np.nan, "anchor_pid_p95": np.nan}
    anchor = encoded[0].astype(np.int16, copy=False)
    values: List[float] = []
    for i in range(1, encoded.shape[0]):
        row = encoded[i].astype(np.int16, copy=False)
        nongap = (anchor != 0) & (row != 0)
        denom = int(np.sum(nongap))
        pid = 100.0 * float(np.sum(anchor[nongap] == row[nongap])) / float(denom) if denom else 0.0
        values.append(pid)
    arr = np.asarray(values, dtype=np.float64)
    return {
        "anchor_pid_mean": float(np.mean(arr)) if arr.size else np.nan,
        "anchor_pid_median": float(np.median(arr)) if arr.size else np.nan,
        "anchor_pid_p95": float(np.quantile(arr, 0.95)) if arr.size else np.nan,
    }


def compute_mi_apc_summaries(char_array: np.ndarray, top_quantile: float) -> Dict[str, float]:
    """
    Compute APC-corrected mutual information summaries.
    """
    mi_raw = np.asarray(buildMutinfoMatrix(char_array))
    mi_apc = np.asarray(applyMutinfoCorr(mi_raw, corr="apc"))
    tri = np.triu_indices_from(mi_apc, k=1)
    vals = mi_apc[tri]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "mi_apc_topq_mean": np.nan,
            "mi_apc_topq_median": np.nan,
            "mi_apc_topq_count": np.nan,
            "mi_apc_topL_mean": np.nan,
            "mi_apc_topL5_mean": np.nan,
        }
    qv = float(np.quantile(vals, float(top_quantile)))
    sel_q = vals[vals >= qv]
    l_cols = int(char_array.shape[1])
    order = np.argsort(vals)[::-1]
    k_topl = int(min(vals.size, max(1, l_cols)))
    k_topl5 = int(min(vals.size, max(1, max(1, l_cols // 5))))
    return {
        "mi_apc_topq_mean": float(np.mean(sel_q)) if sel_q.size else np.nan,
        "mi_apc_topq_median": float(np.median(sel_q)) if sel_q.size else np.nan,
        "mi_apc_topq_count": float(sel_q.size),
        "mi_apc_topL_mean": float(np.mean(vals[order[:k_topl]])) if k_topl > 0 else np.nan,
        "mi_apc_topL5_mean": float(np.mean(vals[order[:k_topl5]])) if k_topl5 > 0 else np.nan,
    }


def summarize_array(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Summarize a 1D numeric array with common statistics.
    """
    if values.size == 0 or np.all(~np.isfinite(values)):
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    v = values[np.isfinite(values)]
    return {
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_median": float(np.median(v)),
        f"{prefix}_p10": float(np.quantile(v, 0.10)),
        f"{prefix}_p90": float(np.quantile(v, 0.90)),
        f"{prefix}_min": float(np.min(v)),
        f"{prefix}_max": float(np.max(v)),
    }


def compute_pairwise_pid_statistics(pid_buffer: ac.UpperTrianglePID) -> dict:
    values = pid_buffer.values.astype(float, copy=False)
    valid = pid_buffer.valid.astype(bool, copy=False)
    sel = values[valid]
    if sel.size == 0:
        return {
            'pid_pair_mean': float('nan'),
            'pid_pair_median': float('nan'),
            'pid_pair_std': float('nan'),
            'pid_pair_p90': float('nan'),
            'pid_pair_p95': float('nan'),
            'pid_pair_p99': float('nan'),
            'pid_pair_prop_ge_0p70': float('nan'),
            'pid_pair_prop_ge_0p80': float('nan'),
            'pid_pair_prop_ge_0p90': float('nan'),
        }
    mean_v = float(sel.mean() * 100.0)
    median_v = float(np.median(sel) * 100.0)
    std_v = float(sel.std(ddof=0) * 100.0)
    p90_v = float(np.quantile(sel, 0.90) * 100.0)
    p95_v = float(np.quantile(sel, 0.95) * 100.0)
    p99_v = float(np.quantile(sel, 0.99) * 100.0)
    prop70 = float(np.mean(sel >= 0.70))
    prop80 = float(np.mean(sel >= 0.80))
    prop90 = float(np.mean(sel >= 0.90))
    return {
        'pid_pair_mean': mean_v,
        'pid_pair_median': median_v,
        'pid_pair_std': std_v,
        'pid_pair_p90': p90_v,
        'pid_pair_p95': p95_v,
        'pid_pair_p99': p99_v,
        'pid_pair_prop_ge_0p70': prop70,
        'pid_pair_prop_ge_0p80': prop80,
        'pid_pair_prop_ge_0p90': prop90,
    }


def compute_snr_derived_from_pid(pid_buffer: ac.UpperTrianglePID) -> dict:
    ra = ac.build_row_accumulators(pid_buffer)
    snr, mean_pct, sd_pct = ac.snr_from_moments(ra.total_sum, ra.total_sumsq, ra.total_pairs)
    return {
        'snr_derived': float(snr),
        'snr_mean_pid_pct': float(mean_pct),
        'snr_sd_pid_pct': float(sd_pct),
    }


def compute_singleton_and_weight_entropy(weights: np.ndarray) -> dict:
    if weights.size == 0:
        return {
            'singleton_fraction_theta_0p70': float('nan'),
            'row_weight_entropy_theta_0p70': float('nan'),
        }
    singleton_mask = weights >= 0.999
    singleton_fraction = float(np.mean(singleton_mask))
    w = weights.astype(np.float64, copy=False)
    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return {
            'singleton_fraction_theta_0p70': singleton_fraction,
            'row_weight_entropy_theta_0p70': float('nan'),
        }
    p = w / w_sum
    with np.errstate(divide='ignore', invalid='ignore'):
        logp = np.where(p > 0.0, np.log2(p), 0.0)
        ent = -float(np.sum(p * logp))
    return {
        'singleton_fraction_theta_0p70': singleton_fraction,
        'row_weight_entropy_theta_0p70': ent,
    }


def build_msa_similarity_graph(encoded: np.ndarray, pid_threshold: float) -> nx.Graph:
    """
    Build an undirected MSA similarity graph where an edge exists if no-gap PID(i,j) >= pid_threshold.
    """
    n_rows, _ = encoded.shape
    graph = nx.Graph()
    graph.add_nodes_from(range(n_rows))
    gaps = (encoded == 0)
    block = 256
    for i0 in range(0, n_rows, block):
        i1 = min(n_rows, i0 + block)
        for j0 in range(i0 + 1, n_rows, block):
            j1 = min(n_rows, j0 + block)
            for i in range(i0, i1):
                j_start = max(j0, i + 1)
                if j_start >= j1:
                    continue
                nongap_i = ~gaps[i]
                row_i = encoded[i]
                for j in range(j_start, j1):
                    nongap = nongap_i & (~gaps[j])
                    denom = int(nongap.sum())
                    if denom == 0:
                        continue
                    pid = (row_i[nongap] == encoded[j, nongap]).sum() / float(denom)
                    if pid >= pid_threshold:
                        graph.add_edge(i, j)
    if nx.number_of_selfloops(graph) != 0:
        raise ValueError("MSA similarity graph contains self-loops")
    return graph


def graph_metrics_from_nx(graph: nx.Graph, core_cuts: List[int]) -> Dict[str, float]:
    """
    Compute graph diagnostics. Spectral and path metrics are computed on the LCC.
    """
    out: Dict[str, float] = {
        "msa_graph_singleton_fraction": np.nan,
        "msa_graph_components": np.nan,
        "msa_graph_lcc_fraction": np.nan,
        "msa_graph_core_degeneracy": np.nan,
        "msa_graph_clustering_mean": np.nan,
        "msa_graph_transitivity": np.nan,
        "msa_graph_laplacian_lambda2": np.nan,
        "msa_graph_edge_density": np.nan,
        "msa_graph_apl_lcc": np.nan,
        "msa_graph_articulation_fraction": np.nan,
        "msa_graph_bridge_fraction": np.nan,
        "msa_graph_modularity": np.nan,
        "msa_graph_communities": np.nan,
    }
    if graph.number_of_nodes() == 0:
        return out
    n = graph.number_of_nodes()
    degs = dict(graph.degree())
    singletons = sum(1 for d in degs.values() if d == 0)
    out["msa_graph_singleton_fraction"] = float(singletons / max(1, n))
    out["msa_graph_components"] = float(nx.number_connected_components(graph))
    comps = sorted(nx.connected_components(graph), key=len, reverse=True)
    lcc = graph.subgraph(comps[0]).copy() if comps else nx.Graph()
    out["msa_graph_lcc_fraction"] = float(lcc.number_of_nodes() / max(1, n))
    out["msa_graph_edge_density"] = float(nx.density(graph))
    try:
        core_num = nx.core_number(graph)
        degeneracy = max(core_num.values()) if core_num else 0
        out["msa_graph_core_degeneracy"] = float(degeneracy)
        for k in core_cuts:
            frac = float(sum(1 for v in core_num.values() if v >= k) / n) if core_num else np.nan
            out[f"msa_graph_core_profile_ge{k}"] = frac
    except Exception:
        out["msa_graph_core_degeneracy"] = np.nan
    try:
        out["msa_graph_clustering_mean"] = float(nx.average_clustering(graph))
        out["msa_graph_transitivity"] = float(nx.transitivity(graph))
    except Exception:
        pass
    try:
        arts = list(nx.articulation_points(graph))
        out["msa_graph_articulation_fraction"] = float(len(arts) / max(1, n))
    except Exception:
        pass
    try:
        m_edges = graph.number_of_edges()
        bridges = list(nx.bridges(graph))
        out["msa_graph_bridge_fraction"] = float(len(bridges) / max(1, m_edges))
    except Exception:
        pass
    if lcc.number_of_nodes() >= 2 and lcc.number_of_edges() >= 1:
        try:
            lap = nx.laplacian_matrix(lcc)
            w = eigsh(lap.asfptype(), k=2, which="SM", return_eigenvectors=False, tol=1e-3)
            w_sorted = np.sort(w)
            lam2 = float(w_sorted[1]) if w_sorted.size >= 2 else float("nan")
            out["msa_graph_laplacian_lambda2"] = lam2
        except Exception:
            pass
        try:
            apl = nx.average_shortest_path_length(lcc)
            out["msa_graph_apl_lcc"] = float(apl)
        except Exception:
            pass
    return out


def format_theta_suffix(theta_value: float) -> str:
    """
    Format a theta value as a suffix like '0p80'.
    """
    rounded = f"{theta_value:.2f}"
    return rounded.replace(".", "p").lstrip()


def compute_meff_sweep_from_pid_buffer(pid_buffer: np.ndarray, n_rows: int, thetas: List[float]) -> Dict[str, float]:
    """
    Compute Meff and Meff/N for a list of theta values using a precomputed PID buffer.
    """
    out: Dict[str, float] = {}
    for theta in thetas:
        _, meff_value = ac.compute_meff_weights(pid_buffer, pid_threshold=float(theta))
        meff_fraction = float(meff_value / max(1, n_rows))
        suffix = format_theta_suffix(float(theta))
        out[f"meff_theta_{suffix}"] = float(meff_value)
        out[f"meff_fraction_theta_{suffix}"] = meff_fraction
    return out


def compute_group_record_task(task: Dict[str, object]) -> Tuple[str, int, Optional[Dict[str, object]], Optional[str]]:
    """
    Compute all metrics for a single group task. Returns (run_root_key, group_index, record or None, error or None).
    """
    try:
        pkl_path = Path(str(task["pkl_path"]))
        encoded = load_compressed_alignment_matrix(pkl_path)
        reverse_lut = build_reverse_residue_lut()
        char_array = int_matrix_to_char_array(encoded, reverse_lut)
        entropy_bits, occupancy = compute_entropy_and_coverage(char_array)
        bits_site_unweighted = compute_bits_site_from_entropy(entropy_bits)
        bits_site_weighted, bits_site_weighted_eff_mean = compute_weighted_bits_site_theta070(encoded, occupancy)
        anchor_pid = compute_anchor_pid_summaries(encoded)
        mi_apc = compute_mi_apc_summaries(char_array, top_quantile=float(task["mi_top_quantile"]))
        eff_mask = occupancy >= float(task["effective_col_min_occupancy"])
        prop_gt_2p2_unw = float(np.mean(bits_site_unweighted[eff_mask] > 2.2)) if np.any(eff_mask) else np.nan
        prop_gt_2p2_w = float(np.mean(bits_site_weighted[eff_mask] > 2.2)) if np.any(eff_mask) else np.nan
        try:
            graph = build_msa_similarity_graph(encoded, pid_threshold=float(task["pid_threshold"]))
            graph_metrics = graph_metrics_from_nx(graph, core_cuts=list(task["core_cuts"]))
        except Exception:
            graph_metrics = {}
        record: Dict[str, object] = {
            "group_key": str(task["group_key"]),
            "group_size": int(encoded.shape[0]),
            "n_sequences": int(encoded.shape[0]),
            "alignment_length": int(encoded.shape[1]),
            "meff": np.nan,
            "meff_fraction": np.nan,
            "gap_fraction": float(np.mean(1.0 - occupancy)) if occupancy.size else np.nan,
            "effective_length_ge_0p5": int(np.sum(occupancy >= 0.5)),
            **summarize_array(entropy_bits, "entropy_bits"),
            **summarize_array(bits_site_unweighted, "bits_site"),
            "bits_site_weighted_mean": bits_site_weighted_eff_mean,
            **summarize_array(bits_site_weighted, "bits_site_weighted"),
            "prop_columns_gt_2p2_unweighted": prop_gt_2p2_unw,
            "prop_columns_gt_2p2_weighted": prop_gt_2p2_w,
            **anchor_pid,
            **mi_apc,
        }
        group_meta: Dict[str, object] = dict(task["group_meta"])
        record.update(group_meta)
        record.update(graph_metrics)
        try:
            pid_buffer = ac.build_upper_triangle_pid_buffer(encoded)
            theta_list = [float(x) for x in task.get("meff_thetas", [])]
            meff_sweep = compute_meff_sweep_from_pid_buffer(pid_buffer, int(encoded.shape[0]), theta_list)
            record.update(meff_sweep)
            if "meff_theta_0p70" in meff_sweep:
                record["meff"] = float(meff_sweep["meff_theta_0p70"])
                record["meff_fraction"] = float(meff_sweep["meff_fraction_theta_0p70"])
        except Exception:
            pass
        try:
            pair_stats = compute_pairwise_pid_statistics(pid_buffer)
            record.update(pair_stats)
            snr_block = compute_snr_derived_from_pid(pid_buffer)
            record.update(snr_block)
            weights_070, _meff_unused = ac.compute_meff_weights(pid_buffer, pid_threshold=0.70)
            singleton_block = compute_singleton_and_weight_entropy(weights_070)
            record.update(singleton_block)
        except Exception:
            pass
        try:
            anchor_vals = []
            if encoded.shape[0] >= 2:
                anchor = encoded[0].astype(np.int16, copy=False)
                for i in range(1, encoded.shape[0]):
                    row = encoded[i].astype(np.int16, copy=False)
                    nongap = (anchor != 0) & (row != 0)
                    denom = int(np.sum(nongap))
                    pidv = 100.0 * float(np.sum(anchor[nongap] == row[nongap])) / float(denom) if denom else 0.0
                    anchor_vals.append(pidv)
            if anchor_vals:
                av = np.asarray(anchor_vals, dtype=np.float64)
                record['anchor_pid_p90'] = float(np.quantile(av, 0.90))
                record['anchor_pid_p99'] = float(np.quantile(av, 0.99))
                record['anchor_pid_std'] = float(av.std(ddof=0))
        except Exception:
            pass
        try:
            eff_mask_bits = occupancy >= float(task["effective_col_min_occupancy"])
            if np.any(eff_mask_bits):
                bs_unw = compute_bits_site_from_entropy(entropy_bits)
                record['bits_site_p1_eff'] = float(np.quantile(bs_unw[eff_mask_bits], 0.01))
                record['bits_site_p5_eff'] = float(np.quantile(bs_unw[eff_mask_bits], 0.05))
                bs_w = bits_site_weighted
                bs_w_eff = bs_w[eff_mask_bits]
                bs_w_eff = bs_w_eff[np.isfinite(bs_w_eff)]
                record['bits_site_weighted_p1_eff'] = float(np.quantile(bs_w_eff, 0.01)) if bs_w_eff.size else float('nan')
                record['bits_site_weighted_p5_eff'] = float(np.quantile(bs_w_eff, 0.05)) if bs_w_eff.size else float('nan')
            else:
                record['bits_site_p1_eff'] = float('nan')
                record['bits_site_p5_eff'] = float('nan')
                record['bits_site_weighted_p1_eff'] = float('nan')
                record['bits_site_weighted_p5_eff'] = float('nan')
        except Exception:
            pass
        return str(task["run_root_key"]), int(task["group_index"]), record, None
    except Exception as ex:
        return str(task["run_root_key"]), int(task["group_index"]), None, f"{ex}"


def assemble_and_write_outputs_for_run_root(
    context: Dict[str, object],
    group_records_sorted: List[Dict[str, object]],
) -> List[Path]:
    """
    Assemble DataFrames and write artifacts for one run root.
    """
    run_root: Path = context["run_root"]
    msa_tag: str = context["msa_tag"]
    families: List[str] = context["families"]
    group_by: List[str] = context["group_by"]
    fold_df: pd.DataFrame = context["fold_df"]
    protein_name: str = context["protein_name"]
    write_csv: bool = context["write_csv"]
    write_xlsx: bool = context["write_xlsx"]
    write_parquet: bool = context["write_parquet"]
    mi_top_quantile: float = context["mi_top_quantile"]
    effective_col_min_occupancy: float = context["effective_col_min_occupancy"]
    pid_threshold: float = context["pid_threshold"]
    core_cuts: List[int] = context["core_cuts"]
    out_dir = run_root / "msa_statistics" / msa_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    by_group_df = pd.DataFrame(group_records_sorted)
    if not fold_df.empty and not by_group_df.empty:
        keys = [c for c in group_by if c in by_group_df.columns and c in fold_df.columns]
        nonkey_cols = [c for c in by_group_df.columns if c not in keys + ["group_key", "group_size"]]
        if keys and nonkey_cols:
            collisions = [c for c in nonkey_cols if c in fold_df.columns]
            msa_cols = keys + [c for c in nonkey_cols if c not in collisions]
            by_sample_df = fold_df.merge(by_group_df[msa_cols], on=keys, how="left")
        else:
            by_sample_df = fold_df.copy()
    else:
        by_sample_df = fold_df.copy() if not fold_df.empty else pd.DataFrame()
    base = f"{protein_name}_{msa_tag}"
    if not by_sample_df.empty and write_csv:
        path_csv = out_dir / f"{base}_by_sample.csv"
        by_sample_df.to_csv(path_csv, index=False)
        written.append(path_csv)
    if not by_group_df.empty and write_csv:
        path_group_csv = out_dir / f"{base}_by_group.csv"
        by_group_df.to_csv(path_group_csv, index=False)
        written.append(path_group_csv)
    if not by_sample_df.empty and write_parquet:
        path_parquet = out_dir / f"{base}_by_sample.parquet"
        by_sample_df.to_parquet(path_parquet, index=False)
        written.append(path_parquet)
    if write_xlsx and (not by_sample_df.empty or not by_group_df.empty):
        path_xlsx = out_dir / f"{base}_by_sample.xlsx"
        with pd.ExcelWriter(path_xlsx, engine="openpyxl") as xlw:
            if not by_sample_df.empty:
                by_sample_df.to_excel(xlw, sheet_name="by_sample", index=False)
            if not by_group_df.empty:
                by_group_df.to_excel(xlw, sheet_name="by_group", index=False)
        written.append(path_xlsx)
    provenance = {
        "run_root": str(run_root),
        "msa_tag": msa_tag,
        "families": families,
        "group_by": group_by,
        "mi_top_quantile": float(mi_top_quantile),
        "folding_budget": {
            "effective_col_min_occupancy": float(effective_col_min_occupancy),
            "weighted_bits_site": {"row_weights": "Meff theta=0.70", "gaps_excluded": True},
            "reference_bits_site": 2.2,
        },
        "msa_graph": {
            "pid_threshold_no_gap": float(pid_threshold),
            "core_profile_cuts": core_cuts,
            "spectral_on_lcc": True,
        },
        "methods": {
            "meff": {"theta": 0.70},
            "entropy_bits": "prody_calcShannonEntropy",
            "coverage": "prody_calcMSAOccupancy_res",
            "bits_site": "log2(20)-H",
            "mi_apc": "prody_buildMutinfoMatrix + applyMutinfoCorr(apc)",
        },
        "outputs": [],
    }
    provenance["outputs"] = [str(p) for p in written]
    prov_path = out_dir / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    written.append(prov_path)
    return written


def build_group_tasks_for_run_root(
    run_root: Path,
    msa_tag: str,
    families: List[str],
    group_by: List[str],
    mi_top_quantile: float,
    effective_col_min_occupancy: float,
    pid_threshold: float,
    core_cuts: List[int],
    write_csv: bool,
    write_xlsx: bool,
    write_parquet: bool,
    meff_thetas: List[float],
) -> Tuple[List[Dict[str, object]], Optional[Dict[str, object]]]:
    """
    Build per-group tasks and run-root context. Returns (tasks, context).
    """
    manifest_paths = discover_shard_manifests(run_root, families=families)
    manifest = read_manifest_csvs(manifest_paths)
    if manifest.empty:
        return [], None
    protein_name = (
        manifest["protein_name"].dropna().iloc[0]
        if ("protein_name" in manifest.columns and manifest["protein_name"].notna().any())
        else run_root.name
    )
    if "compressed_msa_pkl_path" not in manifest.columns:
        return [], None
    manifest = manifest.dropna(subset=["compressed_msa_pkl_path"]).copy()
    usable = [c for c in group_by if c in manifest.columns]
    manifest["group_key"] = manifest[usable].astype(str).agg("_".join, axis=1) if usable else "ALL"
    fold_df = find_fold_queue_merged(run_root, protein_name)
    if not fold_df.empty and "sample_id" in fold_df.columns:
        fold_df = fold_df.drop_duplicates(subset=["sample_id"], keep="last")
    tasks: List[Dict[str, object]] = []
    group_iter = manifest.groupby("group_key", sort=False)
    for group_index, (group_key, gdf) in enumerate(group_iter):
        pkl_paths = gdf["compressed_msa_pkl_path"]
        if pkl_paths.empty:
            continue
        pkl_path = Path(str(pkl_paths.iloc[0]))
        group_meta = {col: gdf[col].iloc[0] for col in group_by if col in gdf.columns}
        task: Dict[str, object] = {
            "run_root_key": str(run_root.resolve()),
            "group_index": int(group_index),
            "group_key": str(group_key),
            "group_size": int(gdf.shape[0]),
            "pkl_path": str(pkl_path),
            "group_meta": group_meta,
            "mi_top_quantile": float(mi_top_quantile),
            "effective_col_min_occupancy": float(effective_col_min_occupancy),
            "pid_threshold": float(pid_threshold),
            "core_cuts": list(core_cuts),
            "meff_thetas": list(meff_thetas),
        }
        tasks.append(task)
    if not tasks:
        return [], None
    context: Dict[str, object] = {
        "run_root": run_root,
        "msa_tag": msa_tag,
        "families": list(families),
        "group_by": list(group_by),
        "fold_df": fold_df,
        "protein_name": protein_name,
        "write_csv": bool(write_csv),
        "write_xlsx": bool(write_xlsx),
        "write_parquet": bool(write_parquet),
        "mi_top_quantile": float(mi_top_quantile),
        "effective_col_min_occupancy": float(effective_col_min_occupancy),
        "pid_threshold": float(pid_threshold),
        "core_cuts": list(core_cuts),
    }
    return tasks, context


def build_global_tasks_and_contexts(
    run_roots: List[Path],
    msa_tag: str,
    families: List[str],
    group_by: List[str],
    mi_top_quantile: float,
    effective_col_min_occupancy: float,
    pid_threshold: float,
    core_cuts: List[int],
    write_csv: bool,
    write_xlsx: bool,
    write_parquet: bool,
    meff_thetas: List[float],
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, object]], List[str]]:
    """
    Build global task list and per-run-root contexts. Returns (tasks, contexts_by_key, run_root_keys_in_order).
    """
    all_tasks: List[Dict[str, object]] = []
    contexts: Dict[str, Dict[str, object]] = {}
    ordered_keys: List[str] = []
    for run_root in run_roots:
        tasks, context = build_group_tasks_for_run_root(
            run_root=run_root,
            msa_tag=msa_tag,
            families=families,
            group_by=group_by,
            mi_top_quantile=mi_top_quantile,
            effective_col_min_occupancy=effective_col_min_occupancy,
            pid_threshold=pid_threshold,
            core_cuts=core_cuts,
            write_csv=write_csv,
            write_xlsx=write_xlsx,
            write_parquet=write_parquet,
            meff_thetas=meff_thetas,
        )
        if context is None or not tasks:
            continue
        run_root_key = str(run_root.resolve())
        contexts[run_root_key] = context
        ordered_keys.append(run_root_key)
        all_tasks.extend(tasks)
    return all_tasks, contexts, ordered_keys


def execute_tasks_parallel(
    tasks: List[Dict[str, object]],
    workers: int,
) -> Tuple[Dict[str, List[Tuple[int, Dict[str, object]]]], Dict[str, List[str]]]:
    """
    Execute tasks in parallel or sequentially. Returns (records_by_run_root_key, errors_by_run_root_key).
    """
    records_by_run_root: Dict[str, List[Tuple[int, Dict[str, object]]]] = {}
    errors_by_run_root: Dict[str, List[str]] = {}
    if not tasks:
        return records_by_run_root, errors_by_run_root
    if int(workers) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for run_root_key, group_index, record, err in ex.map(compute_group_record_task, tasks, chunksize=1):
                if err is not None:
                    errors_by_run_root.setdefault(run_root_key, []).append(err)
                    continue
                records_by_run_root.setdefault(run_root_key, []).append((group_index, record))
    else:
        for task in tasks:
            run_root_key, group_index, record, err = compute_group_record_task(task)
            if err is not None:
                errors_by_run_root.setdefault(run_root_key, []).append(err)
                continue
            records_by_run_root.setdefault(run_root_key, []).append((group_index, record))
    return records_by_run_root, errors_by_run_root


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the compressed-MSA statistics CLI with global task orchestration.
    """
    args = parse_args(argv)
    set_global_seed(int(args.seed))
    root_path = Path(args.root)
    run_roots = find_run_roots(root_path)
    if not run_roots:
        print(f"No runs found under {root_path}")
        return 0
    core_cuts = [int(x) for x in str(args.graph_core_cuts).split(",") if str(x).strip()]
    meff_thetas = [float(x) for x in str(args.meff_thetas).split(",") if str(x).strip()]
    tasks, contexts, ordered_keys = build_global_tasks_and_contexts(
        run_roots=run_roots,
        msa_tag=str(args.msa_tag),
        families=list(args.families),
        group_by=list(args.group_by),
        mi_top_quantile=float(args.mi_top_quantile),
        effective_col_min_occupancy=float(args.effective_col_min_occupancy),
        pid_threshold=float(args.pid_threshold),
        core_cuts=core_cuts,
        write_csv=bool(args.write_csv),
        write_xlsx=bool(args.write_xlsx),
        write_parquet=bool(args.write_parquet),
        meff_thetas=meff_thetas,
    )
    if not tasks:
        print("No eligible groups found for analysis")
        return 0
    records_by_key, errors_by_key = execute_tasks_parallel(tasks=tasks, workers=int(args.workers))
    outputs: List[Path] = []
    for run_root_key in ordered_keys:
        run_root_path = Path(run_root_key)
        if run_root_key in errors_by_key and errors_by_key[run_root_key]:
            print(f"ERROR analyzing {run_root_path}: {errors_by_key[run_root_key][0]}")
            continue
        recs = records_by_key.get(run_root_key, [])
        if not recs:
            continue
        recs_sorted = [rec for _, rec in sorted(recs, key=lambda t: t[0])]
        context = contexts[run_root_key]
        try:
            written = assemble_and_write_outputs_for_run_root(context, recs_sorted)
            outputs.extend(written)
        except Exception as ex:
            print(f"ERROR analyzing {run_root_path}: {ex}")
            continue
    print(f"Wrote {len(outputs)} artifacts")
    for p in outputs:
        print(f" - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
