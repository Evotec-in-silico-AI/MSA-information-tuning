from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from skdim.id import TwoNN


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for latent geometry analysis.
    """
    parser = argparse.ArgumentParser(description="Post-hoc latent geometry analysis for protein runs.")
    parser.add_argument("--root", type=str, default="runs", help="Root directory containing run folders.")
    parser.add_argument(
        "--latent_spaces", type=str, nargs="+", default=["model_latents"],
        help="Latent spaces to analyze (tag for output folder names).",
    )
    parser.add_argument(
        "--group_by", type=str, nargs="+",
        default=["shard_group", "p10_target", "decode_steps", "family", "control_rep"],
        help="Grouping columns present in shard manifests.",
    )
    parser.add_argument("--knn_k", type=int, default=10, help="k for k-NN radius metric.")
    parser.add_argument("--twonn_discard", type=float, default=0.10, help="TwoNN discard_fraction.")
    parser.add_argument("--graph_k", type=int, default=15, help="k (n_neighbors) for k-NN graph connectivity checks.")
    parser.add_argument("--whiten", action="store_true", default=False, help="Whiten latents before geometry metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--write_csv", action="store_true", default=True, help="Write CSV outputs.")
    parser.add_argument("--write_xlsx", action="store_true", default=True, help="Write XLSX outputs.")
    parser.add_argument("--write_parquet", action="store_true", default=True, help="Write Parquet outputs.")
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


def discover_shard_manifests(run_root: Path) -> List[Path]:
    """
    Locate shard manifest CSV files under a run root.
    """
    return sorted((run_root / "experiments").rglob("shard_manifest.csv"))


def read_manifest_csvs(manifest_paths: List[Path]) -> pd.DataFrame:
    """
    Read and combine shard manifest CSVs into a single DataFrame.
    """
    frames: List[pd.DataFrame] = []
    for path in manifest_paths:
        try:
            df = pd.read_csv(path)
            df["manifest_path"] = str(path)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def read_fold_queue_merged(run_root: Path, protein_name: Optional[str]) -> pd.DataFrame:
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


def load_latent_payload(latent_file: Path) -> Optional[dict]:
    """
    Load the latent payload dictionary from a pickle file.
    """
    import pickle
    try:
        with open(latent_file, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "latents" in payload and isinstance(payload["latents"], np.ndarray):
            return payload
        return None
    except Exception:
        return None


def build_latent_matrix_from_manifest(manifest_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Construct latent matrix X, aligned metadata, and per-row anchor vectors A from a manifest.
    """
    required_cols = ["sample_id", "latent_path", "latent_index"]
    for col in required_cols:
        if col not in manifest_df.columns:
            raise ValueError(f"Manifest missing required column: {col}")
    working = manifest_df.dropna(subset=required_cols).copy()
    working["latent_index"] = working["latent_index"].astype(int)
    latent_rows: List[np.ndarray] = []
    anchor_rows: List[np.ndarray] = []
    meta_rows: List[pd.Series] = []
    cache: Dict[str, dict] = {}
    for _, row in working.iterrows():
        latent_path_str = str(row["latent_path"])
        latent_index = int(row["latent_index"])
        if not latent_path_str:
            continue
        payload = cache.get(latent_path_str)
        if payload is None:
            payload = load_latent_payload(Path(latent_path_str))
            if payload is None:
                continue
            cache[latent_path_str] = payload
        latents = payload.get("latents", None)
        anchor_vec = payload.get("anchor_latent", None)
        if isinstance(latents, np.ndarray) and 0 <= latent_index < latents.shape[0]:
            vec = latents[latent_index]
            latent_rows.append(np.asarray(vec, dtype=np.float32).ravel())
            if isinstance(anchor_vec, np.ndarray):
                anchor_rows.append(np.asarray(anchor_vec, dtype=np.float32).ravel())
            else:
                anchor_rows.append(np.zeros_like(latent_rows[-1], dtype=np.float32))
            meta_rows.append(row)
    if not latent_rows:
        return working.head(0), np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    meta_aligned = pd.DataFrame(meta_rows).reset_index(drop=True)
    X = np.vstack(latent_rows).astype(np.float32)
    A = np.vstack(anchor_rows).astype(np.float32)
    return meta_aligned, X, A


def center_latents(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center a latent matrix by subtracting the column mean.
    """
    if X.size == 0:
        return X, np.zeros((0,), dtype=np.float32)
    mu = X.mean(axis=0)
    return X - mu, mu


def whiten_latents(X_centered: np.ndarray) -> np.ndarray:
    """
    Apply covariance-based whitening so that the transformed covariance is approximately identity.
    """
    if X_centered.size == 0:
        return X_centered
    lw = LedoitWolf().fit(X_centered)
    cov = 0.5 * (lw.covariance_ + lw.covariance_.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-6, None)
    inv_sqrt = 1.0 / np.sqrt(vals)
    X_whitened = X_centered @ (vecs @ (inv_sqrt[:, None] * vecs.T))
    return X_whitened


def compute_eigenspectrum_ledoit_wolf(X_centered: np.ndarray) -> np.ndarray:
    """
    Compute an eigenspectrum from Ledoit–Wolf covariance on centered data.
    """
    if X_centered.shape[0] < 2:
        return np.array([], dtype=np.float64)
    lw = LedoitWolf().fit(X_centered)
    cov = 0.5 * (lw.covariance_ + lw.covariance_.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    return np.sort(eigvals)[::-1].astype(np.float64)


def compute_participation_ratio(eigvals: np.ndarray) -> float:
    """
    Compute Participation Ratio from non-negative eigenvalues.
    """
    if eigvals.size == 0:
        return float("nan")
    s1 = float(np.sum(eigvals))
    s2 = float(np.sum(eigvals * eigvals))
    if s2 <= 0.0:
        return float("nan")
    return (s1 * s1) / s2


def compute_condition_number_from_eigs(eigvals: np.ndarray) -> float:
    """
    Compute covariance condition number from eigenvalues.
    """
    if eigvals.size == 0:
        return float("nan")
    lam_max = float(np.max(eigvals))
    lam_min = float(np.min(eigvals))
    if lam_min <= 0.0:
        return float("inf")
    return lam_max / lam_min


def compute_pcs_to_variance_thresholds(eigvals: np.ndarray, thresholds: Tuple[float, float] = (0.90, 0.95)) -> Dict[float, int]:
    """
    Compute the number of principal components needed to reach variance thresholds.
    """
    pcs: Dict[float, int] = {}
    if eigvals.size == 0:
        for t in thresholds:
            pcs[t] = 0
        return pcs
    total = float(np.sum(eigvals))
    if total <= 0.0:
        for t in thresholds:
            pcs[t] = 0
        return pcs
    frac = np.cumsum(eigvals) / total
    for t in thresholds:
        idx = int(np.searchsorted(frac, t, side="left")) + 1
        pcs[t] = min(idx, eigvals.size)
    return pcs


def compute_knn_kth_radius(X_centered: np.ndarray, k_neighbors: int) -> Tuple[float, float, np.ndarray]:
    """
    Compute the mean and median distance to the k-th nearest neighbor.
    """
    if X_centered.shape[0] <= k_neighbors:
        return float("nan"), float("nan"), np.array([], dtype=np.float32)
    model = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean")
    model.fit(X_centered)
    dists, _ = model.kneighbors(X_centered, n_neighbors=k_neighbors + 1, return_distance=True)
    kth = dists[:, k_neighbors]
    return float(np.mean(kth)), float(np.median(kth)), kth


def compute_multi_k_knn_radii(X_centered: np.ndarray, ks: Tuple[int, ...]) -> Dict[str, float]:
    """
    Compute k-NN k-th radius statistics for multiple k values.
    """
    out: Dict[str, float] = {}
    for k in ks:
        mean_k, med_k, _ = compute_knn_kth_radius(X_centered, k)
        out[f"lat_knn_radius_k{k}_mean"] = float(mean_k)
        out[f"lat_knn_radius_k{k}_median"] = float(med_k)
    return out


def estimate_twonn_global_id(X_centered: np.ndarray, discard_fraction: float) -> float:
    """
    Estimate intrinsic dimension using TwoNN.
    """
    if X_centered.shape[0] < 10:
        return float("nan")
    try:
        estimator = TwoNN(discard_fraction=float(discard_fraction), dist=False)
        estimator.fit(X_centered)
        dim = getattr(estimator, "dimension_", None)
        return float(dim) if dim is not None else float("nan")
    except Exception:
        return float("nan")


def calculate_bartlett_sphericity_stat(X_centered: np.ndarray) -> Tuple[float, float]:
    """
    Compute Bartlett’s test of sphericity on the correlation matrix.
    """
    try:
        chi2, pval = calculate_bartlett_sphericity(X_centered)
        return float(chi2), float(pval)
    except Exception:
        return float("nan"), float("nan")


def compute_knn_graph_connectivity(X_centered: np.ndarray, n_neighbors: int) -> Tuple[int, bool, csr_matrix]:
    """
    Build a k-NN adjacency graph and report connectivity; return symmetrized adjacency.
    """
    n = int(X_centered.shape[0])
    if n <= 1:
        return 0, False, csr_matrix((n, n))
    k = min(int(n_neighbors), n - 1)
    if k <= 0:
        return 0, False, csr_matrix((n, n))
    adjacency = kneighbors_graph(X_centered, n_neighbors=k, mode="connectivity", include_self=False)
    adjacency_undirected = adjacency.maximum(adjacency.T)
    n_comp, _ = connected_components(adjacency_undirected, directed=False)
    is_connected = bool(int(n_comp) == 1)
    return int(n_comp), is_connected, adjacency_undirected


def compute_effective_rank(eigvals: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute spectral entropy, effective rank, and stable rank from eigenvalues.
    """
    if eigvals.size == 0:
        return float("nan"), float("nan"), float("nan")
    s = float(np.sum(eigvals))
    if s <= 0.0:
        return float("nan"), float("nan"), float("nan")
    p = eigvals / s
    p = p[p > 0]
    spectral_entropy = float(-(p * np.log(p)).sum())
    effective_rank = float(np.exp(spectral_entropy))
    denom = float(np.max(eigvals)) ** 2 if eigvals.size else 0.0
    stable_rank = float(np.sum(eigvals * eigvals) / denom) if denom > 0.0 else float("nan")
    return spectral_entropy, effective_rank, stable_rank


def compute_hubness_metrics(X_centered: np.ndarray, n_neighbors: int) -> Dict[str, float]:
    """
    Compute hubness statistics from k-NN neighbor occurrences.
    """
    out: Dict[str, float] = {"lat_hubness_mean": np.nan, "lat_hubness_std": np.nan, "lat_hubness_skew": np.nan}
    if X_centered.shape[0] == 0 or n_neighbors <= 0 or X_centered.shape[0] <= n_neighbors:
        return out
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean").fit(X_centered)
    ind = nn.kneighbors(return_distance=False)[:, 1:]
    counts = np.bincount(ind.ravel(), minlength=X_centered.shape[0]).astype(np.float64)
    mean_c = float(counts.mean())
    std_c = float(counts.std(ddof=0))
    skew_c = float(((counts - mean_c) ** 3).mean() / (std_c ** 3 + 1e-12)) if std_c > 0 else float("nan")
    out["lat_hubness_mean"] = mean_c
    out["lat_hubness_std"] = std_c
    out["lat_hubness_skew"] = skew_c
    return out


def summarize_anchor_distances(X_group: np.ndarray, A_group: np.ndarray) -> Dict[str, float]:
    """
    Summarize distances to anchor vectors for a group.
    """
    out: Dict[str, float] = {"lat_dist_anchor_mean": float("nan"), "lat_dist_anchor_median": float("nan"), "lat_dist_anchor_p95": float("nan")}
    if X_group.size == 0 or A_group.size == 0 or X_group.shape != A_group.shape:
        return out
    d = np.linalg.norm(X_group - A_group, axis=1).astype(np.float64)
    out["lat_dist_anchor_mean"] = float(np.nanmean(d)) if d.size else float("nan")
    out["lat_dist_anchor_median"] = float(np.nanmedian(d)) if d.size else float("nan")
    out["lat_dist_anchor_p95"] = float(np.nanpercentile(d, 95)) if d.size else float("nan")
    return out


def build_group_key(df: pd.DataFrame, group_by: List[str]) -> pd.Series:
    """
    Build a stable group key from available columns.
    """
    usable = [c for c in group_by if c in df.columns]
    if not usable:
        return pd.Series(["ALL"] * df.shape[0], index=df.index)
    return df[usable].astype(str).agg("_".join, axis=1)


def nx_graph_metrics_from_knn(adjacency_undirected: csr_matrix) -> Dict[str, float]:
    """
    On symmetrized kNN adjacency, compute clustering, transitivity, core/degeneracy, λ2, assortativity, degree gini.
    """
    out = {
        "lat_knn_clustering_mean": np.nan,
        "lat_knn_transitivity": np.nan,
        "lat_knn_core_degeneracy": np.nan,
        "lat_knn_laplacian_lambda2": np.nan,
        "lat_knn_assortativity_degree": np.nan,
        "lat_knn_degree_gini": np.nan,
    }
    if adjacency_undirected.shape[0] == 0:
        return out
    G = nx.from_scipy_sparse_matrix(adjacency_undirected)
    if G.number_of_nodes() == 0:
        return out
    try:
        out["lat_knn_clustering_mean"] = float(nx.average_clustering(G))
        out["lat_knn_transitivity"] = float(nx.transitivity(G))
    except Exception:
        pass
    try:
        core_num = nx.core_number(G)
        degeneracy = max(core_num.values()) if core_num else 0
        out["lat_knn_core_degeneracy"] = float(degeneracy)
        for k in (1, 2, 3, 5):
            frac = float(sum(1 for v in core_num.values() if v >= k) / G.number_of_nodes()) if core_num else np.nan
            out[f"lat_knn_core_profile_ge{k}"] = frac
    except Exception:
        pass
    try:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if comps:
            lcc = G.subgraph(comps[0]).copy()
            if lcc.number_of_nodes() >= 2 and lcc.number_of_edges() >= 1:
                L = nx.laplacian_matrix(lcc)
                w = eigsh(L.asfptype(), k=2, which="SM", return_eigenvectors=False, tol=1e-3)
                w_sorted = np.sort(w)
                out["lat_knn_laplacian_lambda2"] = float(w_sorted[1]) if w_sorted.size >= 2 else float("nan")
    except Exception:
        pass
    try:
        degs = np.array([d for _, d in G.degree()], dtype=float)
        if degs.size:
            s = np.sort(degs)
            n = s.size
            gini = (np.sum((2 * np.arange(1, n + 1) - n - 1) * s)) / (n * np.sum(s) + 1e-12)
            out["lat_knn_degree_gini"] = float(gini)
    except Exception:
        pass
    try:
        out["lat_knn_assortativity_degree"] = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        pass
    return out


def compute_group_metrics(
    X: np.ndarray,
    A: np.ndarray,
    knn_k: int,
    graph_k: int,
    twonn_discard: float,
    whiten: bool,
) -> Dict[str, object]:
    """
    Compute group-level latent metrics on a block of rows.
    """
    X_centered, _ = center_latents(X)
    if whiten:
        X_centered = whiten_latents(X_centered)
    eigvals = compute_eigenspectrum_ledoit_wolf(X_centered)
    pr_value = compute_participation_ratio(eigvals)
    n_features = int(X.shape[1]) if X.ndim == 2 else 0
    sphericity_norm = float(pr_value / n_features) if (n_features > 0 and np.isfinite(pr_value)) else float("nan")
    cond_value = compute_condition_number_from_eigs(eigvals)
    pcs_map = compute_pcs_to_variance_thresholds(eigvals, thresholds=(0.90, 0.95))
    mean_kth, median_kth, _ = compute_knn_kth_radius(X_centered, knn_k)
    id_twonn = estimate_twonn_global_id(X_centered, discard_fraction=float(twonn_discard))
    chi2_stat, p_value = calculate_bartlett_sphericity_stat(X_centered)
    n_components, is_connected, adj_undirected = compute_knn_graph_connectivity(X_centered, n_neighbors=graph_k)
    spectral_entropy, effective_rank, stable_rank = compute_effective_rank(eigvals)
    multi_k = compute_multi_k_knn_radii(X_centered, ks=(5, 10, 20))
    hubness = compute_hubness_metrics(X_centered, n_neighbors=graph_k)
    anchor_stats = summarize_anchor_distances(X, A)
    knn_graph_metrics = nx_graph_metrics_from_knn(adj_undirected)
    metrics: Dict[str, object] = {
        "lat_pr": float(pr_value),
        "lat_sphericity_pr_norm": float(sphericity_norm),
        "lat_condition_number": float(cond_value),
        "lat_pca90_n": int(pcs_map.get(0.90, 0)),
        "lat_pca95_n": int(pcs_map.get(0.95, 0)),
        f"lat_knn_radius_k{knn_k}_mean": float(mean_kth),
        f"lat_knn_radius_k{knn_k}_median": float(median_kth),
        "lat_twonn_id_global": float(id_twonn),
        "lat_bartlett_chi2": float(chi2_stat),
        "lat_bartlett_p": float(p_value),
        "lat_graph_components": int(n_components),
        "lat_graph_connected": bool(is_connected),
        "lat_spec_entropy": float(spectral_entropy),
        "lat_effective_rank": float(effective_rank),
        "lat_stable_rank": float(stable_rank),
    }
    metrics.update(multi_k)
    metrics.update(hubness)
    metrics.update(anchor_stats)
    metrics.update(knn_graph_metrics)
    return metrics


def analyze_run_root(
    run_root: Path,
    latent_tag: str,
    group_by: List[str],
    knn_k: int,
    graph_k: int,
    twonn_discard: float,
    whiten: bool,
    write_csv: bool,
    write_xlsx: bool,
    write_parquet: bool,
) -> List[Path]:
    """
    Analyze a single run root and write latent-analysis artifacts.
    """
    written: List[Path] = []
    manifest_paths = discover_shard_manifests(run_root)
    if not manifest_paths:
        return written
    manifest = read_manifest_csvs(manifest_paths)
    essential = ["sample_id", "latent_path", "latent_index"]
    for col in essential:
        if col not in manifest.columns:
            return written
    manifest = manifest.drop_duplicates(subset=["sample_id"], keep="last")
    protein_name = (manifest["protein_name"].dropna().iloc[0]
                    if ("protein_name" in manifest.columns and manifest["protein_name"].notna().any())
                    else run_root.name)
    meta_df, X, A = build_latent_matrix_from_manifest(manifest)
    if X.size == 0 or meta_df.empty:
        return written
    usable_groups = [c for c in group_by if c in meta_df.columns]
    if not usable_groups:
        fallback = ["shard_group", "p10_target", "decode_steps", "family", "control_rep"]
        usable_groups = [c for c in fallback if c in meta_df.columns]
    meta_df["group_key"] = build_group_key(meta_df, usable_groups)
    out_dir = run_root / "latent_analysis" / latent_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df = read_fold_queue_merged(run_root, protein_name)
    fold_small = (fold_df.drop_duplicates(subset=["sample_id"], keep="last")
                  if (not fold_df.empty and "sample_id" in fold_df.columns)
                  else pd.DataFrame())
    group_rows: List[Dict[str, object]] = []
    sample_rows: List[Dict[str, object]] = []
    for gkey, gdf in meta_df.groupby("group_key", sort=False):
        idxs = gdf.index.values
        Xg = X[idxs]
        Ag = A[idxs]
        mask_finite = np.all(np.isfinite(Xg), axis=1)
        Xg_clean = Xg[mask_finite]
        Ag_clean = Ag[mask_finite]
        metrics = compute_group_metrics(
            X=Xg_clean, A=Ag_clean,
            knn_k=int(knn_k),
            graph_k=int(graph_k),
            twonn_discard=float(twonn_discard),
            whiten=bool(whiten),
        )
        group_record: Dict[str, object] = {
            "group_key": gkey,
            "group_size": int(Xg.shape[0]),
            "n_unique_sample_ids": int(gdf["sample_id"].nunique()),
        }
        for col in usable_groups:
            group_record[col] = gdf[col].iloc[0]
        group_record.update(metrics)
        group_rows.append(group_record)
        norms = np.linalg.norm(Xg, axis=1)
        anchor_norms = np.linalg.norm(Ag, axis=1)
        dists = np.linalg.norm(Xg - Ag, axis=1)
        for pos, (_, row) in enumerate(gdf.iterrows()):
            rec = row.to_dict()
            rec.update(metrics)
            rec["group_size"] = int(Xg.shape[0])
            rec["n_unique_sample_ids"] = int(gdf["sample_id"].nunique())
            rec["latent_norm"] = float(norms[pos]) if norms.size else float("nan")
            rec["anchor_latent_norm"] = float(anchor_norms[pos]) if anchor_norms.size else float("nan")
            rec["dist_to_anchor"] = float(dists[pos]) if dists.size else float("nan")
            sample_rows.append(rec)
    by_group_df = pd.DataFrame(group_rows)
    by_sample_df = pd.DataFrame(sample_rows)
    if not fold_small.empty and "sample_id" in by_sample_df.columns:
        add_cols = [c for c in fold_small.columns if c != "sample_id" and c not in by_sample_df.columns]
        if add_cols:
            by_sample_df = by_sample_df.merge(fold_small[["sample_id"] + add_cols], on="sample_id", how="left")
    base = f"{protein_name}_{latent_tag}"
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
        "analysis_version": "1.2.4",
        "run_root": str(run_root),
        "latent_tag": latent_tag,
        "group_by": usable_groups,
        "preprocessing": {"center": True, "whiten": bool(whiten)},
        "covariance": {"estimator": "LedoitWolf"},
        "knn_radius": {"k": int(knn_k), "metric": "euclidean"},
        "twonn": {"discard_fraction": float(twonn_discard)},
        "graph": {"k": int(graph_k), "builder": "kneighbors_graph", "symmetrized": True, "spectral_on_lcc": True},
        "multi_k_knn": {"ks": [5, 10, 20]},
        "outputs": [str(p) for p in written],
    }
    prov_path = out_dir / "provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    written.append(prov_path)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the latent geometry analysis CLI.
    """
    args = parse_args(argv)
    set_global_seed(int(args.seed))
    root_path = Path(args.root)
    run_roots = find_run_roots(root_path)
    if not run_roots:
        print(f"No runs found under {root_path}")
        return 0
    outputs: List[Path] = []
    for run_root in run_roots:
        for latent_tag in args.latent_spaces:
            try:
                paths = analyze_run_root(
                    run_root=run_root,
                    latent_tag=latent_tag,
                    group_by=list(args.group_by),
                    knn_k=int(args.knn_k),
                    graph_k=int(args.graph_k),
                    twonn_discard=float(args.twonn_discard),
                    whiten=bool(args.whiten),
                    write_csv=bool(args.write_csv),
                    write_xlsx=bool(args.write_xlsx),
                    write_parquet=bool(args.write_parquet),
                )
                outputs.extend(paths)
            except Exception as ex:
                print(f"ERROR analyzing {run_root}: {ex}")
                continue
    print(f"Wrote {len(outputs)} artifacts")
    for p in outputs:
        print(f" - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
