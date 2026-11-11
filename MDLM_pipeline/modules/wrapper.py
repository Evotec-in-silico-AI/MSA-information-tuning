from __future__ import annotations
import hashlib
import json
import shutil
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from Bio import SeqIO
from . import alignment
from . import alignment_compression
from . import blastp
from . import diagnostics
from . import identity_analysis
from . import locations
from . import make_proteins
from . import msa_context
from . import train_mdlm
from .io_utils import atomic_pickle_dump, atomic_write_csv_df, atomic_write_numpy_array, atomic_write_text
from .utils import parse_conditioning_types, parse_corruption_rates_text

PIPELINE_VERSION = "1.0"
MAX_COMPRESSED_LENGTH = 2048
RATE_NDP = 6
RATE_TAG_SCALE = 10**RATE_NDP


def stable_seed_from_text(text: str) -> int:
    """Return a stable 32-bit seed derived from text."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return int(value % ((2**32) - 1))


def format_float_tag(value: float, ndp: int = 2) -> str:
    """Format a float into a filesystem-friendly tag with fixed decimal places."""
    return f"{float(value):.{int(ndp)}f}".replace(".", "p")


def quantize_rate(value: Any, ndp: int = RATE_NDP) -> float:
    """Quantize a corruption rate to a fixed decimal precision for IDs."""
    return round(float(value), int(ndp))


def make_p10_tag(target_p10: float) -> str:
    """Create a directory-friendly p10 identifier."""
    return f"p10_{format_float_tag(target_p10, ndp=2)}"


def read_json_dict(path: str) -> Dict[str, Any]:
    """Load JSON object from disk; return empty dict on failure or absence."""
    json_path = Path(str(path))
    if not json_path.exists():
        return {}
    try:
        payload = json.loads(json_path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def validate_input_fasta_canonical(input_fasta_path: str) -> None:
    """Validate that every FASTA record contains only canonical amino acids."""
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    fasta_path = Path(str(input_fasta_path))
    if not fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta_path}")
    invalid_records: List[str] = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        seq = str(record.seq).upper()
        invalid = sorted({ch for ch in seq if ch not in canonical})
        if invalid:
            invalid_records.append(f"{record.id}: {invalid}")
    if invalid_records:
        joined = "; ".join(invalid_records[:20])
        raise ValueError(f"Input FASTA contains non-canonical residues in one or more records: {joined}")


def build_sampling_settings(args: Any) -> Dict[str, Any]:
    """Collect sampling settings used by candidate generation and for provenance."""
    try:
        corruption_rates = parse_corruption_rates_text(getattr(args, "corruption_rates", ""))
    except Exception as exc:
        raise ValueError(f"Failed to parse --corruption_rates; expected CSV like 0.5,0.65 or a JSON list. Error: {exc}")
    decode_steps = int(getattr(args, "decode_steps", 12))
    temperature = float(getattr(args, "temperature", 1.0))
    candidate_pool_size = int(getattr(args, "candidate_pool_size", 15))
    select_k_per_rate = int(getattr(args, "select_k_per_rate", 5))
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be >= 1")
    if select_k_per_rate < 1:
        raise ValueError("select_k_per_rate must be >= 1")
    if select_k_per_rate > candidate_pool_size:
        raise ValueError("select_k_per_rate must be <= candidate_pool_size")
    return {
        "corruption_rates": [float(v) for v in corruption_rates],
        "decode_steps": int(decode_steps),
        "temperature": float(temperature),
        "candidate_pool_size": int(candidate_pool_size),
        "select_k_per_rate": int(select_k_per_rate),
        "init_state": "masked_anchor",
        "clamping": "strict",
        "order": "random",
        "commit_schedule": "cosine",
        "decode_steps_semantics": "iterations",
        "selection_policy": "wt_max_pid_then_kcenter_latent_cosine_then_seed",
    }


def compute_row_shuffle_seed(args: Any) -> int:
    """Compute a deterministic row shuffle seed from run inputs."""
    text = f"{int(args.seed)}__{args.name}__{args.blast_database}"
    return stable_seed_from_text(text)


def shard_done_marker(shard_dir: Path) -> Path:
    """Return the path used as a 'done' marker for a shard."""
    return shard_dir / "DONE"


def shard_skipped_marker(shard_dir: Path) -> Path:
    """Return the path used as a 'skipped' marker for a shard."""
    return shard_dir / "SKIPPED"


def cleanup_skipped_shard_directory(shard_dir: Path) -> None:
    """Remove all files under shard_dir except SKIPPED and meta.json."""
    keep_names = {"SKIPPED", "meta.json"}
    for entry in list(shard_dir.iterdir()):
        if entry.name in keep_names:
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


def mark_shard_skipped(
    shard_dir: Path,
    *,
    family: str,
    shard_id: str,
    p10_target: float,
    p10_achieved: float,
    reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a skip marker and metadata; remove partial artifacts for a shard."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "status": "skipped",
        "reason": str(reason),
        "family": str(family),
        "shard_id": str(shard_id),
        "p10_target": float(p10_target),
        "p10_achieved": float(p10_achieved) if np.isfinite(float(p10_achieved)) else float("nan"),
        "pipeline_version": str(PIPELINE_VERSION),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    atomic_write_text(shard_dir / "meta.json", json.dumps(payload, indent=2))
    atomic_write_text(shard_skipped_marker(shard_dir), "skipped\n")
    cleanup_skipped_shard_directory(shard_dir)


def shard_skip_status(shard_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    """Return skip status and metadata for a shard directory."""
    skip_path = shard_skipped_marker(shard_dir)
    meta_path = shard_dir / "meta.json"
    if not skip_path.exists() or not meta_path.exists():
        return False, {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    if isinstance(meta, dict) and meta.get("status") == "skipped":
        return True, meta
    return False, {}


def checkpoint_complete(model_dir: Path, weights_prefix: str = "weights") -> bool:
    """Check that a TensorFlow checkpoint is complete for a given directory."""
    index_path = model_dir / f"{weights_prefix}.index"
    data_files = list(model_dir.glob(f"{weights_prefix}.data-*"))
    return index_path.exists() and (len(data_files) > 0)


def shard_is_complete(shard_dir: Path, required_rates: List[float], expected_candidates_per_rate: int) -> bool:
    """Return True if a shard directory appears complete and safe for analysis."""
    is_skipped, _ = shard_skip_status(shard_dir)
    if is_skipped:
        return True
    results_path = shard_dir / "results.csv"
    manifest_path = shard_dir / "shard_manifest.csv"
    meta_path = shard_dir / "meta.json"
    samples_dir = shard_dir / "samples"
    fasta_path = samples_dir / "generated.fasta"
    latents_path = samples_dir / "latents.pkl"
    fold_queue_path = samples_dir / "fold_queue.csv"
    model_dir = shard_dir / "model"
    config_path = model_dir / "config.json"
    curve_path = model_dir / "training_curve.csv"
    summary_path = model_dir / "training_summary.json"
    required_ok = all(
        [
            results_path.exists(),
            manifest_path.exists(),
            meta_path.exists(),
            samples_dir.exists(),
            fasta_path.exists(),
            latents_path.exists(),
            fold_queue_path.exists(),
            model_dir.exists(),
            config_path.exists(),
            curve_path.exists(),
            summary_path.exists(),
            checkpoint_complete(model_dir, weights_prefix="weights"),
        ]
    )
    if not required_ok:
        return False
    try:
        results_df = pd.read_csv(results_path)
    except Exception:
        return False
    if results_df.empty:
        return False
    if "corruption_rate_init" not in results_df.columns or "candidate_index" not in results_df.columns:
        return False
    required_rate_set = {quantize_rate(v) for v in required_rates}
    got_pairs = {
        (quantize_rate(rate_value), int(candidate_value))
        for rate_value, candidate_value in zip(results_df["corruption_rate_init"], results_df["candidate_index"])
        if pd.notna(rate_value) and pd.notna(candidate_value)
    }
    expected_pairs = {
        (quantize_rate(rate_value), int(candidate_index))
        for rate_value in required_rate_set
        for candidate_index in range(int(expected_candidates_per_rate))
    }
    if not expected_pairs.issubset(got_pairs):
        return False
    try:
        meta_payload = json.loads(meta_path.read_text())
    except Exception:
        return False
    for key in ("compressed_msa", "compression_json", "msa_context"):
        path_value = meta_payload.get(key, "") if isinstance(meta_payload, dict) else ""
        if not path_value:
            return False
        if not Path(str(path_value)).exists():
            return False
    if not {"selected_for_folding", "nll_cond_path", "entropy_cond_path", "jsd_cond_uncond_path", "commit_step_index_path"}.issubset(set(results_df.columns)):
        return False
    sel = results_df[results_df["selected_for_folding"] == 1]
    for _, row in sel.iterrows():
        for key in ("nll_cond_path", "entropy_cond_path", "jsd_cond_uncond_path", "commit_step_index_path"):
            p = Path(str(row.get(key, "")))
            if not p.exists():
                return False
    shard_feats = shard_dir / "samples" / "shard_feats" / "bits_site_weighted_theta0p70.npy"
    if not shard_feats.exists():
        return False
    return True


def iter_paths(root: Path, pattern: str) -> List[Path]:
    """Return a sorted list of paths under a root matching a glob pattern."""
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def merge_skipped_shards(run_root: Path, family: str) -> Path:
    """Write a table listing all skipped shards for a family under merged/."""
    shards_root = run_root / "experiments" / family
    out_path = run_root / "merged" / f"{family}_skipped.csv"
    rows: List[Dict[str, Any]] = []
    for skipped_path in iter_paths(shards_root, "SKIPPED"):
        meta_path = skipped_path.parent / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        if not isinstance(meta, dict) or meta.get("status") != "skipped":
            continue
        row = dict(meta)
        row["shard_path"] = str(skipped_path.parent)
        rows.append(row)
    atomic_write_csv_df(out_path, pd.DataFrame(rows))
    return out_path


def merge_results(run_root: Path, family: str) -> None:
    """Merge per-shard results into merged/family_results.csv and enriched variant."""
    shards_root = run_root / "experiments" / family
    out_path = run_root / "merged" / f"{family}_results.csv"
    out_enriched_path = run_root / "merged" / f"{family}_results_enriched.csv"
    parts: List[pd.DataFrame] = []
    for results_path in iter_paths(shards_root, "results.csv"):
        try:
            parts.append(pd.read_csv(results_path))
        except Exception:
            continue
    if not parts:
        return
    merged = pd.concat(parts, axis=0, ignore_index=True)
    if "sample_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["sample_id"], keep="last")
    atomic_write_csv_df(out_path, merged)
    manifest_enriched_path = run_root / "merged" / f"{family}_manifest_enriched.csv"
    manifest_path = run_root / "merged" / f"{family}_manifest.csv"
    manifest_df: Optional[pd.DataFrame] = None
    if manifest_enriched_path.exists():
        try:
            manifest_df = pd.read_csv(manifest_enriched_path)
        except Exception:
            manifest_df = None
    elif manifest_path.exists():
        try:
            manifest_df = pd.read_csv(manifest_path)
        except Exception:
            manifest_df = None
    if manifest_df is None or manifest_df.empty:
        return
    if "sample_id" not in manifest_df.columns or "sample_id" not in merged.columns:
        return
    cols_to_add = [col for col in manifest_df.columns if col != "sample_id" and col not in merged.columns]
    manifest_small = manifest_df[["sample_id"] + cols_to_add].copy()
    enriched = merged.merge(manifest_small, on="sample_id", how="left")
    atomic_write_csv_df(out_enriched_path, enriched)


def merge_manifests(run_root: Path, family: str) -> None:
    """Merge per-shard manifests into merged/family_manifest.csv and trigger enrich/merge."""
    shards_root = run_root / "experiments" / family
    out_path = run_root / "merged" / f"{family}_manifest.csv"
    parts: List[pd.DataFrame] = []
    for manifest_path in iter_paths(shards_root, "shard_manifest.csv"):
        try:
            parts.append(pd.read_csv(manifest_path))
        except Exception:
            continue
    if not parts:
        merge_skipped_shards(run_root, family)
        return
    merged = pd.concat(parts, axis=0, ignore_index=True)
    if not merged.empty and "sample_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["sample_id"], keep="last")
    atomic_write_csv_df(out_path, merged)
    diagnostics.enrich_merged_manifest(run_root, family)
    merge_results(run_root, family)
    merge_skipped_shards(run_root, family)


def write_samples(
    samples_dir: Path, sample_pairs: List[Tuple[str, str]], latents: List[np.ndarray], anchor_info: Dict[str, Any]
) -> None:
    """Write candidate sequences and latent bundle to samples/ for a shard."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    latents_path = samples_dir / "latents.pkl"
    fasta_path = samples_dir / "generated.fasta"
    latent_matrix = np.stack(latents, axis=0) if latents else np.zeros((0, 0), dtype=np.float32)
    sample_ids = [sid for sid, _seq in sample_pairs]
    payload = {
        "latents": latent_matrix,
        "sample_ids": sample_ids,
        "anchor_latent": np.asarray(anchor_info["anchor_latent"], dtype=np.float32),
        "anchor_rate": float(anchor_info["anchor_rate"]),
        "anchor_sequence": str(anchor_info["anchor_sequence"]),
        "anchor_tokens": np.asarray(anchor_info["anchor_tokens"], dtype=np.int16),
        "anchor_row_index": int(anchor_info.get("anchor_row_index", 0)),
    }
    atomic_pickle_dump(latents_path, payload)
    fasta_lines: List[str] = []
    for sample_id_value, sample_sequence in sample_pairs:
        fasta_lines.append(f">{sample_id_value}\n{sample_sequence}\n")
    atomic_write_text(fasta_path, "".join(fasta_lines))


def write_fold_queue_shard(samples_dir: Path, selected_rows: List[Dict[str, Any]]) -> Path:
    """Write the per-shard fold queue to samples/fold_queue.csv."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / "fold_queue.csv"
    atomic_write_csv_df(out_path, pd.DataFrame(selected_rows))
    return out_path


def update_fold_queue_run(run_root: Path, protein_name: str, selected_rows: List[Dict[str, Any]]) -> Path:
    """Append selected rows to the run-level fold queue and drop duplicates by sample_id."""
    out_path = Path(run_root) / "merged" / f"{protein_name}_fold_queue.csv"
    incoming = pd.DataFrame(selected_rows)
    if out_path.exists():
        try:
            existing = pd.read_csv(out_path)
        except Exception:
            existing = pd.DataFrame([])
        combined = pd.concat([existing, incoming], axis=0, ignore_index=True)
    else:
        combined = incoming
    if "sample_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    atomic_write_csv_df(out_path, combined)
    return out_path


def kmer_set(sequence: str, k: int) -> set:
    """Return the set of all k-mers for a sequence; empty set if length < k."""
    s = str(sequence)
    if not s or len(s) < int(k):
        return set()
    return {s[i: i + int(k)] for i in range(len(s) - int(k) + 1)}


def pairwise_kmer_jaccard_stats(seqs: List[str], k: int = 5) -> Tuple[float, float]:
    """Return mean and max pairwise Jaccard similarity over k-mer sets."""
    kmer_sets = [kmer_set(s, int(k)) for s in seqs]
    num = len(kmer_sets)
    if num < 2:
        return 0.0, 0.0
    sims: List[float] = []
    for i in range(num - 1):
        set_a = kmer_sets[i]
        for j in range(i + 1, num):
            set_b = kmer_sets[j]
            if not set_a and not set_b:
                sim = 1.0
            else:
                inter = len(set_a.intersection(set_b))
                union = len(set_a.union(set_b))
                sim = float(inter / union) if union > 0 else 0.0
            sims.append(sim)
    return float(np.mean(sims)), float(np.max(sims))


def latent_cosine_distance_stats(latents: List[np.ndarray]) -> Tuple[float, float]:
    """Return mean and min pairwise cosine distance among latent vectors."""
    if len(latents) < 2:
        return 0.0, 0.0
    matrix = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in latents], axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    normed = matrix / norms
    sims = normed @ normed.T
    dists: List[float] = []
    for i in range(sims.shape[0] - 1):
        for j in range(i + 1, sims.shape[0]):
            dists.append(float(1.0 - sims[i, j]))
    return float(np.mean(dists)), float(np.min(dists))


def diversity_metrics(seqs: List[str], latents: List[np.ndarray]) -> Dict[str, Any]:
    """Compute sequence and latent diversity summaries for a set of candidates."""
    unique_count = len(set(seqs))
    unique_fraction = float(unique_count / max(1, len(seqs)))
    j_mean, j_max = pairwise_kmer_jaccard_stats(seqs, k=5)
    cos_mean, cos_min = latent_cosine_distance_stats(latents)
    return {
        "gen_unique_sequences": int(unique_count),
        "gen_unique_fraction": float(unique_fraction),
        "gen_pairwise_kmer_jaccard_mean": float(j_mean),
        "gen_pairwise_kmer_jaccard_max": float(j_max),
        "gen_latent_cosine_dist_mean": float(cos_mean),
        "gen_latent_cosine_dist_min": float(cos_min),
    }


def compute_sequence_quality_metrics(sequence: str) -> Dict[str, Any]:
    """Compute basic sequence quality metrics used in results reporting."""
    s = str(sequence)
    length = int(len(s))
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    total = float(max(1, length))
    entropy_bits = 0.0
    for count in counts.values():
        p = float(count) / total
        if p > 0:
            entropy_bits -= p * float(np.log2(p))
    max_run = 0
    run = 0
    last = None
    for ch in s:
        if ch == last:
            run += 1
        else:
            run = 1
            last = ch
        if run > max_run:
            max_run = run
    frac_x = float(s.count("X") / total)
    return {
        "seq_length": int(length),
        "seq_aa_entropy_bits": float(entropy_bits),
        "seq_max_homopolymer_run": int(max_run),
        "seq_frac_X": float(frac_x),
    }


def select_diverse_top_k(
    candidate_rows: List[Dict[str, Any]],
    latents: List[np.ndarray],
    k: int,
) -> Tuple[List[int], List[int]]:
    """Return selection ranks and flags for the top-k by minimal WT-max-PID with latent k-center and seed tie-breakers."""
    num = len(candidate_rows)
    if num == 0:
        return [], []
    k_value = int(max(1, min(int(k), num)))
    max_pid_arr = np.array([float(row.get("max_protein_id", np.inf)) for row in candidate_rows], dtype=np.float64)
    seed_arr = np.array([int(row.get("candidate_seed", 0)) for row in candidate_rows], dtype=np.int64)
    order = np.argsort(max_pid_arr, kind="stable")
    cutoff_pid = max_pid_arr[order[k_value - 1]]
    below_mask = max_pid_arr < cutoff_pid
    equal_mask = max_pid_arr == cutoff_pid
    selected_idx: List[int] = np.where(below_mask)[0].tolist()
    remaining = k_value - len(selected_idx)
    if remaining > 0:
        tie_indices = np.where(equal_mask)[0].tolist()
        if len(tie_indices) <= remaining:
            selected_idx.extend(tie_indices)
        else:
            pool_latents = [latents[i] for i in tie_indices]
            chosen_local = kcenter_pick(pool_latents, remaining, [latents[i] for i in selected_idx] if selected_idx else [])
            chosen_global = [tie_indices[i] for i in chosen_local]
            if len(chosen_global) < remaining:
                leftovers = sorted(set(tie_indices) - set(chosen_global), key=lambda i: seed_arr[i])[: remaining - len(chosen_global)]
                chosen_global.extend(leftovers)
            selected_idx.extend(chosen_global)
    if len(selected_idx) < k_value:
        remaining_pool = [i for i in range(num) if i not in selected_idx]
        remaining_sorted = sorted(remaining_pool, key=lambda i: (max_pid_arr[i], seed_arr[i]))
        selected_idx.extend(remaining_sorted[: k_value - len(selected_idx)])
    selection_ranks = [0] * num
    selected_flags = [0] * num
    full_order = np.argsort(max_pid_arr, kind="stable")
    for rank, idx in enumerate(full_order.tolist()):
        selection_ranks[int(idx)] = int(rank + 1)
    for idx in selected_idx:
        selected_flags[int(idx)] = 1
    return selection_ranks, selected_flags


def kcenter_pick(pool_latents: List[np.ndarray], m: int, prefixed_latents: List[np.ndarray]) -> List[int]:
    """Pick m indices from pool_latents by greedy farthest-first traversal using cosine distance with optional prefixed seeds."""
    if m <= 0 or not pool_latents:
        return []
    matrix = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in pool_latents], axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    normed = matrix / norms
    chosen: List[int] = []
    if prefixed_latents:
        pref = np.stack([np.asarray(v, dtype=np.float32).ravel() for v in prefixed_latents], axis=0)
        pref_norms = np.linalg.norm(pref, axis=1, keepdims=True) + 1e-8
        pref_normed = pref / pref_norms
        sims = normed @ pref_normed.T
        dists = 1.0 - np.max(sims, axis=1)
    else:
        dists = np.ones((normed.shape[0],), dtype=np.float32)
    first = int(np.argmax(dists))
    chosen.append(first)
    while len(chosen) < m and len(chosen) < normed.shape[0]:
        sims = normed @ normed[chosen].T
        min_d = 1.0 - np.max(sims, axis=1)
        min_d[chosen] = -1.0
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
    return chosen[:m]


def training_summary_fields(model_dir: str) -> Dict[str, Any]:
    """Extract compact training provenance fields from a model directory."""
    summary_path = Path(str(model_dir)) / "training_summary.json"
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    budget = data.get("training_budget", {}) if isinstance(data.get("training_budget", {}), dict) else {}
    outcome = data.get("outcome", {}) if isinstance(data.get("outcome", {}), dict) else {}
    best = data.get("best", {}) if isinstance(data.get("best", {}), dict) else {}
    return {
        "training_selection_metric": str(data.get("selection_metric", "val_loss_masked")),
        "training_val_draws": data.get("val_draws", np.nan),
        "train_max_steps": budget.get("max_steps", np.nan),
        "train_eval_interval_steps": budget.get("eval_interval_steps", np.nan),
        "train_early_stop_patience": budget.get("early_stop_patience", np.nan),
        "train_early_stop_min_delta": budget.get("early_stop_min_delta", np.nan),
        "train_early_stop_burn_in_steps": budget.get("early_stop_burn_in_steps", np.nan),
        "train_total_steps_run": outcome.get("total_steps_run", np.nan),
        "train_early_stopped": outcome.get("early_stopped", np.nan),
        "train_stop_reason": outcome.get("stop_reason", ""),
        "train_best_step": best.get("best_step", np.nan),
        "val_loss_masked_best": best.get("best_val_loss_masked", np.nan),
        "training_summary_json_path": str(summary_path),
    }


def ensure_max_length(num_cols: int, label: str) -> None:
    """Enforce a maximum compressed length for safety and resource limits."""
    if int(num_cols) > int(MAX_COMPRESSED_LENGTH):
        raise ValueError(f"Compressed alignment length L={int(num_cols)} exceeds {int(MAX_COMPRESSED_LENGTH)} for {label}.")


def validate_anchor_row_zero(compressed_alignment_path: str, compression_json_path: str) -> None:
    """Validate anchor row index 0 and gap-free anchor in a compressed alignment."""
    alignment_path = Path(str(compressed_alignment_path))
    if not alignment_path.exists():
        raise FileNotFoundError(f"Compressed alignment not found: {compressed_alignment_path}")
    with open(str(alignment_path), "rb") as handle:
        matrix = np.asarray(pickle.load(handle), dtype=np.int32)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"Compressed alignment must be a non-empty 2D array; got shape {matrix.shape}")
    anchor_row = matrix[0]
    if np.any(anchor_row == 0):
        zero_positions = np.nonzero(anchor_row == 0)[0]
        preview = zero_positions[:20].astype(int).tolist()
        print(f"[validate_anchor_row_zero] Anchor has {int(zero_positions.size)} gap tokens; first positions={preview}")
        raise ValueError("Anchor row at index 0 contains gap tokens after compression")
    meta_path = Path(str(compression_json_path))
    if meta_path.exists():
        meta = read_json_dict(str(meta_path))
        selected_ids = meta.get("selected_ids", None)
        if isinstance(selected_ids, list) and selected_ids:
            if int(selected_ids[0]) != 0:
                raise ValueError("Compression metadata does not record anchor id at selected_ids[0]")


def build_shard_paths(shared: Dict[str, Any], shard_dir: Path) -> Dict[str, Any]:
    """Return a dict of per-shard paths based on shared paths and shard directory."""
    shard_paths = dict(shared)
    shard_paths["compressed_alignment"] = str(shard_dir / "compressed_alignment.pkl")
    shard_paths["msa_context"] = str(shard_dir / "msa_context.pkl")
    shard_paths["diffusion_model_dir"] = str(shard_dir / "model")
    return shard_paths


def run_alignment_to_p10_for_shard(
    shard_paths: Dict[str, Any],
    *,
    p10_target: float,
    p10_tolerance: float,
    apply_row_pruning: bool,
    max_row_drops_fraction: float,
    swap_budget: int,
    row_shuffle_seed: int,
) -> Tuple[int, int, float, str, float, float, float, float]:
    """Tune alignment rows to reach p10_target within tolerance; return sizes, achieved p10, JSON path, and SNR/p10 provenance."""
    num_rows, num_cols, p10_achieved, compression_json_path, snr_initial, snr_used, p10_initial, p10_used = alignment_compression.tune_to_p10(
        paths=shard_paths,
        p10_target=float(p10_target),
        p10_tolerance=float(p10_tolerance),
        apply_row_pruning=bool(apply_row_pruning),
        max_row_drops_fraction=float(max_row_drops_fraction),
        do_swap_refinement=True,
        swap_budget=int(swap_budget),
        row_shuffle_seed=int(row_shuffle_seed),
    )
    ensure_max_length(int(num_cols), label=f"shard {Path(str(shard_paths['compressed_alignment'])).parent}")
    return int(num_rows), int(num_cols), float(p10_achieved), str(compression_json_path), float(snr_initial), float(snr_used), float(p10_initial), float(p10_used)


def build_sample_id(
    protein_name: str,
    family: str,
    shard_label: str,
    rate: float,
    candidate_index: int,
    seed: int,
    null_rep: Optional[int] = None,
) -> str:
    """Return a stable sample identifier string for a candidate."""
    rate_tag = format_float_tag(rate, ndp=RATE_NDP)
    if null_rep is None:
        return f"{protein_name}__{family}__{shard_label}__rate_{rate_tag}__cand{int(candidate_index)}__seed{int(seed)}"
    return f"{protein_name}__{family}__{shard_label}__nullr{int(null_rep)}__rate_{rate_tag}__cand{int(candidate_index)}__seed{int(seed)}"


def build_candidate_seed(base_seed: int, rate: float, candidate_index: int) -> int:
    """Return the deterministic per-candidate seed based on base seed and rate."""
    rate_tag = int(round(quantize_rate(rate) * float(RATE_TAG_SCALE)))
    return (int(base_seed) + 1000003 * int(rate_tag) + int(candidate_index)) % ((2**32) - 1)


def gather_shard_msa_metrics(compressed_alignment_path: str, compression_json_path: str) -> Dict[str, Any]:
    """Aggregate MSA metrics and compression metadata for a shard."""
    rate_metrics = diagnostics.compute_msa_rate_metrics(compressed_alignment_path)
    rate_metrics = {k: v for k, v in rate_metrics.items() if "snr" not in str(k).lower()}
    meta = read_json_dict(compression_json_path)
    out = {
        **rate_metrics,
        "compression_json_path": str(compression_json_path),
        "compressed_msa_pkl_path": str(compressed_alignment_path),
        "meff_theta0p70": meta.get("meff_theta0p70", np.nan),
        "meff_fraction_theta0p70": meta.get("meff_fraction_theta0p70", np.nan),
        "meff_theta0p80": meta.get("meff_theta0p80", np.nan),
        "meff_fraction_theta0p80": meta.get("meff_fraction_theta0p80", np.nan),
        "row_weights_theta0p70_path": meta.get("row_weights_theta0p70_path", ""),
        "row_weights_theta0p80_path": meta.get("row_weights_theta0p80_path", ""),
        "bits_site_p10_target": meta.get("bits_site_p10_target", np.nan),
        "bits_site_p10_initial": meta.get("bits_site_p10_initial", np.nan),
        "bits_site_p10_used": meta.get("bits_site_p10_used", np.nan),
        "bits_site_p10_achieved": meta.get("bits_site_p10_achieved", np.nan),
    }
    anchor_map = meta.get("anchor_seq_index_by_selected_col", [])
    if isinstance(anchor_map, list):
        out["anchor_seq_index_by_selected_col_len"] = int(len(anchor_map))
    else:
        out["anchor_seq_index_by_selected_col_len"] = 0
    out["snr_initial"] = meta.get("snr_initial", np.nan)
    out["snr_used"] = meta.get("snr_used", np.nan)
    return out


def run_training_for_shard(args: Any, shard_paths: Dict[str, Any]) -> float:
    """Train the MDLM for a shard and return the best validation value."""
    conditioning_types = parse_conditioning_types(getattr(args, "conditioning_types", "msa"))
    best_val = train_mdlm.train_mdlm(
        output_paths=shard_paths,
        max_steps=int(args.max_steps),
        seed=int(args.seed),
        early_stop_patience=int(args.early_stop_patience),
        early_stop_min_delta=float(args.early_stop_min_delta),
        early_stop_burn_in_steps=int(args.early_stop_burn_in_steps),
        p_null=0.10,
        conditioning_enabled=bool(getattr(args, "conditioning_enabled", True)),
        conditioning_types=conditioning_types,
        val_draws=3,
        pipeline_version=str(PIPELINE_VERSION),
    )
    return float(best_val)


def write_shard_tables(shard_dir: Path, results_rows: List[Dict[str, Any]], manifest_rows: List[Dict[str, Any]]) -> None:
    """Write shard-level results and manifest CSVs."""
    atomic_write_csv_df(shard_dir / "results.csv", pd.DataFrame(results_rows))
    atomic_write_csv_df(shard_dir / "shard_manifest.csv", pd.DataFrame(manifest_rows))


def append_rows_with_latent_indices(
    *,
    results_rows: List[Dict[str, Any]],
    manifest_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
    shard_dir: Path,
    model_dir: str,
    training_fields: Dict[str, Any],
    compression_json_path: str,
    compressed_msa_path: str,
    latent_path: str,
    latent_index_offset: int,
) -> None:
    """Append candidate rows to shard tables and add latent indices to manifest."""
    for idx, row in enumerate(candidate_rows):
        filtered = {key: row[key] for key in row if key not in ("protein_name", "outer_label")}
        results_rows.append(filtered)
        manifest_rows.append(
            {
                **row,
                "latent_path": str(latent_path),
                "latent_index": int(latent_index_offset + idx),
                "anchor_latent_path": str(latent_path),
                "anchor_latent_field": "anchor_latent",
                "model_dir": str(model_dir),
                "training_curve_csv_path": str(Path(str(model_dir)) / "training_curve.csv"),
                **training_fields,
                "shard_path": str(shard_dir),
                "compression_json_path": str(compression_json_path),
                "compressed_msa_pkl_path": str(compressed_msa_path),
            }
        )


def build_sidecar_paths_for_candidate(shard_dir: Path, quantized_rate: float, candidate_index: int, trace_path_from_sampler: Optional[str]) -> Tuple[str, str, str]:
    """Return mask, commit, and trace file paths for a given candidate and rate."""
    rate_tag = str(quantized_rate).replace(".", "p")
    masks_dir = shard_dir / "samples" / "masks"
    commit_dir = shard_dir / "samples" / "commit"
    trace_dir = shard_dir / "samples" / "trace"
    mask_path = masks_dir / f"cand{int(candidate_index)}_rate_{rate_tag}_mask.npy"
    commit_hist_path = commit_dir / f"cand{int(candidate_index)}_rate_{rate_tag}_commit.json"
    if trace_path_from_sampler:
        trace_path = Path(trace_path_from_sampler)
    else:
        trace_path = trace_dir / f"cand{int(candidate_index)}_rate_{rate_tag}_trace.npz"
    return str(mask_path), str(commit_hist_path), str(trace_path)


def compute_weighted_bits_site_theta070(encoded: np.ndarray, row_weights: np.ndarray) -> np.ndarray:
    """Compute Meff-weighted bits/site using θ=0.70 row weights for the optimized compressed MSA."""
    n_rows, n_cols = encoded.shape
    w = np.asarray(row_weights, dtype=np.float64).reshape(n_rows, 1)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return np.full((n_cols,), np.nan, dtype=np.float32)
    w = w / w_sum
    out = np.full((n_cols,), np.nan, dtype=np.float64)
    for j in range(n_cols):
        col = encoded[:, j]
        nongap = (col != 0)
        if not np.any(nongap):
            continue
        w_eff = w[nongap, 0]
        toks = col[nongap]
        hist = np.bincount(toks, weights=w_eff, minlength=21)[1:21]
        p = hist / max(np.sum(hist), 1e-12)
        p = p[p > 0]
        hj = -np.sum(p * (np.log(p) / np.log(2.0)))
        out[j] = np.log2(20.0) - hj
    return out.astype(np.float32)


def write_shard_weighted_bits_site(shard_dir: Path, compressed_alignment_path: str, weights_path_theta070: str) -> str:
    """Write shard-level weighted bits/site (θ=0.70) to samples/shard_feats/ and return the file path."""
    with open(str(compressed_alignment_path), "rb") as fh:
        encoded = np.asarray(pickle.load(fh), dtype=np.int32)
    weights = np.load(str(weights_path_theta070)).astype(np.float64)
    bits_w = compute_weighted_bits_site_theta070(encoded, weights)
    out_dir = shard_dir / "samples" / "shard_feats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bits_site_weighted_theta0p70.npy"
    atomic_write_numpy_array(out_path, bits_w)
    return str(out_path)


def build_candidate_rows_for_rate(
    *,
    shared: Dict[str, Any],
    shard_paths: Dict[str, Any],
    protein_name: str,
    family: str,
    outer_label: str,
    p10_target: float,
    p10_achieved: float,
    n_selected_rows: int,
    n_selected_cols: int,
    corruption_rate_init: float,
    decode_steps: int,
    temperature: float,
    base_seed: int,
    val_loss: float,
    control_type: str,
    control_rep: int,
    shard_group: str,
    candidate_pool_size: int,
    select_k_per_rate: int,
    shard_msa_metrics: Dict[str, Any],
    anchor_info: Dict[str, Any],
    snr_initial: float,
    snr_used: float,
    p10_initial: float,
    p10_used: float,
    null_rep: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]], List[np.ndarray]]:
    """Sample and assemble candidate rows for a single corruption rate."""
    sequences, embeddings, decode_nll_means, edited_counts, sum_nll_cond, sum_nll_uncond, commit_hists, masked_positions_list, trace_paths, invalid_token_resamples_list, anchor_mismatch_violations_list, nll_paths, entropy_paths, jsd_paths, stepindex_paths = make_proteins.sample_variants(
        output_paths=shard_paths,
        corruption_rate_init=float(corruption_rate_init),
        num_samples=int(candidate_pool_size),
        seed=int(base_seed),
        decode_steps=int(decode_steps),
        temperature=float(temperature),
    )
    max_id_values = identity_analysis.process_de_novo(shared["input"], shared["blast_results"], list(sequences))
    candidate_rows: List[Dict[str, Any]] = []
    sample_pairs: List[Tuple[str, str]] = []
    latents: List[np.ndarray] = []
    shard_dir = Path(str(Path(shard_paths["compressed_alignment"]).parent))
    quantized_rate_value = quantize_rate(corruption_rate_init)
    for candidate_index, sequence in enumerate(sequences):
        embedding = embeddings[candidate_index]
        decode_nll_mean = float(decode_nll_means[candidate_index]) if candidate_index < len(decode_nll_means) else float("nan")
        max_pid = float(max_id_values[candidate_index]) if candidate_index < len(max_id_values) else float("nan")
        candidate_seed_value = build_candidate_seed(int(base_seed), float(quantized_rate_value), int(candidate_index))
        mask_path_str, commit_hist_path_str, trace_path_str = build_sidecar_paths_for_candidate(
            shard_dir=shard_dir,
            quantized_rate=float(quantized_rate_value),
            candidate_index=int(candidate_index),
            trace_path_from_sampler=trace_paths[candidate_index] if candidate_index < len(trace_paths) else None,
        )
        sample_id_value = build_sample_id(
            protein_name=str(protein_name),
            family=str(family),
            shard_label=str(outer_label),
            rate=float(corruption_rate_init),
            candidate_index=int(candidate_index),
            seed=int(base_seed),
            null_rep=null_rep,
        )
        row = {
            "sample_id": str(sample_id_value),
            "protein_name": str(protein_name),
            "family": str(family),
            "outer_label": str(outer_label),
            "outer_value": float(p10_target),
            "p10_target": float(p10_target),
            "p10_achieved": float(p10_achieved),
            "p10_initial": float(p10_initial),
            "p10_used": float(p10_used),
            "snr_initial": float(snr_initial),
            "snr_used": float(snr_used),
            "n_selected_rows": int(n_selected_rows),
            "n_selected_cols": int(n_selected_cols),
            "corruption_rate_init": float(corruption_rate_init),
            "decode_steps": int(decode_steps),
            "temperature": float(temperature),
            "seed": int(base_seed),
            "candidate_seed": int(candidate_seed_value),
            "candidate_index": int(candidate_index),
            "sequence": str(sequence),
            "max_protein_id": float(max_pid),
            "val_loss": float(val_loss),
            "control_type": str(control_type),
            "control_rep": int(control_rep),
            "shard_group": str(shard_group),
            "sample_group": f"{protein_name}__{shard_group}__rate_{format_float_tag(corruption_rate_init, RATE_NDP)}__seed{int(base_seed)}",
            "pipeline_version": str(PIPELINE_VERSION),
            "decode_nll_mean": float(decode_nll_mean),
            "candidate_pool_size": int(candidate_pool_size),
            "select_k_per_rate": int(select_k_per_rate),
            "anchor_rate": float(anchor_info["anchor_rate"]),
            "anchor_sequence": str(anchor_info["anchor_sequence"]),
            "anchor_row_index": int(anchor_info.get("anchor_row_index", 0)),
            "nll_cond_path": nll_paths[candidate_index] if candidate_index < len(nll_paths) else "",
            "entropy_cond_path": entropy_paths[candidate_index] if candidate_index < len(entropy_paths) else "",
            "jsd_cond_uncond_path": jsd_paths[candidate_index] if candidate_index < len(jsd_paths) else "",
            "commit_step_index_path": stepindex_paths[candidate_index] if candidate_index < len(stepindex_paths) else "",
            **compute_sequence_quality_metrics(sequence),
            **shard_msa_metrics,
        }
        if candidate_index < len(edited_counts):
            row["edited_token_count"] = int(edited_counts[candidate_index])
        if candidate_index < len(sum_nll_cond):
            row["sum_neglogp_cond_nats"] = float(sum_nll_cond[candidate_index])
        if candidate_index < len(sum_nll_uncond):
            row["sum_neglogp_uncond_nats"] = float(sum_nll_uncond[candidate_index])
        if candidate_index < len(commit_hists):
            row["commit_hist_path"] = str(commit_hist_path_str)
        if candidate_index < len(masked_positions_list):
            row["mask_path"] = str(mask_path_str)
        if candidate_index < len(trace_paths):
            row["trace_path"] = str(trace_path_str)
        if candidate_index < len(invalid_token_resamples_list):
            row["invalid_token_resamples"] = int(invalid_token_resamples_list[candidate_index])
        if candidate_index < len(anchor_mismatch_violations_list):
            row["anchor_mismatch_violations"] = int(anchor_mismatch_violations_list[candidate_index])
        candidate_rows.append(row)
        sample_pairs.append((str(sample_id_value), str(sequence)))
        latents.append(np.asarray(embedding, dtype=np.float32))
    return candidate_rows, sample_pairs, latents


def run_sampling_for_shard(
    *,
    args: Any,
    shared: Dict[str, Any],
    sampling: Dict[str, Any],
    shard_dir: Path,
    shard_paths: Dict[str, Any],
    family: str,
    outer_label: str,
    p10_target: float,
    p10_achieved: float,
    n_selected_rows: int,
    n_selected_cols: int,
    control_type: str,
    control_rep: int,
    shard_group: str,
    compression_json_path: str,
    best_val: float,
    anchor_info: Dict[str, Any],
    shard_msa_metrics: Dict[str, Any],
    snr_initial: float,
    snr_used: float,
    p10_initial: float,
    p10_used: float,
    null_rep: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Tuple[str, str]], List[np.ndarray]]:
    """Sample candidates across rates and assemble shard tables with sidecars and metrics."""
    required_rates = list(sampling["corruption_rates"])
    candidate_pool_size = int(sampling["candidate_pool_size"])
    decode_steps = int(sampling["decode_steps"])
    temperature = float(sampling["temperature"])
    select_k_per_rate = int(sampling["select_k_per_rate"])
    results_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    selected_fold_rows: List[Dict[str, Any]] = []
    sample_pairs: List[Tuple[str, str]] = []
    latents: List[np.ndarray] = []
    training_fields = training_summary_fields(shard_paths["diffusion_model_dir"])
    for rate in required_rates:
        candidate_rows, rate_pairs, rate_latents = build_candidate_rows_for_rate(
            shared=shared,
            shard_paths=shard_paths,
            protein_name=str(args.name),
            family=str(family),
            outer_label=str(outer_label),
            p10_target=float(p10_target),
            p10_achieved=float(p10_achieved),
            n_selected_rows=int(n_selected_rows),
            n_selected_cols=int(n_selected_cols),
            corruption_rate_init=float(rate),
            decode_steps=int(decode_steps),
            temperature=float(temperature),
            base_seed=int(args.seed),
            val_loss=float(best_val),
            control_type=str(control_type),
            control_rep=int(control_rep),
            shard_group=str(shard_group),
            candidate_pool_size=int(candidate_pool_size),
            select_k_per_rate=int(select_k_per_rate),
            shard_msa_metrics=shard_msa_metrics,
            anchor_info=anchor_info,
            snr_initial=float(snr_initial),
            snr_used=float(snr_used),
            p10_initial=float(p10_initial),
            p10_used=float(p10_used),
            null_rep=None if family == "msa_p10_sweep" else control_rep,
        )
        selection_ranks, selected_flags = select_diverse_top_k(candidate_rows, rate_latents, k=int(select_k_per_rate))
        for idx, row in enumerate(candidate_rows):
            row["selection_rank"] = int(selection_ranks[idx]) if idx < len(selection_ranks) else 0
            row["selected_for_folding"] = int(selected_flags[idx]) if idx < len(selected_flags) else 0
            row["selection_policy"] = str(sampling["selection_policy"])
            if row["selected_for_folding"] == 1:
                selected_fold_rows.append(dict(row))
        latent_offset = int(len(latents))
        append_rows_with_latent_indices(
            results_rows=results_rows,
            manifest_rows=manifest_rows,
            candidate_rows=candidate_rows,
            shard_dir=shard_dir,
            model_dir=str(shard_paths["diffusion_model_dir"]),
            training_fields=training_fields,
            compression_json_path=str(compression_json_path),
            compressed_msa_path=str(shard_paths["compressed_alignment"]),
            latent_path=str(shard_dir / "samples" / "latents.pkl"),
            latent_index_offset=latent_offset,
        )
        sample_pairs.extend(rate_pairs)
        latents.extend(rate_latents)
        metrics = diversity_metrics([pair[1] for pair in sample_pairs], latents)
        for row in results_rows:
            row.update(metrics)
        for row in manifest_rows:
            row.update(metrics)
        for row in selected_fold_rows:
            row.update(metrics)
    return results_rows, manifest_rows, selected_fold_rows, sample_pairs, latents


def ensure_null_base_full(
    *,
    shared: Dict[str, Any],
    run_root: Path,
    p10_tag: str,
    p10_target: float,
    p10_tolerance: float,
    row_shuffle_seed: int,
) -> Tuple[str, str]:
    """Ensure a cached baseline compressed alignment exists for the null family."""
    base_dir = run_root / "experiments" / "msa_p10_sweep_null" / p10_tag / "base_full"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_comp_path = base_dir / "compressed_alignment.pkl"
    json_candidates = sorted(base_dir.glob("compressed_alignment_p10_*.json"))
    base_json_path = str(json_candidates[-1]) if json_candidates else ""
    if base_comp_path.exists() and base_json_path and Path(base_json_path).exists():
        return str(base_comp_path), str(base_json_path)
    base_paths = dict(shared)
    base_paths["compressed_alignment"] = str(base_comp_path)
    num_rows, num_cols, p10_achieved, json_path, snr_initial, snr_used, p10_initial, p10_used = run_alignment_to_p10_for_shard(
        base_paths,
        p10_target=float(p10_target),
        p10_tolerance=float(p10_tolerance),
        apply_row_pruning=False,
        max_row_drops_fraction=0.0,
        swap_budget=0,
        row_shuffle_seed=int(row_shuffle_seed),
    )
    ensure_max_length(int(num_cols), label=f"null base {p10_tag}")
    return str(base_comp_path), str(json_path)


def build_null_compressed_msa(
    *,
    shared: Dict[str, Any],
    run_root: Path,
    p10_tag: str,
    p10_target: float,
    p10_tolerance: float,
    targeted_comp_json_path: str,
    null_rep: int,
    row_shuffle_seed: int,
) -> Tuple[str, str]:
    """Create a null compressed alignment by random row selection after column selection."""
    null_dir = run_root / "experiments" / "msa_p10_sweep_null" / p10_tag / f"null_r{int(null_rep)}"
    null_dir.mkdir(parents=True, exist_ok=True)
    null_comp_path = null_dir / "compressed_alignment.pkl"
    target_meta = read_json_dict(str(targeted_comp_json_path))
    target_keep = int(target_meta.get("n_selected_rows", 0))
    if target_keep <= 1:
        raise ValueError(f"Targeted shard metadata missing n_selected_rows for {p10_tag}")
    base_comp_path, base_json_path = ensure_null_base_full(
        shared=shared,
        run_root=run_root,
        p10_tag=p10_tag,
        p10_target=float(p10_target),
        p10_tolerance=float(p10_tolerance),
        row_shuffle_seed=int(row_shuffle_seed),
    )
    with open(base_comp_path, "rb") as handle:
        full_matrix = np.asarray(pickle.load(handle), dtype=np.int32)
    if full_matrix.ndim != 2:
        raise ValueError(f"Null base compressed MSA must be 2D; got shape {full_matrix.shape}")
    full_rows, full_cols = full_matrix.shape
    ensure_max_length(int(full_cols), label=f"null base {p10_tag}")
    target_keep = min(int(target_keep), int(full_rows))
    null_seed = stable_seed_from_text(f"{p10_tag}__nullrep{int(null_rep)}")
    rng = np.random.default_rng(int(null_seed))
    candidates = np.arange(1, int(full_rows), dtype=np.int64)
    chosen = rng.choice(candidates, size=int(target_keep - 1), replace=False)
    kept = np.array([0] + sorted(int(v) for v in chosen.tolist()), dtype=np.int64)
    null_matrix = full_matrix[kept].astype(np.int32, copy=False)
    atomic_pickle_dump(null_comp_path, null_matrix)
    pid_buffer = alignment_compression.build_upper_triangle_pid_buffer(null_matrix)
    row_weights_070, meff_070 = alignment_compression.compute_meff_weights(pid_buffer, pid_threshold=0.70)
    row_weights_080, meff_080 = alignment_compression.compute_meff_weights(pid_buffer, pid_threshold=0.80)
    weights_path_070 = Path(str(null_comp_path.with_suffix("")) + "_row_weights_theta0p70.npy")
    weights_path_080 = Path(str(null_comp_path.with_suffix("")) + "_row_weights_theta0p80.npy")
    atomic_write_numpy_array(weights_path_070, row_weights_070)
    atomic_write_numpy_array(weights_path_080, row_weights_080)
    rounded = round(float(p10_target), 2)
    null_json_path = str(null_comp_path.with_suffix("")) + f"_p10_{rounded}.json"
    base_meta = read_json_dict(base_json_path)
    selected_cols = base_meta.get("selected_cols", [])
    base_ids = base_meta.get("selected_ids", [])
    base_rows = base_meta.get("selected_rows", [])
    selected_ids: List[Any] = []
    selected_rows: List[Any] = []
    if isinstance(base_ids, list) and isinstance(base_rows, list) and len(base_ids) == int(full_rows) and len(base_rows) == int(full_rows):
        selected_ids = [base_ids[int(i)] for i in kept.tolist()]
        selected_rows = [base_rows[int(i)] for i in kept.tolist()]
    bits_site_full, occupancy_full, entropy_full, _ = alignment_compression.compute_bits_site_and_masks(full_matrix)
    eff_mask_full = occupancy_full >= 0.5
    p10_initial_value = float(np.quantile(bits_site_full[eff_mask_full], 0.10)) if np.any(eff_mask_full) else float("nan")
    bits_site_null, occupancy_null, entropy_null, _ = alignment_compression.compute_bits_site_and_masks(null_matrix)
    eff_mask_null = occupancy_null >= 0.5
    p10_used_value = float(np.quantile(bits_site_null[eff_mask_null], 0.10)) if np.any(eff_mask_null) else float("nan")
    ra_full = alignment_compression.build_row_accumulators(alignment_compression.build_upper_triangle_pid_buffer(full_matrix))
    snr_initial_value, _, _ = alignment_compression.snr_from_moments(ra_full.total_sum, ra_full.total_sumsq, ra_full.total_pairs)
    ra_null = alignment_compression.build_row_accumulators(alignment_compression.build_upper_triangle_pid_buffer(null_matrix))
    snr_used_value, _, _ = alignment_compression.snr_from_moments(ra_null.total_sum, ra_null.total_sumsq, ra_null.total_pairs)
    json_meta = {
        "p10_target": float(p10_target),
        "p10_achieved": float("nan"),
        "p10_tolerance": float(p10_tolerance),
        "p10_initial": float(p10_initial_value),
        "p10_used": float(p10_used_value),
        "snr_initial": float(snr_initial_value),
        "snr_used": float(snr_used_value),
        "n_selected_rows": int(target_keep),
        "n_selected_cols": int(len(selected_cols)) if isinstance(selected_cols, list) else int(base_meta.get("n_selected_cols", 0)),
        "selected_cols": selected_cols,
        "selected_ids": selected_ids,
        "selected_rows": selected_rows,
        "array_path": str(null_comp_path),
        "control_type": "null_random",
        "control_rep": int(null_rep),
        "null_seed": int(null_seed),
        "targeted_reference_json": str(targeted_comp_json_path),
        "base_full_alignment": str(base_comp_path),
        "base_full_json": str(base_json_path),
        "meff_theta0p70": float(meff_070),
        "meff_fraction_theta0p70": float(meff_070 / max(1, int(target_keep))),
        "meff_theta0p80": float(meff_080),
        "meff_fraction_theta0p80": float(meff_080 / max(1, int(target_keep))),
        "row_weights_theta0p70_path": str(weights_path_070),
        "row_weights_theta0p80_path": str(weights_path_080),
        "compression_version": "prody_cov_entropy+snr_v1",
    }
    atomic_write_text(Path(null_json_path), json.dumps(json_meta, indent=2))
    validate_anchor_row_zero(str(null_comp_path), str(null_json_path))
    return str(null_comp_path), str(null_json_path)


def run_targeted_shard(
    *,
    args: Any,
    shared: Dict[str, Any],
    run_root: Path,
    p10_target: float,
    p10_tolerance: float,
    sampling: Dict[str, Any],
    row_shuffle_seed: int,
) -> Dict[str, Any]:
    """Run a targeted shard end-to-end: compression to p10, training, sampling, and tables."""
    shard_id = make_p10_tag(float(p10_target))
    shard_dir = run_root / "experiments" / "msa_p10_sweep" / shard_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    required_rates = list(sampling["corruption_rates"])
    candidate_pool_size = int(sampling["candidate_pool_size"])
    is_skipped, _ = shard_skip_status(shard_dir)
    if is_skipped:
        return {"shard_id": shard_id, "skipped": True, "compression_json": "", "p10_achieved": float("nan"), "skip_reason": "marked_skipped"}
    if shard_is_complete(shard_dir, required_rates, expected_candidates_per_rate=candidate_pool_size):
        meta_payload = read_json_dict(str(shard_dir / "meta.json"))
        meta_compression_json = meta_payload.get("compression_json", "") if isinstance(meta_payload, dict) else ""
        compression_json = str(meta_compression_json) if meta_compression_json and Path(str(meta_compression_json)).exists() else ""
        if not compression_json:
            json_candidates = sorted(shard_dir.glob("compressed_alignment_p10_*.json"))
            compression_json = str(json_candidates[-1]) if json_candidates else ""
        p10_achieved_value = float(meta_payload.get("p10_achieved", float("nan")))
        return {"shard_id": shard_id, "skipped": False, "compression_json": compression_json, "p10_achieved": p10_achieved_value, "skip_reason": ""}
    shard_paths = build_shard_paths(shared, shard_dir)
    max_drop_fraction = float(getattr(args, "max_row_drops_fraction", 0.75))
    num_rows, num_cols, p10_achieved, compression_json_path, snr_initial, snr_used, p10_initial, p10_used = run_alignment_to_p10_for_shard(
        shard_paths,
        p10_target=float(p10_target),
        p10_tolerance=float(p10_tolerance),
        apply_row_pruning=True,
        max_row_drops_fraction=float(max_drop_fraction),
        swap_budget=250,
        row_shuffle_seed=int(row_shuffle_seed),
    )
    validate_anchor_row_zero(shard_paths["compressed_alignment"], str(compression_json_path))
    comp_meta = read_json_dict(str(compression_json_path))
    if comp_meta.get("row_weights_theta0p70_path"):
        shard_paths["row_weights_theta0p70_path"] = str(comp_meta.get("row_weights_theta0p70_path"))
    if comp_meta.get("row_weights_theta0p80_path"):
        shard_paths["row_weights_theta0p80_path"] = str(comp_meta.get("row_weights_theta0p80_path"))
    if not (abs(float(p10_achieved) - float(p10_target)) <= float(p10_tolerance)):
        mark_shard_skipped(
            shard_dir,
            family="msa_p10_sweep",
            shard_id=shard_id,
            p10_target=float(p10_target),
            p10_achieved=float(p10_achieved),
            reason="p10_unreachable_under_constraints",
            extra={"max_row_drops_fraction": float(max_drop_fraction)},
        )
        return {"shard_id": shard_id, "skipped": True, "compression_json": "", "p10_achieved": float(p10_achieved), "skip_reason": "p10_unreachable_under_constraints"}
    msa_context.compute_msa_features(shard_paths)
    best_val = run_training_for_shard(args, shard_paths)
    anchor_info = make_proteins.compute_anchor_latent(shard_paths)
    shard_msa_metrics = gather_shard_msa_metrics(shard_paths["compressed_alignment"], compression_json_path)
    results_rows, manifest_rows, selected_fold_rows, sample_pairs, latents = run_sampling_for_shard(
        args=args,
        shared=shared,
        sampling=sampling,
        shard_dir=shard_dir,
        shard_paths=shard_paths,
        family="msa_p10_sweep",
        outer_label=shard_id,
        p10_target=float(p10_target),
        p10_achieved=float(p10_achieved),
        n_selected_rows=int(num_rows),
        n_selected_cols=int(num_cols),
        control_type="targeted",
        control_rep=0,
        shard_group=shard_id,
        compression_json_path=str(compression_json_path),
        best_val=float(best_val),
        anchor_info=anchor_info,
        shard_msa_metrics=shard_msa_metrics,
        snr_initial=float(snr_initial),
        snr_used=float(snr_used),
        p10_initial=float(p10_initial),
        p10_used=float(p10_used),
        null_rep=None,
    )
    write_shard_tables(shard_dir, results_rows, manifest_rows)
    write_samples(shard_dir / "samples", sample_pairs, latents, anchor_info)
    fold_queue_shard_path = write_fold_queue_shard(shard_dir / "samples", selected_fold_rows)
    run_fold_queue_path = update_fold_queue_run(run_root, str(args.name), selected_fold_rows)
    training_fields = training_summary_fields(shard_paths["diffusion_model_dir"])
    diversity = diversity_metrics([pair[1] for pair in sample_pairs], latents)
    if "row_weights_theta0p70_path" in shard_paths:
        write_shard_weighted_bits_site(shard_dir, shard_paths["compressed_alignment"], shard_paths["row_weights_theta0p70_path"])
    meta_payload = {
        "family": "msa_p10_sweep",
        "control_type": "targeted",
        "control_rep": 0,
        "shard_group": shard_id,
        "shard_id": shard_id,
        "p10_target": float(p10_target),
        "p10_initial": float(p10_initial),
        "p10_used": float(p10_used),
        "p10_achieved": float(p10_achieved),
        "snr_initial": float(snr_initial),
        "snr_used": float(snr_used),
        "sampling": sampling,
        "sampling_version": "poslik_v1",
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "val_loss": float(best_val),
        "compressed_msa": str(shard_paths["compressed_alignment"]),
        "compression_json": str(compression_json_path),
        "msa_context": str(shard_paths["msa_context"]),
        "model_dir": str(shard_paths["diffusion_model_dir"]),
        "pipeline_version": str(PIPELINE_VERSION),
        "fold_queue_shard_csv": str(fold_queue_shard_path),
        "fold_queue_run_csv": str(run_fold_queue_path),
        "anchor_rate": float(anchor_info["anchor_rate"]),
        "anchor_sequence": str(anchor_info["anchor_sequence"]),
        "anchor_row_index": int(anchor_info.get("anchor_row_index", 0)),
        **training_fields,
        **shard_msa_metrics,
        **diversity,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_text(shard_dir / "meta.json", json.dumps(meta_payload, indent=2))
    atomic_write_text(shard_done_marker(shard_dir), "ok\n")
    return {"shard_id": shard_id, "skipped": False, "compression_json": str(compression_json_path), "p10_achieved": float(p10_achieved), "skip_reason": ""}


def write_auto_skipped_target_and_nulls(
    *,
    run_root: Path,
    p10_target: float,
    reason: str,
    p10_achieved_hint: float,
    args: Any,
) -> None:
    """Write SKIPPED markers for a targeted shard and all its null replicates with a shared reason."""
    shard_id = make_p10_tag(float(p10_target))
    targ_dir = run_root / "experiments" / "msa_p10_sweep" / shard_id
    mark_shard_skipped(
        targ_dir,
        family="msa_p10_sweep",
        shard_id=shard_id,
        p10_target=float(p10_target),
        p10_achieved=float(p10_achieved_hint),
        reason=str(reason),
        extra={},
    )
    null_reps = max(1, int(getattr(args, "msa_p10_sweep_null_reps", 1)))
    null_start = max(0, int(getattr(args, "msa_p10_sweep_null_start_rep", 0)))
    for rep in range(null_start, null_start + null_reps):
        null_dir = run_root / "experiments" / "msa_p10_sweep_null" / shard_id / f"null_r{int(rep)}"
        mark_shard_skipped(
            null_dir,
            family="msa_p10_sweep_null",
            shard_id=f"{shard_id}__null_r{int(rep)}",
            p10_target=float(p10_target),
            p10_achieved=float("nan"),
            reason="auto_skip_due_to_target_skipped",
            extra={},
        )


def run_null_shard(
    *,
    args: Any,
    shared: Dict[str, Any],
    run_root: Path,
    p10_target: float,
    p10_tolerance: float,
    targeted_info: Dict[str, Any],
    sampling: Dict[str, Any],
    null_rep: int,
    row_shuffle_seed: int,
) -> Dict[str, Any]:
    """Run a null shard end-to-end: build null compressed MSA with random rows, train with identical budget, sample, and write outputs."""
    shard_id = make_p10_tag(float(p10_target))
    shard_dir = run_root / "experiments" / "msa_p10_sweep_null" / shard_id / f"null_r{int(null_rep)}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    required_rates = list(sampling["corruption_rates"])
    candidate_pool_size = int(sampling["candidate_pool_size"])
    is_skipped, _ = shard_skip_status(shard_dir)
    if is_skipped:
        return {"shard_id": f"{shard_id}__null_r{int(null_rep)}", "skipped": True, "compression_json": "", "p10_achieved": float("nan"), "skip_reason": "marked_skipped"}
    if shard_is_complete(shard_dir, required_rates, expected_candidates_per_rate=candidate_pool_size):
        json_candidates = sorted(shard_dir.glob("compressed_alignment_p10_*.json"))
        compression_json = str(json_candidates[-1]) if json_candidates else ""
        return {"shard_id": f"{shard_id}__null_r{int(null_rep)}", "skipped": False, "compression_json": compression_json, "p10_achieved": float("nan"), "skip_reason": ""}
    targeted_json = str(targeted_info.get("compression_json", ""))
    if not targeted_json:
        raise ValueError("Targeted shard compression_json is missing for null shard construction")
    null_comp_path, null_json_path = build_null_compressed_msa(
        shared=shared,
        run_root=run_root,
        p10_tag=shard_id,
        p10_target=float(p10_target),
        p10_tolerance=float(p10_tolerance),
        targeted_comp_json_path=str(targeted_json),
        null_rep=int(null_rep),
        row_shuffle_seed=int(row_shuffle_seed),
    )
    with open(null_comp_path, "rb") as fh:
        matrix = np.asarray(pickle.load(fh), dtype=np.int32)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("Null compressed alignment is invalid")
    num_rows, num_cols = int(matrix.shape[0]), int(matrix.shape[1])
    shard_paths = build_shard_paths(shared, shard_dir)
    shard_paths["compressed_alignment"] = str(null_comp_path)
    comp_meta = read_json_dict(str(null_json_path))
    if comp_meta.get("row_weights_theta0p70_path"):
        shard_paths["row_weights_theta0p70_path"] = str(comp_meta.get("row_weights_theta0p70_path"))
    if comp_meta.get("row_weights_theta0p80_path"):
        shard_paths["row_weights_theta0p80_path"] = str(comp_meta.get("row_weights_theta0p80_path"))
    msa_context.compute_msa_features(shard_paths)
    best_val = run_training_for_shard(args, shard_paths)
    anchor_info = make_proteins.compute_anchor_latent(shard_paths)
    shard_msa_metrics = gather_shard_msa_metrics(shard_paths["compressed_alignment"], null_json_path)
    base_meta = read_json_dict(str(comp_meta.get("base_full_json", "")))
    p10_initial = float(base_meta.get("bits_site_p10_used", base_meta.get("bits_site_p10_initial", np.nan)))
    snr_initial = float(base_meta.get("snr_used", base_meta.get("snr_initial", np.nan)))
    p10_used = float(comp_meta.get("p10_used", np.nan))
    snr_used = float(comp_meta.get("snr_used", np.nan))
    results_rows, manifest_rows, selected_fold_rows, sample_pairs, latents = run_sampling_for_shard(
        args=args,
        shared=shared,
        sampling=sampling,
        shard_dir=shard_dir,
        shard_paths=shard_paths,
        family="msa_p10_sweep_null",
        outer_label=f"{shard_id}__null_r{int(null_rep)}",
        p10_target=float(p10_target),
        p10_achieved=float("nan"),
        n_selected_rows=int(num_rows),
        n_selected_cols=int(num_cols),
        control_type="null_random",
        control_rep=int(null_rep),
        shard_group=shard_id,
        compression_json_path=str(null_json_path),
        best_val=float(best_val),
        anchor_info=anchor_info,
        shard_msa_metrics=shard_msa_metrics,
        snr_initial=float(snr_initial),
        snr_used=float(snr_used),
        p10_initial=float(p10_initial),
        p10_used=float(p10_used),
        null_rep=int(null_rep),
    )
    write_shard_tables(shard_dir, results_rows, manifest_rows)
    write_samples(shard_dir / "samples", sample_pairs, latents, anchor_info)
    fold_queue_shard_path = write_fold_queue_shard(shard_dir / "samples", selected_fold_rows)
    run_fold_queue_path = update_fold_queue_run(run_root, str(args.name), selected_fold_rows)
    training_fields = training_summary_fields(shard_paths["diffusion_model_dir"])
    diversity = diversity_metrics([pair[1] for pair in sample_pairs], latents)
    if "row_weights_theta0p70_path" in shard_paths:
        write_shard_weighted_bits_site(shard_dir, shard_paths["compressed_alignment"], shard_paths["row_weights_theta0p70_path"])
    meta_payload = {
        "family": "msa_p10_sweep_null",
        "control_type": "null_random",
        "control_rep": int(null_rep),
        "shard_group": shard_id,
        "shard_id": f"{shard_id}__null_r{int(null_rep)}",
        "p10_target": float(p10_target),
        "p10_initial": float(p10_initial),
        "p10_used": float(p10_used),
        "p10_achieved": float("nan"),
        "snr_initial": float(snr_initial),
        "snr_used": float(snr_used),
        "sampling": sampling,
        "sampling_version": "poslik_v1",
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "val_loss": float(best_val),
        "compressed_msa": str(shard_paths["compressed_alignment"]),
        "compression_json": str(null_json_path),
        "msa_context": str(shard_paths["msa_context"]),
        "model_dir": str(shard_paths["diffusion_model_dir"]),
        "pipeline_version": str(PIPELINE_VERSION),
        "fold_queue_shard_csv": str(fold_queue_shard_path),
        "fold_queue_run_csv": str(run_fold_queue_path),
        "anchor_rate": float(anchor_info["anchor_rate"]),
        "anchor_sequence": str(anchor_info["anchor_sequence"]),
        "anchor_row_index": int(anchor_info.get("anchor_row_index", 0)),
        **training_fields,
        **shard_msa_metrics,
        **diversity,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_text(shard_dir / "meta.json", json.dumps(meta_payload, indent=2))
    atomic_write_text(shard_done_marker(shard_dir), "ok\n")
    return {"shard_id": f"{shard_id}__null_r{int(null_rep)}", "skipped": False, "compression_json": str(null_json_path), "p10_achieved": float("nan"), "skip_reason": ""}


def diffusion_wrapper(args: Any) -> None:
    """Run the requested experiment families for a single protein across p10 budgets."""
    shared = locations.get_output_paths(args.name, args.blast_database)
    validate_input_fasta_canonical(shared["input"])
    run_root = locations.get_run_root(args.name, args.blast_database, int(args.max_steps))
    (run_root / "experiments" / "msa_p10_sweep").mkdir(parents=True, exist_ok=True)
    (run_root / "experiments" / "msa_p10_sweep_null").mkdir(parents=True, exist_ok=True)
    if not Path(shared["blast_results"]).exists():
        blastp.blast_sequence(shared["input"], 5000, shared["blast_results"], args.blast_database)
    if not Path(shared["blast_analysis"]).exists():
        identity_analysis.process_blast(shared)
    if not Path(shared["alignment"]).exists():
        alignment.muscle_alignment(shared)
    row_shuffle_seed = compute_row_shuffle_seed(args)
    sampling = build_sampling_settings(args)
    max_shards = int(getattr(args, "max_shards", 0) or 0)
    targets_in = [float(t) for t in getattr(args, "p10_targets")]
    p10_targets = sorted(set(targets_in))
    auto_skip_lower_floor: Optional[float] = None
    auto_skip_upper_remainder: bool = False
    shards_done = 0
    protein_name = str(args.name)
    for p10_target in p10_targets:
        if auto_skip_upper_remainder:
            write_auto_skipped_target_and_nulls(
                run_root=run_root,
                p10_target=float(p10_target),
                reason="auto_skip_due_to_prior_high_end_failure",
                p10_achieved_hint=float("nan"),
                args=args,
            )
            shards_done += 1
            if max_shards > 0 and shards_done >= max_shards:
                merge_manifests(run_root, "msa_p10_sweep")
                merge_manifests(run_root, "msa_p10_sweep_null")
                return
            continue
        if auto_skip_lower_floor is not None and float(p10_target) < float(auto_skip_lower_floor):
            write_auto_skipped_target_and_nulls(
                run_root=run_root,
                p10_target=float(p10_target),
                reason="auto_skip_due_to_floor_from_prior_failure",
                p10_achieved_hint=float(auto_skip_lower_floor),
                args=args,
            )
            shards_done += 1
            if max_shards > 0 and shards_done >= max_shards:
                merge_manifests(run_root, "msa_p10_sweep")
                merge_manifests(run_root, "msa_p10_sweep_null")
                return
            continue
        targeted_info = run_targeted_shard(
            args=args,
            shared=shared,
            run_root=run_root,
            p10_target=float(p10_target),
            p10_tolerance=float(getattr(args, "p10_tolerance", 0.01)),
            sampling=sampling,
            row_shuffle_seed=row_shuffle_seed,
        )
        merge_manifests(run_root, "msa_p10_sweep")
        shards_done += 1
        if max_shards > 0 and shards_done >= max_shards:
            merge_manifests(run_root, "msa_p10_sweep_null")
            return
        if bool(targeted_info.get("skipped", False)) and targeted_info.get("skip_reason") == "p10_unreachable_under_constraints":
            achieved = float(targeted_info.get("p10_achieved", float("nan")))
            tolerance = float(getattr(args, "p10_tolerance", 0.01))
            if np.isfinite(achieved):
                if achieved - float(p10_target) > tolerance:
                    auto_skip_lower_floor = achieved
                elif float(p10_target) - achieved > tolerance:
                    auto_skip_upper_remainder = True
            continue
        comp_json_path = str(targeted_info.get("compression_json", ""))
        if not comp_json_path or not Path(comp_json_path).exists():
            continue
        null_reps = max(1, int(getattr(args, "msa_p10_sweep_null_reps", 1)))
        null_start = max(0, int(getattr(args, "msa_p10_sweep_null_start_rep", 0)))
        for rep in range(null_start, null_start + null_reps):
            run_null_shard(
                args=args,
                shared=shared,
                run_root=run_root,
                p10_target=float(p10_target),
                p10_tolerance=float(getattr(args, "p10_tolerance", 0.01)),
                targeted_info=targeted_info,
                sampling=sampling,
                null_rep=int(rep),
                row_shuffle_seed=row_shuffle_seed,
            )
            shards_done += 1
            if max_shards > 0 and shards_done >= max_shards:
                merge_manifests(run_root, "msa_p10_sweep_null")
                return
    merge_manifests(run_root, "msa_p10_sweep_null")
