from __future__ import annotations
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from . import conditioning
from .train_mdlm import MDLMConfig, MDLMDenoiser, checkpoint_complete
from .utils import candidate_seed_from_base, quantize_rate_value
from .io_utils import atomic_write_numpy_array, atomic_write_text

INT_TO_AA = {
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

TRACE_SCHEMA_VERSION = "v1"
TRACE_TOPK = 20


def set_global_seed(seed: int) -> None:
    """Seed NumPy and TensorFlow RNGs for repeatable behavior."""
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def decode_tokens(tokens: np.ndarray) -> str:
    """Convert a token vector to an amino acid sequence string, skipping gap tokens."""
    token_array = np.asarray(tokens, dtype=np.int32).ravel()
    sequence_chars = [INT_TO_AA.get(int(value), "X") for value in token_array if int(value) != 0]
    return "".join(sequence_chars)


def validate_anchor_tokens(anchor_tokens: np.ndarray) -> None:
    """Validate that the anchor token vector is non-empty and contains no gaps."""
    tokens = np.asarray(anchor_tokens, dtype=np.int32).ravel()
    if tokens.size == 0:
        raise ValueError("Anchor token vector is empty")
    if np.any(tokens == 0):
        raise ValueError("Anchor token vector contains gap tokens")


def assert_strict_clamping(anchor_tokens: np.ndarray, decoded_tokens: np.ndarray, masked_positions: np.ndarray) -> None:
    """Ensure decoded tokens equal anchor tokens at all unmasked positions."""
    anchor_vector = np.asarray(anchor_tokens, dtype=np.int32).ravel()
    decoded_vector = np.asarray(decoded_tokens, dtype=np.int32).ravel()
    if anchor_vector.shape != decoded_vector.shape:
        raise ValueError(f"Anchor/decoded length mismatch: {anchor_vector.shape} vs {decoded_vector.shape}")
    clamp_mask = np.ones(anchor_vector.shape[0], dtype=bool)
    masked_vec = np.asarray(masked_positions, dtype=np.int64).ravel() if masked_positions is not None else np.zeros((0,), dtype=np.int64)
    if masked_vec.size > 0:
        clamp_mask[masked_vec] = False
    mismatch = clamp_mask & (decoded_vector != anchor_vector)
    if np.any(mismatch):
        mismatch_positions = np.nonzero(mismatch)[0].astype(int).tolist()
        raise ValueError(f"Strict clamping violated at {len(mismatch_positions)} positions; first few: {mismatch_positions[:20]}")


def load_model_config(model_dir: Path) -> Dict[str, Any]:
    """Load the model configuration JSON from the model directory."""
    return json.loads((model_dir / "config.json").read_text())


def describe_model_dir(model_dir: Path) -> List[str]:
    """Return a sorted list of file names present in the model directory."""
    if not model_dir.exists():
        return []
    try:
        return sorted([p.name for p in model_dir.iterdir()])
    except Exception:
        return []


def build_model_from_config(cfg: Dict[str, Any], model_dir: Path, length: int, context_dim: int) -> Tuple[MDLMDenoiser, MDLMConfig]:
    """Instantiate the denoiser, materialize shapes, and load weights from a checkpoint."""
    if not checkpoint_complete(model_dir, weights_prefix="weights"):
        directory_listing = describe_model_dir(model_dir)
        raise FileNotFoundError(
            f"Incomplete checkpoint under {model_dir}. Expected weights.index and weights.data-*. Directory contents: {directory_listing}"
        )
    model_config = MDLMConfig(**{k: v for k, v in cfg.items() if k in MDLMConfig.__annotations__})
    model = MDLMDenoiser(model_config, context_dim=context_dim)
    dummy_tokens = tf.zeros((1, int(length)), tf.int32)
    dummy_rate = tf.constant([0.5], tf.float32)
    _ = model(dummy_tokens, dummy_rate, context=None, training=False)
    model.load_weights(str(model_dir / "weights")).expect_partial()
    return model, model_config


def load_sampling_context(output_paths: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[tf.Tensor]:
    """Load the requested conditioning context and return a batched tensor if present."""
    cond_config = cfg.get("conditioning", {})
    is_enabled = bool(cond_config.get("enabled", True)) if isinstance(cond_config, dict) else True
    requested_types = cond_config.get("requested_types", ["msa"]) if isinstance(cond_config, dict) else ["msa"]
    context_value, _meta = conditioning.load_conditioning(
        output_paths, enabled=is_enabled, requested_types=list(requested_types)
    )
    if context_value is None:
        return None
    if not isinstance(context_value, tf.Tensor):
        context_value = tf.convert_to_tensor(context_value, dtype=tf.float32)
    return tf.expand_dims(context_value, 0)


def load_anchor_and_model(output_paths: Dict[str, Any]) -> Tuple[MDLMDenoiser, np.ndarray, Optional[tf.Tensor], Dict[str, Any]]:
    """Load model weights, anchor token vector, sampling context, and configuration."""
    model_dir = Path(str(output_paths["diffusion_model_dir"]))
    cfg = load_model_config(model_dir)
    with open(str(output_paths["compressed_alignment"]), "rb") as handle:
        msa = np.asarray(pickle.load(handle), dtype=np.int32)
    anchor_tokens = msa[0].copy()
    validate_anchor_tokens(anchor_tokens)
    seq_length = int(anchor_tokens.shape[0])
    context_tensor_batched = load_sampling_context(output_paths, cfg)
    context_dim = int(context_tensor_batched.shape[-1]) if context_tensor_batched is not None else 0
    model, _model_cfg = build_model_from_config(cfg, model_dir, length=seq_length, context_dim=context_dim)
    return model, anchor_tokens, context_tensor_batched, cfg


def compute_anchor_latent(output_paths: Dict[str, Any]) -> Dict[str, Any]:
    """Encode the unclamped anchor and return pooled latent and anchor details."""
    model, anchor_tokens, context_tensor_batched, cfg = load_anchor_and_model(output_paths)
    anchor_rate = float(cfg.get("r_train_min", 0.50))
    rate_tensor = tf.constant([anchor_rate], tf.float32)
    token_tensor = tf.convert_to_tensor(anchor_tokens[None, :], tf.int32)
    pooled, position_embeddings = model.encode(token_tensor, rate_tensor, context=context_tensor_batched, training=False)
    del position_embeddings
    anchor_latent = np.asarray(pooled.numpy()[0], dtype=np.float32)
    anchor_sequence = decode_tokens(anchor_tokens)
    return {
        "anchor_latent": anchor_latent,
        "anchor_rate": float(anchor_rate),
        "anchor_sequence": str(anchor_sequence),
        "anchor_tokens": np.asarray(anchor_tokens, dtype=np.int16),
        "anchor_row_index": 0,
    }


def mask_anchor_positions(anchor_tokens: np.ndarray, corruption_rate_init: float, rng: np.random.Generator, mask_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Mask non-gap anchor positions proportional to the requested rate."""
    anchor_vector = np.asarray(anchor_tokens, dtype=np.int32).copy()
    eligible_positions = np.where(anchor_vector != 0)[0].astype(np.int64)
    if eligible_positions.size == 0:
        return anchor_vector, np.zeros((0,), dtype=np.int64)
    clamped_rate = float(np.clip(float(corruption_rate_init), 0.0, 1.0))
    mask_count = int(np.round(clamped_rate * float(eligible_positions.size)))
    mask_count = max(1, min(int(eligible_positions.size), mask_count))
    chosen_positions = rng.choice(eligible_positions, size=mask_count, replace=False)
    masked_positions = np.asarray(chosen_positions, dtype=np.int64)
    masked_positions.sort()
    masked_tokens = anchor_vector.copy()
    masked_tokens[masked_positions] = int(mask_id)
    return masked_tokens, masked_positions


def sample_categorical(rng: np.random.Generator, probabilities: np.ndarray) -> int:
    """Sample a token index from a probability vector."""
    prob_vec = np.asarray(probabilities, dtype=np.float64)
    cdf = np.cumsum(prob_vec)
    draw_value = float(rng.random()) * float(cdf[-1])
    return int(np.searchsorted(cdf, draw_value, side="right"))


def cosine_commit_targets(total_positions: int, decode_steps: int) -> List[int]:
    """Return cumulative commit targets per step using a cosine schedule."""
    total = int(max(0, int(total_positions)))
    steps = max(1, int(decode_steps))
    targets = []
    for step_index in range(1, steps + 1):
        t_frac = float(step_index) / float(steps)
        fraction = 0.5 * (1.0 - math.cos(math.pi * t_frac))
        targets.append(int(math.ceil(fraction * total)))
    if targets:
        targets[-1] = total
    return targets


def safe_softmax_rows(logits_batch: np.ndarray, temperature: float, forbidden_token_ids: List[int]) -> np.ndarray:
    """Compute softmax with temperature, zero out forbidden token ids, and renormalize per position."""
    temp_value = float(max(1e-6, float(temperature)))
    probs_full = tf.nn.softmax(logits_batch / temp_value, axis=-1).numpy()
    if forbidden_token_ids:
        for position_index in range(probs_full.shape[1]):
            row_probs = probs_full[0, position_index]
            for token_id in forbidden_token_ids:
                if 0 <= int(token_id) < row_probs.shape[0]:
                    row_probs[int(token_id)] = 0.0
            row_sum = float(row_probs.sum())
            if row_sum > 0.0:
                row_probs[:] = row_probs / row_sum
            else:
                row_probs[:] = 0.0
    return probs_full[0]


def entropy_of_vector(probabilities: np.ndarray) -> float:
    """Return entropy in nats for a non-negative probability vector."""
    valid = probabilities[probabilities > 0]
    if valid.size == 0:
        return 0.0
    return float(-(valid * np.log(valid)).sum())


def kl_divergence_nats(p: np.ndarray, q: np.ndarray) -> float:
    """Return KL(p || q) in nats for vectors with positive overlap."""
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return float((p[mask] * (np.log(p[mask]) - np.log(q[mask]))).sum())


def jsd_nats(p: np.ndarray, q: np.ndarray) -> float:
    """Return Jensen-Shannon divergence in nats between two probability vectors."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_nats(p, m) + 0.5 * kl_divergence_nats(q, m)


def decode_reference_conditioned_with_trace(
    model: MDLMDenoiser,
    masked_tokens: np.ndarray,
    anchor_tokens: np.ndarray,
    masked_positions: np.ndarray,
    corruption_rate_init: float,
    context_batched: Optional[tf.Tensor],
    seed: int,
    decode_steps: int,
    temperature: float,
    mask_id: int,
    eos_id: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Decode masked positions with strict clamping and capture a per-edit trace."""
    set_global_seed(seed)
    rng = np.random.default_rng(int(seed))
    current_tokens = np.asarray(masked_tokens, dtype=np.int32).copy()
    anchor_vector = np.asarray(anchor_tokens, dtype=np.int32).copy()
    masked_pos_vec = np.asarray(masked_positions, dtype=np.int64)
    clamp_mask = np.ones(current_tokens.shape[0], dtype=bool)
    clamp_mask[masked_pos_vec] = False
    current_tokens[clamp_mask] = anchor_vector[clamp_mask]
    rate_tensor = tf.constant([float(corruption_rate_init)], tf.float32)
    if masked_pos_vec.size == 0:
        pooled_once, embeddings_unused = model.encode(
            tf.convert_to_tensor(current_tokens[None, :], tf.int32),
            rate_tensor,
            context=context_batched,
            training=False,
        )
        del embeddings_unused
        empty_trace = {
            "positions": np.zeros((0,), dtype=np.int64),
            "token_ids": np.zeros((0,), dtype=np.int32),
            "logp_cond": np.zeros((0,), dtype=np.float32),
            "logp_uncond": np.zeros((0,), dtype=np.float32),
            "entropy_cond": np.zeros((0,), dtype=np.float32),
            "entropy_uncond": np.zeros((0,), dtype=np.float32),
            "jsd_cond_uncond": np.zeros((0,), dtype=np.float32),
            "topk_ids": np.zeros((0, TRACE_TOPK), dtype=np.int16),
            "topk_logp_cond": np.zeros((0, TRACE_TOPK), dtype=np.float32),
            "topk_logp_uncond": np.zeros((0, TRACE_TOPK), dtype=np.float32),
            "step_index": np.zeros((0,), dtype=np.int16),
            "commit_batch": np.zeros((0,), dtype=np.int16),
            "decode_order": np.zeros((0,), dtype=np.int64),
            "step_temperature": np.zeros((decode_steps,), dtype=np.float32),
            "step_rate": np.zeros((decode_steps,), dtype=np.float32),
            "step_mean_entropy_cond": np.zeros((decode_steps,), dtype=np.float32),
            "step_mean_entropy_uncond": np.zeros((decode_steps,), dtype=np.float32),
            "invalid_token_resamples": 0,
            "anchor_mismatch_violations": 0,
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "commit_hist": [],
        }
        return current_tokens, pooled_once.numpy()[0], 0.0, empty_trace
    decode_order = rng.permutation(masked_pos_vec)
    step_targets = cosine_commit_targets(total_positions=int(decode_order.size), decode_steps=int(decode_steps))
    total_committed = 0
    pooled_last = None
    nll_sum = 0.0
    nll_count = 0
    positions_trace: List[int] = []
    token_ids_trace: List[int] = []
    logp_cond_trace: List[float] = []
    logp_uncond_trace: List[float] = []
    entropy_cond_trace: List[float] = []
    entropy_uncond_trace: List[float] = []
    jsd_trace: List[float] = []
    topk_ids_matrix = np.zeros((decode_order.size, TRACE_TOPK), dtype=np.int16)
    topk_logp_cond_matrix = np.zeros((decode_order.size, TRACE_TOPK), dtype=np.float32)
    topk_logp_uncond_matrix = np.zeros((decode_order.size, TRACE_TOPK), dtype=np.float32)
    step_index_trace: List[int] = []
    commit_batch_trace: List[int] = []
    decode_order_vec = np.asarray(decode_order, dtype=np.int64)
    step_mean_entropy_cond = np.zeros((decode_steps,), dtype=np.float32)
    step_mean_entropy_uncond = np.zeros((decode_steps,), dtype=np.float32)
    step_used_mask = np.zeros((decode_steps,), dtype=np.int32)
    commit_history: List[Dict[str, int]] = []
    invalid_token_resamples = 0
    anchor_mismatch_violations = 0
    for step_idx, step_target in enumerate(step_targets, start=1):
        commit_now = int(step_target - total_committed)
        if commit_now <= 0:
            continue
        chunk_positions = decode_order[total_committed: total_committed + commit_now]
        total_committed += commit_now
        current_tokens[clamp_mask] = anchor_vector[clamp_mask]
        logits_cond = model(
            tf.convert_to_tensor(current_tokens[None, :], tf.int32),
            rate_tensor,
            context=context_batched,
            training=False,
        ).numpy()
        logits_uncond = model(
            tf.convert_to_tensor(current_tokens[None, :], tf.int32),
            rate_tensor,
            context=None,
            training=False,
        ).numpy()
        forbidden_ids = [0, int(mask_id), int(eos_id)]
        probs_cond = safe_softmax_rows(logits_cond, temperature=float(temperature), forbidden_token_ids=forbidden_ids)
        probs_uncond = safe_softmax_rows(logits_uncond, temperature=float(temperature), forbidden_token_ids=forbidden_ids)
        entropies_cond_step: List[float] = []
        entropies_uncond_step: List[float] = []
        for position_value in chunk_positions.tolist():
            prob_row_cond = probs_cond[int(position_value)].copy()
            prob_row_uncond = probs_uncond[int(position_value)].copy()
            entropy_cond_value = entropy_of_vector(prob_row_cond)
            entropy_uncond_value = entropy_of_vector(prob_row_uncond)
            entropies_cond_step.append(entropy_cond_value)
            entropies_uncond_step.append(entropy_uncond_value)
            if prob_row_cond.sum() <= 0.0:
                sampled_token = int(anchor_vector[int(position_value)])
                selected_prob_cond = 1e-12
                invalid_token_resamples += 1
            else:
                sampled_token = sample_categorical(rng, prob_row_cond)
                selected_prob_cond = float(prob_row_cond[int(sampled_token)]) if 0 <= int(sampled_token) < prob_row_cond.shape[0] else 1e-12
            if sampled_token in (0, int(mask_id), int(eos_id)):
                sampled_token = int(anchor_vector[int(position_value)])
                selected_prob_cond = 1e-12
                invalid_token_resamples += 1
            selected_prob_uncond = float(prob_row_uncond[int(sampled_token)]) if 0 <= int(sampled_token) < prob_row_uncond.shape[0] else 1e-12
            current_tokens[int(position_value)] = int(sampled_token)
            positions_trace.append(int(position_value))
            token_ids_trace.append(int(sampled_token))
            logp_cond_trace.append(float(np.log(max(1e-12, selected_prob_cond))))
            logp_uncond_trace.append(float(np.log(max(1e-12, selected_prob_uncond))))
            entropy_cond_trace.append(float(entropy_cond_value))
            entropy_uncond_trace.append(float(entropy_uncond_value))
            jsd_trace.append(float(jsd_nats(prob_row_cond, prob_row_uncond)))
            if TRACE_TOPK > 0:
                top_k = min(TRACE_TOPK, prob_row_cond.shape[0])
                top_ids_cond = np.argpartition(prob_row_cond, -top_k)[-top_k:]
                top_ids_cond = top_ids_cond[np.argsort(prob_row_cond[top_ids_cond])[::-1]]
                top_ids_uncond = np.argpartition(prob_row_uncond, -top_k)[-top_k:]
                top_ids_uncond = top_ids_uncond[np.argsort(prob_row_uncond[top_ids_uncond])[::-1]]
                row_index = len(positions_trace) - 1
                topk_ids_matrix[row_index, :top_k] = top_ids_cond.astype(np.int16)
                topk_logp_cond_matrix[row_index, :top_k] = np.log(np.maximum(prob_row_cond[top_ids_cond], 1e-12)).astype(np.float32)
                topk_logp_uncond_matrix[row_index, :top_k] = np.log(np.maximum(prob_row_uncond[top_ids_uncond], 1e-12)).astype(np.float32)
            nll_sum += -float(np.log(max(1e-12, selected_prob_cond)))
            nll_count += 1
            step_index_trace.append(int(step_idx))
            commit_batch_trace.append(int(commit_now))
        if entropies_cond_step:
            step_mean_entropy_cond[step_idx - 1] = float(np.mean(entropies_cond_step))
            step_mean_entropy_uncond[step_idx - 1] = float(np.mean(entropies_uncond_step))
            step_used_mask[step_idx - 1] = 1
        pooled_tokens, pooled_positions = model.encode(
            tf.convert_to_tensor(current_tokens[None, :], tf.int32),
            rate_tensor,
            context=context_batched,
            training=False,
        )
        del pooled_positions
        pooled_last = pooled_tokens.numpy()[0]
        commit_history.append({"step": int(step_idx), "committed": int(commit_now)})
        if total_committed >= int(decode_order.size):
            break
    if pooled_last is None:
        pooled_last = np.zeros((int(getattr(model.cfg, "d_model", 256)),), dtype=np.float32)
    mean_nll = float(nll_sum / max(1, nll_count))
    try:
        assert_strict_clamping(anchor_vector, current_tokens, masked_pos_vec)
    except Exception:
        anchor_mismatch_violations = 1
        raise
    trace_payload = {
        "positions": np.asarray(positions_trace, dtype=np.int64),
        "token_ids": np.asarray(token_ids_trace, dtype=np.int32),
        "logp_cond": np.asarray(logp_cond_trace, dtype=np.float32),
        "logp_uncond": np.asarray(logp_uncond_trace, dtype=np.float32),
        "entropy_cond": np.asarray(entropy_cond_trace, dtype=np.float32),
        "entropy_uncond": np.asarray(entropy_uncond_trace, dtype=np.float32),
        "jsd_cond_uncond": np.asarray(jsd_trace, dtype=np.float32),
        "topk_ids": topk_ids_matrix,
        "topk_logp_cond": topk_logp_cond_matrix,
        "topk_logp_uncond": topk_logp_uncond_matrix,
        "step_index": np.asarray(step_index_trace, dtype=np.int16),
        "commit_batch": np.asarray(commit_batch_trace, dtype=np.int16),
        "decode_order": decode_order_vec,
        "step_temperature": np.asarray([float(temperature)] * int(decode_steps), dtype=np.float32),
        "step_rate": np.asarray([float(corruption_rate_init)] * int(decode_steps), dtype=np.float32),
        "step_mean_entropy_cond": step_mean_entropy_cond,
        "step_mean_entropy_uncond": step_mean_entropy_uncond,
        "invalid_token_resamples": int(invalid_token_resamples),
        "anchor_mismatch_violations": int(anchor_mismatch_violations),
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "commit_hist": commit_history,
    }
    return current_tokens, pooled_last, mean_nll, trace_payload


def ensure_sample_dirs(shard_dir: Path) -> Dict[str, Path]:
    """Create and return per-sample sidecar directories."""
    samples_dir = shard_dir / "samples"
    masks_dir = samples_dir / "masks"
    commit_dir = samples_dir / "commit"
    trace_dir = samples_dir / "trace"
    masks_dir.mkdir(parents=True, exist_ok=True)
    commit_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    return {"masks": masks_dir, "commit": commit_dir, "trace": trace_dir}


def shard_dir_from_paths(output_paths: Dict[str, Any]) -> Path:
    """Infer the shard directory from the compressed alignment path."""
    comp_path = Path(str(output_paths["compressed_alignment"]))
    return comp_path.parent


def build_full_length_position_arrays(anchor_length: int, trace_payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build full-length per-position arrays for nll_cond, entropy_cond, jsd_cond_uncond, and 1-based commit step index."""
    nll = np.full((anchor_length,), np.nan, dtype=np.float32)
    ent = np.full((anchor_length,), np.nan, dtype=np.float32)
    jsd = np.full((anchor_length,), np.nan, dtype=np.float32)
    step = np.full((anchor_length,), -1, dtype=np.int16)
    positions = np.asarray(trace_payload.get("positions", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    logp_cond = np.asarray(trace_payload.get("logp_cond", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    entropy_cond = np.asarray(trace_payload.get("entropy_cond", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    jsd_vec = np.asarray(trace_payload.get("jsd_cond_uncond", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    step_index = np.asarray(trace_payload.get("step_index", np.zeros((0,), dtype=np.int16)), dtype=np.int16)
    if positions.size:
        nll[positions] = -logp_cond
        ent[positions] = entropy_cond
        jsd[positions] = jsd_vec
        step[positions] = step_index
    return nll, ent, jsd, step


def sample_variants(
    output_paths: Dict[str, Any],
    corruption_rate_init: float,
    num_samples: int,
    seed: int,
    decode_steps: int = 12,
    temperature: float = 1.0,
) -> Tuple[
    List[str],
    List[np.ndarray],
    List[float],
    List[int],
    List[float],
    List[float],
    List[List[Dict[str, int]]],
    List[np.ndarray],
    List[str],
    List[int],
    List[int],
    List[str],
    List[str],
    List[str],
    List[str],
]:
    """Sample sequence variants and persist sidecars for each candidate."""
    model_dir = Path(str(output_paths["diffusion_model_dir"]))
    cfg = load_model_config(model_dir)
    with open(str(output_paths["compressed_alignment"]), "rb") as handle:
        msa = np.asarray(pickle.load(handle), dtype=np.int32)
    anchor_tokens = msa[0].copy()
    anchor_length = int(anchor_tokens.shape[0])
    context_batched = load_sampling_context(output_paths, cfg)
    context_dim = int(context_batched.shape[-1]) if context_batched is not None else 0
    model, _model_cfg = build_model_from_config(cfg, model_dir, length=anchor_length, context_dim=context_dim)
    mask_id = int(cfg.get("mask_id", 21))
    eos_id = int(cfg.get("eos_id", 22))
    sequences_out: List[str] = []
    latents_out: List[np.ndarray] = []
    decode_nll_means_out: List[float] = []
    edited_counts_out: List[int] = []
    sum_neglogp_cond_nats_out: List[float] = []
    sum_neglogp_uncond_nats_out: List[float] = []
    commit_hists_out: List[List[Dict[str, int]]] = []
    masked_positions_out: List[np.ndarray] = []
    trace_paths_out: List[str] = []
    invalid_resamples_out: List[int] = []
    anchor_mismatch_violations_out: List[int] = []
    nll_paths_out: List[str] = []
    entropy_paths_out: List[str] = []
    jsd_paths_out: List[str] = []
    stepindex_paths_out: List[str] = []
    candidate_count = max(1, int(num_samples))
    quantized_rate = quantize_rate_value(corruption_rate_init)
    shard_dir = shard_dir_from_paths(output_paths)
    sidecar_dirs = ensure_sample_dirs(shard_dir)
    position_dir = shard_dir / "samples" / "position"
    position_dir.mkdir(parents=True, exist_ok=True)
    for candidate_index in range(candidate_count):
        sample_seed = candidate_seed_from_base(int(seed), quantized_rate, int(candidate_index))
        rng = np.random.default_rng(int(sample_seed))
        masked_tokens, masked_positions = mask_anchor_positions(anchor_tokens, float(quantized_rate), rng, mask_id)
        decoded_tokens, pooled_vector, mean_nll_value, trace_payload = decode_reference_conditioned_with_trace(
            model=model,
            masked_tokens=masked_tokens,
            anchor_tokens=anchor_tokens,
            masked_positions=masked_positions,
            corruption_rate_init=float(quantized_rate),
            context_batched=context_batched,
            seed=int(sample_seed),
            decode_steps=int(decode_steps),
            temperature=float(temperature),
            mask_id=mask_id,
            eos_id=eos_id,
        )
        assert_strict_clamping(anchor_tokens, decoded_tokens, masked_positions)
        sequence_string = decode_tokens(decoded_tokens)
        sequences_out.append(sequence_string)
        latents_out.append(np.asarray(pooled_vector, dtype=np.float32))
        decode_nll_means_out.append(float(mean_nll_value))
        edited_counts_out.append(int(masked_positions.size))
        if trace_payload["logp_cond"].size:
            sum_neglogp_cond_nats_out.append(float(-np.sum(trace_payload["logp_cond"])))
        else:
            sum_neglogp_cond_nats_out.append(0.0)
        if trace_payload["logp_uncond"].size:
            sum_neglogp_uncond_nats_out.append(float(-np.sum(trace_payload["logp_uncond"])))
        else:
            sum_neglogp_uncond_nats_out.append(0.0)
        commit_hist_current = [{"step": int(h["step"]), "committed": int(h["committed"])} for h in trace_payload.get("commit_hist", [])]
        commit_hists_out.append(commit_hist_current)
        masked_positions_out.append(np.asarray(masked_positions, dtype=np.int64))
        invalid_resamples_out.append(int(trace_payload.get("invalid_token_resamples", 0)))
        anchor_mismatch_violations_out.append(int(trace_payload.get("anchor_mismatch_violations", 0)))
        mask_vector = np.zeros((anchor_length,), dtype=np.uint8)
        if masked_positions.size:
            valid_positions = masked_positions[(masked_positions >= 0) & (masked_positions < anchor_length)]
            mask_vector[valid_positions] = 1
        rate_tag = str(quantized_rate).replace(".", "p")
        mask_file = sidecar_dirs["masks"] / f"cand{int(candidate_index)}_rate_{rate_tag}_mask.npy"
        atomic_write_numpy_array(mask_file, mask_vector)
        commit_hist_file = sidecar_dirs["commit"] / f"cand{int(candidate_index)}_rate_{rate_tag}_commit.json"
        atomic_write_text(commit_hist_file, json.dumps(commit_hist_current, indent=2))
        trace_file = sidecar_dirs["trace"] / f"cand{int(candidate_index)}_rate_{rate_tag}_trace.npz"
        np.savez_compressed(
            trace_file,
            positions=trace_payload["positions"],
            token_ids=trace_payload["token_ids"],
            logp_cond=trace_payload["logp_cond"],
            logp_uncond=trace_payload["logp_uncond"],
            entropy_cond=trace_payload["entropy_cond"],
            entropy_uncond=trace_payload["entropy_uncond"],
            jsd_cond_uncond=trace_payload["jsd_cond_uncond"],
            topk_ids=trace_payload["topk_ids"],
            topk_logp_cond=trace_payload["topk_logp_cond"],
            topk_logp_uncond=trace_payload["topk_logp_uncond"],
            step_index=trace_payload["step_index"],
            commit_batch=trace_payload["commit_batch"],
            decode_order=trace_payload["decode_order"],
            step_temperature=trace_payload["step_temperature"],
            step_rate=trace_payload["step_rate"],
            step_mean_entropy_cond=trace_payload["step_mean_entropy_cond"],
            step_mean_entropy_uncond=trace_payload["step_mean_entropy_uncond"],
            invalid_token_resamples=np.asarray([trace_payload.get("invalid_token_resamples", 0)], dtype=np.int32),
            anchor_mismatch_violations=np.asarray([trace_payload.get("anchor_mismatch_violations", 0)], dtype=np.int32),
            trace_schema_version=np.asarray([TRACE_SCHEMA_VERSION]),
        )
        trace_paths_out.append(str(trace_file))
        nll_vec, ent_vec, jsd_vec, step_vec = build_full_length_position_arrays(anchor_length, trace_payload)
        nll_path = position_dir / f"cand{int(candidate_index)}_rate_{rate_tag}_nll_cond.npy"
        ent_path = position_dir / f"cand{int(candidate_index)}_rate_{rate_tag}_entropy_cond.npy"
        jsd_path = position_dir / f"cand{int(candidate_index)}_rate_{rate_tag}_jsd_cond_uncond.npy"
        step_path = position_dir / f"cand{int(candidate_index)}_rate_{rate_tag}_commit_step_index.npy"
        atomic_write_numpy_array(nll_path, nll_vec)
        atomic_write_numpy_array(ent_path, ent_vec)
        atomic_write_numpy_array(jsd_path, jsd_vec)
        atomic_write_numpy_array(step_path, step_vec)
        nll_paths_out.append(str(nll_path))
        entropy_paths_out.append(str(ent_path))
        jsd_paths_out.append(str(jsd_path))
        stepindex_paths_out.append(str(step_path))
    return (
        sequences_out,
        latents_out,
        decode_nll_means_out,
        edited_counts_out,
        sum_neglogp_cond_nats_out,
        sum_neglogp_uncond_nats_out,
        commit_hists_out,
        masked_positions_out,
        trace_paths_out,
        invalid_resamples_out,
        anchor_mismatch_violations_out,
        nll_paths_out,
        entropy_paths_out,
        jsd_paths_out,
        stepindex_paths_out,
    )
