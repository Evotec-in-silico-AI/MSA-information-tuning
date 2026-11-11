from __future__ import annotations
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from . import conditioning
from .io_utils import atomic_write_csv_df, atomic_write_text


@dataclass
class MDLMConfig:
    """Hyperparameters for rate-conditioned denoiser training."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    cross_attn_every: int = 3
    vocab_base: int = 21
    mask_id: int = 21
    eos_id: int = 22
    vocab_size: int = 23
    lr: float = 3e-4
    batch_size: int = 16
    p_null: float = 0.10
    r_train_min: float = 0.50
    r_train_max: float = 0.95


def set_global_seed(seed: int) -> None:
    """Seed NumPy and TensorFlow RNGs."""
    np.random.seed(int(seed))
    tf.random.set_seed(int(seed))


def load_compressed_msa(output_paths: Dict[str, Any]) -> np.ndarray:
    """Load the compressed alignment token matrix from disk and validate shape."""
    path_value = output_paths.get("compressed_alignment")
    if not path_value:
        raise FileNotFoundError("output_paths['compressed_alignment'] is required")
    import pickle
    with open(str(path_value), "rb") as handle:
        matrix = np.asarray(pickle.load(handle), dtype=np.int32)
    if matrix.ndim != 2:
        raise ValueError(f"Compressed alignment must be 2D; got {matrix.shape}")
    return matrix


def sample_training_rates(batch_size: int, r_min: float, r_max: float) -> tf.Tensor:
    """Sample per-example corruption rates uniformly on [r_min, r_max]."""
    return tf.random.uniform(
        (int(batch_size),),
        minval=float(r_min),
        maxval=float(r_max),
        dtype=tf.float32,
    )


def make_stateless_seeds(base_seed: int, index_tensor: tf.Tensor, draw_id: int, salt0: int, salt1: int) -> tf.Tensor:
    """Build a (B, 2) seed tensor keyed by example index and draw identifier."""
    seed0 = tf.cast(int(base_seed) + int(salt0) + 1000 * int(draw_id), tf.int32)
    seed1 = tf.cast(int(salt1), tf.int32) + tf.cast(index_tensor, tf.int32)
    return tf.stack([tf.fill(tf.shape(index_tensor), seed0), seed1], axis=1)


def sample_validation_rates(seeds: tf.Tensor, r_min: float, r_max: float) -> tf.Tensor:
    """Sample per-example rates deterministically from stateless seeds."""
    seed_list = tf.unstack(seeds, axis=0)
    out: List[tf.Tensor] = []
    for seed in seed_list:
        out.append(
            tf.random.stateless_uniform(
                (),
                seed=seed,
                minval=float(r_min),
                maxval=float(r_max),
                dtype=tf.float32,
            )
        )
    return tf.stack(out, axis=0)


def corrupt_tokens_stateful(x0: tf.Tensor, rate: tf.Tensor, cfg: MDLMConfig) -> tf.Tensor:
    """Apply stateful corruption by masking a Bernoulli fraction of eligible tokens."""
    rate = tf.reshape(tf.cast(rate, tf.float32), (-1, 1))
    is_gap = tf.equal(x0, 0)
    rnd = tf.random.uniform(tf.shape(x0), 0.0, 1.0, dtype=tf.float32)
    do_mask = tf.logical_and(tf.logical_not(is_gap), rnd < rate)
    return tf.where(do_mask, tf.fill(tf.shape(x0), int(cfg.mask_id)), x0)


def corrupt_tokens_one_example_stateless(x0_1d: tf.Tensor, rate_scalar: tf.Tensor, seed_2: tf.Tensor, cfg: MDLMConfig) -> tf.Tensor:
    """Apply deterministic corruption for one example using a stateless RNG."""
    r = tf.cast(rate_scalar, tf.float32)
    is_gap = tf.equal(x0_1d, 0)
    rnd = tf.random.stateless_uniform(tf.shape(x0_1d), seed=seed_2, minval=0.0, maxval=1.0, dtype=tf.float32)
    do_mask = tf.logical_and(tf.logical_not(is_gap), rnd < r)
    return tf.where(do_mask, tf.fill(tf.shape(x0_1d), int(cfg.mask_id)), x0_1d)


def corrupt_tokens_stateless(x0: tf.Tensor, rate: tf.Tensor, seeds: tf.Tensor, cfg: MDLMConfig) -> tf.Tensor:
    """Apply deterministic corruption for a batch using per-example stateless seeds."""
    x0_list = tf.unstack(x0, axis=0)
    r_list = tf.unstack(rate, axis=0)
    seed_list = tf.unstack(seeds, axis=0)
    out: List[tf.Tensor] = []
    for x0_i, r_i, seed_i in zip(x0_list, r_list, seed_list):
        out.append(corrupt_tokens_one_example_stateless(x0_i, r_i, seed_i, cfg))
    return tf.stack(out, axis=0)


class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block with optional cross-attention over conditioning context."""

    def __init__(self, cfg: MDLMConfig, use_cross_attention: bool):
        """Initialize attention and feed-forward sublayers."""
        super().__init__()
        self.use_cross_attention = bool(use_cross_attention)
        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=int(cfg.n_heads),
            key_dim=int(cfg.d_model // cfg.n_heads),
            dropout=float(cfg.dropout),
        )
        self.cross_attention = None
        if self.use_cross_attention:
            self.cross_attention = tf.keras.layers.MultiHeadAttention(
                num_heads=int(cfg.n_heads),
                key_dim=int(cfg.d_model // cfg.n_heads),
                dropout=float(cfg.dropout),
            )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(cfg.d_ff), activation="gelu"),
                tf.keras.layers.Dropout(float(cfg.dropout)),
                tf.keras.layers.Dense(int(cfg.d_model)),
            ]
        )
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop = tf.keras.layers.Dropout(float(cfg.dropout))

    def call(self, x: tf.Tensor, context: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        """Run self-attention, optional cross-attention, and feed-forward layers with residuals."""
        h = self.norm1(x)
        h = self.self_attention(h, h, training=training)
        x = x + self.drop(h, training=training)
        if self.use_cross_attention and context is not None and self.cross_attention is not None:
            h = self.norm2(x)
            h = self.cross_attention(h, context, training=training)
            x = x + self.drop(h, training=training)
        h = self.norm3(x)
        h = self.ffn(h, training=training)
        x = x + self.drop(h, training=training)
        return x


class MDLMDenoiser(tf.keras.Model):
    """Token-space denoiser conditioned on a scalar corruption rate."""

    def __init__(self, cfg: MDLMConfig, context_dim: int):
        """Construct embeddings, transformer blocks, and prediction head."""
        super().__init__()
        self.cfg = cfg
        self.context_dim = int(context_dim)
        self.token_embedding = tf.keras.layers.Embedding(int(cfg.vocab_size), int(cfg.d_model))
        self.position_embedding = tf.keras.layers.Embedding(2048, int(cfg.d_model))
        self.rate_projection = tf.keras.layers.Dense(int(cfg.d_model), activation=None)
        self.context_projection = None
        self.context_position_embedding = None
        if self.context_dim > 0:
            self.context_projection = tf.keras.layers.Dense(int(cfg.d_model), activation=None)
            self.context_position_embedding = tf.keras.layers.Embedding(2048, int(cfg.d_model))
        self.blocks: List[TransformerBlock] = []
        for layer_index in range(int(cfg.n_layers)):
            use_cross = (
                self.context_dim > 0
                and int(cfg.cross_attn_every) > 0
                and (layer_index % int(cfg.cross_attn_every) == 0)
            )
            self.blocks.append(TransformerBlock(cfg, use_cross_attention=use_cross))
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.head = tf.keras.layers.Dense(int(cfg.vocab_size))

    def encode(
        self,
        xt: tf.Tensor,
        rate: tf.Tensor,
        context: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode tokens with positional, rate, and optional context embeddings."""
        batch_size = tf.shape(xt)[0]
        length = tf.shape(xt)[1]
        positions = tf.tile(tf.expand_dims(tf.range(length), 0), [batch_size, 1])
        x = self.token_embedding(xt) + self.position_embedding(positions)
        rate = tf.reshape(tf.cast(rate, tf.float32), (-1, 1))
        rate_emb = self.rate_projection(rate)
        x = x + tf.expand_dims(rate_emb, 1)
        ctx = None
        if self.context_projection is not None and context is not None:
            projected = self.context_projection(context)
            if self.context_position_embedding is not None:
                context_length = tf.shape(projected)[1]
                context_positions = tf.tile(tf.expand_dims(tf.range(context_length), 0), [batch_size, 1])
                projected = projected + self.context_position_embedding(context_positions)
            ctx = projected
        for block in self.blocks:
            x = block(x, context=ctx, training=training)
        x = self.norm(x)
        pooled = tf.reduce_mean(x, axis=1)
        return pooled, x

    def call(
        self,
        xt: tf.Tensor,
        rate: tf.Tensor,
        context: Optional[tf.Tensor] = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Compute per-position logits given possibly corrupted tokens, rate, and optional context."""
        pooled_unused, x = self.encode(xt, rate, context=context, training=training)
        return self.head(x)


def build_conditioning_batch(context_base: Optional[tf.Tensor], batch_size: int, p_null: float) -> Optional[tf.Tensor]:
    """Tile a base context across the batch with classifier-free dropout."""
    if context_base is None:
        return None
    context_batch = tf.tile(tf.expand_dims(context_base, 0), [int(batch_size), 1, 1])
    drop_mask = tf.random.uniform((int(batch_size), 1, 1), 0.0, 1.0) < float(p_null)
    return tf.where(drop_mask, tf.zeros_like(context_batch), context_batch)


def compute_masked_and_nongap_losses(
    x0: tf.Tensor,
    xt: tf.Tensor,
    per_pos_loss: tf.Tensor,
    mask_id: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compute masked-position and non-gap losses with a stable fallback when no masked tokens exist."""
    non_gap = tf.not_equal(x0, 0)
    is_masked = tf.equal(xt, int(mask_id))
    mask = tf.logical_and(non_gap, is_masked)
    any_masked = tf.reduce_any(mask, axis=1, keepdims=True)
    mask = tf.where(any_masked, mask, non_gap)
    per_pos_masked = tf.where(mask, per_pos_loss, tf.zeros_like(per_pos_loss))
    denom_masked = tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-8
    loss_masked = tf.reduce_sum(per_pos_masked) / denom_masked
    per_pos_nongap = tf.where(non_gap, per_pos_loss, tf.zeros_like(per_pos_loss))
    denom_nongap = tf.reduce_sum(tf.cast(non_gap, tf.float32)) + 1e-8
    loss_nongap = tf.reduce_sum(per_pos_nongap) / denom_nongap
    return loss_masked, loss_nongap


def ensure_minimum_one_mask_stateful(x0: tf.Tensor, xt: tf.Tensor, cfg: MDLMConfig) -> tf.Tensor:
    """Ensure each example has at least one masked token by force-masking one eligible position when needed."""
    def fix_row(row_inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        row_x0, row_xt = row_inputs
        has_mask = tf.reduce_any(tf.equal(row_xt, int(cfg.mask_id)))
        def no_op() -> tf.Tensor:
            return row_xt
        def apply_force_mask() -> tf.Tensor:
            eligible = tf.where(tf.not_equal(row_x0, 0))[:, 0]
            def mask_first() -> tf.Tensor:
                k = tf.shape(eligible)[0]
                idx = tf.random.uniform((), minval=0, maxval=k, dtype=tf.int32)
                pos = eligible[idx]
                updated = tf.tensor_scatter_nd_update(
                    row_xt,
                    indices=tf.reshape(pos, (1, 1)),
                    updates=tf.constant([int(cfg.mask_id)], dtype=row_xt.dtype),
                )
                return updated
            def return_original() -> tf.Tensor:
                return row_xt
            return tf.cond(tf.size(eligible) > 0, mask_first, return_original)
        return tf.cond(has_mask, no_op, apply_force_mask)
    fixed_body = tf.map_fn(
        fix_row,
        (x0, xt),
        fn_output_signature=tf.int32,
        parallel_iterations=10,
    )
    fixed = tf.nest.map_structure(tf.stop_gradient, fixed_body)
    return fixed


def run_training_step(
    model: MDLMDenoiser,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: Any,
    x0: tf.Tensor,
    cfg: MDLMConfig,
    context_base: Optional[tf.Tensor],
) -> Tuple[float, float]:
    """Run one optimizer step and return masked and non-gap losses."""
    batch_size = int(x0.shape[0])
    rate = sample_training_rates(batch_size, cfg.r_train_min, cfg.r_train_max)
    xt = corrupt_tokens_stateful(x0, rate, cfg)
    xt = ensure_minimum_one_mask_stateful(x0, xt, cfg)
    context_in = build_conditioning_batch(context_base, batch_size, cfg.p_null)
    with tf.GradientTape() as tape:
        logits = model(xt, rate, context=context_in, training=True)
        per_pos = loss_fn(x0, logits)
        loss_masked, loss_nongap = compute_masked_and_nongap_losses(x0, xt, per_pos, cfg.mask_id)
        grads = tape.gradient(loss_masked, model.trainable_variables)
    grad_var_pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    optimizer.apply_gradients(grad_var_pairs)
    return float(loss_masked.numpy()), float(loss_nongap.numpy())


def run_validation_draw(
    model: MDLMDenoiser,
    loss_fn: Any,
    x0: tf.Tensor,
    val_index: tf.Tensor,
    cfg: MDLMConfig,
    base_seed: int,
    draw_id: int,
    context_base: Optional[tf.Tensor],
) -> Tuple[float, float]:
    """Compute validation losses for one deterministic corruption draw keyed by sequence index."""
    batch_size = int(x0.shape[0])
    rate_seeds = make_stateless_seeds(int(base_seed), val_index, int(draw_id), salt0=1103515245, salt1=12345)
    rate = sample_validation_rates(rate_seeds, cfg.r_train_min, cfg.r_train_max)
    mask_seeds = make_stateless_seeds(int(base_seed), val_index, int(draw_id), salt0=214013, salt1=54321)
    xt = corrupt_tokens_stateless(x0, rate, mask_seeds, cfg)
    context_in = None
    if context_base is not None:
        context_in = tf.tile(tf.expand_dims(context_base, 0), [batch_size, 1, 1])
    logits = model(xt, rate, context=context_in, training=False)
    per_pos = loss_fn(x0, logits)
    loss_masked, loss_nongap = compute_masked_and_nongap_losses(x0, xt, per_pos, cfg.mask_id)
    return float(loss_masked.numpy()), float(loss_nongap.numpy())


def write_training_curve(curve_path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write the training curve to CSV using an atomic write."""
    import pandas as pd
    df = pd.DataFrame(rows)
    atomic_write_csv_df(curve_path, df)


def checkpoint_complete(model_dir: Path, weights_prefix: str = "weights") -> bool:
    """Return True when a TensorFlow checkpoint appears complete for a given prefix."""
    index_path = model_dir / f"{weights_prefix}.index"
    data_files = list(model_dir.glob(f"{weights_prefix}.data-*"))
    return index_path.exists() and (len(data_files) > 0)


def save_weights_safely(model_dir: Path, model: MDLMDenoiser, weights_prefix: str = "weights") -> None:
    """Save TensorFlow weights via a temporary prefix, then atomically rename into place."""
    model_dir.mkdir(parents=True, exist_ok=True)
    tmp_prefix = f"{weights_prefix}.tmp"
    tmp_prefix_path = model_dir / tmp_prefix
    tmp_index_path = model_dir / f"{tmp_prefix}.index"
    for temp_path in [tmp_index_path] + list(model_dir.glob(f"{tmp_prefix}.data-*")):
        if temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    model.save_weights(str(tmp_prefix_path))
    tmp_index_path = model_dir / f"{tmp_prefix}.index"
    tmp_data_files = list(model_dir.glob(f"{tmp_prefix}.data-*"))
    if not tmp_index_path.exists() or len(tmp_data_files) == 0:
        existing = sorted([p.name for p in model_dir.iterdir()]) if model_dir.exists() else []
        raise FileNotFoundError(
            f"Checkpoint save did not produce expected files for prefix '{tmp_prefix_path}'. "
            f"Directory contents: {existing}"
        )
    final_index_path = model_dir / f"{weights_prefix}.index"
    for final_path in [final_index_path] + list(model_dir.glob(f"{weights_prefix}.data-*")):
        if final_path.exists():
            try:
                final_path.unlink()
            except FileNotFoundError:
                pass
    tmp_index_path.replace(final_index_path)
    for tmp_file in tmp_data_files:
        final_name = tmp_file.name.replace(f"{tmp_prefix}.data-", f"{weights_prefix}.data-")
        tmp_file.replace(model_dir / final_name)
    if not checkpoint_complete(model_dir, weights_prefix=weights_prefix):
        existing = sorted([p.name for p in model_dir.iterdir()]) if model_dir.exists() else []
        raise FileNotFoundError(
            f"Checkpoint rename did not produce a complete checkpoint at prefix '{model_dir / weights_prefix}'. "
            f"Directory contents: {existing}"
        )


def ensure_model_checkpoint(model_dir: Path, model: MDLMDenoiser, length: int) -> None:
    """Ensure a valid checkpoint exists by writing one if necessary."""
    if checkpoint_complete(model_dir, weights_prefix="weights"):
        return
    dummy_xt = tf.zeros((1, int(length)), tf.int32)
    dummy_rate = tf.constant([0.5], tf.float32)
    _ = model(dummy_xt, dummy_rate, context=None, training=False)
    save_weights_safely(model_dir, model, weights_prefix="weights")


def train_mdlm_epochs(
    output_paths: Dict[str, Any],
    epochs: int,
    seed: int,
    p_null: float,
    conditioning_enabled: bool = True,
    conditioning_types: Sequence[str] = ("msa",),
    val_draws: int = 3,
    pipeline_version: str = "1.0",
) -> float:
    """Train for a fixed number of epochs and return best validation masked loss."""
    set_global_seed(int(seed))
    X = load_compressed_msa(output_paths)
    num_rows, length = X.shape
    cfg = MDLMConfig()
    cfg.p_null = float(p_null)
    context_base, conditioning_meta = conditioning.load_conditioning(
        output_paths,
        enabled=bool(conditioning_enabled),
        requested_types=conditioning_types,
    )
    if context_base is not None and not isinstance(context_base, tf.Tensor):
        context_base = tf.convert_to_tensor(context_base, dtype=tf.float32)
    context_dim = int(conditioning_meta.context_dim) if conditioning_meta else 0
    model = MDLMDenoiser(cfg, context_dim=context_dim)

    batch_size = int(min(cfg.batch_size, max(1, num_rows)))
    val_n = max(1, int(0.05 * num_rows))
    train_X = X[:-val_n] if num_rows > val_n else X
    val_X = X[-val_n:]

    train_ds = tf.data.Dataset.from_tensor_slices(train_X)
    train_ds = train_ds.shuffle(min(len(train_X), 10000), seed=int(seed), reshuffle_each_iteration=True)
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    val_len = int(val_X.shape[0])
    val_ds = tf.data.Dataset.from_tensor_slices((val_X, tf.range(val_len, dtype=tf.int32)))
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg.lr))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    model_dir = Path(str(output_paths["diffusion_model_dir"]))
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    curve_path = model_dir / "training_curve.csv"
    summary_path = model_dir / "training_summary.json"

    cfg_dict = asdict(cfg)
    conditioning_dict = conditioning.meta_as_dict(conditioning_meta)
    if isinstance(conditioning_dict, dict):
        conditioning_dict["context_positional_encoding"] = bool(context_dim > 0)
        conditioning_dict["row_weights_theta_used"] = 0.70 if context_dim > 0 else 0.0
    cfg_dict["conditioning"] = conditioning_dict
    cfg_dict["pipeline_version"] = str(pipeline_version)
    atomic_write_text(config_path, json.dumps(cfg_dict, indent=2))

    ensure_model_checkpoint(model_dir, model, length)

    K = int(max(1, min(int(val_draws), 8)))
    curve_rows: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1

    for ep in range(int(epochs)):
        train_masked_vals: List[float] = []
        train_nongap_vals: List[float] = []
        for batch in train_ds:
            lm, lng = run_training_step(model, optimizer, loss_fn, batch, cfg, context_base)
            train_masked_vals.append(lm)
            train_nongap_vals.append(lng)
        train_loss_masked = float(np.mean(train_masked_vals)) if train_masked_vals else float("nan")
        train_loss_nongap = float(np.mean(train_nongap_vals)) if train_nongap_vals else float("nan")

        val_masked_vals: List[float] = []
        val_nongap_vals: List[float] = []
        for vb, vidx in val_ds:
            draw_masked: List[float] = []
            draw_nongap: List[float] = []
            for draw_id in range(K):
                lm, lng = run_validation_draw(model, loss_fn, vb, vidx, cfg, int(seed), int(draw_id), context_base)
                draw_masked.append(lm)
                draw_nongap.append(lng)
            val_masked_vals.append(float(np.mean(draw_masked)))
            val_nongap_vals.append(float(np.mean(draw_nongap)))
        val_loss_masked = float(np.mean(val_masked_vals)) if val_masked_vals else float("nan")
        val_loss_nongap = float(np.mean(val_nongap_vals)) if val_nongap_vals else float("nan")

        is_best = 0
        if np.isfinite(val_loss_masked) and val_loss_masked < best_val:
            best_val = float(val_loss_masked)
            best_epoch = int(ep) + 1
            is_best = 1
            save_weights_safely(model_dir, model, weights_prefix="weights")

        curve_rows.append(
            {
                "epoch": int(ep) + 1,
                "train_loss_masked": float(train_loss_masked),
                "train_loss_nongap": float(train_loss_nongap),
                "val_loss_masked": float(val_loss_masked),
                "val_loss_nongap": float(val_loss_nongap),
                "val_draws": int(K),
                "selection_metric": "val_loss_masked",
                "is_best": int(is_best),
            }
        )
        write_training_curve(curve_path, curve_rows)

    summary = {
        "selection_metric": "val_loss_masked",
        "val_draws": int(K),
        "best": {"best_epoch": int(best_epoch), "best_val_loss_masked": float(best_val)},
    }
    atomic_write_text(summary_path, json.dumps(summary, indent=2))
    return float(best_val)


def train_mdlm(
    *,
    output_paths: Dict[str, Any],
    max_steps: int,
    seed: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    early_stop_burn_in_steps: int,
    p_null: float,
    conditioning_enabled: bool = True,
    conditioning_types: Sequence[str] = ("msa",),
    val_draws: int = 3,
    pipeline_version: str = "1.0",
) -> float:
    """Train for a fixed optimizer step budget and return best validation masked loss."""
    set_global_seed(int(seed))
    X = load_compressed_msa(output_paths)
    num_rows, length = X.shape
    cfg = MDLMConfig()
    cfg.p_null = float(p_null)

    context_base, conditioning_meta = conditioning.load_conditioning(
        output_paths,
        enabled=bool(conditioning_enabled),
        requested_types=conditioning_types,
    )
    if context_base is not None and not isinstance(context_base, tf.Tensor):
        context_base = tf.convert_to_tensor(context_base, dtype=tf.float32)
    context_dim = int(conditioning_meta.context_dim) if conditioning_meta else 0
    model = MDLMDenoiser(cfg, context_dim=context_dim)

    batch_size = int(min(cfg.batch_size, max(1, num_rows)))
    val_n = max(1, int(0.05 * num_rows))
    train_X = X[:-val_n] if num_rows > val_n else X
    val_X = X[-val_n:]

    train_ds = tf.data.Dataset.from_tensor_slices(train_X)
    train_ds = train_ds.shuffle(min(len(train_X), 10000), seed=int(seed), reshuffle_each_iteration=True)
    train_ds = train_ds.repeat().batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    val_len = int(val_X.shape[0])
    val_ds = tf.data.Dataset.from_tensor_slices((val_X, tf.range(val_len, dtype=tf.int32)))
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(cfg.lr))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    model_dir = Path(str(output_paths["diffusion_model_dir"]))
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    curve_path = model_dir / "training_curve.csv"
    summary_path = model_dir / "training_summary.json"

    eval_interval_steps = int(max(50, min(500, int(max_steps) // 20 if int(max_steps) > 0 else 200)))
    val_draws = int(max(1, min(int(val_draws), 8)))

    cfg_dict = asdict(cfg)
    conditioning_dict = conditioning.meta_as_dict(conditioning_meta)
    if isinstance(conditioning_dict, dict):
        conditioning_dict["context_positional_encoding"] = bool(context_dim > 0)
        conditioning_dict["row_weights_theta_used"] = 0.70 if context_dim > 0 else 0.0
    cfg_dict["conditioning"] = conditioning_dict
    cfg_dict["training_budget"] = {
        "max_steps": int(max_steps),
        "eval_interval_steps": int(eval_interval_steps),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stop_burn_in_steps": int(early_stop_burn_in_steps),
    }
    cfg_dict["validation"] = {
        "selection_metric": "val_loss_masked",
        "deterministic": True,
        "val_draws": int(val_draws),
        "keyed_by": "sequence_index",
    }
    cfg_dict["pipeline_version"] = str(pipeline_version)
    atomic_write_text(config_path, json.dumps(cfg_dict, indent=2))

    ensure_model_checkpoint(model_dir, model, length)

    iterator_train = iter(train_ds)
    curve_rows: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_step = -1
    patience_left = int(max(1, early_stop_patience))
    early_stopped = False
    stop_reason = ""
    window_masked: List[float] = []
    window_nongap: List[float] = []
    step = 0
    eval_id = 0

    while step < int(max_steps):
        step += 1
        batch = next(iterator_train)
        loss_masked, loss_nongap = run_training_step(model, optimizer, loss_fn, batch, cfg, context_base)
        window_masked.append(loss_masked)
        window_nongap.append(loss_nongap)

        if step % eval_interval_steps != 0 and step != int(max_steps):
            continue

        eval_id += 1
        train_loss_masked = float(np.mean(window_masked)) if window_masked else float("nan")
        train_loss_nongap = float(np.mean(window_nongap)) if window_nongap else float("nan")
        window_masked = []
        window_nongap = []

        val_masked_vals: List[float] = []
        val_nongap_vals: List[float] = []
        for vb, vidx in val_ds:
            draw_masked: List[float] = []
            draw_nongap: List[float] = []
            for draw_index in range(val_draws):
                lm, lng = run_validation_draw(model, loss_fn, vb, vidx, cfg, int(seed), int(draw_index), context_base)
                draw_masked.append(lm)
                draw_nongap.append(lng)
            val_masked_vals.append(float(np.mean(draw_masked)))
            val_nongap_vals.append(float(np.mean(draw_nongap)))
        val_loss_masked = float(np.mean(val_masked_vals)) if val_masked_vals else float("nan")
        val_loss_nongap = float(np.mean(val_nongap_vals)) if val_nongap_vals else float("nan")

        is_best = 0
        if np.isfinite(val_loss_masked) and (val_loss_masked + float(early_stop_min_delta)) < best_val:
            best_val = float(val_loss_masked)
            best_step = int(step)
            is_best = 1
            save_weights_safely(model_dir, model, weights_prefix="weights")
            patience_left = int(max(1, early_stop_patience))
        else:
            if step >= int(early_stop_burn_in_steps):
                patience_left -= 1

        curve_rows.append(
            {
                "eval_id": int(eval_id),
                "step": int(step),
                "train_loss_masked": float(train_loss_masked),
                "train_loss_nongap": float(train_loss_nongap),
                "val_loss_masked": float(val_loss_masked),
                "val_loss_nongap": float(val_loss_nongap),
                "val_draws": int(val_draws),
                "selection_metric": "val_loss_masked",
                "is_best": int(is_best),
            }
        )
        write_training_curve(curve_path, curve_rows)

        if step >= int(early_stop_burn_in_steps) and patience_left <= 0:
            early_stopped = True
            stop_reason = "early_stop_patience_exhausted"
            break

    total_steps_run = int(step)
    if not checkpoint_complete(model_dir, weights_prefix="weights"):
        save_weights_safely(model_dir, model, weights_prefix="weights")

    summary = {
        "selection_metric": "val_loss_masked",
        "val_draws": int(val_draws),
        "deterministic": True,
        "keyed_by": "sequence_index",
        "training_budget": {
            "max_steps": int(max_steps),
            "eval_interval_steps": int(eval_interval_steps),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
            "early_stop_burn_in_steps": int(early_stop_burn_in_steps),
        },
        "outcome": {
            "total_steps_run": int(total_steps_run),
            "early_stopped": bool(early_stopped),
            "stop_reason": str(stop_reason),
        },
        "best": {
            "best_step": int(best_step),
            "best_val_loss_masked": float(best_val) if np.isfinite(best_val) else float("nan"),
        },
        "paths": {
            "training_curve_csv": str(curve_path),
            "training_summary_json": str(summary_path),
            "weights_prefix": str(model_dir / "weights"),
            "config_json": str(config_path),
        },
    }
    atomic_write_text(summary_path, json.dumps(summary, indent=2))
    return float(best_val)
