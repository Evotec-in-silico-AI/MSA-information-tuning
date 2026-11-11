from __future__ import annotations

import sys
import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import tensorflow as tf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cohort_builder_common as common
from modules.train_mdlm import MDLMDenoiser, MDLMConfig



class GpuBuilder:
    """Run GPU-stage all-site confidence inference."""
    def __init__(self, repo_root: Path, cohort_xlsx: Path, device: str):
        self.repo_root = Path(repo_root)
        self.cohort_xlsx = Path(cohort_xlsx)
        self.device = str(device)
        self.paths = common.build_paths(self.repo_root)
        self.logger = common.BuilderLogger(self.paths.logs_dir / "gpu_builder.jsonl")
        self.runner = GpuRunner(self.device, self.logger)

    def run(self, limit: int) -> None:
        """Compute GPU outputs for all samples not yet checkpointed."""
        cohort_df = common.read_cohort_by_sample(self.cohort_xlsx)
        common.write_build_manifest(self.paths, common.build_manifest(self.cohort_xlsx, cohort_df))

        resolved_by_sample_id = common.resolve_cohort(self.cohort_xlsx, self.logger)

        sample_ids = cohort_df["sample_id"].astype(str).tolist()
        if int(limit) > 0:
            sample_ids = sample_ids[: int(limit)]

        pending: List[common.ResolvedSample] = []
        missing_resolution = 0
        for sample_id in sample_ids:
            if common.is_checkpoint_complete(self.paths, "gpu", sample_id):
                continue
            resolved = resolved_by_sample_id.get(sample_id)
            if resolved is None:
                missing_resolution += 1
                continue
            pending.append(resolved)

        self.logger.log(
            "gpu_builder_start",
            {
                "total_samples": int(len(sample_ids)),
                "pending": int(len(pending)),
                "missing_resolution": int(missing_resolution),
                "device": self.device,
            },
        )

        processed = 0
        failures = 0
        for resolved in pending:
            sample_id = resolved.sample_id
            if common.is_checkpoint_complete(self.paths, "gpu", sample_id):
                continue

            started = time.time()
            out_paths = common.build_gpu_output_paths(self.paths, resolved.protein_name, sample_id)

            try:
                table, qc = self.runner.score(resolved)
                if table is None:
                    failures += 1
                    self.logger.log("gpu_sample_failed", {"sample_id": sample_id, **qc})
                    continue

                common.atomic_write_parquet(table, out_paths["per_site_conf"])

                duration = float(time.time() - started)
                common.mark_checkpoint_complete(
                    self.paths,
                    "gpu",
                    sample_id,
                    {
                        "sample_id": sample_id,
                        "duration_sec": duration,
                        **qc,
                    },
                )

                processed += 1
                if processed % 200 == 0:
                    self.logger.log(
                        "gpu_progress",
                        {
                            "processed": int(processed),
                            "failures": int(failures),
                            "remaining": int(len(pending) - processed),
                        },
                    )

            except Exception as exc:
                failures += 1
                self.logger.log("gpu_sample_exception", {"sample_id": sample_id, "error": str(exc)})

        self.logger.log("gpu_builder_done", {"processed": int(processed), "failures": int(failures)})


class GpuRunner:
    """Score samples while reusing model/context across model_dir groups."""
    def __init__(self, device: str, logger: common.BuilderLogger):
        self.device = str(device)
        self.logger = logger
        self.current_model_dir = ""
        self.model: Optional[Any] = None
        self.cfg_dict: Dict[str, Any] = {}
        self.context_array: Optional[np.ndarray] = None

    def score(self, resolved: common.ResolvedSample) -> Tuple[Optional[pa.Table], Dict[str, Any]]:
        """Compute per-site NLL/entropy/JSD for one sample."""
        self.ensure_loaded(resolved)
        return self.score_one(resolved)

    def ensure_loaded(self, resolved: common.ResolvedSample) -> None:
        """Load weights/context when model_dir changes."""
        model_dir = str(Path(resolved.model_dir))
        if self.model is not None and model_dir == self.current_model_dir:
            return

        self.model, self.cfg_dict = self.load_model_and_cfg(Path(model_dir))
        self.context_array = self.load_msa_context(Path(resolved.shard_path))
        self.current_model_dir = model_dir

        self.logger.log(
            "gpu_model_loaded",
            {"model_dir": self.current_model_dir, "context_present": bool(self.context_array is not None)},
        )

    def load_model_and_cfg(self, model_dir: Path) -> Tuple[Any, Dict[str, Any]]:
        """Load MDLM model and config.json."""

        cfg_path = Path(model_dir) / "config.json"
        cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))

        cfg_fields = {k: v for k, v in cfg_dict.items() if k in MDLMConfig.__annotations__}
        cfg = MDLMConfig(**cfg_fields)

        conditioning = cfg_dict.get("conditioning", {})
        context_dim = int(conditioning.get("context_dim", 0) or 0) if isinstance(conditioning, dict) else 0

        model = MDLMDenoiser(cfg, context_dim=context_dim)
        dummy_xt = tf.zeros((1, 2), tf.int32)
        dummy_rate = tf.constant([0.5], tf.float32)
        _ = model(dummy_xt, dummy_rate, context=None, training=False)
        model.load_weights(str(Path(model_dir) / "weights")).expect_partial()
        return model, cfg_dict

    def load_msa_context(self, shard_path: Path) -> Optional[np.ndarray]:
        """Load msa_context.pkl if present."""
        path = Path(shard_path) / "msa_context.pkl"
        if not path.exists():
            return None

        with open(str(path), "rb") as handle:
            obj = pickle.load(handle)

        if isinstance(obj, np.ndarray):
            return np.asarray(obj, dtype=np.float32)

        if isinstance(obj, dict):
            for key in ["context", "msa_context", "features", "x", "tensor"]:
                if key in obj:
                    return np.asarray(obj[key], dtype=np.float32)

        return np.asarray(obj, dtype=np.float32)

    def score_one(self, resolved: common.ResolvedSample) -> Tuple[Optional[pa.Table], Dict[str, Any]]:
        """Score one sample with conditioned and unconditioned passes."""
        qc: Dict[str, Any] = {}

        with open(str(resolved.compressed_msa_pkl_path), "rb") as handle:
            msa_tokens = np.asarray(pickle.load(handle), dtype=np.int32)

        anchor_tokens = msa_tokens[0].astype(np.int32, copy=False)
        length = int(anchor_tokens.shape[0])

        mask_vec = common.read_mask_vector(resolved.mask_path)
        if mask_vec is None:
            qc["mask_missing"] = True
            return None, qc

        if int(mask_vec.shape[0]) != length:
            qc["mask_length_mismatch"] = {"expected": length, "got": int(mask_vec.shape[0])}
            if mask_vec.shape[0] > length:
                mask_vec = mask_vec[:length]
            else:
                mask_vec = np.pad(mask_vec, (0, length - mask_vec.shape[0]), constant_values=0)

        final_tokens = self.encode_sequence_to_tokens(resolved.sequence, length)

        mask_id = int(self.cfg_dict.get("mask_id", 21))
        eos_id = int(self.cfg_dict.get("eos_id", 22))
        forbidden = [0, mask_id, eos_id]

        xt = anchor_tokens.copy()
        xt[mask_vec.astype(bool)] = int(mask_id)

        rate = tf.constant([float(resolved.corruption_rate_init)], tf.float32)

        ctx_batched = None
        if self.context_array is not None:
            ctx_batched = tf.convert_to_tensor(self.context_array[None, :, :], dtype=tf.float32)

        with tf.device(self.device):
            logits_cond = self.model(tf.convert_to_tensor(xt[None, :], tf.int32), rate, context=ctx_batched, training=False)
            logits_uncond = self.model(tf.convert_to_tensor(xt[None, :], tf.int32), rate, context=None, training=False)

        probs_cond = tf.nn.softmax(logits_cond[0], axis=-1).numpy().astype(np.float32)
        probs_uncond = tf.nn.softmax(logits_uncond[0], axis=-1).numpy().astype(np.float32)

        probs_cond = self.apply_forbidden_and_renormalize(probs_cond, forbidden)
        probs_uncond = self.apply_forbidden_and_renormalize(probs_uncond, forbidden)

        ent_cond = self.compute_entropy_nats(probs_cond)
        ent_uncond = self.compute_entropy_nats(probs_uncond)

        tok_idx = final_tokens.astype(np.int64)
        p_tok_cond = probs_cond[np.arange(length), tok_idx]
        p_tok_uncond = probs_uncond[np.arange(length), tok_idx]

        nll_cond = (-np.log(np.maximum(p_tok_cond, 1e-12))).astype(np.float32)
        nll_uncond = (-np.log(np.maximum(p_tok_uncond, 1e-12))).astype(np.float32)

        jsd = self.compute_jsd_nats(probs_cond, probs_uncond)

        qc["length"] = length

        table = pa.table(
            {
                "sample_id": pa.array([resolved.sample_id] * length, pa.string()),
                "protein_name": pa.array([resolved.protein_name] * length, pa.string()),
                "site_index": pa.array(np.arange(length, dtype=np.int32), pa.int32()),
                "nll_cond_all": pa.array(nll_cond, pa.float32()),
                "entropy_cond_all_nats": pa.array(ent_cond, pa.float32()),
                "nll_uncond_all": pa.array(nll_uncond, pa.float32()),
                "entropy_uncond_all_nats": pa.array(ent_uncond, pa.float32()),
                "jsd_cond_uncond_all_nats": pa.array(jsd, pa.float32()),
            }
        )

        return table, qc

    def encode_sequence_to_tokens(self, sequence: str, length: int) -> np.ndarray:
        """Encode a protein sequence to token ids."""
        seq = (sequence or "").strip().upper()
        tokens = np.zeros((length,), dtype=np.int32)
        for idx in range(length):
            ch = seq[idx] if idx < len(seq) else "-"
            tokens[idx] = int(common.AA_TO_TOKEN.get(ch, 0))
        return tokens

    def compute_entropy_nats(self, probs: np.ndarray) -> np.ndarray:
        """Compute entropy in nats per position."""
        p = np.asarray(probs, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(p > 0.0, np.log(p), 0.0)
        return (-np.sum(p * logp, axis=1)).astype(np.float32)

    def compute_jsd_nats(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Compute Jensen-Shannon divergence in nats per position."""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where((p > 0.0) & (m > 0.0), np.log(p) - np.log(m), 0.0)
            logq = np.where((q > 0.0) & (m > 0.0), np.log(q) - np.log(m), 0.0)
        kl_pm = np.sum(np.where(p > 0.0, p * logp, 0.0), axis=1)
        kl_qm = np.sum(np.where(q > 0.0, q * logq, 0.0), axis=1)
        return (0.5 * (kl_pm + kl_qm)).astype(np.float32)

    def apply_forbidden_and_renormalize(self, probs: np.ndarray, forbidden_ids: List[int]) -> np.ndarray:
        """Zero forbidden ids and renormalize probabilities."""
        out = np.asarray(probs, dtype=np.float64).copy()
        for token_id in forbidden_ids:
            if 0 <= int(token_id) < out.shape[1]:
                out[:, int(token_id)] = 0.0
        row_sum = np.sum(out, axis=1, keepdims=True)
        ok = row_sum[:, 0] > 0.0
        out[ok] = out[ok] / row_sum[ok]
        out[~ok] = 0.0
        return out.astype(np.float32)


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="GPU-stage all-site confidence builder")
    parser.add_argument("--repo_root", default=".", help="Repository root containing cohort/ and runs/")
    parser.add_argument("--cohort_xlsx", default="cohort/cohort.xlsx", help="Path to cohort.xlsx")
    parser.add_argument("--device", default="/GPU:0", help="TensorFlow device string")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of samples")
    return parser.parse_args()


def main() -> None:
    """Run GPU builder."""
    args = parse_args()
    builder = GpuBuilder(Path(args.repo_root).resolve(), Path(args.cohort_xlsx).resolve(), str(args.device))
    builder.run(int(args.limit))


if __name__ == "__main__":
    main()
