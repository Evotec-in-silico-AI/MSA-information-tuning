from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa

import cohort_builder_common as common


class CpuBuilder:
    """Compute CPU-stage compiled datasets."""
    def __init__(self, repo_root: Path, cohort_xlsx: Path, workers: int, max_in_flight: int):
        self.repo_root = Path(repo_root)
        self.cohort_xlsx = Path(cohort_xlsx)
        self.workers = int(workers)
        self.max_in_flight = int(max_in_flight)
        self.paths = common.build_paths(self.repo_root)
        self.logger = common.BuilderLogger(self.paths.logs_dir / "cpu_builder.jsonl")
        self.worker_logger_path = str(self.paths.logs_dir / "cpu_workers.jsonl")

    def run(self, limit: int) -> None:
        """Run the CPU stage."""
        cohort_df = common.read_cohort_by_sample(self.cohort_xlsx)
        common.write_build_manifest(self.paths, common.build_manifest(self.cohort_xlsx, cohort_df))
        resolved_by_sample_id = common.resolve_cohort(self.cohort_xlsx, self.logger)
        sample_ids = cohort_df["sample_id"].astype(str).tolist()
        if int(limit) > 0:
            sample_ids = sample_ids[: int(limit)]
        resolved_samples: List[common.ResolvedSample] = []
        missing_resolution = 0
        for sample_id in sample_ids:
            resolved = resolved_by_sample_id.get(sample_id)
            if resolved is None:
                missing_resolution += 1
                continue
            resolved_samples.append(resolved)
        self.logger.log(
            "cpu_builder_start",
            {
                "total_samples": int(len(sample_ids)),
                "resolved_samples": int(len(resolved_samples)),
                "missing_resolution": int(missing_resolution),
                "workers": int(self.workers),
                "max_in_flight": int(self.max_in_flight),
            },
        )
        self.run_pool(resolved_samples)
        self.logger.log("cpu_builder_done", {})

    def run_pool(self, resolved_samples: List[common.ResolvedSample]) -> None:
        """Run worker pool with bounded in-flight tasks."""
        max_workers = max(1, int(self.workers))
        max_in_flight = max(max_workers, int(self.max_in_flight))
        submitted = 0
        completed = 0
        failures = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            index = 0
            while index < len(resolved_samples) or futures:
                while index < len(resolved_samples) and len(futures) < max_in_flight:
                    futures.append(executor.submit(CpuWorker.run_one, self.paths, resolved_samples[index], self.worker_logger_path))
                    index += 1
                    submitted += 1
                for future in as_completed(list(futures)):
                    sample_id, ok, err = future.result()
                    futures.remove(future)
                    completed += 1
                    if not ok:
                        failures += 1
                        self.logger.log("cpu_sample_failed", {"sample_id": sample_id, "error": err})
                    if completed % 200 == 0:
                        self.logger.log(
                            "cpu_progress",
                            {
                                "submitted": int(submitted),
                                "completed": int(completed),
                                "failures": int(failures),
                                "remaining": int(len(resolved_samples) - completed),
                            },
                        )
                    break


class CpuWorker:
    """Process one sample for CPU stage."""
    @staticmethod
    def run_one(paths: common.BuilderPaths, resolved: common.ResolvedSample, worker_log_path: str) -> Tuple[str, bool, str]:
        """Run CPU work for one sample."""
        sample_id = str(resolved.sample_id)
        worker_logger = common.BuilderLogger(Path(worker_log_path))
        if common.is_checkpoint_complete(paths, "cpu", sample_id):
            return sample_id, True, ""
        started = time.time()
        try:
            out_paths = common.build_cpu_output_paths(paths, resolved.protein_name, sample_id)
            per_site_table, qc = CpuWorker.build_per_site_table(resolved)
            common.atomic_write_parquet(per_site_table, out_paths["per_site"])
            edit_events_table = CpuWorker.build_edit_events_table(resolved)
            if edit_events_table.num_rows > 0:
                common.atomic_write_parquet(edit_events_table, out_paths["edit_events"])
            commit_step = np.asarray(per_site_table.column("commit_step_index").to_numpy(), dtype=np.int16)
            per_step_table = CpuWorker.build_per_step_table(resolved, commit_step)
            if per_step_table.num_rows > 0:
                common.atomic_write_parquet(per_step_table, out_paths["per_step"])
            duration = float(time.time() - started)
            common.mark_checkpoint_complete(paths, "cpu", sample_id, {"sample_id": sample_id, "duration_sec": duration, **qc})
            worker_logger.log("cpu_sample_complete", {"sample_id": sample_id, "protein_name": resolved.protein_name, "duration_sec": duration})
            return sample_id, True, ""
        except Exception as exc:
            worker_logger.log("cpu_sample_exception", {"sample_id": sample_id, "error": str(exc)})
            return sample_id, False, str(exc)

    @staticmethod
    def build_per_site_table(resolved: common.ResolvedSample) -> Tuple[pa.Table, Dict[str, Any]]:
        """Build per-site CPU table for one sample."""
        qc: Dict[str, Any] = {}
        import pickle
        with open(str(resolved.compressed_msa_pkl_path), "rb") as handle:
            msa_tokens = np.asarray(pickle.load(handle), dtype=np.int32)
        if msa_tokens.ndim != 2 or msa_tokens.size == 0:
            raise ValueError(f"Compressed MSA invalid shape {msa_tokens.shape}")
        anchor_tokens = msa_tokens[0].astype(np.int32, copy=False)
        length = int(anchor_tokens.shape[0])
        occupancy, entropy_bits, bits_site, effective_alphabet = common.compute_msa_site_metrics_from_compressed_msa(msa_tokens)
        mask_vec = common.read_mask_vector(resolved.mask_path)
        if mask_vec is None:
            qc["mask_missing"] = True
            mask_vec = np.zeros((length,), dtype=np.uint8)
        mask_vec = CpuWorker.pad_or_trim(mask_vec.astype(np.uint8), length, 0, qc, "mask")
        commit_step = common.read_numpy_vector(resolved.commit_step_index_path, dtype=np.int16)
        if commit_step is None:
            qc["commit_step_missing"] = True
            commit_step = np.full((length,), -1, dtype=np.int16)
        commit_step = CpuWorker.pad_or_trim(commit_step.astype(np.int16), length, -1, qc, "commit_step")
        nll_edit = common.read_numpy_vector(resolved.nll_cond_path, dtype=np.float32)
        if nll_edit is None:
            qc["nll_edit_missing"] = True
            nll_edit = np.full((length,), np.nan, dtype=np.float32)
        nll_edit = CpuWorker.pad_or_trim(nll_edit.astype(np.float32), length, np.nan, qc, "nll_edit")
        ent_edit = common.read_numpy_vector(resolved.entropy_cond_path, dtype=np.float32)
        if ent_edit is None:
            qc["entropy_edit_missing"] = True
            ent_edit = np.full((length,), np.nan, dtype=np.float32)
        ent_edit = CpuWorker.pad_or_trim(ent_edit.astype(np.float32), length, np.nan, qc, "entropy_edit")
        jsd_edit = common.read_numpy_vector(resolved.jsd_cond_uncond_path, dtype=np.float32)
        if jsd_edit is None:
            qc["jsd_edit_missing"] = True
            jsd_edit = np.full((length,), np.nan, dtype=np.float32)
        jsd_edit = CpuWorker.pad_or_trim(jsd_edit.astype(np.float32), length, np.nan, qc, "jsd_edit")
        sidecar_dir = common.build_sidecar_dir(resolved.run_root, resolved.sample_id)
        plddt_paths = common.list_plddt_slot_paths(sidecar_dir)
        pae_paths = common.list_pae_slot_paths(sidecar_dir)
        plddt_mean, plddt_std = common.compute_plddt_mean_std_from_slots(plddt_paths)
        if plddt_mean is None or plddt_std is None:
            qc["plddt_slots_missing"] = True
            plddt_mean = np.full((length,), np.nan, dtype=np.float32)
            plddt_std = np.full((length,), np.nan, dtype=np.float32)
        pae_mean, pae_std = common.compute_pae_flattened_mean_std_from_slots(pae_paths)
        if pae_mean is None or pae_std is None:
            qc["pae_slots_missing"] = True
            pae_mean = np.full((length,), np.nan, dtype=np.float32)
            pae_std = np.full((length,), np.nan, dtype=np.float32)
        anchor_aa, sample_aa, is_mut = common.build_mutation_vectors(resolved.anchor_sequence, resolved.sequence, length)
        qc["length"] = length
        qc["sidecar_dir"] = str(sidecar_dir)
        qc["plddt_slots"] = int(len(plddt_paths))
        qc["pae_slots"] = int(len(pae_paths))
        qc["trace_present"] = bool(common.read_trace_npz(resolved.trace_path) is not None)
        table = pa.table(
            {
                "sample_id": pa.array([resolved.sample_id] * length, pa.string()),
                "protein_name": pa.array([resolved.protein_name] * length, pa.string()),
                "family": pa.array([resolved.family] * length, pa.string()),
                "control_type": pa.array([resolved.control_type] * length, pa.string()),
                "control_rep": pa.array([int(resolved.control_rep)] * length, pa.int32()),
                "p10_target": pa.array([float(resolved.p10_target)] * length, pa.float64()),
                "corruption_rate_init": pa.array([float(resolved.corruption_rate_init)] * length, pa.float64()),
                "site_index": pa.array(np.arange(length, dtype=np.int32), pa.int32()),
                "msa_occupancy": pa.array(occupancy.astype(np.float32), pa.float32()),
                "msa_entropy_bits": pa.array(entropy_bits.astype(np.float32), pa.float32()),
                "msa_bits_per_site": pa.array(bits_site.astype(np.float32), pa.float32()),
                "msa_effective_alphabet": pa.array(effective_alphabet.astype(np.float32), pa.float32()),
                "mask_initial": pa.array(mask_vec.astype(np.uint8), pa.uint8()),
                "commit_step_index": pa.array(commit_step.astype(np.int16), pa.int16()),
                "nll_cond_edit": pa.array(nll_edit.astype(np.float32), pa.float32()),
                "entropy_cond_edit": pa.array(ent_edit.astype(np.float32), pa.float32()),
                "jsd_cond_uncond_edit": pa.array(jsd_edit.astype(np.float32), pa.float32()),
                "plddt_mean": pa.array(plddt_mean.astype(np.float32), pa.float32()),
                "plddt_std": pa.array(plddt_std.astype(np.float32), pa.float32()),
                "pae_mean": pa.array(pae_mean.astype(np.float32), pa.float32()),
                "pae_std": pa.array(pae_std.astype(np.float32), pa.float32()),
                "anchor_aa": pa.array(anchor_aa.tolist(), pa.string()),
                "sample_aa": pa.array(sample_aa.tolist(), pa.string()),
                "is_mutation": pa.array(is_mut.astype(np.uint8), pa.uint8()),
            }
        )
        return table, qc

    @staticmethod
    def pad_or_trim(vec: np.ndarray, length: int, fill_value: Any, qc: Dict[str, Any], name: str) -> np.ndarray:
        """Pad or trim a vector to match length."""
        if int(vec.shape[0]) == int(length):
            return vec
        qc[f"{name}_length_mismatch"] = {"expected": int(length), "got": int(vec.shape[0])}
        if vec.shape[0] > length:
            return vec[:length]
        pad_width = int(length - vec.shape[0])
        return np.pad(vec, (0, pad_width), constant_values=fill_value)

    @staticmethod
    def build_edit_events_table(resolved: common.ResolvedSample) -> pa.Table:
        """Build edit-event table from trace."""
        trace_data = common.read_trace_npz(resolved.trace_path)
        if trace_data is None:
            return pa.table({"sample_id": pa.array([], pa.string())})
        positions = np.asarray(trace_data.get("positions", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        step_index = np.asarray(trace_data.get("step_index", np.zeros((0,), dtype=np.int16)), dtype=np.int16)
        token_ids = np.asarray(trace_data.get("token_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32)
        logp_cond = np.asarray(trace_data.get("logp_cond", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        logp_uncond = np.asarray(trace_data.get("logp_uncond", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        entropy_cond = np.asarray(trace_data.get("entropy_cond", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        entropy_uncond = np.asarray(trace_data.get("entropy_uncond", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        jsd = np.asarray(trace_data.get("jsd_cond_uncond", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        topk_ids = np.asarray(trace_data.get("topk_ids", np.zeros((0, 0), dtype=np.int16)), dtype=np.int16)
        topk_logp_cond = np.asarray(trace_data.get("topk_logp_cond", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        topk_logp_uncond = np.asarray(trace_data.get("topk_logp_uncond", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        n = int(positions.size)
        if n == 0:
            return pa.table({"sample_id": pa.array([], pa.string())})
        topk_k = int(topk_ids.shape[1]) if topk_ids.ndim == 2 else 0
        return pa.table(
            {
                "sample_id": pa.array([resolved.sample_id] * n, pa.string()),
                "protein_name": pa.array([resolved.protein_name] * n, pa.string()),
                "family": pa.array([resolved.family] * n, pa.string()),
                "position": pa.array(positions.astype(np.int32), pa.int32()),
                "step_index": pa.array(step_index.astype(np.int16), pa.int16()),
                "token_id": pa.array(token_ids.astype(np.int16), pa.int16()),
                "logp_cond": pa.array(logp_cond.astype(np.float32), pa.float32()),
                "logp_uncond": pa.array(logp_uncond.astype(np.float32), pa.float32()),
                "entropy_cond_nats": pa.array(entropy_cond.astype(np.float32), pa.float32()),
                "entropy_uncond_nats": pa.array(entropy_uncond.astype(np.float32), pa.float32()),
                "jsd_cond_uncond_nats": pa.array(jsd.astype(np.float32), pa.float32()),
                "topk_k": pa.array([topk_k] * n, pa.int16()),
                "topk_ids": pa.array(topk_ids.tolist(), pa.list_(pa.int16())),
                "topk_logp_cond": pa.array(topk_logp_cond.tolist(), pa.list_(pa.float32())),
                "topk_logp_uncond": pa.array(topk_logp_uncond.tolist(), pa.list_(pa.float32())),
            }
        )

    @staticmethod
    def build_per_step_table(resolved: common.ResolvedSample, commit_step: np.ndarray) -> pa.Table:
        """Build per-step commit counts from commit_step_index."""
        steps = int(np.max(commit_step)) if commit_step.size else -1
        if steps <= 0:
            return pa.table({"sample_id": pa.array([], pa.string())})
        sample_ids: List[str] = []
        proteins: List[str] = []
        families: List[str] = []
        step_indices: List[int] = []
        n_committed: List[int] = []
        for step in range(1, steps + 1):
            sample_ids.append(resolved.sample_id)
            proteins.append(resolved.protein_name)
            families.append(resolved.family)
            step_indices.append(int(step))
            n_committed.append(int(np.sum(commit_step == step)))
        return pa.table(
            {
                "sample_id": pa.array(sample_ids, pa.string()),
                "protein_name": pa.array(proteins, pa.string()),
                "family": pa.array(families, pa.string()),
                "step_index": pa.array(step_indices, pa.int32()),
                "n_committed": pa.array(n_committed, pa.int32()),
            }
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="CPU-stage cohort builder")
    parser.add_argument("--repo_root", default=".", help="Repository root containing cohort/ and runs/")
    parser.add_argument("--cohort_xlsx", default="cohort/cohort.xlsx", help="Path to cohort.xlsx")
    parser.add_argument("--workers", type=int, default=32, help="CPU worker processes")
    parser.add_argument("--max_in_flight", type=int, default=256, help="Maximum in-flight tasks")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of samples")
    return parser.parse_args()


def main() -> None:
    """Run the CPU builder."""
    args = parse_args()
    builder = CpuBuilder(Path(args.repo_root).resolve(), Path(args.cohort_xlsx).resolve(), int(args.workers), int(args.max_in_flight))
    builder.run(int(args.limit))


if __name__ == "__main__":
    main()
