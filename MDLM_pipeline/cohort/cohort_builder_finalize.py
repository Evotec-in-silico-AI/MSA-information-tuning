from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import cohort_builder_common as common


class FinalizeBuilder:
    """Merge CPU and GPU outputs into final datasets."""
    def __init__(self, repo_root: Path, cohort_xlsx: Path):
        self.repo_root = Path(repo_root)
        self.cohort_xlsx = Path(cohort_xlsx)
        self.paths = common.build_paths(self.repo_root)
        self.logger = common.BuilderLogger(self.paths.logs_dir / "finalize.jsonl")

    def run(self, archive_intermediates: bool) -> None:
        """Run final merge and optional archive."""
        cohort_df = common.read_cohort_by_sample(self.cohort_xlsx)
        cpu_root = self.paths.cpu_dir / "per_site"
        gpu_root = self.paths.gpu_dir / "per_site_conf"
        if not self.dataset_has_parquet(cpu_root):
            raise RuntimeError(f"CPU per_site dataset missing: {cpu_root}")
        if not self.dataset_has_parquet(gpu_root):
            raise RuntimeError(f"GPU per_site_conf dataset missing: {gpu_root}")
        cpu_ds = ds.dataset(str(cpu_root), format="parquet", partitioning="hive")
        gpu_ds = ds.dataset(str(gpu_root), format="parquet", partitioning="hive")
        final_root = self.paths.final_dir / "per_site"
        final_root.mkdir(parents=True, exist_ok=True)
        self.logger.log("finalize_start", {"cpu_root": str(cpu_root), "gpu_root": str(gpu_root)})

        for protein_name, group in cohort_df.groupby("protein_name"):
            protein = str(protein_name)
            sample_ids = group["sample_id"].astype(str).tolist()
            out_dir = final_root / f"protein_name={protein}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "per_site.parquet"
            merged_table = self.merge_per_site(cpu_ds, gpu_ds, sample_ids)
            pq.write_table(merged_table, out_file)
            self.logger.log("finalize_protein_done", {"protein_name": protein, "samples": int(len(sample_ids)), "rows": int(merged_table.num_rows)})

        common.atomic_write_json(
            self.paths.final_dir / "final_manifest.json",
            {
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "cohort_xlsx": str(self.cohort_xlsx),
                "proteins": int(cohort_df["protein_name"].nunique()),
                "samples": int(len(cohort_df)),
            },
        )

        if bool(archive_intermediates):
            self.archive_intermediate_dirs()

        self.logger.log("finalize_done", {"final_dir": str(self.paths.final_dir)})

    def dataset_has_parquet(self, root: Path) -> bool:
        """Return True if any parquet exists under root."""
        root = Path(root)
        return root.exists() and any(root.rglob("*.parquet"))

    def merge_per_site(self, cpu_ds: ds.Dataset, gpu_ds: ds.Dataset, sample_ids: List[str]) -> pa.Table:
        """Join CPU and GPU per-site tables."""
        cpu_table = cpu_ds.to_table(filter=ds.field("sample_id").isin(sample_ids))
        gpu_table = gpu_ds.to_table(filter=ds.field("sample_id").isin(sample_ids))
        cpu_df = cpu_table.to_pandas()
        gpu_df = gpu_table.to_pandas()
        merged_df = cpu_df.merge(gpu_df, on=["sample_id", "site_index", "protein_name"], how="left", validate="one_to_one")
        return pa.Table.from_pandas(merged_df, preserve_index=False)

    def archive_intermediate_dirs(self) -> None:
        """Archive cpu/ and gpu/ to archive/timestamp."""
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        archive_dir = self.paths.compiled_dir / "archive" / ts
        archive_dir.mkdir(parents=True, exist_ok=True)
        for name in ["cpu", "gpu"]:
            src = self.paths.compiled_dir / name
            if src.exists():
                shutil.move(str(src), str(archive_dir / name))
        self.logger.log("finalize_archived", {"archive_dir": str(archive_dir)})


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Finalize cohort builder outputs into final/")
    parser.add_argument("--repo_root", default=".", help="Repository root")
    parser.add_argument("--cohort_xlsx", default="cohort/cohort.xlsx", help="Path to cohort.xlsx")
    parser.add_argument("--archive_intermediates", action="store_true", help="Archive cpu/ and gpu/ into archive/<timestamp>/")
    return parser.parse_args()


def main() -> None:
    """Run finalize merge."""
    args = parse_args()
    builder = FinalizeBuilder(Path(args.repo_root).resolve(), Path(args.cohort_xlsx).resolve())
    builder.run(bool(args.archive_intermediates))


if __name__ == "__main__":
    main()
