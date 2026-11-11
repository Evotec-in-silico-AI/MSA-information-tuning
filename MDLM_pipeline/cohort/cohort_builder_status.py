from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import cohort_builder_common as common


class StatusReporter:
    """Report progress for CPU/GPU/final stages."""
    def __init__(self, repo_root: Path, cohort_xlsx: Path):
        self.repo_root = Path(repo_root)
        self.cohort_xlsx = Path(cohort_xlsx)
        self.paths = common.build_paths(self.repo_root)
        self.logger = common.BuilderLogger(self.paths.logs_dir / "status.jsonl")

    def run(self, per_protein: bool, print_unresolved: bool) -> None:
        """Run status reporting."""
        cohort_df = common.read_cohort_by_sample(self.cohort_xlsx)
        resolved_by_sample_id = common.resolve_cohort(self.cohort_xlsx, self.logger)
        cohort_sample_ids = cohort_df["sample_id"].astype(str).tolist()
        cpu_done = self.load_checkpoint_set(self.paths.checkpoints_dir / "cpu")
        gpu_done = self.load_checkpoint_set(self.paths.checkpoints_dir / "gpu")
        resolved_ids = set(resolved_by_sample_id.keys())
        unresolved_ids = [sid for sid in cohort_sample_ids if sid not in resolved_ids]
        cpu_complete_count = sum(1 for sid in cohort_sample_ids if sid in cpu_done)
        gpu_complete_count = sum(1 for sid in cohort_sample_ids if sid in gpu_done)

        print("Cohort status")
        print(f"  samples_total: {len(cohort_sample_ids)}")
        print(f"  proteins_total: {cohort_df['protein_name'].nunique()}")
        print(f"  resolved_samples: {len(resolved_ids)}")
        print(f"  unresolved_samples: {len(unresolved_ids)}")
        print(f"  cpu_done: {cpu_complete_count}")
        print(f"  cpu_remaining: {len(cohort_sample_ids) - cpu_complete_count}")
        print(f"  gpu_done: {gpu_complete_count}")
        print(f"  gpu_remaining: {len(cohort_sample_ids) - gpu_complete_count}")

        if print_unresolved and unresolved_ids:
            print("\nUnresolved sample_ids (no match in merged manifest/results):")
            for sid in unresolved_ids[:200]:
                print(f"  {sid}")
            if len(unresolved_ids) > 200:
                print(f"  ... ({len(unresolved_ids) - 200} more)")

        if per_protein:
            print("\nPer-protein status (cpu_done/gpu_done):")
            for protein_name, group in cohort_df.groupby("protein_name"):
                sids = group["sample_id"].astype(str).tolist()
                cpu_ok = sum(1 for sid in sids if sid in cpu_done)
                gpu_ok = sum(1 for sid in sids if sid in gpu_done)
                print(f"  {protein_name}: samples={len(sids)} cpu_done={cpu_ok} gpu_done={gpu_ok}")

        self.logger.log(
            "status_snapshot",
            {
                "samples_total": int(len(cohort_sample_ids)),
                "proteins_total": int(cohort_df["protein_name"].nunique()),
                "resolved_samples": int(len(resolved_ids)),
                "unresolved_samples": int(len(unresolved_ids)),
                "cpu_done": int(cpu_complete_count),
                "gpu_done": int(gpu_complete_count),
            },
        )

    def load_checkpoint_set(self, checkpoints_dir: Path) -> Set[str]:
        """Load checkpoint stems for a stage."""
        d = Path(checkpoints_dir)
        if not d.exists():
            return set()
        return {p.stem for p in d.glob("*.json") if p.is_file()}

def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Cohort builder status helper (read-only)")
    parser.add_argument("--repo_root", default=".", help="Repository root containing cohort/ and runs/")
    parser.add_argument("--cohort_xlsx", default="cohort/cohort.xlsx", help="Path to cohort.xlsx")
    parser.add_argument("--print_unresolved", action="store_true", help="Print unresolved sample_ids")
    parser.add_argument("--per_protein", action="store_true", help="Print per-protein status")
    return parser.parse_args()


def main() -> None:
    """Run status report."""
    args = parse_args()
    reporter = StatusReporter(Path(args.repo_root).resolve(), Path(args.cohort_xlsx).resolve())
    reporter.run(bool(args.per_protein), bool(args.print_unresolved))


if __name__ == "__main__":
    main()
