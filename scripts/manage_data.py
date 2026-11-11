import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def repository_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> list[dict[str, str]]:
    """Load the artifact manifest."""
    with (root / "data" / "artifact_manifest.tsv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="	"))


def validate_relative_path(value: str, field: str) -> Path:
    """Validate a repository-relative manifest path."""
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid {field}: {value}")
    return path


def status(root: Path, rows: list[dict[str, str]]) -> int:
    """Print artifact status."""
    failures = 0
    for row in rows:
        target = root / row["install_path"]
        state = "missing"
        if target.is_file():
            expected = row["sha256"].strip()
            state = "verified" if expected and sha256_file(target) == expected else "present"
        print(f"{row['artifact_id']}	{state}	{row['install_path']}")
        if row["status"] == "included" and state == "missing":
            failures += 1
    return 1 if failures else 0


def install(root: Path, data_root: Path, rows: list[dict[str, str]], dry_run: bool, copy_files: bool) -> None:
    """Install external artifacts into their declared paths."""
    for row in rows:
        if row["status"] != "required_external":
            continue
        source = data_root / validate_relative_path(row["source_path"], "source_path")
        target = root / validate_relative_path(row["install_path"], "install_path")
        if not source.is_file():
            raise FileNotFoundError(f"Missing companion-data file: {source}")
        expected = row["sha256"].strip()
        if expected and sha256_file(source) != expected:
            raise ValueError(f"Checksum mismatch: {source}")
        if target.exists():
            if target.is_file() and sha256_file(target) == sha256_file(source):
                print(f"KEEP {target}")
                continue
            raise FileExistsError(f"Refusing to replace conflicting path: {target}")
        action = "COPY" if copy_files else "LINK"
        print(f"{action} {target} <- {source}")
        if dry_run:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy_files:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source.resolve())


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Install and verify paper companion data.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--data-root", required=True, type=Path)
    install_parser.add_argument("--dry-run", action="store_true")
    install_parser.add_argument("--copy", action="store_true", dest="copy_files")
    return parser.parse_args()


def main() -> None:
    """Run the selected data command."""
    args = parse_arguments()
    root = repository_root()
    rows = load_manifest(root)
    if args.command == "status":
        raise SystemExit(status(root, rows))
    install(root, args.data_root.resolve(), rows, args.dry_run, args.copy_files)


if __name__ == "__main__":
    main()
