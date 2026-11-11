import argparse
import hashlib
import os
from pathlib import Path
from typing import Iterable


def parse_arguments() -> argparse.Namespace:
    """Parse checksum-generation arguments."""
    parser = argparse.ArgumentParser(description="Regenerate the release-wide SHA-256 manifest.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("RELEASE_CHECKSUMS.sha256"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files(root: Path, output: Path) -> Iterable[Path]:
    """Yield release files in deterministic order."""
    excluded_suffixes = {".pyc", ".pyo", ".out", ".err", ".log"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix.lower() in excluded_suffixes:
            continue
        if relative == output:
            continue
        yield path


def main() -> None:
    """Write and verify the deterministic release checksum manifest."""
    args = parse_arguments()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = args.output
    if output.is_absolute():
        raise ValueError("Output must be repository-relative")
    rows = []
    for path in release_files(root, output):
        relative = path.relative_to(root).as_posix()
        rows.append(f"{sha256_file(path)}  ./{relative}")
    output_path = root / output
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} checksums to {output_path}")


if __name__ == "__main__":
    main()
