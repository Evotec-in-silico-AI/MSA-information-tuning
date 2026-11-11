import argparse
import csv
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path


def repository_root() -> Path:
    """Return and validate the repository root."""
    root = Path(__file__).resolve().parent.parent
    required = [root / "MDLM_pipeline", root / "cofolding_study", root / "figure_generation"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Repository layout is incomplete: {missing}")
    return root


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA or A3M file."""
    records: dict[str, str] = {}
    current = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].split()[0]
            if not current or current in records:
                raise RuntimeError(f"Invalid FASTA identifier in {path}")
            records[current] = ""
        elif current:
            records[current] += line
        else:
            raise RuntimeError(f"Sequence encountered before header in {path}")
    if not records or any(not sequence for sequence in records.values()):
        raise RuntimeError(f"Incomplete FASTA file: {path}")
    return records


def alignment_fixture_summary(reference_sequence: str, alignment: dict[str, str]) -> dict[str, int]:
    """Calculate deterministic summary values for the bundled alignment fixture."""
    sequences = list(alignment.values())
    alignment_width = len(reference_sequence)
    gap_count = sum(sequence.count("-") for sequence in sequences)
    substitutions = sum(
        residue != reference_residue and residue != "-"
        for sequence in sequences[1:]
        for reference_residue, residue in zip(reference_sequence, sequence)
    )
    non_gap_characters = sum(residue != "-" for sequence in sequences for residue in sequence)
    return {
        "depth": len(sequences),
        "width": alignment_width,
        "gaps": gap_count,
        "substitutions": substitutions,
        "non_gap_characters": non_gap_characters,
        "total_characters": len(sequences) * alignment_width,
    }


def verify_locked_code(root: Path) -> None:
    """Verify the immutable manuscript code manifest."""
    manifest = root / "LOCKED_CODE_CHECKSUMS.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Locked-code checksum mismatch: {relative}")


def test_repository(root: Path) -> None:
    """Run the lightweight repository test."""
    reference = parse_fasta(root / "example_data/reference/example.fasta")
    alignment = parse_fasta(root / "example_data/parental_msa/example.a3m")
    reference_id, reference_sequence = next(iter(reference.items()))
    if reference_id not in alignment or alignment[reference_id].replace("-", "") != reference_sequence:
        raise RuntimeError("Example reference and alignment disagree")
    if len({len(sequence) for sequence in alignment.values()}) != 1:
        raise RuntimeError("Example alignment lengths are inconsistent")
    observed_summary = alignment_fixture_summary(reference_sequence, alignment)
    expected_summary = {
        "depth": 6,
        "width": 33,
        "gaps": 1,
        "substitutions": 3,
        "non_gap_characters": 197,
        "total_characters": 198,
    }
    if observed_summary != expected_summary:
        raise RuntimeError(f"Alignment fixture summary mismatch: {observed_summary}")
    missing_packages = [name for name in ["numpy", "pandas", "matplotlib", "scipy", "sklearn", "openpyxl"] if importlib.util.find_spec(name) is None]
    if missing_packages:
        raise RuntimeError(f"Missing analysis packages: {missing_packages}")
    verify_locked_code(root)
    print("Repository test passed")


def parse_figure_numbers(value: str) -> list[int]:
    """Parse and validate a comma-separated figure list."""
    numbers = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = sorted(set(numbers).difference(range(1, 8)))
    if not numbers or invalid:
        raise ValueError(f"Invalid figure numbers: {invalid}")
    return list(dict.fromkeys(numbers))


def figure_scripts(numbers: list[int], supplementary: bool, sensitivity: bool) -> list[str]:
    """Return selected figure scripts in manuscript order."""
    mapping = {
        1: "figure1_plot.py",
        2: "figure2_plot.py",
        3: "figure3_plot.py",
        4: "figure4_plot.py",
        5: "figure5_plot.py",
        6: "figure6_plot.py",
        7: "figure7_plot.py",
    }
    scripts = [mapping[number] for number in numbers]
    if supplementary:
        if 6 in numbers:
            scripts.append("figure6_mediation_plot.py")
        if 7 in numbers:
            scripts.extend(["figure7g_plot.py", "supplementary_figure5cd_plot.py"])
    if sensitivity:
        scripts.append("sensitivity_analysis.py")
    return scripts

def run_figures(root: Path, numbers: list[int], check_only: bool, supplementary: bool, sensitivity: bool) -> None:
    """Run locked figure scripts from their expected working directory."""
    figure_root = root / "figure_generation"
    for required in [figure_root / "cohort.xlsx", figure_root / "data_summary.xlsx"]:
        if not required.is_file():
            raise FileNotFoundError(required)
    if any(number in {5, 6} for number in numbers):
        per_site = figure_root / "compiled" / "final" / "per_site"
        if not per_site.is_dir():
            raise FileNotFoundError(f"Figures 5 and 6 require companion residue-level data at {per_site}")
    commands = [[sys.executable, script] for script in figure_scripts(numbers, supplementary, sensitivity)]
    for command in commands:
        if not (figure_root / command[1]).is_file():
            raise FileNotFoundError(figure_root / command[1])
        print("COMMAND", " ".join(command))
        if not check_only:
            subprocess.run(command, cwd=figure_root, check=True)


def example_commands(root: Path) -> list[list[str]]:
    """Build the two manuscript example commands."""
    common = [sys.executable, "main.py", "--blast_database", "nr", "--max_steps", "4000", "--seed", "42"]
    return [
        common + ["--name", "P29972", "--p10_targets", "2.05,2.10,2.15,2.20,2.25,2.30,2.35,2.40,2.45,2.50,2.55,2.60,2.65,2.70,2.75"],
        common + ["--name", "P07740", "--p10_targets", "2.35,2.40,2.45,2.50,2.55,2.60,2.65,2.70,2.75,2.80,2.85,2.90,2.95,3.00,3.05,3.10"],
    ]


def run_examples(root: Path, check_only: bool) -> None:
    """Run or inspect the selected manuscript examples."""
    required = [
        "MDLM_pipeline/input/P29972.fasta",
        "MDLM_pipeline/analysis/blast/P29972_nr.csv",
        "MDLM_pipeline/data/blast_results/P29972_nr.fasta",
        "MDLM_pipeline/data/alignments/P29972_nr.fasta",
        "MDLM_pipeline/input/P07740.fasta",
        "MDLM_pipeline/analysis/blast/P07740_nr.csv",
        "MDLM_pipeline/data/blast_results/P07740_nr.fasta",
        "MDLM_pipeline/data/alignments/P07740_nr.fasta",
    ]
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise RuntimeError("Missing example inputs:\n" + "\n".join(missing))
    for command in example_commands(root):
        print("COMMAND", " ".join(command))
        if not check_only:
            subprocess.run(command, cwd=root / "MDLM_pipeline", check=True)


def parse_arguments() -> argparse.Namespace:
    """Parse workflow arguments."""
    parser = argparse.ArgumentParser(description="Run paper repository workflows.")
    parser.add_argument("workflow", choices=["test", "figures", "examples"])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--figures", default="1,2,3,4,5,6,7", help="Comma-separated main figure numbers")
    parser.add_argument("--include-supplementary", action="store_true")
    parser.add_argument("--include-sensitivity", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the selected workflow."""
    args = parse_arguments()
    root = repository_root()
    if args.workflow == "test":
        test_repository(root)
    elif args.workflow == "figures":
        run_figures(root, parse_figure_numbers(args.figures), args.check_only, args.include_supplementary, args.include_sensitivity)
    else:
        run_examples(root, args.check_only)


if __name__ == "__main__":
    main()
