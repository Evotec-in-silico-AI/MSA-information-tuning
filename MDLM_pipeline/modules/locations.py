from __future__ import annotations
import os
import shutil
from pathlib import Path

locations = {
    "blastp": "/data/projects/protein-db/ncbi-blast-2.13.0+/bin/blastp",
    "uniprot_swissprot_db": "/data/projects/protein-db/protein_db/uniprot_sprot",
    "non_redundant_db": "/data/projects/protein-db/nr_data/nr",
    "uniparc": "/data/projects/protein-db/uniparc/uniparc",
    "muscle": "/data/projects/protein-db/muscle/muscle5.1.linux_intel64",
    "blastdbcommand": "/data/projects/protein-db/ncbi-blast-2.13.0+/bin/blastdbcmd",
}


def resolve_binary_path(env_variable: str, default_path: str, *fallback_names: str) -> str:
    """Resolve a binary path using an environment variable, an existing default path, or PATH discovery."""
    env_value = os.environ.get(env_variable, "").strip()
    if env_value:
        path_obj = Path(env_value)
        if path_obj.exists():
            return str(path_obj)
    if default_path:
        default_obj = Path(default_path)
        if default_obj.exists():
            return str(default_obj)
    for name in fallback_names:
        discovered = shutil.which(name)
        if discovered:
            return discovered
    return default_path


locations["blastp"] = resolve_binary_path("BLASTP", locations["blastp"], "blastp")
locations["blastdbcommand"] = resolve_binary_path("BLASTDBCMD", locations["blastdbcommand"], "blastdbcmd")
locations["muscle"] = resolve_binary_path("MUSCLE", locations["muscle"], "muscle")


def repo_root() -> Path:
    """Return the repository root directory path."""
    return Path(__file__).parents[1]


def get_run_root(name: str, database: str, max_steps: int) -> Path:
    """Create and return the run root directory for a protein, database, and step budget."""
    root = repo_root() / "runs" / str(name) / str(database) / f"steps_{int(max_steps)}"
    (root / "experiments").mkdir(parents=True, exist_ok=True)
    (root / "merged").mkdir(parents=True, exist_ok=True)
    return root


def get_output_paths(name: str, database: str) -> dict:
    """Return common output paths for BLAST, alignment, and conditioning artifacts."""
    working_dir = repo_root()
    name = str(name)
    database = str(database)
    output_paths = {
        "input": str(working_dir / "input" / f"{name}.fasta"),
        "blast_results": str(working_dir / "data" / "blast_results" / f"{name}_{database}.fasta"),
        "blast_analysis": str(working_dir / "analysis" / "blast" / f"{name}_{database}.csv"),
        "alignment": str(working_dir / "data" / "alignments" / f"{name}_{database}.fasta"),
        "alignment_analysis": str(working_dir / "analysis" / "alignment" / f"{name}_{database}.csv"),
        "structure_context": str(working_dir / "data" / "conditioning" / "structure" / f"{name}.pkl"),
        "text_context": str(working_dir / "data" / "conditioning" / "text" / f"{name}.pkl"),
        "msa_context": str(working_dir / "data" / "conditioning" / "msa" / f"{name}.pkl"),
    }
    for path_text in output_paths.values():
        Path(path_text).parent.mkdir(parents=True, exist_ok=True)
    return output_paths
