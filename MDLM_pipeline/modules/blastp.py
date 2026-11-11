from __future__ import annotations
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List
from Bio import SeqIO
from .io_utils import atomic_write_text
from .locations import locations
from .logger import logger


def run_command(cmd: List[str], capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command and raise an error on non-zero exit."""
    kwargs: dict = {}
    if capture:
        kwargs.update(dict(capture_output=True, text=True))
    proc = subprocess.run(cmd, **kwargs)
    if proc.returncode != 0:
        command_str = " ".join(cmd)
        stderr_text = getattr(proc, "stderr", "") or ""
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {command_str}\n{stderr_text}")
    return proc


def blast_sequence(fasta: str, num_sequences: int, output_path: str, database: str) -> None:
    """Run BLASTP and write query plus hits to a FASTA file."""
    fasta_path = Path(str(fasta))
    if not fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {fasta}")
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError("Input FASTA contained no sequences")
    query_record = records[0]
    if database not in {"uniprot", "nr", "uniparc"}:
        raise ValueError("Invalid database selection: choose from 'uniprot', 'nr', 'uniparc'.")
    if database == "uniprot":
        db_location = locations["uniprot_swissprot_db"]
        evalue = "1E-05"
    elif database == "nr":
        db_location = locations["non_redundant_db"]
        evalue = "1E-02"
    else:
        db_location = locations["uniparc"]
        evalue = "5E-06"
    threads = int(os.environ.get("BLAST_NUM_THREADS", "16"))
    num_sequences = int(num_sequences)
    Path(str(output_path)).parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting BLASTP (db={database}) max_target_seqs={num_sequences}")
    cmd = [
        locations["blastp"],
        "-db",
        db_location,
        "-query",
        str(fasta_path),
        "-outfmt",
        "6 sseqid",
        "-evalue",
        evalue,
        "-max_target_seqs",
        str(num_sequences),
        "-max_hsps",
        "1",
        "-num_threads",
        str(threads),
    ]
    proc = run_command(cmd, capture=True)
    ids = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if not ids:
        logger.warning("BLASTP returned zero hits; output FASTA will contain only query sequence.")
    hit_records = []
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        hits_fasta = temp_dir / "hits.fasta"
        if ids:
            id_file = temp_dir / "ids.txt"
            id_file.write_text("\n".join(ids) + "\n")
            cmd2 = [
                locations["blastdbcommand"],
                "-db",
                db_location,
                "-entry_batch",
                str(id_file),
                "-outfmt",
                "%f",
                "-out",
                str(hits_fasta),
            ]
            run_command(cmd2, capture=False)
        if hits_fasta.exists():
            hit_records = list(SeqIO.parse(str(hits_fasta), "fasta"))
    lines: List[str] = []
    lines.append(f">{query_record.id}\n{str(query_record.seq)}\n")
    for hit in hit_records:
        lines.append(f">{hit.id}\n{str(hit.seq)}\n")
    atomic_write_text(output_path, "".join(lines))
    logger.info(f"BLAST FASTA written: {output_path}")
