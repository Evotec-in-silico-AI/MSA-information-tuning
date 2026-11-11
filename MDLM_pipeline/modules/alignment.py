from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from Bio import SeqIO

from .io_utils import atomic_write_text
from .locations import locations
from .logger import logger


def validate_canonical_sequence(seq: str, *, record_id: str) -> None:
    """Raise if seq contains non-canonical residues."""
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    seq_u = str(seq).upper()
    invalid = sorted({ch for ch in seq_u if ch not in canonical})
    if invalid:
        raise ValueError(f"Non-canonical residues in record '{record_id}': {invalid}")


def muscle_alignment(output_paths: dict) -> None:
    """Run MUSCLE on filtered BLAST results to create an MSA.

    - The query/anchor (first sequence) must be strictly canonical (fail fast; no sanitization).
    - Additional sequences are filtered to strictly canonical and deduplicated.
    - Writes a filtered FASTA with numeric IDs (0...N-1) for MUSCLE, ensuring the anchor is ID 0.
    """
    blast_results = output_paths["blast_results"]
    alignment_output = output_paths["alignment"]

    blast_path = Path(str(blast_results))
    aln_path = Path(str(alignment_output))
    aln_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_path = blast_path.with_name(blast_path.stem + "_filtered.fasta")

    parsed = list(SeqIO.parse(str(blast_path), "fasta"))
    if not parsed:
        raise ValueError(f"BLAST results FASTA contained no sequences: {blast_path}")

    # Anchor/query must be canonical and becomes ID 0 for downstream pipeline invariants.
    query_rec = parsed[0]
    query_seq = str(query_rec.seq).upper()
    validate_canonical_sequence(query_seq, record_id=str(query_rec.id))

    seen = set()
    sequences: List[str] = []

    seen.add(query_seq)
    sequences.append(query_seq)

    # Filter and dedupe remaining sequences (canonical only).
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    for rec in parsed[1:]:
        seq_u = str(rec.seq).upper()
        if seq_u in seen:
            continue
        if not set(seq_u).issubset(canonical):
            continue
        seen.add(seq_u)
        sequences.append(seq_u)

    if not sequences:
        raise ValueError(f"No valid sequences remained after filtering: {blast_path}")

    # Write filtered FASTA with numeric IDs and query first (ID=0).
    lines: List[str] = []
    for counter, seq in enumerate(sequences):
        lines.append(f">{counter}\n{seq}\n")
    atomic_write_text(filtered_path, "".join(lines))
    output_paths["blast_results_filtered"] = str(filtered_path)

    logger.info("Aligning sequences with MUSCLE.")
    alignment_cmd = [
        locations["muscle"],
        "-super5",
        str(filtered_path),
        "-output",
        str(aln_path),
    ]
    proc = subprocess.run(alignment_cmd, capture_output=True, text=True, check=False)

    aln_ok = aln_path.exists() and aln_path.stat().st_size > 0
    if proc.returncode != 0:
        msg = (
            f"MUSCLE returned non-zero exit status rc={proc.returncode}. "
            f"stdout={(proc.stdout or '')[:5000]} "
            f"stderr={(proc.stderr or '')[:5000]}"
        )
        if aln_ok:
            logger.warning(msg + " (continuing because alignment output exists)")
        else:
            raise RuntimeError(msg + " (no usable alignment output produced)")

    if not aln_ok:
        raise RuntimeError(f"MUSCLE completed but alignment output missing/empty: {aln_path}")

    logger.info("Alignment completed.")
