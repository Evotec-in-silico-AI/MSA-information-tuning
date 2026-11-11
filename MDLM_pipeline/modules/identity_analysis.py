from __future__ import annotations
import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices
from .io_utils import atomic_write_csv_df

logger = logging.getLogger("identity_analysis")

ALIGNER = PairwiseAligner()
ALIGNER.mode = "global"
ALIGNER.open_gap_score = -10
ALIGNER.extend_gap_score = -1
ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")

PERMITTED_RESIDUES = set("AELMCDGHKNPQRSWYTVIF")
EXTRA_ALLOWED_RESIDUES = set("XBZUOJ")


def aligner_allowed_alphabet() -> set[str]:
    """Return the set of characters accepted by the configured pairwise aligner."""
    alphabet: Optional[str] = None
    try:
        matrix = getattr(ALIGNER, "substitution_matrix", None)
        alphabet = getattr(matrix, "alphabet", None) if matrix is not None else None
    except Exception:
        alphabet = None
    if isinstance(alphabet, str) and alphabet:
        logger.info("Using substitution matrix alphabet for aligner.")
        return set(alphabet)
    logger.info("Falling back to default aligner alphabet.")
    return set("ACDEFGHIKLMNPQRSTVWYBXZ")


ALIGNER_ALLOWED = aligner_allowed_alphabet()


def sanitize_sequence(seq: str) -> str:
    """Uppercase and replace unsupported residues with 'X' for pipeline handling."""
    s = str(seq).upper()
    out: List[str] = []
    allowed = PERMITTED_RESIDUES.union(EXTRA_ALLOWED_RESIDUES).union(set("-"))
    for ch in s:
        if ch in allowed:
            out.append(ch)
        else:
            out.append("X")
    return "".join(out)


def sanitize_for_aligner(seq: str) -> str:
    """Sanitize a sequence for PairwiseAligner by removing gaps and mapping unknowns."""
    s = sanitize_sequence(seq).replace("-", "")
    out: List[str] = []
    for ch in s:
        out.append(ch if ch in ALIGNER_ALLOWED else "X")
    return "".join(out)


def is_valid_protein(seq_str: str) -> bool:
    """Return True if a sequence uses only allowed residues (canonical, ambiguous, gap)."""
    allowed = PERMITTED_RESIDUES.union(EXTRA_ALLOWED_RESIDUES).union(set("-"))
    return set(seq_str).issubset(allowed)


def percent_identity_from_aligned(aln_ref: str, aln_query: str) -> float:
    """Compute percent identity over aligned columns excluding gaps in either sequence."""
    matches = 0
    denom = 0
    for r, q in zip(aln_ref, aln_query):
        if r == "-" or q == "-":
            continue
        denom += 1
        if r == q:
            matches += 1
    if denom == 0:
        return 0.0
    return 100.0 * (matches / denom)


def reconstruct_aligned_strings(ref_seq: str, query_seq: str) -> Tuple[str, str]:
    """Perform alignment and reconstruct aligned strings using aligner block coordinates."""
    alns = ALIGNER.align(ref_seq, query_seq)
    best = alns[0]
    ref_blocks, qry_blocks = best.aligned
    ref_out: List[str] = []
    qry_out: List[str] = []
    ref_pos = 0
    qry_pos = 0
    for (r_start, r_end), (q_start, q_end) in zip(ref_blocks, qry_blocks):
        if r_start > ref_pos:
            gap_len = r_start - ref_pos
            ref_out.extend(list(ref_seq[ref_pos:r_start]))
            qry_out.extend(["-"] * gap_len)
            ref_pos = r_start
        if q_start > qry_pos:
            gap_len = q_start - qry_pos
            ref_out.extend(["-"] * gap_len)
            qry_out.extend(list(query_seq[qry_pos:q_start]))
            qry_pos = q_start
        ref_out.extend(list(ref_seq[r_start:r_end]))
        qry_out.extend(list(query_seq[q_start:q_end]))
        ref_pos = r_end
        qry_pos = q_end
    if ref_pos < len(ref_seq):
        ref_out.extend(list(ref_seq[ref_pos:]))
        qry_out.extend(["-"] * (len(ref_seq) - ref_pos))
    if qry_pos < len(query_seq):
        ref_out.extend(["-"] * (len(query_seq) - qry_pos))
        qry_out.extend(list(query_seq[qry_pos:]))
    return "".join(ref_out), "".join(qry_out)


def get_percent_identity(ref_seq: str, query_seq: str) -> float:
    """Compute global alignment identity between reference and query; round to two decimals."""
    ref_seq_s = sanitize_for_aligner(ref_seq)
    query_seq_s = sanitize_for_aligner(query_seq)
    aln_ref, aln_query = reconstruct_aligned_strings(ref_seq_s, query_seq_s)
    pid = percent_identity_from_aligned(aln_ref, aln_query)
    return round(float(pid), 2)


def get_identities(ref_seq: str, query_sequences: List[str]) -> List[float]:
    """Compute percent identities for each query sequence vs the reference sequence."""
    return [get_percent_identity(ref_seq, q) for q in query_sequences]


def get_target_pid(ref_fasta_path: str, query_sequences: List[str]) -> List[float]:
    """Load the reference sequence from FASTA and compute identities against queries."""
    ref_records = [str(r.seq) for r in SeqIO.parse(ref_fasta_path, "fasta")]
    if not ref_records:
        raise ValueError("Reference FASTA contained no sequences.")
    ref = sanitize_sequence(ref_records[0])
    return get_identities(ref, query_sequences)


def process_blast(output_paths: Dict[str, str]) -> None:
    """Compute percent identity of WT pool plus input sequence vs the reference; enforce minimal pool size."""
    fasta = output_paths["input"]
    blast_results = output_paths["blast_results"]
    blast_analysis = output_paths["blast_analysis"]
    min_required_env = os.environ.get("IDENTITY_MIN_WT", "").strip()
    try:
        min_required = int(min_required_env) if min_required_env else 100
    except Exception:
        min_required = 100
    seen = set()
    query_sequences: List[str] = []
    query_ids: List[str] = []
    input_records = list(SeqIO.parse(fasta, "fasta"))
    if not input_records:
        raise ValueError("Reference sequence could not be determined from input FASTA.")
    ref_rec = input_records[0]
    ref_seq = sanitize_sequence(str(ref_rec.seq))
    ref_id = str(ref_rec.id)
    seen.add(ref_seq)
    query_sequences.append(ref_seq)
    query_ids.append(ref_id)
    for rec in input_records[1:]:
        s = sanitize_sequence(str(rec.seq))
        if s in seen:
            continue
        if not is_valid_protein(s):
            continue
        seen.add(s)
        query_sequences.append(s)
        query_ids.append(str(rec.id))
    for rec in SeqIO.parse(blast_results, "fasta"):
        s = sanitize_sequence(str(rec.seq))
        if s in seen:
            continue
        if not is_valid_protein(s):
            continue
        seen.add(s)
        query_sequences.append(s)
        query_ids.append(str(rec.id))
    if len(query_sequences) < min_required:
        raise ValueError(
            f"WT set size {len(query_sequences)} is below the required minimum {min_required}. "
            "Adjust IDENTITY_MIN_WT if this threshold must be overridden."
        )
    identities = get_identities(ref_seq, query_sequences)
    out_df = pd.DataFrame(
        {
            "seq_id": query_ids,
            "sequence": query_sequences,
            "percent_id_to_input": identities,
        }
    )
    atomic_write_csv_df(blast_analysis, out_df)


def stable_hash32(text: str) -> int:
    """Return a stable 32-bit hash for a string."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def stable_hash64_bytes(data: bytes) -> int:
    """Return a stable 64-bit hash for bytes."""
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def kmer_hashes(sequence: str, k: int) -> np.ndarray:
    """Return unique 32-bit hashes of k-mers from a sequence as uint64."""
    if k <= 0 or len(sequence) < k:
        return np.empty((0,), dtype=np.uint64)
    hashes = [stable_hash32(sequence[i: i + k]) for i in range(len(sequence) - k + 1)]
    if not hashes:
        return np.empty((0,), dtype=np.uint64)
    unique_hashes = np.fromiter(set(hashes), dtype=np.uint64)
    return unique_hashes


@dataclass(frozen=True)
class MinHashParams:
    """Parameters controlling MinHash signature size and LSH retrieval behavior."""
    k: int = 5
    num_perm: int = 128
    bands: int = 32
    rows: int = 4
    top_k_candidates: int = 200
    min_candidates: int = 50
    seed: int = 42


class MinHashLSHIndex:
    """MinHash signatures with LSH buckets for fast candidate retrieval."""

    def __init__(self, sequences: List[str], params: MinHashParams):
        """Build an index over sequences using MinHash signatures and LSH banding."""
        if params.bands * params.rows != params.num_perm:
            raise ValueError("bands * rows must equal num_perm")
        self.params = params
        self.sequences = sequences
        self.prime_modulus = np.uint64(4294967311)
        rng = np.random.default_rng(params.seed)
        self.coeff_a = rng.integers(1, int(self.prime_modulus - 1), size=params.num_perm, dtype=np.uint64)
        self.coeff_b = rng.integers(0, int(self.prime_modulus - 1), size=params.num_perm, dtype=np.uint64)
        self.signatures = self.build_signatures(sequences)
        self.buckets = self.build_buckets(self.signatures)

    def build_signatures(self, sequences: List[str]) -> np.ndarray:
        """Return MinHash signature matrix of shape (N, num_perm) as uint64."""
        n = len(sequences)
        sig = np.full((n, self.params.num_perm), self.prime_modulus - 1, dtype=np.uint64)
        for i, seq in enumerate(sequences):
            kmers = kmer_hashes(seq, self.params.k)
            if kmers.size == 0:
                continue
            hv = (self.coeff_a[:, None] * kmers[None, :] + self.coeff_b[:, None]) % self.prime_modulus
            sig[i, :] = hv.min(axis=1)
        return sig

    def band_hash(self, band_slice: np.ndarray) -> int:
        """Compute a stable band hash from a (rows,) signature slice."""
        return stable_hash64_bytes(band_slice.tobytes())

    def build_buckets(self, signatures: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
        """Build LSH buckets keyed by (band_index, band_hash) to sequence indices."""
        buckets: Dict[Tuple[int, int], List[int]] = {}
        b = self.params.bands
        r = self.params.rows
        for idx in range(signatures.shape[0]):
            for band_idx in range(b):
                start = band_idx * r
                end = start + r
                band_slice = signatures[idx, start:end]
                band_key = (band_idx, self.band_hash(band_slice))
                buckets.setdefault(band_key, []).append(idx)
        return buckets

    def signature_for(self, sequence: str) -> np.ndarray:
        """Compute a MinHash signature for a sequence."""
        sig = np.full((self.params.num_perm,), self.prime_modulus - 1, dtype=np.uint64)
        kmers = kmer_hashes(sequence, self.params.k)
        if kmers.size == 0:
            return sig
        hv = (self.coeff_a[:, None] * kmers[None, :] + self.coeff_b[:, None]) % self.prime_modulus
        sig[:] = hv.min(axis=1)
        return sig

    def query_candidates(self, query_sequence: str) -> List[int]:
        """Return candidate indices for a query sequence using LSH buckets."""
        query_sig = self.signature_for(query_sequence)
        b = self.params.bands
        r = self.params.rows
        candidate_set = set()
        for band_idx in range(b):
            start = band_idx * r
            end = start + r
            band_slice = query_sig[start:end]
            band_key = (band_idx, self.band_hash(band_slice))
            for idx in self.buckets.get(band_key, []):
                candidate_set.add(idx)
        return list(candidate_set)

    def rank_by_minhash(self, query_sequence: str, candidate_indices: Optional[List[int]] = None) -> List[int]:
        """Rank candidates by MinHash similarity in descending order."""
        query_sig = self.signature_for(query_sequence)
        if candidate_indices is None:
            candidate_indices = list(range(self.signatures.shape[0]))
        cand_idx = np.asarray(candidate_indices, dtype=np.int64)
        cand_sigs = self.signatures[cand_idx]
        eq = (cand_sigs == query_sig[None, :]).sum(axis=1)
        sim = eq.astype(np.float32) / float(self.params.num_perm)
        order = np.argsort(-sim)
        return [candidate_indices[int(i)] for i in order]


MINHASH_CACHE: Dict[Tuple[str, str, int, int, int, int, int], MinHashLSHIndex] = {}


def load_wt_sequences(input_fasta_path: str, blast_results_path: str) -> List[str]:
    """Load and validate the WT sequence set from input FASTA and BLAST results."""
    wt_sequences: List[str] = []
    seen = set()
    input_records = list(SeqIO.parse(input_fasta_path, "fasta"))
    if not input_records:
        return wt_sequences
    ref = sanitize_sequence(str(input_records[0].seq))
    seen.add(ref)
    wt_sequences.append(ref)
    for rec in input_records[1:]:
        s = sanitize_sequence(str(rec.seq))
        if s in seen:
            continue
        if not is_valid_protein(s):
            continue
        seen.add(s)
        wt_sequences.append(s)
    for rec in SeqIO.parse(blast_results_path, "fasta"):
        s = sanitize_sequence(str(rec.seq))
        if s in seen:
            continue
        if not is_valid_protein(s):
            continue
        seen.add(s)
        wt_sequences.append(s)
    return wt_sequences


def get_or_build_index(input_fasta_path: str, blast_results_path: str, params: MinHashParams) -> MinHashLSHIndex:
    """Return a cached MinHashLSHIndex for a WT set, building it if necessary."""
    key = (
        str(input_fasta_path),
        str(blast_results_path),
        params.k,
        params.num_perm,
        params.bands,
        params.rows,
        params.seed,
    )
    cached = MINHASH_CACHE.get(key)
    if cached is not None:
        return cached
    wt_sequences = load_wt_sequences(input_fasta_path, blast_results_path)
    logger.info(
        "Building MinHash LSH index for WT set (n=%d) with params: k=%d, num_perm=%d, bands=%d, rows=%d, seed=%d.",
        len(wt_sequences),
        params.k,
        params.num_perm,
        params.bands,
        params.rows,
        params.seed,
    )
    index = MinHashLSHIndex(wt_sequences, params)
    MINHASH_CACHE[key] = index
    return index


def process_de_novo(fasta: str, blast_results: str, de_novo_sequences: List[str]) -> List[float]:
    """Compute max identity for each de novo sequence vs WT set using exact refinement or full scan."""
    params = MinHashParams(
        k=5,
        num_perm=128,
        bands=32,
        rows=4,
        top_k_candidates=200,
        min_candidates=50,
        seed=42,
    )
    index = get_or_build_index(fasta, blast_results, params)
    wt_sequences = index.sequences
    max_percent_identities: List[float] = []
    for de_novo in de_novo_sequences:
        de_novo_s = sanitize_sequence(de_novo)
        if len(wt_sequences) <= 6000:
            ids_all = get_identities(de_novo_s, wt_sequences)
            max_pid = round(float(np.max(ids_all)) if ids_all else 0.0, 2)
            max_percent_identities.append(max_pid)
            continue
        candidates = index.query_candidates(de_novo_s)
        if len(candidates) < params.min_candidates:
            ranked_all = index.rank_by_minhash(de_novo_s, candidate_indices=None)
        else:
            ranked_all = index.rank_by_minhash(de_novo_s, candidate_indices=candidates)
        refine_cap = max(params.top_k_candidates, min(2000, 10 * params.top_k_candidates))
        refine_indices = ranked_all[: min(len(ranked_all), refine_cap)]
        candidate_sequences = [wt_sequences[i] for i in refine_indices] if wt_sequences else []
        ids = get_identities(de_novo_s, candidate_sequences)
        max_pid = round(float(np.max(ids)) if ids else 0.0, 2)
        max_percent_identities.append(max_pid)
    return max_percent_identities
