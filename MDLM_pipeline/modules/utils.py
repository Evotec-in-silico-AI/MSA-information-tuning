import argparse
import json
from typing import Any, List

RATE_NDP = 6
RATE_TAG_SCALE = 10 ** RATE_NDP


def parse_conditioning_types(text_value: str) -> List[str]:
    """
    Parse comma-separated conditioning modalities into a list of normalized tokens.
    Tokens are stripped and lowercased; empty entries are removed.
    """
    tokens = [t.strip().lower() for t in str(text_value).split(",")]
    return [t for t in tokens if t]


def quantize_rate_value(value: Any, ndp: int = RATE_NDP) -> float:
    """
    Quantize a corruption rate to a fixed number of decimal places.
    """
    return round(float(value), int(ndp))


def parse_numeric_list(text_value: Any) -> List[float]:
    """
    Parse a flexible numeric list from various input styles:
    """
    if isinstance(text_value, (list, tuple)):
        return [float(x) for x in text_value]
    text = str(text_value).strip()
    if not text:
        return []
    try:
        if text.startswith("[") and text.endswith("]"):
            parsed_json = json.loads(text)
            if isinstance(parsed_json, list):
                return [float(x) for x in parsed_json]
    except Exception:
        pass
    cleaned = (
        text.replace("[", " ")
        .replace("]", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("{", " ")
        .replace("}", " ")
        .strip()
    )
    normalized = cleaned.replace(";", ",").replace(" ", ",")
    tokens = [tok for tok in normalized.split(",") if tok]
    values: List[float] = []
    for tok in tokens:
        t = tok.strip().rstrip(".,;")
        if not t:
            continue
        try:
            values.append(float(t))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric token '{tok}' in list: {text_value}") from exc
    return values


def parse_corruption_rates_text(text_value: str) -> List[float]:
    """
    Parse corruption rates into a list of floats with validation in [0, 1].
    Accepts JSON lists, CSV, and space-separated inputs.
    """
    values = parse_numeric_list(text_value)
    out_of_range = [v for v in values if not (0.0 <= float(v) <= 1.0)]
    if out_of_range:
        raise ValueError(f"corruption_rates must be within [0, 1]; invalid values: {out_of_range}")
    return [float(v) for v in values]


def parse_p10_targets_text(text_value: str) -> List[float]:
    """
    Parse bits/site p10 targets into a list of floats.
    Accepts JSON lists, CSV, and space-separated inputs.
    """
    values = parse_numeric_list(text_value)
    return [float(v) for v in values]


def rate_tag_from_value(value: Any, ndp: int = RATE_NDP) -> int:
    """
    Return an integer tag for a corruption rate using a fixed decimal precision.
    """
    quantized_rate = quantize_rate_value(value, ndp=int(ndp))
    scale = 10 ** int(ndp)
    return int(round(float(quantized_rate) * float(scale)))


def candidate_seed_from_base(base_seed: int, rate: Any, candidate_index: int, ndp: int = RATE_NDP) -> int:
    """
    Return a deterministic candidate seed from a base seed, rate, and candidate index.
    """
    rate_tag = rate_tag_from_value(rate, ndp=int(ndp))
    modulus = (2 ** 32) - 1
    return (int(base_seed) + 1000003 * int(rate_tag) + int(candidate_index)) % int(modulus)


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command-line arguments for the p10-budgeted pipeline.
    """
    parser = argparse.ArgumentParser(description="Protein diffusion model experiments (p10-budgeted)")
    parser.add_argument("--name", type=str, required=True, help="Name of the protein family")
    parser.add_argument(
        "--blast_database",
        type=str,
        default="nr",
        choices=["uniprot", "nr", "uniparc"],
        help="BLAST database to query",
    )
    parser.add_argument("--max_steps", type=int, default=4000, help="Maximum optimizer steps for denoiser training")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Number of eval points without improvement")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3, help="Minimum validation improvement")
    parser.add_argument("--early_stop_burn_in_steps", type=int, default=1000, help="Steps before early stopping is allowed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for training and sampling")
    parser.add_argument(
        "--conditioning_enabled",
        dest="conditioning_enabled",
        action="store_true",
        default=True,
        help="Enable conditioning if context features are present",
    )
    parser.add_argument(
        "--no_conditioning",
        dest="conditioning_enabled",
        action="store_false",
        help="Disable conditioning even if context features are present",
    )
    parser.add_argument(
        "--conditioning_types",
        type=str,
        default="msa",
        help="Comma-separated conditioning modalities to request (e.g. msa,structure,text)",
    )
    parser.add_argument(
        "--p10_targets",
        type=str,
        default="2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0",
        help="Comma-separated list of bits/site p10 targets to run",
    )
    parser.add_argument(
        "--p10_tolerance",
        type=float,
        default=0.01,
        help="Absolute tolerance for p10 acceptance",
    )
    parser.add_argument(
        "--corruption_rates",
        type=str,
        default="0.5,0.65,0.75,0.85,0.90,0.95",
        help="Comma-separated list of corruption rates for sampling",
    )
    parser.add_argument("--decode_steps", type=int, default=12, help="Number of decoding steps for sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for sampling")
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=15,
        help="Number of candidates generated per corruption rate before selection",
    )
    parser.add_argument(
        "--select_k_per_rate",
        type=int,
        default=5,
        help="Number of candidates selected per corruption rate for fold queue",
    )
    parser.add_argument(
        "--max_row_drops_fraction",
        type=float,
        default=0.75,
        help="Maximum fraction of alignment rows that may be dropped during p10 targeting",
    )
    parser.add_argument(
        "--msa_p10_sweep_null_reps",
        type=int,
        default=1,
        help="Number of null-control repetitions per p10 target",
    )
    parser.add_argument(
        "--msa_p10_sweep_null_start_rep",
        type=int,
        default=0,
        help="Starting rep index for null-control runs",
    )
    parser.add_argument("--max_shards", type=int, default=0, help="Maximum total shards to run (0 means no limit)")
    args = parser.parse_args()
    if int(args.max_steps) < 1:
        raise ValueError("--max_steps must be >= 1")
    if int(args.early_stop_patience) < 1:
        raise ValueError("--early_stop_patience must be >= 1")
    if float(args.early_stop_min_delta) < 0.0:
        raise ValueError("--early_stop_min_delta must be >= 0")
    if int(args.early_stop_burn_in_steps) < 0:
        raise ValueError("--early_stop_burn_in_steps must be >= 0")
    if int(args.candidate_pool_size) < 1:
        raise ValueError("--candidate_pool_size must be >= 1")
    if int(args.select_k_per_rate) < 1:
        raise ValueError("--select_k_per_rate must be >= 1")
    if int(args.select_k_per_rate) > int(args.candidate_pool_size):
        raise ValueError("--select_k_per_rate must be <= --candidate_pool_size")
    if int(args.msa_p10_sweep_null_reps) < 1:
        raise ValueError("--msa_p10_sweep_null_reps must be >= 1")
    if int(args.msa_p10_sweep_null_start_rep) < 0:
        raise ValueError("--msa_p10_sweep_null_start_rep must be >= 0")
    if float(args.p10_tolerance) <= 0.0:
        raise ValueError("--p10_tolerance must be > 0")
    args.p10_targets = parse_p10_targets_text(args.p10_targets)
    if not args.p10_targets:
        raise ValueError("--p10_targets must contain at least one value")
    args.corruption_rates = parse_corruption_rates_text(args.corruption_rates)
    if not args.corruption_rates:
        raise ValueError("--corruption_rates must contain at least one value")
    return args
