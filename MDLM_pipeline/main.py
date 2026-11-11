from __future__ import annotations

from modules.utils import parse_arguments
from modules.wrapper import diffusion_wrapper


def main() -> None:
    """Entry point for running experiment families from the command line."""
    args = parse_arguments()

    print(f"[main] name: {args.name}")
    print(f"[main] blast_database: {args.blast_database}")
    print(f"[main] max_steps: {args.max_steps}")
    print(f"[main] early_stop_patience: {args.early_stop_patience}")
    print(f"[main] early_stop_min_delta: {args.early_stop_min_delta}")
    print(f"[main] early_stop_burn_in_steps: {args.early_stop_burn_in_steps}")
    print(f"[main] seed: {args.seed}")
    print(f"[main] conditioning_enabled: {args.conditioning_enabled}")
    print(f"[main] conditioning_types: {args.conditioning_types}")
    print(f"[main] corruption_rates: {args.corruption_rates}")
    print(f"[main] decode_steps: {args.decode_steps}")
    print(f"[main] temperature: {args.temperature}")
    print(f"[main] max_row_drops_fraction: {args.max_row_drops_fraction}")

    diffusion_wrapper(args)


if __name__ == "__main__":
    main()
