import argparse
import json
from pathlib import Path

from heva.config import load_config
from heva.runner import describe_experiment, run_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the HEVA main experiment.",
    )
    parser.add_argument(
        "--config",
        default="configs/main.yaml",
        help="Path to the main configuration file.",
    )
    parser.add_argument(
        "--dataset",
        choices=("deam", "pmemo"),
        required=True,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument(
        "--branch",
        choices=("both", "valence", "arousal"),
        default="both",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "cuda"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--model-output",
        help=(
            "Save a complete model package for direct inference "
            "(single seed and both branches only)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    frozen_path = Path(config["runtime"]["frozen_config"])
    if not frozen_path.is_absolute():
        frozen_path = config_path.parent.parent / frozen_path
    frozen = load_config(frozen_path)

    if args.all_seeds:
        if args.seed is not None:
            raise ValueError("--seed and --all-seeds cannot be combined")
        seeds = [int(value) for value in config["experiment"]["seeds"]]
    else:
        if args.seed is None:
            raise ValueError("--seed is required unless --all-seeds is used")
        seeds = [int(args.seed)]
    if args.model_output is not None:
        if args.all_seeds:
            raise ValueError("--model-output cannot be combined with --all-seeds")
        if args.branch != "both":
            raise ValueError("--model-output requires --branch both")
        if args.dry_run:
            raise ValueError("--model-output cannot be combined with --dry-run")

    for seed in seeds:
        if args.dry_run:
            result = describe_experiment(
                config,
                frozen,
                args.dataset,
                seed,
                args.branch,
                args.backend,
            )
        else:
            result = run_experiment(
                config,
                frozen,
                args.dataset,
                seed,
                args.branch,
                backend=args.backend,
                model_output=args.model_output,
            )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
