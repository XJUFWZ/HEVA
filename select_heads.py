# Command-line entry point for training-only arousal head selection.
# The selection algorithm and workflow live in heva/head_selection.py; this
# file only parses arguments, loads the configuration, runs the selection, and
# writes the JSON result. Run `python select_heads.py --help` for the options.

import argparse
import json
from pathlib import Path

from heva.config import load_config
from heva.head_selection import (
    DEFAULT_HEAD_ALPHA_GRID,
    HeadSelectionOptions,
    run_head_selection,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train-only selection of arousal attention heads."
    )
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--dataset", required=True, choices=("deam", "pmemo"))
    parser.add_argument(
        "--ranking-json",
        default="",
        help="optional single training-only per-head ranking JSON for the pool",
    )
    parser.add_argument(
        "--ranking-template",
        default="",
        help="per-seed ranking path containing a '{seed}' placeholder",
    )
    parser.add_argument(
        "--selection-protocol",
        default="",
        help="per-seed starting baseline heads and fixed alphas JSON",
    )
    parser.add_argument("--frozen", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--pool-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--stability-penalty", type=float, default=0.5)
    parser.add_argument("--rank-points", type=int, default=8)
    parser.add_argument("--rank-alpha", type=float, default=1.0)
    parser.add_argument(
        "--head-alpha-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_HEAD_ALPHA_GRID),
    )
    parser.add_argument(
        "--refit-alpha",
        action="store_true",
        help="diagnostically re-pick the head alpha on the selected set under "
        "the baseline RMSE guard; does not change the selected heads",
    )
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="")
    parser.add_argument(
        "--cache-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="device for the head Gram caches; cpu fits any machine, cuda is faster",
    )
    return parser.parse_args()


def _progress(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    frozen_arg = Path(args.frozen) if args.frozen else Path(
        config["runtime"]["frozen_config"]
    )
    frozen_path = frozen_arg if frozen_arg.is_absolute() else (
        config_path.parent.parent / frozen_arg
    )
    frozen = load_config(frozen_path)

    options = HeadSelectionOptions(
        seeds=args.seeds,
        pool_size=args.pool_size,
        top_k=args.top_k,
        outer_splits=args.outer_splits,
        stability_penalty=args.stability_penalty,
        rank_points=args.rank_points,
        rank_alpha=args.rank_alpha,
        head_alpha_grid=tuple(float(value) for value in args.head_alpha_grid),
        refit_alpha=args.refit_alpha,
        backend=args.backend,
        cache_device=args.cache_device,
        ranking_json=args.ranking_json,
        ranking_template=args.ranking_template,
        selection_protocol=args.selection_protocol,
    )

    payload = run_head_selection(
        config, frozen, args.dataset, options, progress=_progress
    )
    output = (
        Path(args.output)
        if args.output
        else Path("outputs") / "head_selection" / (args.dataset + "_head_selection.json")
    )
    write_json(output, payload)
    print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
