"""Select the arousal attention heads from training data, and optionally the
head Ridge alpha.

This is the training-only head-selection routine behind the arousal entries of
configs/frozen.yaml. Test features and test labels are never loaded.

Two input modes are supported:

* Exact reproduction. For every seed, a training-only per-head ranking JSON
  supplies the macro-CCC shortlist and a selection-protocol JSON supplies the
  starting baseline heads together with the fixed head/acoustic Ridge alphas.
  The candidate pool is the per-seed shortlist union the per-seed baseline.
  Provide these with --ranking-template and --selection-protocol.

* Self-contained probe. When no ranking or protocol is given, the shortlist is
  estimated on the fly from the head shards and the frozen configuration is
  used as the baseline. This needs no extra files but only approximates the
  original run because the on-the-fly probe uses a pooled ranking.

For every seed, the complete arousal branch is evaluated inside grouped
cross-validation: the head and handcrafted-acoustic streams each get a
first-stage Ridge, their out-of-fold predictions are fused with a fixed convex
weight, and a temporal Ridge produces the validation prediction. Heads are
grown greedily from an empty set on the mean fold CCC minus a stability
penalty on its sample standard deviation, using a fixed head Ridge alpha. The
greedy trajectories of all seeds are combined into one consensus set.

With --refit-alpha, the selected set is additionally scored on a fixed alpha
grid and an alpha is reported only when its mean RMSE does not exceed the
baseline mean RMSE; this is a diagnostic and never changes the selected heads.

The result is written to JSON. configs/frozen.yaml is not modified; the
reported heads are meant to be compared against it manually.
"""

from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from heva.config import get_seed_config, load_config
from heva.datasets import build_content_group_map, groups_for_rows
from heva.features import (
    align_acoustic_rows,
    load_acoustic_features,
    load_sampled_point_head_feature_shards,
    load_selected_point_head_features,
)
from heva.metrics import track_macro_metrics
from heva.models.arousal import _grouped_folds, _oof_prediction, rank_arousal_heads
from heva.models.ridge import fit_ridge_state, predict_ridge_state
from heva.models.temporal import build_temporal_features


METRIC_NAMES = ("pcc", "ccc", "rmse")
OUTER_SEED_BASE = 29000
OOF_SEED_BASE = {"deam": 19000, "pmemo": 31000}
HEAD_WIDTH = 128
DEFAULT_HEAD_ALPHA_GRID = (100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_ridge_cache(
    fit_features: np.ndarray,
    fit_target: np.ndarray,
    validation_features: np.ndarray,
    device: str,
) -> dict:
    # Cache the standardized Gram matrix once so each greedy trial only solves
    # a small Cholesky system on the selected head columns.
    fit = np.asarray(fit_features, dtype=np.float32)
    validation = np.asarray(validation_features, dtype=np.float32)
    target = np.asarray(fit_target, dtype=np.float32).reshape(-1)
    if fit.ndim != 3 or validation.ndim != 3:
        raise ValueError("candidate features must have shape [rows, heads, width]")
    if fit.shape[1:] != validation.shape[1:]:
        raise ValueError("fit and validation candidate grids differ")
    head_count = int(fit.shape[1])
    head_width = int(fit.shape[2])
    torch_device = torch.device(device)
    fit_tensor = torch.as_tensor(
        fit.reshape(len(fit), head_count * head_width),
        dtype=torch.float32,
        device=torch_device,
    )
    validation_tensor = torch.as_tensor(
        validation.reshape(len(validation), head_count * head_width),
        dtype=torch.float32,
        device=torch_device,
    )
    target_tensor = torch.as_tensor(target, dtype=torch.float32, device=torch_device)
    feature_mean = fit_tensor.mean(dim=0)
    centered = fit_tensor - feature_mean
    feature_scale = torch.sqrt(torch.mean(centered.square(), dim=0))
    feature_scale = torch.where(
        feature_scale == 0.0,
        torch.ones_like(feature_scale),
        feature_scale,
    )
    standardized_fit = centered / feature_scale
    standardized_validation = (validation_tensor - feature_mean) / feature_scale
    target_mean = target_tensor.mean()
    centered_target = target_tensor - target_mean
    return {
        "gram": standardized_fit.T @ standardized_fit,
        "right_hand_side": standardized_fit.T @ centered_target,
        "validation": standardized_validation,
        "target_mean": target_mean,
        "head_width": head_width,
        "device": torch_device,
    }


def _subset_prediction(cache: dict, selected_indices: list[int], alpha: float) -> np.ndarray:
    width = int(cache["head_width"])
    columns = []
    for candidate_index in selected_indices:
        begin = int(candidate_index) * width
        columns.extend(range(begin, begin + width))
    column_index = torch.as_tensor(
        columns,
        dtype=torch.long,
        device=cache["device"],
    )
    gram = cache["gram"].index_select(0, column_index)
    gram = gram.index_select(1, column_index)
    rhs = cache["right_hand_side"].index_select(0, column_index)
    system = gram + float(alpha) * torch.eye(
        len(columns),
        dtype=gram.dtype,
        device=cache["device"],
    )
    factor, info = torch.linalg.cholesky_ex(system)
    if int(info.item()) != 0:
        raise RuntimeError("head Ridge system is not positive definite")
    weights = torch.cholesky_solve(rhs[:, None], factor).squeeze(1)
    prediction = cache["validation"].index_select(1, column_index) @ weights
    prediction = prediction + cache["target_mean"]
    return prediction.detach().cpu().numpy().astype(np.float32, copy=False)


def _aggregate_fold_metrics(fold_metrics: list[dict], stability_penalty: float) -> dict:
    if not fold_metrics:
        raise ValueError("at least one fold metric is required")
    if stability_penalty < 0:
        raise ValueError("stability_penalty must be non-negative")
    values = {
        metric: [float(row[metric]) for row in fold_metrics]
        for metric in METRIC_NAMES
    }
    means = {metric: float(mean(rows)) for metric, rows in values.items()}
    sample_stds = {
        metric: float(stdev(rows)) if len(rows) > 1 else 0.0
        for metric, rows in values.items()
    }
    return {
        "folds": [dict(row) for row in fold_metrics],
        "mean": means,
        "sample_std": sample_stds,
        "objective": means["ccc"] - float(stability_penalty) * sample_stds["ccc"],
    }


class _OuterFold:
    def __init__(
        self,
        head_grid: np.ndarray,
        acoustic: np.ndarray,
        target: np.ndarray,
        tracks: np.ndarray,
        fold_groups: np.ndarray,
        times: np.ndarray,
        fit_rows: np.ndarray,
        validation_rows: np.ndarray,
        default_head_alpha: float,
        acoustic_alpha: float,
        fusion_weight: float,
        context_radius: int,
        temporal_alpha: float,
        inner_splits: int,
        inner_seed: int,
        backend: str,
        cache_device: str,
    ) -> None:
        self.fit_rows = np.asarray(fit_rows, dtype=np.int64)
        self.validation_rows = np.asarray(validation_rows, dtype=np.int64)
        self.fit_target = np.asarray(target[self.fit_rows], dtype=np.float32).reshape(-1)
        self.validation_target = np.asarray(
            target[self.validation_rows], dtype=np.float32
        ).reshape(-1)
        self.fit_tracks = tracks[self.fit_rows]
        self.validation_tracks = tracks[self.validation_rows]
        self.fit_times = times[self.fit_rows]
        self.validation_times = times[self.validation_rows]
        self.fusion_weight = float(fusion_weight)
        self.context_radius = int(context_radius)
        self.temporal_alpha = float(temporal_alpha)
        self.backend = backend
        self.default_head_alpha = float(default_head_alpha)

        fit_acoustic = np.asarray(acoustic[self.fit_rows], dtype=np.float32)
        validation_acoustic = np.asarray(
            acoustic[self.validation_rows], dtype=np.float32
        )
        fit_groups = fold_groups[self.fit_rows]

        # The acoustic stream is fixed across head trials, so build its
        # out-of-fold and outer-validation predictions once per outer fold.
        self.acoustic_oof, coverage, overlap = _oof_prediction(
            fit_acoustic,
            self.fit_target,
            fit_groups,
            float(acoustic_alpha),
            int(inner_seed),
            int(inner_splits),
            backend,
        )
        if not np.all(np.asarray(coverage) == 1) or int(overlap) != 0:
            raise RuntimeError("outer-fit acoustic OOF audit failed")
        acoustic_state = fit_ridge_state(
            fit_acoustic,
            self.fit_target,
            float(acoustic_alpha),
            "lsqr",
            backend,
        )
        self.acoustic_validation = predict_ridge_state(
            acoustic_state, validation_acoustic
        ).reshape(-1)

        fit_heads = head_grid[self.fit_rows]
        self.outer_cache = _prepare_ridge_cache(
            fit_heads,
            self.fit_target,
            head_grid[self.validation_rows],
            cache_device,
        )
        self.inner_folds = _grouped_folds(
            fit_groups, int(inner_splits), int(inner_seed)
        )
        self.inner_caches = []
        inner_coverage = np.zeros(len(self.fit_rows), dtype=np.int8)
        for inner_fit, inner_validation in self.inner_folds:
            self.inner_caches.append(
                _prepare_ridge_cache(
                    fit_heads[inner_fit],
                    self.fit_target[inner_fit],
                    fit_heads[inner_validation],
                    cache_device,
                )
            )
            inner_coverage[inner_validation] += 1
        if not np.all(inner_coverage == 1):
            raise RuntimeError("inner head folds do not cover each fit row once")

    def evaluate(self, selected_indices: list[int], head_alpha: float | None = None) -> dict:
        resolved_alpha = (
            self.default_head_alpha if head_alpha is None else float(head_alpha)
        )
        head_oof = np.empty(len(self.fit_rows), dtype=np.float32)
        coverage = np.zeros(len(self.fit_rows), dtype=np.int8)
        for cache, (_, inner_validation) in zip(self.inner_caches, self.inner_folds):
            head_oof[inner_validation] = _subset_prediction(
                cache, selected_indices, resolved_alpha
            )
            coverage[inner_validation] += 1
        if not np.all(coverage == 1):
            raise RuntimeError("inner head OOF coverage is not exactly one")
        head_validation = _subset_prediction(
            self.outer_cache, selected_indices, resolved_alpha
        )

        weight = self.fusion_weight
        fit_signal = weight * self.acoustic_oof + (1.0 - weight) * head_oof
        validation_signal = (
            weight * self.acoustic_validation + (1.0 - weight) * head_validation
        )
        fit_temporal = build_temporal_features(
            fit_signal, self.fit_tracks, self.fit_times, self.context_radius
        )
        validation_temporal = build_temporal_features(
            validation_signal,
            self.validation_tracks,
            self.validation_times,
            self.context_radius,
        )
        temporal_state = fit_ridge_state(
            fit_temporal,
            self.fit_target,
            self.temporal_alpha,
            "lsqr",
            self.backend,
        )
        prediction = predict_ridge_state(
            temporal_state, validation_temporal
        ).reshape(-1)
        metrics = track_macro_metrics(
            prediction, self.validation_target, self.validation_tracks
        )
        return {name: float(metrics[name]) for name in METRIC_NAMES}


class _CrossValidator:
    def __init__(self, folds: list[_OuterFold], stability_penalty: float) -> None:
        if not folds:
            raise ValueError("at least one outer fold is required")
        self.folds = tuple(folds)
        self.stability_penalty = float(stability_penalty)

    def evaluate(
        self, selected_indices: list[int], head_alpha: float | None = None
    ) -> dict:
        fold_metrics = [
            fold.evaluate(selected_indices, head_alpha) for fold in self.folds
        ]
        return _aggregate_fold_metrics(fold_metrics, self.stability_penalty)


def _greedy_select(
    cv: _CrossValidator,
    coordinates: list[tuple[int, int]],
    top_k: int,
    head_alpha: float,
) -> tuple[list[int], list[dict]]:
    if top_k <= 0 or top_k > len(coordinates):
        raise ValueError("top_k must be within the candidate count")
    selected: list[int] = []
    steps: list[dict] = []
    previous_objective = 0.0
    previous_ccc = 0.0
    for step_index in range(int(top_k)):
        candidates = []
        remaining = [
            index for index in range(len(coordinates)) if index not in selected
        ]
        for candidate_index in remaining:
            trial_indices = sorted(selected + [candidate_index])
            metrics = cv.evaluate(trial_indices, float(head_alpha))
            layer, head = coordinates[candidate_index]
            candidates.append(
                {
                    "candidate_index": int(candidate_index),
                    "layer": int(layer),
                    "head": int(head),
                    "metrics": metrics,
                }
            )
        candidates.sort(
            key=lambda row: (
                -float(row["metrics"]["objective"]),
                float(row["metrics"]["mean"]["rmse"]),
                -float(row["metrics"]["mean"]["pcc"]),
                int(row["layer"]),
                int(row["head"]),
            )
        )
        chosen = candidates[0]
        selected.append(int(chosen["candidate_index"]))
        objective = float(chosen["metrics"]["objective"])
        ccc = float(chosen["metrics"]["mean"]["ccc"])
        steps.append(
            {
                "step": int(step_index + 1),
                "selected": {
                    "layer": int(chosen["layer"]),
                    "head": int(chosen["head"]),
                },
                "cross_validation_metrics": chosen["metrics"],
                "marginal_objective_gain": objective - previous_objective,
                "marginal_ccc_gain": ccc - previous_ccc,
            }
        )
        previous_objective = objective
        previous_ccc = ccc
        print(
            "greedy step",
            step_index + 1,
            "chosen",
            (int(chosen["layer"]), int(chosen["head"])),
            "objective",
            round(objective, 6),
            flush=True,
        )
    return selected, steps


def _choose_head_alpha(
    cv: _CrossValidator,
    selected_indices: list[int],
    head_alpha_grid: list[float],
    baseline_metrics: dict,
) -> tuple[dict, list[dict], bool]:
    rmse_limit = float(baseline_metrics["mean"]["rmse"])
    candidates = []
    for alpha in head_alpha_grid:
        metrics = cv.evaluate(selected_indices, float(alpha))
        candidates.append({"head_alpha": float(alpha), "metrics": metrics})
    eligible = [
        row
        for row in candidates
        if float(row["metrics"]["mean"]["rmse"]) <= rmse_limit
    ]
    guard_satisfied = bool(eligible)
    pool = eligible if eligible else candidates
    pool.sort(
        key=lambda row: (
            -float(row["metrics"]["objective"]),
            float(row["metrics"]["mean"]["rmse"]),
            -float(row["metrics"]["mean"]["pcc"]),
            float(row["head_alpha"]),
        )
    )
    return pool[0], candidates, guard_satisfied


def _rank_consensus(seed_steps: list[list[dict]], top_k: int) -> list[tuple[int, int]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    records: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"frequency": 0, "objective_gains": [], "ccc_gains": []}
    )
    for steps in seed_steps:
        seen: set[tuple[int, int]] = set()
        for step in steps:
            coordinate = (
                int(step["selected"]["layer"]),
                int(step["selected"]["head"]),
            )
            if coordinate in seen:
                raise ValueError("a seed selection contains a duplicate head")
            seen.add(coordinate)
            record = records[coordinate]
            record["frequency"] = int(record["frequency"]) + 1
            record["objective_gains"].append(float(step["marginal_objective_gain"]))
            record["ccc_gains"].append(float(step["marginal_ccc_gain"]))
    if len(records) < top_k:
        raise ValueError("fewer unique selected heads than top_k")

    def sort_key(item: tuple[tuple[int, int], dict]) -> tuple:
        coordinate, record = item
        return (
            -int(record["frequency"]),
            -float(mean(record["objective_gains"])),
            -float(mean(record["ccc_gains"])),
            coordinate[0],
            coordinate[1],
        )

    return [coordinate for coordinate, _ in sorted(records.items(), key=sort_key)[:top_k]]


def _compact_metrics(metrics: dict) -> dict:
    return {
        "mean": dict(metrics["mean"]),
        "sample_std": dict(metrics["sample_std"]),
        "objective": float(metrics["objective"]),
    }


def _inspect_head_grid(head_dir: str) -> tuple[int, int, int]:
    paths = sorted(Path(head_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError("no head feature shards found in " + head_dir)
    with np.load(paths[0], allow_pickle=False) as data:
        shape = np.asarray(data["head_features"]).shape
    if len(shape) != 4:
        raise RuntimeError("expected [rows, layers, heads, width] head cache")
    return int(shape[1]), int(shape[2]), int(shape[3])


def _first_present(row: dict, names: tuple[str, ...]) -> float:
    for name in names:
        if name in row and row[name] is not None:
            return float(row[name])
    raise KeyError("ranking row lacks any of " + str(names))


def _shortlist_from_ranking_json(path: Path, pool_size: int) -> list[tuple[int, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selection = payload.get("head_selection", payload)
    if bool(selection.get("uses_test_features", False)) or bool(
        selection.get("uses_test_labels", False)
    ):
        raise RuntimeError("ranking file is not training-only: " + str(path))
    if int(selection.get("outer_group_overlap_count", 0)) != 0:
        raise RuntimeError("ranking file contains group overlap: " + str(path))
    rows = list(selection["all_scores"])
    rows.sort(
        key=lambda row: (
            _first_present(row, ("macro_ccc", "score", "ccc")),
            _first_present(row, ("macro_pcc", "pcc")),
            -_first_present(row, ("macro_rmse", "rmse")),
            -int(row["layer"]),
            -int(row["head"]),
        ),
        reverse=True,
    )
    shortlist: list[tuple[int, int]] = []
    for row in rows[: int(pool_size)]:
        coordinate = (int(row["layer"]), int(row["head"]))
        if coordinate not in shortlist:
            shortlist.append(coordinate)
    return shortlist


def _load_training_split(
    config: dict,
    dataset: str,
    baseline_union: list[tuple[int, int]],
    pool_size: int,
    rank_points: int,
    rank_alpha: float,
    backend: str,
    ranking_json: Path | None = None,
) -> dict:
    paths = config["datasets"][dataset]["paths"]
    head_dir = paths["head_features"]
    # PMEmo pairs head shards with global-feature markers and groups tracks by
    # the audio content hash; DEAM groups by track identifier.
    counterpart = paths.get("global_features") if dataset == "pmemo" else None
    track_to_group = None
    if dataset == "pmemo":
        track_to_group, _ = build_content_group_map(paths["manifest"])

    def fold_groups_for(track_ids: np.ndarray) -> np.ndarray:
        if dataset == "pmemo":
            return groups_for_rows(track_ids, track_to_group)
        return np.asarray(track_ids).astype(str)

    layer_count, head_count, width = _inspect_head_grid(head_dir)
    if width != HEAD_WIDTH:
        raise RuntimeError("unexpected attention-head width: " + str(width))
    if ranking_json is not None:
        # Reuse a training-only per-head ranking when one is supplied, so the
        # candidate pool matches the original selection run exactly.
        shortlist = _shortlist_from_ranking_json(ranking_json, pool_size)
    else:
        # Otherwise rank the full head grid on a small uniform sample so long
        # tracks cannot dominate the shortlist probe.
        sampled = load_sampled_point_head_feature_shards(
            head_dir,
            "train",
            int(rank_points),
            counterpart_dir=counterpart,
        )
        ranking = rank_arousal_heads(
            sampled.head_features,
            sampled.targets,
            fold_groups_for(np.asarray(sampled.track_ids).astype(str)),
            int(pool_size),
            alpha=float(rank_alpha),
            backend=backend,
        )
        shortlist = [
            (int(row["layer"]), int(row["head"])) for row in ranking["heads"]
        ]
        del sampled, ranking
        gc.collect()

    pool: list[tuple[int, int]] = []
    for coordinate in shortlist:
        if coordinate not in pool:
            pool.append(coordinate)
    for coordinate in baseline_union:
        if coordinate not in pool:
            pool.append(coordinate)
    for layer, head in pool:
        if layer >= layer_count or head >= head_count:
            raise RuntimeError("candidate coordinate is outside the head grid")

    # Load every training row for the pooled heads; the greedy search itself
    # runs on the full training split, not on the shortlist sample.
    selected = load_selected_point_head_features(
        head_dir,
        "train",
        pool,
        counterpart_dir=counterpart,
    )
    row_count = int(selected.features.shape[0])
    if selected.features.shape[1] != len(pool) * HEAD_WIDTH:
        raise RuntimeError("selected head feature width does not match the pool")
    head_grid = np.asarray(selected.features, dtype=np.float32).reshape(
        row_count, len(pool), HEAD_WIDTH
    )
    acoustic = align_acoustic_rows(
        selected.track_ids,
        selected.target_times,
        selected.targets,
        load_acoustic_features(paths["acoustic_train"], split="train"),
    ).features
    tracks = np.asarray(selected.track_ids).astype(str)
    return {
        "head_grid": head_grid,
        "acoustic": np.asarray(acoustic, dtype=np.float32),
        "target": np.asarray(selected.targets)[:, 1].astype(np.float32),
        "tracks": tracks,
        "times": np.asarray(selected.target_times, dtype=np.float32),
        "groups": fold_groups_for(tracks),
        "pool": pool,
        "shortlist": shortlist,
        "layer_count": layer_count,
        "head_count": head_count,
        "row_count": row_count,
    }


def _load_selection_protocol(
    path: str | None, dataset: str, seeds: list[int], branch: dict
) -> dict | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    block = payload.get(dataset)
    if not isinstance(block, dict):
        raise RuntimeError("selection protocol has no block for " + dataset)
    protocol: dict[int, dict] = {}
    for seed in seeds:
        record = block.get(str(seed))
        if record is None:
            raise RuntimeError("selection protocol missing seed " + str(seed))
        protocol[int(seed)] = {
            "baseline_heads": [
                (int(head[0]), int(head[1])) for head in record["baseline_heads"]
            ],
            "head_alpha": float(record["head_alpha"]),
            "acoustic_alpha": float(record["acoustic_alpha"]),
            "fusion_weight": float(
                record.get("fusion_weight", branch["fusion_weight"])
            ),
        }
    return protocol


def _evaluate_seed(
    data: dict,
    seed: int,
    baseline_heads: list[tuple[int, int]],
    head_alpha: float,
    acoustic_alpha: float,
    fusion_weight: float,
    branch: dict,
    args: argparse.Namespace,
    backend: str,
    cache_device: str,
    oof_seed_base: int,
) -> tuple[dict, list[dict]]:
    pool = data["pool"]
    coordinate_to_index = {coordinate: index for index, coordinate in enumerate(pool)}
    missing = [
        coordinate
        for coordinate in baseline_heads
        if coordinate not in coordinate_to_index
    ]
    if missing:
        raise RuntimeError(
            "baseline heads absent from the candidate pool: " + str(missing)
        )
    baseline_indices = [
        coordinate_to_index[coordinate] for coordinate in baseline_heads
    ]

    outer_folds = _grouped_folds(
        data["groups"], int(args.outer_splits), OUTER_SEED_BASE + int(seed)
    )
    outer_coverage = np.zeros(len(data["tracks"]), dtype=np.int8)
    folds = []
    for fold_index, (fit_rows, validation_rows) in enumerate(outer_folds):
        fit_groups = set(data["groups"][fit_rows].tolist())
        validation_groups = set(data["groups"][validation_rows].tolist())
        if fit_groups & validation_groups:
            raise RuntimeError("outer-fold group overlap detected")
        outer_coverage[validation_rows] += 1
        inner_seed = oof_seed_base + int(seed) + 1000 * fold_index
        folds.append(
            _OuterFold(
                data["head_grid"],
                data["acoustic"],
                data["target"],
                data["tracks"],
                data["groups"],
                data["times"],
                fit_rows,
                validation_rows,
                float(head_alpha),
                float(acoustic_alpha),
                float(fusion_weight),
                int(branch["context_radius"]),
                float(branch["temporal_alpha"]),
                int(branch["oof_splits"]),
                inner_seed,
                backend,
                cache_device,
            )
        )
    if not np.all(outer_coverage == 1):
        raise RuntimeError("outer-fold validation coverage is not exactly one")

    cv = _CrossValidator(folds, float(args.stability_penalty))
    baseline_cv = cv.evaluate(sorted(baseline_indices))
    greedy_indices, steps = _greedy_select(
        cv, pool, int(args.top_k), float(head_alpha)
    )
    selected_heads = [pool[index] for index in greedy_indices]
    selected_cv = steps[-1]["cross_validation_metrics"]

    record = {
        "protocol_baseline": {
            "heads": [list(coordinate) for coordinate in baseline_heads],
            "head_alpha": float(head_alpha),
            "acoustic_alpha": float(acoustic_alpha),
            "fusion_weight": float(fusion_weight),
        },
        "candidate_shortlist": [
            list(coordinate) for coordinate in data["shortlist"]
        ],
        "candidate_pool": [list(coordinate) for coordinate in pool],
        "baseline_cross_validation": _compact_metrics(baseline_cv),
        "greedy_steps": [
            {
                "step": int(step["step"]),
                "selected": dict(step["selected"]),
                "cross_validation_metrics": _compact_metrics(
                    step["cross_validation_metrics"]
                ),
                "marginal_objective_gain": float(
                    step["marginal_objective_gain"]
                ),
                "marginal_ccc_gain": float(step["marginal_ccc_gain"]),
            }
            for step in steps
        ],
        "selected_heads": [list(coordinate) for coordinate in selected_heads],
        "selected_head_alpha": float(head_alpha),
        "selected_cross_validation": _compact_metrics(selected_cv),
    }

    if bool(args.refit_alpha):
        chosen, candidates, guard = _choose_head_alpha(
            cv,
            greedy_indices,
            [float(value) for value in args.head_alpha_grid],
            baseline_cv,
        )
        record["refit_alpha"] = {
            "selected_head_alpha": float(chosen["head_alpha"]),
            "rmse_guard_satisfied": bool(guard),
            "head_alpha_candidates": [
                {
                    "head_alpha": float(row["head_alpha"]),
                    "cross_validation_metrics": _compact_metrics(row["metrics"]),
                }
                for row in candidates
            ],
        }

    return record, steps


def _run_dataset(args: argparse.Namespace, config: dict, frozen: dict, dataset: str) -> dict:
    branch = config["model"]["arousal"]
    dataset_config = config["datasets"][dataset]
    seeds = [int(seed) for seed in (args.seeds or config["experiment"]["seeds"])]
    backend = args.backend if args.backend else str(config["runtime"]["backend"])
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cache_device = backend if args.cache_device == "auto" else args.cache_device
    if cache_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA head cache was requested but is unavailable")
    oof_seed_base = OOF_SEED_BASE[dataset]

    # The released frozen configuration is the final cross-seed consensus; it
    # is the comparison target, not the starting baseline of the selection run.
    final_heads_by_seed = {
        seed: [
            (int(row["layer"]), int(row["head"]))
            for row in get_seed_config(config, frozen, dataset, seed)["heads"]
        ]
        for seed in seeds
    }
    final_alpha_by_seed = {
        seed: float(get_seed_config(config, frozen, dataset, seed)["head_alpha"])
        for seed in seeds
    }

    protocol = _load_selection_protocol(
        args.selection_protocol, dataset, seeds, branch
    )
    per_seed_mode = protocol is not None or bool(args.ranking_template)

    per_seed: dict[str, dict] = {}
    seed_steps: list[list[dict]] = []

    def finalize_seed(seed: int, record: dict, steps: list[dict]) -> None:
        selected = [tuple(coordinate) for coordinate in record["selected_heads"]]
        record["matches_final_frozen_heads"] = sorted(selected) == sorted(
            final_heads_by_seed[seed]
        )
        record["matches_final_frozen_alpha"] = (
            record["selected_head_alpha"] == final_alpha_by_seed[seed]
        )
        per_seed[str(seed)] = record
        seed_steps.append(steps)

    if per_seed_mode:
        # Each seed gets its own ranking shortlist and its own starting
        # baseline, matching the original per-seed selection run exactly.
        for seed in seeds:
            if protocol is not None:
                setup = protocol[seed]
                baseline_heads = list(setup["baseline_heads"])
                head_alpha = float(setup["head_alpha"])
                acoustic_alpha = float(setup["acoustic_alpha"])
                fusion_weight = float(setup["fusion_weight"])
            else:
                seed_config = get_seed_config(config, frozen, dataset, seed)
                baseline_heads = [
                    (int(row["layer"]), int(row["head"]))
                    for row in seed_config["heads"]
                ]
                head_alpha = float(seed_config["head_alpha"])
                acoustic_alpha = float(seed_config["acoustic_alpha"])
                fusion_weight = float(branch["fusion_weight"])

            if args.ranking_template:
                ranking_path = Path(args.ranking_template.format(seed=seed))
            elif args.ranking_json:
                ranking_path = Path(args.ranking_json)
            else:
                ranking_path = None

            print(
                "loading training split for",
                dataset,
                "seed",
                seed,
                flush=True,
            )
            data = _load_training_split(
                config,
                dataset,
                baseline_heads,
                int(args.pool_size),
                int(args.rank_points),
                float(args.rank_alpha),
                backend,
                ranking_json=ranking_path,
            )
            print(
                dataset,
                "seed",
                seed,
                "head grid",
                (data["layer_count"], data["head_count"]),
                "candidate pool",
                len(data["pool"]),
                "training rows",
                data["row_count"],
                flush=True,
            )
            record, steps = _evaluate_seed(
                data,
                seed,
                baseline_heads,
                head_alpha,
                acoustic_alpha,
                fusion_weight,
                branch,
                args,
                backend,
                cache_device,
                oof_seed_base,
            )
            finalize_seed(seed, record, steps)
            print(
                "seed",
                seed,
                "fixed head alpha",
                head_alpha,
                "selected",
                record["selected_heads"],
                flush=True,
            )
            del data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # Self-contained probe: one shared pool from a single ranking or the
        # on-the-fly probe, with the released frozen configuration as baseline.
        baseline_union: list[tuple[int, int]] = []
        for seed in seeds:
            for coordinate in final_heads_by_seed[seed]:
                if coordinate not in baseline_union:
                    baseline_union.append(coordinate)

        print("loading training split for", dataset, flush=True)
        ranking_json = Path(args.ranking_json) if args.ranking_json else None
        data = _load_training_split(
            config,
            dataset,
            baseline_union,
            int(args.pool_size),
            int(args.rank_points),
            float(args.rank_alpha),
            backend,
            ranking_json=ranking_json,
        )
        print(
            dataset,
            "head grid",
            (data["layer_count"], data["head_count"]),
            "candidate pool",
            len(data["pool"]),
            "training rows",
            data["row_count"],
            flush=True,
        )
        expected = dataset_config.get("expected", {})
        if "train_points" in expected and data["row_count"] != int(expected["train_points"]):
            raise RuntimeError(dataset + " training point count changed")
        train_track_count = len(np.unique(data["tracks"]))
        if "train_tracks" in expected and train_track_count != int(expected["train_tracks"]):
            raise RuntimeError(dataset + " training song count changed")
        if dataset == "pmemo" and "train_content_groups" in expected:
            group_count = len(np.unique(data["groups"]))
            if group_count != int(expected["train_content_groups"]):
                raise RuntimeError(dataset + " training content-group count changed")

        for seed in seeds:
            seed_config = get_seed_config(config, frozen, dataset, seed)
            baseline_heads = list(final_heads_by_seed[seed])
            record, steps = _evaluate_seed(
                data,
                seed,
                baseline_heads,
                float(seed_config["head_alpha"]),
                float(seed_config["acoustic_alpha"]),
                float(branch["fusion_weight"]),
                branch,
                args,
                backend,
                cache_device,
                oof_seed_base,
            )
            finalize_seed(seed, record, steps)
            print(
                "seed",
                seed,
                "fixed head alpha",
                seed_config["head_alpha"],
                "selected",
                record["selected_heads"],
                flush=True,
            )
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    consensus_heads = _rank_consensus(seed_steps, int(args.top_k))
    final_consensus = sorted(final_heads_by_seed[seeds[0]])
    payload = {
        "experiment": "HEVA arousal head selection (training only)",
        "dataset": dataset,
        "seeds": seeds,
        "mode": "per_seed_protocol" if per_seed_mode else "on_the_fly_probe",
        "backend": backend,
        "head_cache_device": cache_device,
        "ranking_template": Path(args.ranking_template).name
        if args.ranking_template
        else None,
        "selection_protocol": Path(args.selection_protocol).name
        if args.selection_protocol
        else None,
        "parameters": {
            "pool_size": int(args.pool_size),
            "top_k": int(args.top_k),
            "outer_splits": int(args.outer_splits),
            "inner_oof_splits": int(branch["oof_splits"]),
            "stability_penalty": float(args.stability_penalty),
            "rank_points_per_track": int(args.rank_points),
            "rank_probe_alpha": float(args.rank_alpha),
            "fusion_weight": float(branch["fusion_weight"]),
            "context_radius": int(branch["context_radius"]),
            "temporal_alpha": float(branch["temporal_alpha"]),
            "outer_seed_base": OUTER_SEED_BASE,
            "inner_oof_seed_base": oof_seed_base,
            "refit_alpha_enabled": bool(args.refit_alpha),
            "head_alpha_grid": [float(value) for value in args.head_alpha_grid],
        },
        "per_seed": per_seed,
        "consensus_heads": [list(coordinate) for coordinate in consensus_heads],
        "final_frozen_heads": [list(coordinate) for coordinate in final_consensus],
        "consensus_matches_final_frozen": sorted(consensus_heads) == final_consensus,
        "uses_test_features": False,
        "uses_test_labels": False,
    }
    print(
        "consensus heads",
        [list(coordinate) for coordinate in consensus_heads],
        flush=True,
    )
    return payload


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
        help="per-seed ranking path containing '{seed}', e.g. "
        "selection_inputs/deam_seed{seed}_arousal.json",
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

    payload = _run_dataset(args, config, frozen, args.dataset)
    output = (
        Path(args.output)
        if args.output
        else Path("outputs") / "head_selection" / (args.dataset + "_head_selection.json")
    )
    write_json(output, payload)
    print("wrote", output, flush=True)


if __name__ == "__main__":
    main()
