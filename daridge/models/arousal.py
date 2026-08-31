from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold

from daridge.metrics import one_dimension_metrics
from daridge.models.ridge import (
    RidgeState,
    fit_ridge_state,
    predict_ridge,
    predict_ridge_state,
)
from daridge.models.temporal import build_temporal_features


def _validate(head_features, targets, group_ids, top_k):
    features = np.asarray(head_features, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(group_ids).astype(str).reshape(-1)

    if features.ndim != 4:
        raise ValueError("head features must have shape [n, layers, heads, d]")
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError("targets must have shape [n, 2]")
    if not (len(features) == len(target) == len(groups)):
        raise ValueError("head features, targets and groups must align")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    if len(np.unique(groups)) < 2:
        raise ValueError("at least two groups are required")
    return features, target, groups


def _folds(groups, max_splits=5):
    split_count = min(int(max_splits), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=split_count)
    rows = np.arange(len(groups))
    return list(splitter.split(rows, groups=groups))


# Score each head with grouped OOF CCC penalized by fold instability.
def _score_head(
    features,
    target,
    folds,
    alpha,
    stability_weight,
    backend,
):
    prediction = np.empty(len(target), dtype=np.float32)
    fold_ccc = []

    for fit_rows, validation_rows in folds:
        fold_prediction = predict_ridge(
            features[fit_rows],
            target[fit_rows],
            features[validation_rows],
            alpha,
            "lsqr",
            backend,
        )
        prediction[validation_rows] = fold_prediction
        fold_metrics = one_dimension_metrics(
            fold_prediction,
            target[validation_rows],
        )
        fold_ccc.append(fold_metrics["ccc"])

    metrics = one_dimension_metrics(prediction, target)
    ccc_std = float(np.std(fold_ccc, ddof=0))
    score = float(metrics["ccc"] - float(stability_weight) * ccc_std)
    return {
        "score": score,
        "ccc": float(metrics["ccc"]),
        "ccc_std": ccc_std,
        "pcc": float(metrics["pcc"]),
        "rmse": float(metrics["rmse"]),
    }


def rank_arousal_heads(
    head_features,
    targets,
    group_ids,
    top_k,
    alpha=1.0,
    stability_weight=0.5,
    backend="cuda",
):
    features, target, groups = _validate(
        head_features,
        targets,
        group_ids,
        top_k,
    )
    if float(stability_weight) < 0.0:
        raise ValueError("stability_weight must be non-negative")

    folds = _folds(groups)
    scores = []

    # Rank heads using arousal labels only.
    for layer in range(features.shape[1]):
        for head in range(features.shape[2]):
            row = _score_head(
                features[:, layer, head, :],
                target[:, 1],
                folds,
                float(alpha),
                float(stability_weight),
                backend,
            )
            scores.append(
                {
                    "layer": int(layer),
                    "head": int(head),
                    **row,
                }
            )

    # Use deterministic metric and coordinate tie breakers.
    scores.sort(
        key=lambda item: (
            item["score"],
            item["ccc"],
            item["pcc"],
            -item["rmse"],
            -item["layer"],
            -item["head"],
        ),
        reverse=True,
    )
    selected = scores[: min(int(top_k), len(scores))]

    return {
        "heads": [{"layer": row["layer"], "head": row["head"]} for row in selected],
        "all_scores": scores,
        "top_k": len(selected),
        "probe_alpha": float(alpha),
        "stability_weight": float(stability_weight),
        "probe_folds": len(folds),
        "selection_target": "arousal_only",
    }


@dataclass(frozen=True)
class ArousalState:
    head_ridge: RidgeState
    acoustic_ridge: RidgeState
    acoustic_weight: float
    temporal_ridge: RidgeState
    context_radius: int
    selected_heads: tuple[tuple[int, int], ...]


# Shuffle complete content groups without mixing rows between folds.
def _grouped_folds(group_ids, n_splits, seed):
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    unique_groups = np.unique(groups)
    split_count = int(n_splits)
    if split_count < 2 or split_count > len(unique_groups):
        raise ValueError("invalid grouped OOF split count")
    shuffled = unique_groups.copy()
    random = np.random.default_rng(int(seed))
    random.shuffle(shuffled)
    folds = []
    for fold_index in range(split_count):
        validation_groups = set(shuffled[fold_index::split_count].tolist())
        validation_mask = np.asarray(
            [value in validation_groups for value in groups],
            dtype=bool,
        )
        folds.append(
            (
                np.flatnonzero(~validation_mask),
                np.flatnonzero(validation_mask),
            )
        )
    return folds


# Produce one leakage-controlled prediction for every training row.
def _oof_prediction(
    features,
    target,
    group_ids,
    alpha,
    seed,
    n_splits,
    backend,
):
    matrix = np.asarray(features, dtype=np.float32)
    values = np.asarray(target, dtype=np.float32).reshape(-1)
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    if matrix.ndim != 2 or not (len(matrix) == len(values) == len(groups)):
        raise ValueError("OOF inputs must be aligned matrices and vectors")

    prediction = np.empty(len(values), dtype=np.float32)
    coverage = np.zeros(len(values), dtype=np.int32)
    overlap_count = 0
    for fit_rows, validation_rows in _grouped_folds(
        groups,
        n_splits,
        seed,
    ):
        prediction[validation_rows] = predict_ridge(
            matrix[fit_rows],
            values[fit_rows],
            matrix[validation_rows],
            float(alpha),
            "lsqr",
            backend,
        )
        coverage[validation_rows] += 1
        overlap_count += len(set(groups[fit_rows]) & set(groups[validation_rows]))
    if not np.all(coverage == 1):
        raise RuntimeError("Arousal OOF must cover each row once")
    if overlap_count:
        raise RuntimeError("Arousal OOF groups overlap")
    return prediction, coverage, overlap_count


def build_oof_fused_signal(
    head_features,
    acoustic_features,
    target,
    group_ids,
    head_alpha,
    acoustic_alpha,
    acoustic_weight,
    seed,
    n_splits=5,
    backend="cuda",
):
    head_signal, head_coverage, head_overlap = _oof_prediction(
        head_features,
        target,
        group_ids,
        head_alpha,
        seed,
        n_splits,
        backend,
    )
    acoustic_signal, acoustic_coverage, acoustic_overlap = _oof_prediction(
        acoustic_features,
        target,
        group_ids,
        acoustic_alpha,
        seed,
        n_splits,
        backend,
    )
    weight = float(acoustic_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Arousal fusion weight must be inside [0, 1]")
    # Combine head and acoustic OOF signals with a fixed convex weight.
    signal = weight * acoustic_signal + (1.0 - weight) * head_signal
    coverage = np.minimum(head_coverage, acoustic_coverage)
    return {
        "signal": np.asarray(signal, dtype=np.float32),
        "coverage": float(np.mean(coverage == 1)),
        "group_overlap_count": int(head_overlap + acoustic_overlap),
    }


def _selected_head_coordinates(seed_config):
    coordinates = []
    for item in seed_config.get("heads", []):
        coordinates.append((int(item["layer"]), int(item["head"])))
    return tuple(coordinates)


def fit_arousal_model(
    train_head_features,
    train_acoustic_features,
    train_target,
    train_track_ids,
    train_group_ids,
    train_target_times,
    seed_config,
    config,
    seed,
    backend="cuda",
):
    train_head = np.asarray(train_head_features, dtype=np.float32)
    train_acoustic = np.asarray(
        train_acoustic_features,
        dtype=np.float32,
    )
    target = np.asarray(train_target, dtype=np.float32).reshape(-1)
    tracks = np.asarray(train_track_ids).astype(str).reshape(-1)
    groups = np.asarray(train_group_ids).astype(str).reshape(-1)
    times = np.asarray(train_target_times, dtype=np.float32).reshape(-1)
    if train_head.ndim != 2 or train_acoustic.ndim != 2:
        raise ValueError("Arousal features must be matrices")
    if not (
        len(train_head)
        == len(train_acoustic)
        == len(target)
        == len(tracks)
        == len(groups)
        == len(times)
    ):
        raise ValueError("training Arousal rows must align")

    oof = build_oof_fused_signal(
        train_head,
        train_acoustic,
        target,
        groups,
        seed_config["head_alpha"],
        seed_config["acoustic_alpha"],
        config["fusion_weight"],
        seed=seed,
        n_splits=config["oof_splits"],
        backend=backend,
    )

    # Refit both static paths on all training rows for test-time prediction.
    head_ridge = fit_ridge_state(
        train_head,
        target,
        seed_config["head_alpha"],
        "lsqr",
        backend,
    )
    acoustic_ridge = fit_ridge_state(
        train_acoustic,
        target,
        seed_config["acoustic_alpha"],
        "lsqr",
        backend,
    )

    # Learn from the current, past-mean, and future-mean static signals.
    train_temporal = build_temporal_features(
        oof["signal"],
        tracks,
        times,
        radius=config["context_radius"],
    )
    temporal_ridge = fit_ridge_state(
        train_temporal,
        target,
        config["temporal_alpha"],
        "lsqr",
        backend,
    )
    state = ArousalState(
        head_ridge=head_ridge,
        acoustic_ridge=acoustic_ridge,
        acoustic_weight=float(config["fusion_weight"]),
        temporal_ridge=temporal_ridge,
        context_radius=int(config["context_radius"]),
        selected_heads=_selected_head_coordinates(seed_config),
    )
    details = {
        "oof_coverage": oof["coverage"],
        "oof_group_overlap_count": oof["group_overlap_count"],
    }
    return state, details


def _predict_static_signal(state, head_features, acoustic_features):
    head_prediction = predict_ridge_state(
        state.head_ridge,
        head_features,
    ).reshape(-1)
    acoustic_prediction = predict_ridge_state(
        state.acoustic_ridge,
        acoustic_features,
    ).reshape(-1)
    weight = float(state.acoustic_weight)
    return (weight * acoustic_prediction + (1.0 - weight) * head_prediction).astype(np.float32)


def predict_arousal_model(
    state,
    head_features,
    acoustic_features,
    track_ids,
    target_times,
):
    head = np.asarray(head_features, dtype=np.float32)
    acoustic = np.asarray(acoustic_features, dtype=np.float32)
    tracks = np.asarray(track_ids).astype(str).reshape(-1)
    times = np.asarray(target_times, dtype=np.float32).reshape(-1)
    if not (len(head) == len(acoustic) == len(tracks) == len(times)):
        raise ValueError("evaluation Arousal rows must align")

    signal = _predict_static_signal(state, head, acoustic)
    temporal = build_temporal_features(
        signal,
        tracks,
        times,
        radius=state.context_radius,
    )
    return predict_ridge_state(state.temporal_ridge, temporal).reshape(-1)


def fit_predict_arousal(
    train_head_features,
    train_acoustic_features,
    train_target,
    train_track_ids,
    train_group_ids,
    train_target_times,
    evaluation_head_features,
    evaluation_acoustic_features,
    evaluation_track_ids,
    evaluation_target_times,
    seed_config,
    config,
    seed,
    backend="cuda",
):
    state, details = fit_arousal_model(
        train_head_features,
        train_acoustic_features,
        train_target,
        train_track_ids,
        train_group_ids,
        train_target_times,
        seed_config,
        config,
        seed,
        backend,
    )
    prediction = predict_arousal_model(
        state,
        evaluation_head_features,
        evaluation_acoustic_features,
        evaluation_track_ids,
        evaluation_target_times,
    )
    return {
        "prediction": np.asarray(prediction, dtype=np.float32).reshape(-1),
        "oof_coverage": details["oof_coverage"],
        "oof_group_overlap_count": details["oof_group_overlap_count"],
        "state": state,
    }
