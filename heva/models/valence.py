from dataclasses import dataclass

import numpy as np

from heva.models.ridge import (
    RidgeState,
    fit_ridge_state,
    predict_ridge,
    predict_ridge_state,
)
from heva.models.temporal import build_temporal_features
from heva.splits import grouped_folds


@dataclass(frozen=True)
class ValenceState:
    base_ridge: RidgeState
    semantic_ridge: RidgeState
    fusion_weight: float
    temporal_ridge: RidgeState
    context_radius: int


def _as_matrix(values, name):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) == 0:
        raise ValueError(name + " must be a non-empty matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(name + " must be finite")
    return matrix


def _as_vector(values, name):
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(vector) == 0:
        raise ValueError(name + " must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(name + " must be finite")
    return vector


def _fuse_predictions(base_signal, semantic_signal, semantic_weight):
    base = _as_vector(base_signal, "base signal")
    semantic = _as_vector(semantic_signal, "semantic signal")
    if len(base) != len(semantic):
        raise ValueError("Valence predictions must align")
    weight = float(semantic_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("Valence fusion weight must be inside [0, 1]")
    return ((1.0 - weight) * base + weight * semantic).astype(np.float32)


# OOF predictions keep the temporal Ridge free of in-sample signals.
def _build_oof_fused_signal(
    base,
    semantic,
    target,
    group_ids,
    config,
    backend,
):
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    signal = np.empty(len(target), dtype=np.float32)
    coverage = np.zeros(len(target), dtype=np.int32)
    for fit_rows, validation_rows in grouped_folds(
        groups,
        max_splits=int(config["oof_splits"]),
    ):
        base_prediction = predict_ridge(
            base[fit_rows],
            target[fit_rows],
            base[validation_rows],
            float(config["base_alpha"]),
            "lsqr",
            backend,
        )
        semantic_prediction = predict_ridge(
            semantic[fit_rows],
            target[fit_rows],
            semantic[validation_rows],
            float(config["semantic_alpha"]),
            "lsqr",
            backend,
        )
        signal[validation_rows] = _fuse_predictions(
            base_prediction,
            semantic_prediction,
            config["fusion_weight"],
        )
        coverage[validation_rows] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("Valence OOF must cover each row once")
    return signal, coverage


def fit_valence_model(
    train_base_features,
    train_semantic_features,
    train_target,
    train_track_ids,
    train_group_ids,
    train_target_times,
    config,
    backend="cuda",
):
    base = _as_matrix(train_base_features, "train base features")
    semantic = _as_matrix(
        train_semantic_features,
        "train semantic features",
    )
    target = _as_vector(train_target, "train target")
    tracks = np.asarray(train_track_ids).astype(str).reshape(-1)
    groups = np.asarray(train_group_ids).astype(str).reshape(-1)
    times = np.asarray(train_target_times, dtype=np.float32).reshape(-1)
    if not (len(base) == len(semantic) == len(target) == len(tracks) == len(groups) == len(times)):
        raise ValueError("training Valence rows must align")

    train_signal, coverage = _build_oof_fused_signal(
        base,
        semantic,
        target,
        groups,
        config,
        backend,
    )

    # Refit both static paths on all training rows for test-time prediction.
    base_ridge = fit_ridge_state(
        base,
        target,
        config["base_alpha"],
        "lsqr",
        backend,
    )
    semantic_ridge = fit_ridge_state(
        semantic,
        target,
        config["semantic_alpha"],
        "lsqr",
        backend,
    )

    # Learn from the current, past-mean, and future-mean static signals.
    train_temporal = build_temporal_features(
        train_signal,
        tracks,
        times,
        radius=config["context_radius"],
    )
    temporal_ridge = fit_ridge_state(
        train_temporal,
        target,
        config["final_alpha"],
        "lsqr",
        backend,
    )
    state = ValenceState(
        base_ridge=base_ridge,
        semantic_ridge=semantic_ridge,
        fusion_weight=float(config["fusion_weight"]),
        temporal_ridge=temporal_ridge,
        context_radius=int(config["context_radius"]),
    )
    details = {
        "nested_oof_coverage": float(np.mean(coverage == 1)),
        "static_weight": float(config["fusion_weight"]),
    }
    return state, details


def _predict_static_signal(state, base, semantic):
    base_signal = predict_ridge_state(state.base_ridge, base).reshape(-1)
    semantic_signal = predict_ridge_state(
        state.semantic_ridge,
        semantic,
    ).reshape(-1)
    return _fuse_predictions(
        base_signal,
        semantic_signal,
        state.fusion_weight,
    )


def predict_valence_model(
    state,
    base_features,
    semantic_features,
    track_ids,
    target_times,
):
    base = _as_matrix(base_features, "evaluation base features")
    semantic = _as_matrix(
        semantic_features,
        "evaluation semantic features",
    )
    tracks = np.asarray(track_ids).astype(str).reshape(-1)
    times = np.asarray(target_times, dtype=np.float32).reshape(-1)
    if not (len(base) == len(semantic) == len(tracks) == len(times)):
        raise ValueError("evaluation Valence rows must align")

    signal = _predict_static_signal(state, base, semantic)
    temporal = build_temporal_features(
        signal,
        tracks,
        times,
        radius=state.context_radius,
    )
    return predict_ridge_state(state.temporal_ridge, temporal).reshape(-1)


def fit_predict_valence(
    train_base_features,
    train_semantic_features,
    train_target,
    train_track_ids,
    train_group_ids,
    train_target_times,
    evaluation_base_features,
    evaluation_semantic_features,
    evaluation_track_ids,
    evaluation_target_times,
    config,
    backend="cuda",
):
    state, details = fit_valence_model(
        train_base_features,
        train_semantic_features,
        train_target,
        train_track_ids,
        train_group_ids,
        train_target_times,
        config,
        backend,
    )
    prediction = predict_valence_model(
        state,
        evaluation_base_features,
        evaluation_semantic_features,
        evaluation_track_ids,
        evaluation_target_times,
    )
    return {
        "prediction": np.asarray(prediction, dtype=np.float32).reshape(-1),
        "static_weight": details["static_weight"],
        "nested_oof_coverage": details["nested_oof_coverage"],
        "state": state,
    }
