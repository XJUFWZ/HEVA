import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heva.models.arousal import ArousalState, predict_arousal_model
from heva.models.ridge import RidgeState
from heva.models.valence import ValenceState, predict_valence_model


@dataclass(frozen=True)
class HEVAPackage:
    metadata: dict
    valence: ValenceState
    arousal: ArousalState


def _store_ridge(arrays, manifest, name, state):
    arrays[name + "_feature_mean"] = np.asarray(
        state.feature_mean,
        dtype=np.float32,
    )
    arrays[name + "_feature_scale"] = np.asarray(
        state.feature_scale,
        dtype=np.float32,
    )
    arrays[name + "_weights"] = np.asarray(
        state.weights,
        dtype=np.float32,
    )
    arrays[name + "_intercept"] = np.asarray(
        state.intercept,
        dtype=np.float32,
    )
    manifest["ridges"][name] = {"alpha": float(state.alpha)}


def _load_ridge(archive, manifest, name):
    intercept = np.asarray(
        archive[name + "_intercept"],
        dtype=np.float32,
    )
    if intercept.ndim == 0:
        intercept = float(intercept.item())
    return RidgeState(
        feature_mean=np.asarray(
            archive[name + "_feature_mean"],
            dtype=np.float32,
        ),
        feature_scale=np.asarray(
            archive[name + "_feature_scale"],
            dtype=np.float32,
        ),
        weights=np.asarray(
            archive[name + "_weights"],
            dtype=np.float32,
        ),
        intercept=intercept,
        alpha=float(manifest["ridges"][name]["alpha"]),
    )


def save_model_package(path, package):
    output = Path(path)

    if output.suffix != ".npz":
        output = Path(str(output) + ".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    manifest = {
        "format_version": 4,
        "metadata": package.metadata,
        "valence": {
            "fusion_weight": float(package.valence.fusion_weight),
            "context_radius": int(package.valence.context_radius),
        },
        "arousal": {
            "acoustic_weight": float(package.arousal.acoustic_weight),
            "context_radius": int(package.arousal.context_radius),
            "selected_heads": [
                [int(layer), int(head)] for layer, head in package.arousal.selected_heads
            ],
        },
        "ridges": {},
    }
    _store_ridge(
        arrays,
        manifest,
        "valence_base",
        package.valence.base_ridge,
    )
    _store_ridge(
        arrays,
        manifest,
        "valence_semantic",
        package.valence.semantic_ridge,
    )
    _store_ridge(
        arrays,
        manifest,
        "valence_temporal",
        package.valence.temporal_ridge,
    )
    _store_ridge(
        arrays,
        manifest,
        "arousal_head",
        package.arousal.head_ridge,
    )
    _store_ridge(
        arrays,
        manifest,
        "arousal_acoustic",
        package.arousal.acoustic_ridge,
    )
    _store_ridge(
        arrays,
        manifest,
        "arousal_temporal",
        package.arousal.temporal_ridge,
    )
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
    )
    np.savez_compressed(output, **arrays)
    return output


def load_model_package(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        if int(manifest.get("format_version", 0)) != 4:
            raise ValueError("unsupported HEVA model package version")
        valence_data = manifest["valence"]
        arousal_data = manifest["arousal"]
        valence = ValenceState(
            base_ridge=_load_ridge(
                archive,
                manifest,
                "valence_base",
            ),
            semantic_ridge=_load_ridge(
                archive,
                manifest,
                "valence_semantic",
            ),
            fusion_weight=float(valence_data["fusion_weight"]),
            temporal_ridge=_load_ridge(
                archive,
                manifest,
                "valence_temporal",
            ),
            context_radius=int(valence_data["context_radius"]),
        )
        arousal = ArousalState(
            head_ridge=_load_ridge(
                archive,
                manifest,
                "arousal_head",
            ),
            acoustic_ridge=_load_ridge(
                archive,
                manifest,
                "arousal_acoustic",
            ),
            acoustic_weight=float(arousal_data["acoustic_weight"]),
            temporal_ridge=_load_ridge(
                archive,
                manifest,
                "arousal_temporal",
            ),
            context_radius=int(arousal_data["context_radius"]),
            selected_heads=tuple(
                (int(layer), int(head)) for layer, head in arousal_data["selected_heads"]
            ),
        )
    return HEVAPackage(
        metadata=manifest["metadata"],
        valence=valence,
        arousal=arousal,
    )


def predict_feature_sequence(package, features, track_id="audio"):
    target_times = np.asarray(features.target_times, dtype=np.float32)
    track_ids = np.full(len(target_times), str(track_id))
    valence = predict_valence_model(
        package.valence,
        features.base,
        features.semantic,
        track_ids,
        target_times,
    )
    arousal = predict_arousal_model(
        package.arousal,
        features.heads,
        features.acoustic,
        track_ids,
        target_times,
    )
    return {
        "time": target_times,
        "valence": np.asarray(valence, dtype=np.float32),
        "arousal": np.asarray(arousal, dtype=np.float32),
    }
