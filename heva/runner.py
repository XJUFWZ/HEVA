import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from heva.audio import FEATURE_PROTOCOL_VERSION
from heva.config import get_seed_config, validate_config
from heva.datasets import load_deam, load_pmemo
from heva.inference import HEVAPackage, save_model_package
from heva.metrics import track_macro_metrics
from heva.models.arousal import fit_predict_arousal
from heva.models.valence import fit_predict_valence
from heva.splits import group_holdout

VALID_BRANCHES = {"both", "valence", "arousal"}


def default_output_path(output_dir, dataset, seed, branch):
    filename = str(dataset) + "_seed" + str(int(seed)) + "_" + str(branch) + ".json"
    return Path(output_dir) / filename


def describe_experiment(
    config,
    frozen,
    dataset,
    seed,
    branch="both",
    backend=None,
):
    validate_config(config)
    dataset_name = str(dataset).lower()
    branch_name = str(branch).lower()
    if dataset_name not in ("deam", "pmemo"):
        raise ValueError("dataset must be deam or pmemo")
    if branch_name not in VALID_BRANCHES:
        raise ValueError("branch must be both, valence or arousal")
    if int(seed) not in [int(value) for value in config["experiment"]["seeds"]]:
        raise ValueError("seed is absent from experiment.seeds")
    arousal = get_seed_config(
        config,
        frozen,
        dataset_name,
        seed,
    )
    arousal.update(deepcopy(config["model"]["arousal"]))
    return {
        "experiment": str(config["experiment"]["name"]),
        "dataset": dataset_name,
        "seed": int(seed),
        "branch": branch_name,
        "backend": (str(config["runtime"]["backend"]) if backend is None else str(backend)),
        "music_flamingo_trainable": False,
        "grouping": config["datasets"][dataset_name]["grouping"],
        "valence": deepcopy(config["model"]["valence"]),
        "arousal": arousal,
    }


def _check_counts(bundle, expected):
    observed = {
        "train_tracks": len(np.unique(bundle.train_track_ids)),
        "train_points": len(bundle.train_track_ids),
        "test_tracks": len(np.unique(bundle.test_track_ids)),
        "test_points": len(bundle.test_track_ids),
    }
    for name, value in observed.items():
        if int(value) != int(expected[name]):
            raise RuntimeError(
                name + " changed: expected " + str(expected[name]) + ", observed " + str(value)
            )
    return {name: int(value) for name, value in observed.items()}


def _load_bundle(config, dataset, seed_config, branch):
    dataset_config = config["datasets"][dataset]
    if dataset == "deam":
        return load_deam(
            dataset_config,
            seed_config["heads"],
            branch=branch,
        )
    return load_pmemo(
        dataset_config,
        seed_config["heads"],
        branch=branch,
    )


def _valence_predictions(bundle, config, backend):
    branch_config = config["model"]["valence"]
    test_result = fit_predict_valence(
        bundle.train_base,
        bundle.train_semantic,
        bundle.train_targets[:, 0],
        bundle.train_track_ids,
        bundle.train_group_ids,
        bundle.train_target_times,
        bundle.test_base,
        bundle.test_semantic,
        bundle.test_track_ids,
        bundle.test_target_times,
        branch_config,
        backend,
    )
    return {
        "prediction": test_result["prediction"],
        "state": test_result["state"],
        "full_train_static_weight": test_result["static_weight"],
        "nested_oof_coverage": test_result["nested_oof_coverage"],
    }


def _arousal_predictions(
    bundle,
    seed_config,
    config,
    dataset,
    seed,
    backend,
):
    branch_config = config["model"]["arousal"]
    seed_offset = 19000 if dataset == "deam" else 31000
    oof_seed = seed_offset + int(seed)
    test_result = fit_predict_arousal(
        bundle.train_head,
        bundle.train_acoustic,
        bundle.train_targets[:, 1],
        bundle.train_track_ids,
        bundle.train_group_ids,
        bundle.train_target_times,
        bundle.test_head,
        bundle.test_acoustic,
        bundle.test_track_ids,
        bundle.test_target_times,
        seed_config,
        branch_config,
        seed=oof_seed,
        backend=backend,
    )
    return {
        "prediction": test_result["prediction"],
        "state": test_result["state"],
        "oof_coverage": test_result["oof_coverage"],
        "oof_group_overlap_count": test_result["oof_group_overlap_count"],
    }


def _write_result(result, arrays, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output.with_suffix(".npz")
    np.savez_compressed(prediction_path, **arrays)
    result["prediction_archive"] = str(prediction_path)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def run_experiment(
    config,
    frozen,
    dataset,
    seed,
    branch="both",
    backend=None,
    output_path=None,
    model_output=None,
):
    description = describe_experiment(
        config,
        frozen,
        dataset,
        seed,
        branch,
        backend,
    )
    dataset_name = description["dataset"]
    branch_name = description["branch"]
    if model_output is not None and branch_name != "both":
        raise ValueError("model export requires branch=both")
    backend_name = str(config["runtime"]["backend"]) if backend is None else str(backend)
    seed_config = get_seed_config(
        config,
        frozen,
        dataset_name,
        seed,
    )

    # Freeze training-derived heads before any test evaluation.
    head_selection = {
        "mode": "frozen_per_seed",
        "heads": deepcopy(seed_config["heads"]),
        "uses_test_features": False,
        "uses_test_labels": False,
    }
    bundle = _load_bundle(
        config,
        dataset_name,
        seed_config,
        branch_name,
    )
    counts = _check_counts(
        bundle,
        config["datasets"][dataset_name]["expected"],
    )

    holdout = group_holdout(
        bundle.train_group_ids,
        config["protocol"]["validation_fraction"],
        seed,
    )

    dimensions = {}
    arrays = {
        "target": bundle.test_targets.astype(np.float32),
        "track_ids": bundle.test_track_ids.astype(str),
        "target_times": bundle.test_target_times.astype(np.float32),
    }
    branch_details = {}
    valence = None
    arousal = None

    if branch_name in ("both", "valence"):
        valence = _valence_predictions(
            bundle,
            config,
            backend_name,
        )
        dimensions["valence"] = track_macro_metrics(
            valence["prediction"],
            bundle.test_targets[:, 0],
            bundle.test_track_ids,
        )
        arrays["prediction_valence"] = valence["prediction"]
        branch_details["valence"] = {
            key: value for key, value in valence.items() if key not in ("prediction", "state")
        }
    if branch_name in ("both", "arousal"):
        arousal = _arousal_predictions(
            bundle,
            seed_config,
            config,
            dataset_name,
            seed,
            backend_name,
        )
        dimensions["arousal"] = track_macro_metrics(
            arousal["prediction"],
            bundle.test_targets[:, 1],
            bundle.test_track_ids,
        )
        arrays["prediction_arousal"] = arousal["prediction"]
        branch_details["arousal"] = {
            key: value for key, value in arousal.items() if key not in ("prediction", "state")
        }

    result = {
        "experiment": "HEVA",
        "dataset": dataset_name,
        "seed": int(seed),
        "branch": branch_name,
        "backend": backend_name,
        "model": description,
        "counts": counts,
        "grouping_audit": bundle.grouping_audit,
        "fit_group_count": len(holdout.fit_groups),
        "validation_group_count": len(holdout.validation_groups),
        "fit_validation_group_overlap_count": 0,
        "metric_aggregation": "per_track_then_arithmetic_mean",
        "metrics": dimensions,
        "branch_details": branch_details,
        "head_selection": head_selection,
        "uses_test_features_for_selection": False,
        "uses_test_labels_for_selection": False,
        "test_labels_role": (
            "cache_integrity_checks_and_metrics_only;" "never_used_for_selection_or_fitting"
        ),
        "music_flamingo_trainable": False,
    }
    if output_path is None:
        output_path = default_output_path(
            config["experiment"]["output_dir"],
            dataset_name,
            seed,
            branch_name,
        )

    # Export only the fixed main-model state and feature protocol.
    if model_output is not None:
        package = HEVAPackage(
            metadata={
                "format_version": 4,
                "dataset": dataset_name,
                "seed": int(seed),
                "sampling_rate": 16000,
                "segment_seconds": 15.0,
                "time_step_seconds": 0.5,
                "time_start_seconds": 15.0,
                "output_scale": "dataset_annotation_scale",
                "music_flamingo_trainable": False,
                "feature_protocol_version": FEATURE_PROTOCOL_VERSION,
            },
            valence=valence["state"],
            arousal=arousal["state"],
        )
        saved_model = save_model_package(model_output, package)
        result["model_package"] = str(saved_model)
    return _write_result(result, arrays, output_path)
