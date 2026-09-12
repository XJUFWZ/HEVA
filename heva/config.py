import os
from pathlib import Path

import yaml


def _expand_environment(value):
    if isinstance(value, dict):
        expanded = {}
        for key, item in value.items():
            expanded[key] = _expand_environment(item)
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a mapping")
    return _expand_environment(payload)


def validate_config(config):
    experiment = config.get("experiment", {})
    seeds = experiment.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("experiment.seeds must be a non-empty list")
    if len({int(seed) for seed in seeds}) != len(seeds):
        raise ValueError("experiment.seeds must be unique")

    model = config.get("model", {})
    if model.get("backbone_trainable") is not False:
        raise ValueError("the main project requires a frozen backbone")

    valence = model.get("valence", {})
    arousal = model.get("arousal", {})
    for name, branch in (("valence", valence), ("arousal", arousal)):
        if int(branch.get("context_radius", 0)) <= 0:
            raise ValueError(name + " context radius must be positive")
        fusion_weight = float(branch.get("fusion_weight", -1.0))
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError(name + " fusion weight must be inside [0, 1]")
        if int(branch.get("oof_splits", 0)) < 2:
            raise ValueError(name + " OOF split count must be at least 2")

    for name in ("base_alpha", "semantic_alpha", "final_alpha"):
        if float(valence.get(name, 0.0)) <= 0.0:
            raise ValueError("valence " + name + " must be positive")
    if int(arousal.get("top_k", 0)) <= 0:
        raise ValueError("arousal top_k must be positive")
    if float(arousal.get("temporal_alpha", 0.0)) <= 0.0:
        raise ValueError("arousal temporal alpha must be positive")

    validation_fraction = float(config.get("protocol", {}).get("validation_fraction", 0.0))
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be inside (0, 1)")

    datasets = config.get("datasets", {})
    for dataset in ("deam", "pmemo"):
        if dataset not in datasets:
            raise ValueError("missing dataset configuration: " + dataset)


# Load training-derived head coordinates frozen for this dataset and seed.
def get_seed_config(config, frozen, dataset, seed):
    validate_config(config)
    dataset_name = str(dataset).lower()
    seed_text = str(int(seed))
    dataset_rows = frozen.get("arousal", {}).get(dataset_name, {})
    if seed_text not in dataset_rows:
        raise ValueError(
            "missing frozen arousal configuration for " + dataset_name + " seed " + seed_text
        )
    row = dict(dataset_rows[seed_text])
    top_k = int(config["model"]["arousal"]["top_k"])
    heads = list(row.get("heads", []))
    if top_k > len(heads):
        raise ValueError("top_k exceeds the frozen head count: " + str(len(heads)))
    row["heads"] = heads[:top_k]
    return row
