import argparse
import csv
import gc
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from heva.audio import (
    ADJECTIVE_GROUPS,
    AUDIO_SAMPLE_RATE,
    BASE_PROMPT,
    CANDIDATE_ADJECTIVES,
    HEAD_DIM,
    PROJECTION_PATTERN,
    PROMPT_TEMPLATES,
    SEGMENT_SAMPLES,
    SEGMENT_SECONDS,
    decode_audio,
    extract_acoustic_features,
    load_music_segment,
)

# Cache schemas and fixed feature definitions.
MANIFEST_COLUMNS = (
    "track_id",
    "audio_path",
    "dataset",
    "start",
    "end",
    "target_time",
    "valence",
    "arousal",
    "split",
)
SEMANTIC_FEATURE_NAMES = (
    "valence_low_arousal",
    "valence_high_arousal",
    "valence_balanced",
    "arousal_disagreement",
    "arousal_evidence",
    "neutral_evidence",
    "group_entropy",
    "confidence_weighted_valence",
)
BASE_ACOUSTIC_NAMES = (
    "log_rms",
    "zcr",
    "spectral_flux",
    "onset_strength",
    "centroid",
    "bandwidth",
    "rolloff85",
    "low_energy",
    "mid_energy",
    "high_energy",
    "high_low_ratio",
)
ACOUSTIC_WINDOWS = (2, 4, 8, 16, 32)
DEAM_DYNAMIC_DIR = Path(
    "annotations/annotations averaged per song/" "dynamic (per second annotations)"
)
SAMPLE_COLUMN = re.compile(r"^sample_(\d+(?:\.\d+)?)ms$")


# Manifest construction and split validation.
def _track_id(value):
    text = Path(str(value).strip()).stem
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _track_sort_key(value):
    text = str(value)
    if text.isdigit():
        return 0, int(text)
    return 1, text


def _finite_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _split_ids(dataset, split, split_root=None):
    if split_root is None:
        base = Path(__file__).resolve().parent / "splits" / str(dataset).lower()
    else:
        base = Path(split_root)
    path = base / (str(split) + ".txt")
    if not path.exists():
        raise FileNotFoundError("missing fixed split file: " + str(path))
    values = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = _track_id(line)
        if value:
            values.add(value)
    return values


def _find_audio(audio_dir, track_id):
    directory = Path(audio_dir)
    for suffix in (".mp3", ".wav", ".flac", ".m4a"):
        candidate = directory / (str(track_id) + suffix)
        if candidate.exists():
            return candidate.resolve()
    matches = sorted(path for path in directory.glob(str(track_id) + ".*") if path.is_file())
    if not matches:
        return None
    return matches[0].resolve()


def _manifest_row(dataset, track_id, audio_path, target_time, valence, arousal, split):
    end = float(target_time)
    start = max(0.0, end - SEGMENT_SECONDS)
    if end <= start:
        return None
    return {
        "track_id": str(track_id),
        "audio_path": str(Path(audio_path).resolve()),
        "dataset": str(dataset).upper() if str(dataset).lower() == "deam" else "PMEmo",
        "start": float(start),
        "end": float(end),
        "target_time": float(end),
        "valence": float(valence),
        "arousal": float(arousal),
        "split": str(split),
    }


def _load_csv_by_track(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header: " + str(path))
        track_column = reader.fieldnames[0]
        rows = {}
        for row in reader:
            key = _track_id(row.get(track_column, ""))
            if key:
                rows[key] = row
    return track_column, tuple(reader.fieldnames), rows


def _deam_manifest_rows(dataset_root, split_root=None):
    root = Path(dataset_root)
    annotation_dir = root / DEAM_DYNAMIC_DIR
    valence_track_column, valence_columns, valence_rows = _load_csv_by_track(
        annotation_dir / "valence.csv"
    )
    arousal_track_column, arousal_columns, arousal_rows = _load_csv_by_track(
        annotation_dir / "arousal.csv"
    )
    if valence_track_column != arousal_track_column:
        raise ValueError("DEAM valence and arousal track columns differ")
    arousal_column_set = set(arousal_columns)
    time_columns = []
    for column in valence_columns:
        match = SAMPLE_COLUMN.match(str(column))
        if match and column in arousal_column_set:
            time_columns.append((column, float(match.group(1)) / 1000.0))
    if not time_columns:
        raise ValueError("DEAM contains no shared dynamic annotation columns")

    rows = []
    for split in ("train", "test"):
        split_tracks = _split_ids("deam", split, split_root)
        available = split_tracks & set(valence_rows) & set(arousal_rows)
        for track_id in sorted(available, key=_track_sort_key):
            audio_path = _find_audio(root / "MEMD_audio", track_id)
            if audio_path is None:
                raise FileNotFoundError("missing DEAM audio for track " + track_id)
            for column, target_time in time_columns:
                valence = _finite_float(valence_rows[track_id].get(column))
                arousal = _finite_float(arousal_rows[track_id].get(column))
                if valence is None or arousal is None:
                    continue
                row = _manifest_row(
                    "deam",
                    track_id,
                    audio_path,
                    target_time,
                    valence,
                    arousal,
                    split,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _pmemo_manifest_rows(dataset_root, split_root=None):
    root = Path(dataset_root)
    track_splits = {}
    for split in ("train", "test"):
        for track_id in _split_ids("pmemo", split, split_root):
            if track_id in track_splits:
                raise ValueError("PMEmo track appears in multiple official splits")
            track_splits[track_id] = split

    rows = []
    with (root / "meta" / "dynamic_annotations.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        reader = csv.DictReader(handle)
        required = {"musicId", "frameTime", "Valence(mean)", "Arousal(mean)"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("PMEmo dynamic annotation columns are incomplete")
        for source in reader:
            track_id = _track_id(source.get("musicId", ""))
            if track_id not in track_splits:
                continue
            target_time = _finite_float(source.get("frameTime"))
            valence = _finite_float(source.get("Valence(mean)"))
            arousal = _finite_float(source.get("Arousal(mean)"))
            if target_time is None or valence is None or arousal is None:
                continue
            audio_path = _find_audio(root / "mp3", track_id)
            if audio_path is None:
                raise FileNotFoundError("missing PMEmo audio for track " + track_id)
            row = _manifest_row(
                "pmemo",
                track_id,
                audio_path,
                target_time,
                valence,
                arousal,
                track_splits[track_id],
            )
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: (_track_sort_key(row["track_id"]), row["target_time"]))
    return rows


def _validate_manifest_rows(rows):
    checked = []
    keys = set()
    for source in rows:
        missing = [name for name in MANIFEST_COLUMNS if name not in source]
        if missing:
            raise ValueError("manifest row is missing columns: " + str(missing))
        row = {
            "track_id": str(source["track_id"]),
            "audio_path": str(source["audio_path"]),
            "dataset": str(source["dataset"]),
            "start": float(source["start"]),
            "end": float(source["end"]),
            "target_time": float(source["target_time"]),
            "valence": float(source["valence"]),
            "arousal": float(source["arousal"]),
            "split": str(source["split"]),
        }
        numeric = [row[name] for name in ("start", "end", "target_time", "valence", "arousal")]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("manifest numeric values must be finite")
        if row["split"] not in ("train", "test"):
            raise ValueError("manifest split must be train or test")
        if row["end"] <= row["start"]:
            raise ValueError("manifest row has a non-positive audio window")
        if row["target_time"] < row["start"] or row["target_time"] > row["end"]:
            raise ValueError("target time lies outside its audio window")
        key = (row["track_id"], round(row["target_time"], 6))
        if key in keys:
            raise ValueError("duplicate track-time row: " + str(key))
        keys.add(key)
        checked.append(row)
    if not checked:
        raise ValueError("manifest must contain at least one row")
    return checked


def write_manifest(rows, output_path):
    checked = _validate_manifest_rows(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(checked)
    temporary.replace(path)
    return path


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return _validate_manifest_rows(rows)


def build_manifest(dataset, dataset_root, output_path, split_root=None):
    name = str(dataset).lower()
    if name == "deam":
        rows = _deam_manifest_rows(dataset_root, split_root)
    elif name == "pmemo":
        rows = _pmemo_manifest_rows(dataset_root, split_root)
    else:
        raise ValueError("dataset must be deam or pmemo")
    return write_manifest(rows, output_path)


# Shard serialization and cache integrity checks.
def shard_slices(total_rows, shard_size, lane_count=1, lane_index=0):
    total = int(total_rows)
    size = int(shard_size)
    lanes = int(lane_count)
    lane = int(lane_index)
    if total < 0 or size <= 0 or lanes <= 0 or lane < 0 or lane >= lanes:
        raise ValueError("invalid shard partition")
    result = []
    for shard_index, start in enumerate(range(0, total, size)):
        if shard_index % lanes == lane:
            result.append((shard_index, start, min(start + size, total)))
    return result


def _row_arrays(rows):
    return {
        "targets": np.asarray(
            [[row["valence"], row["arousal"]] for row in rows], dtype=np.float32
        ),
        "track_ids": np.asarray([row["track_id"] for row in rows]).astype(str),
        "target_times": np.asarray([row["target_time"] for row in rows], dtype=np.float32),
        "starts": np.asarray([row["start"] for row in rows], dtype=np.float32),
        "ends": np.asarray([row["end"] for row in rows], dtype=np.float32),
        "splits": np.asarray([row["split"] for row in rows]).astype(str),
    }


def _atomic_savez(path, **arrays):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(output)


def write_backbone_shard_pair(global_path, head_path, rows, global_features, head_features):
    checked_rows = _validate_manifest_rows(rows)
    global_values = np.asarray(global_features, dtype=np.float32)
    head_values = np.asarray(head_features, dtype=np.float32)
    count = len(checked_rows)
    if global_values.shape != (count, 3584):
        raise ValueError("global features must have shape [n, 3584]")
    if head_values.shape != (count, 28, 28, HEAD_DIM):
        raise ValueError("head features must have shape [n, 28, 28, 128]")
    if not np.all(np.isfinite(global_values)) or not np.all(np.isfinite(head_values)):
        raise ValueError("backbone features must be finite")
    metadata = _row_arrays(checked_rows)
    global_output = Path(global_path)
    head_output = Path(head_path)
    marker = global_output.with_suffix(".ready.json")
    marker.unlink(missing_ok=True)
    _atomic_savez(global_output, features=global_values, **metadata)
    _atomic_savez(
        head_output,
        head_features=head_values.astype(np.float16),
        **metadata,
    )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker_tmp = marker.with_suffix(".tmp.json")
    marker_tmp.write_text(
        json.dumps(
            {
                "global_filename": global_output.name,
                "head_filename": head_output.name,
                "row_count": count,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    marker_tmp.replace(marker)
    return marker


def write_semantic_shard(
    path,
    rows,
    semantic_features,
    audio_pool,
    prompt_valence,
    metadata,
):
    checked_rows = _validate_manifest_rows(rows)
    count = len(checked_rows)
    semantic = np.asarray(semantic_features, dtype=np.float32)
    audio = np.asarray(audio_pool, dtype=np.float32)
    prompt = np.asarray(prompt_valence, dtype=np.float32)
    if semantic.shape != (count, 8):
        raise ValueError("semantic features must have shape [n, 8]")
    if audio.ndim != 2 or audio.shape[0] != count or audio.shape[1] == 0:
        raise ValueError("audio pool must have shape [n, d]")
    if prompt.shape != (count, 2):
        raise ValueError("prompt valence must have shape [n, 2]")
    if not all(np.all(np.isfinite(value)) for value in (semantic, audio, prompt)):
        raise ValueError("semantic cache values must be finite")
    payload = dict(metadata)
    payload.setdefault("feature_protocol", "vasr_v_semantic_probe")
    payload.setdefault("semantic_feature_names", list(SEMANTIC_FEATURE_NAMES))
    payload.setdefault("prompts", list(PROMPT_TEMPLATES))
    payload.setdefault("trainable_parameters", 0)
    payload.setdefault("uses_test_labels", False)
    payload.setdefault("uses_test_features_for_selection", False)
    payload.setdefault("uses_test_labels_for_selection", False)
    _atomic_savez(
        path,
        semantic_features=semantic,
        audio_pool=audio,
        prompt_valence=prompt,
        metadata_json=np.asarray(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        ),
        **_row_arrays(checked_rows),
    )
    return Path(path)


def acoustic_feature_names():
    names = list(BASE_ACOUSTIC_NAMES)
    names.extend(name + "_delta" for name in BASE_ACOUSTIC_NAMES)
    for window in ACOUSTIC_WINDOWS:
        names.extend(name + "_mean_w" + str(window) for name in BASE_ACOUSTIC_NAMES)
        names.extend(name + "_std_w" + str(window) for name in BASE_ACOUSTIC_NAMES)
    return tuple(names)


def write_acoustic_cache(path, rows, features, feature_names):
    checked_rows = _validate_manifest_rows(rows)
    values = np.asarray(features, dtype=np.float32)
    names = tuple(str(name) for name in feature_names)
    if values.shape != (len(checked_rows), 132) or len(names) != 132:
        raise ValueError("acoustic cache must contain 132 features")
    if len(set(names)) != len(names) or not np.all(np.isfinite(values)):
        raise ValueError("acoustic features and names must be valid")
    metadata = _row_arrays(checked_rows)
    metadata.pop("starts")
    metadata.pop("ends")
    _atomic_savez(path, features=values, feature_names=np.asarray(names), **metadata)
    return Path(path)


def _row_key(track_id, target_time):
    return str(track_id), round(float(target_time), 6)


def _manifest_reference(rows, split):
    selected = [row for row in rows if row["split"] == split]
    return {
        _row_key(row["track_id"], row["target_time"]): np.asarray(
            [row["valence"], row["arousal"]], dtype=np.float32
        )
        for row in selected
    }


def _verify_npz_rows(paths, reference, feature_key, expected_tail, label):
    observed = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            features = data[feature_key]
            if tuple(features.shape[1:]) != tuple(expected_tail):
                raise RuntimeError(label + " feature shape changed in " + str(path))
            targets = np.asarray(data["targets"], dtype=np.float32)
            track_ids = np.asarray(data["track_ids"]).astype(str)
            target_times = np.asarray(data["target_times"], dtype=np.float32)
            if not (len(features) == len(targets) == len(track_ids) == len(target_times)):
                raise RuntimeError(label + " row arrays are misaligned")
            for index in range(len(features)):
                key = _row_key(track_ids[index], target_times[index])
                if key in observed:
                    raise RuntimeError(label + " contains duplicate rows")
                observed[key] = targets[index]
    if set(observed) != set(reference):
        raise RuntimeError(label + " row keys do not match the manifest")
    for key, target in observed.items():
        if not np.allclose(target, reference[key], rtol=0.0, atol=1e-6):
            raise RuntimeError(label + " targets do not match the manifest")
    return len(observed)


def verify_prepared_dataset(output_root, dataset):
    name = str(dataset).lower()
    dataset_dir = Path(output_root) / name
    rows = read_manifest(dataset_dir / "manifest.csv")
    summary = {"dataset": name, "splits": {}}
    for split in ("train", "test"):
        reference = _manifest_reference(rows, split)
        global_paths = sorted((dataset_dir / "global").glob("*_" + split + "_*.npz"))
        head_paths = sorted((dataset_dir / "heads").glob("*_" + split + "_*.npz"))
        semantic_paths = sorted((dataset_dir / ("semantic_" + split)).rglob("*.npz"))
        acoustic_path = dataset_dir / ("acoustic_" + split + ".npz")
        if not global_paths:
            raise RuntimeError("global cache is missing for split=" + split)
        if not head_paths:
            raise RuntimeError("head cache is missing for split=" + split)
        if not semantic_paths:
            raise RuntimeError("semantic cache is missing for split=" + split)
        if not acoustic_path.exists():
            raise RuntimeError("acoustic cache is missing for split=" + split)
        global_count = _verify_npz_rows(global_paths, reference, "features", (3584,), "global")
        head_count = _verify_npz_rows(
            head_paths, reference, "head_features", (28, 28, HEAD_DIM), "head"
        )
        semantic_count = _verify_npz_rows(
            semantic_paths, reference, "semantic_features", (8,), "semantic"
        )
        for semantic_path in semantic_paths:
            with np.load(semantic_path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata_json"].item()))
            if int(metadata.get("trainable_parameters", -1)) != 0:
                raise RuntimeError("semantic cache was extracted from a trainable backbone")
            if bool(metadata.get("uses_test_labels", True)):
                raise RuntimeError("semantic cache used test labels")
        acoustic_count = _verify_npz_rows(
            [acoustic_path], reference, "features", (132,), "acoustic"
        )
        counts = {global_count, head_count, semantic_count, acoustic_count, len(reference)}
        if len(counts) != 1:
            raise RuntimeError("prepared feature counts differ")
        summary["splits"][split] = {
            "rows": len(reference),
            "tracks": len({key[0] for key in reference}),
            "global_shards": len(global_paths),
            "head_shards": len(head_paths),
            "semantic_shards": len(semantic_paths),
        }
    return summary


# Frozen Music Flamingo feature extractors.
def _freeze(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _resolve_device(torch, requested):
    name = str(requested)
    if name == "cpu":
        return torch.device("cpu")
    if not name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("the requested CUDA device is unavailable")
    device = torch.device(name)
    index = 0 if device.index is None else int(device.index)
    if index >= torch.cuda.device_count():
        raise RuntimeError("the requested CUDA device is unavailable")
    return torch.device("cuda:" + str(index))


def _move_inputs(torch, inputs, device, dtype):
    prepared = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if key == "input_features":
                prepared[key] = value.to(device=device, dtype=dtype)
            else:
                prepared[key] = value.to(device)
        else:
            prepared[key] = value
    return prepared


class BackboneExtractor:
    def __init__(self, model_path, device):
        self.model_path = Path(model_path)
        self.requested_device = str(device)
        self.torch = None
        self.device = None
        self.dtype = None
        self.processor = None
        self.model = None
        self.projection_modules = ()

    def load(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = _resolve_device(torch, self.requested_device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        load_options = {
            "trust_remote_code": True,
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda":
            load_options["device_map"] = {"": self.device.index or 0}
        self.model = AutoModel.from_pretrained(str(self.model_path), **load_options)
        if self.device.type == "cpu":
            self.model.to(self.device)
        _freeze(self.model)
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        self.projection_modules = tuple(
            (name, module)
            for name, module in self.model.named_modules()
            if name and PROJECTION_PATTERN in name.lower()
        )
        if len(self.projection_modules) != 28:
            raise RuntimeError("Music Flamingo must expose 28 attention projections")
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Music Flamingo is not frozen")

    def _inputs(self, audio):
        text = [str(self.processor.audio_token) + "\n" + BASE_PROMPT]
        inputs = self.processor(
            text=text,
            audio=[audio],
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt",
        )
        return _move_inputs(self.torch, inputs, self.device, self.dtype)

    def extract(self, row):
        if self.model is None:
            self.load()
        audio = load_music_segment(row["audio_path"], row["start"], row["end"])
        if audio.shape != (SEGMENT_SAMPLES,):
            raise ValueError("Music Flamingo requires an exact 15-second segment")
        captures = {}
        handles = []

        def make_hook(name):
            def hook(_module, inputs):
                if not inputs or not self.torch.is_tensor(inputs[0]):
                    return
                value = inputs[0]
                if value.ndim == 3:
                    value = value[:, -1, :]
                captures[name] = value.detach().cpu()

            return hook

        try:
            for name, module in self.projection_modules:
                handles.append(module.register_forward_pre_hook(make_hook(name)))
            with self.torch.inference_mode():
                output = self.model(
                    **self._inputs(audio),
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        if output.hidden_states is None:
            raise RuntimeError("Music Flamingo did not return hidden states")
        missing = [name for name, _module in self.projection_modules if name not in captures]
        if missing:
            raise RuntimeError("attention projection was not captured: " + missing[0])
        head_flat = self.torch.stack(
            [captures[name] for name, _module in self.projection_modules], dim=1
        ).squeeze(0)
        if head_flat.shape != (28, 3584):
            raise RuntimeError("captured attention projections have the wrong shape")
        heads = head_flat.reshape(28, 28, HEAD_DIM).float().numpy()
        global_feature = output.hidden_states[-1][:, -1, :].detach().float().cpu().numpy()[0]
        return global_feature, heads


def _semantic_values(real_logits, null_logits):
    real = np.asarray(real_logits, dtype=np.float64)
    null = np.asarray(null_logits, dtype=np.float64)
    calibrated = real - null[None, :, :]
    prompt_groups = np.empty((len(real), 2, 5), dtype=np.float64)
    for group_index, adjectives in enumerate(ADJECTIVE_GROUPS.values()):
        indices = [CANDIDATE_ADJECTIVES.index(word) for word in adjectives]
        values = calibrated[:, :, indices]
        maximum = np.max(values, axis=2, keepdims=True)
        prompt_groups[:, :, group_index] = np.squeeze(
            maximum + np.log(np.mean(np.exp(values - maximum), axis=2, keepdims=True)),
            axis=2,
        )
    prompt_valence = 0.5 * (
        prompt_groups[:, :, 0]
        - prompt_groups[:, :, 2]
        + prompt_groups[:, :, 1]
        - prompt_groups[:, :, 3]
    )
    groups = np.mean(prompt_groups, axis=1)
    valence_low = groups[:, 0] - groups[:, 2]
    valence_high = groups[:, 1] - groups[:, 3]
    balanced = 0.5 * (valence_low + valence_high)
    disagreement = np.abs(valence_low - valence_high)
    arousal = 0.5 * (groups[:, 1] + groups[:, 3] - groups[:, 0] - groups[:, 2])
    shifted = groups - np.max(groups, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1) / np.log(5.0)
    features = np.column_stack(
        (
            valence_low,
            valence_high,
            balanced,
            disagreement,
            arousal,
            groups[:, 4],
            entropy,
            balanced * (1.0 - entropy),
        )
    )
    return features.astype(np.float32), prompt_valence.astype(np.float32)


class SemanticExtractor:
    def __init__(self, model_path, device):
        self.model_path = Path(model_path)
        self.requested_device = str(device)
        self.torch = None
        self.device = None
        self.dtype = None
        self.processor = None
        self.model = None
        self.token_ids = None
        self.null_logits = None
        self.null_counts = None
        self.null_pool_dimension = None

    def load(self):
        import torch
        from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

        self.torch = torch
        self.device = _resolve_device(torch, self.requested_device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        options = {
            "trust_remote_code": True,
            "dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda":
            options["device_map"] = {"": self.device.index or 0}
        self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
            str(self.model_path), **options
        )
        if self.device.type == "cpu":
            self.model.to(self.device)
        _freeze(self.model)
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )
        self.token_ids = {}
        used = set()
        for word in CANDIDATE_ADJECTIVES:
            encoded = self.processor.tokenizer.encode(" " + word, add_special_tokens=False)
            if len(encoded) != 1 or int(encoded[0]) in used:
                raise ValueError("semantic adjectives must map to unique single tokens")
            self.token_ids[word] = int(encoded[0])
            used.add(int(encoded[0]))
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Music Flamingo is not frozen")

    def _forward(self, audio, prompt):
        text = [str(self.processor.audio_token) + "\n" + str(prompt)]
        inputs = self.processor(
            text=text,
            audio=[audio],
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = _move_inputs(self.torch, inputs, self.device, self.dtype)
        if self.device.type == "cuda":
            precision = self.torch.autocast(device_type="cuda", dtype=self.dtype)
        else:
            from contextlib import nullcontext

            precision = nullcontext()
        with self.torch.inference_mode(), precision:
            output = self.model(
                **inputs,
                logits_to_keep=1,
                return_dict=True,
                use_cache=False,
            )
        logits = output.logits
        if logits.ndim == 3:
            logits = logits[:, 0, :]
        indices = self.torch.tensor(
            [self.token_ids[word] for word in CANDIDATE_ADJECTIVES],
            dtype=self.torch.long,
            device=logits.device,
        )
        selected = logits.index_select(-1, indices).detach().float().cpu().numpy()
        hidden = output.audio_hidden_states
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        if hidden.ndim != 3 or hidden.shape[1] == 0:
            raise RuntimeError("Music Flamingo audio hidden states have the wrong shape")
        pool = (
            self.torch.cat(
                (
                    self.torch.mean(hidden, dim=1),
                    self.torch.std(hidden, dim=1, unbiased=False),
                ),
                dim=1,
            )
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        return selected, pool, int(hidden.shape[1])

    def _ensure_null_reference(self):
        if self.null_logits is not None:
            return
        zero_audio = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        outputs = [self._forward(zero_audio, prompt) for prompt in PROMPT_TEMPLATES]
        self.null_logits = np.stack([item[0][0] for item in outputs], axis=0).astype(np.float32)
        self.null_counts = tuple(item[2] for item in outputs)
        dimensions = {item[1].shape[1] for item in outputs}
        if self.null_logits.shape != (2, 14) or len(dimensions) != 1:
            raise RuntimeError("semantic null reference has the wrong shape")
        self.null_pool_dimension = int(next(iter(dimensions)))

    def metadata(self, split):
        self._ensure_null_reference()
        null_hash = hashlib.sha256(np.asarray(self.null_logits, dtype="<f4").tobytes()).hexdigest()
        total = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "schema_version": 1,
            "feature_protocol": "vasr_v_semantic_probe",
            "source_split": str(split),
            "semantic_feature_names": list(SEMANTIC_FEATURE_NAMES),
            "adjective_groups": {name: list(words) for name, words in ADJECTIVE_GROUPS.items()},
            "prompts": list(PROMPT_TEMPLATES),
            "adjective_token_ids": dict(self.token_ids),
            "null_logits_sha256": null_hash,
            "model_path": str(self.model_path),
            "audio_pool_dimension": int(self.null_pool_dimension),
            "total_parameters": int(total),
            "trainable_parameters": 0,
            "model_frozen": True,
            "uses_external_text_metadata": False,
            "uses_test_labels": False,
            "uses_test_features_for_selection": False,
            "uses_test_labels_for_selection": False,
        }

    def extract(self, row):
        if self.model is None:
            self.load()
        self._ensure_null_reference()
        audio = load_music_segment(row["audio_path"], row["start"], row["end"])
        if audio.shape != (SEGMENT_SAMPLES,):
            raise ValueError("semantic extraction requires an exact 15-second segment")
        outputs = [self._forward(audio, prompt) for prompt in PROMPT_TEMPLATES]
        counts = tuple(item[2] for item in outputs)
        if counts != self.null_counts:
            raise RuntimeError("real and null semantic audio token counts differ")
        logits = np.stack([item[0][0] for item in outputs], axis=0)[None, :, :]
        features, prompt_valence = _semantic_values(logits, self.null_logits)
        audio_pool = np.asarray(outputs[0][1], dtype=np.float32)
        if audio_pool.shape[1] != self.null_pool_dimension:
            raise RuntimeError("semantic audio pool dimension changed")
        return features[0], audio_pool[0], prompt_valence[0]


# Dataset-level preparation pipeline.
def _dataset_dir(output_root, dataset):
    return Path(output_root) / str(dataset).lower()


def _manifest_path(output_root, dataset):
    return _dataset_dir(output_root, dataset) / "manifest.csv"


def _selected_rows(output_root, dataset, split, limit_rows=0):
    rows = [
        row for row in read_manifest(_manifest_path(output_root, dataset)) if row["split"] == split
    ]
    if int(limit_rows) > 0:
        rows = rows[: int(limit_rows)]
    if not rows:
        raise ValueError("manifest has no rows for split=" + str(split))
    return rows


def _backbone_paths(dataset_dir, dataset, split, shard_index):
    prefix = str(dataset).lower()
    global_path = (
        dataset_dir
        / "global"
        / (prefix + "_global_" + split + "_" + format(shard_index, "06d") + ".npz")
    )
    head_path = (
        dataset_dir
        / "heads"
        / (prefix + "_heads_" + split + "_" + format(shard_index, "06d") + ".npz")
    )
    return global_path, head_path


def _semantic_path(dataset_dir, dataset, split, shard_index):
    base = dataset_dir / ("semantic_" + split)
    if str(dataset).lower() == "deam" and split == "train":
        base = base / ("part" + format(shard_index % 10, "02d"))
    return base / (
        str(dataset).lower() + "_semantic_" + split + "_" + format(shard_index, "06d") + ".npz"
    )


def _pair_is_ready(global_path, head_path):
    marker = Path(global_path).with_suffix(".ready.json")
    if not (Path(global_path).exists() and Path(head_path).exists() and marker.exists()):
        return False
    content = json.loads(marker.read_text(encoding="utf-8"))
    return (
        content.get("global_filename") == Path(global_path).name
        and content.get("head_filename") == Path(head_path).name
    )


def prepare_backbone(
    output_root,
    dataset,
    split,
    model_path,
    device="cuda:0",
    shard_size=128,
    lane_count=1,
    lane_index=0,
    resume=False,
    limit_rows=0,
):
    rows = _selected_rows(output_root, dataset, split, limit_rows)
    slices = shard_slices(len(rows), shard_size, lane_count, lane_index)
    dataset_dir = _dataset_dir(output_root, dataset)
    pending = []
    for shard_index, start, end in slices:
        global_path, head_path = _backbone_paths(dataset_dir, dataset, split, shard_index)
        if resume and _pair_is_ready(global_path, head_path):
            continue
        pending.append((shard_index, start, end, global_path, head_path))
    if not pending:
        print("No backbone shards require extraction")
        return
    extractor = BackboneExtractor(model_path, device)
    extractor.load()
    for shard_index, start, end, global_path, head_path in pending:
        shard_rows = rows[start:end]
        global_values = []
        head_values = []
        started = time.time()
        for position, row in enumerate(shard_rows, start=1):
            global_feature, heads = extractor.extract(row)
            global_values.append(global_feature)
            head_values.append(heads)
            if position % 16 == 0 or position == len(shard_rows):
                print(
                    "backbone "
                    + split
                    + " shard="
                    + str(shard_index)
                    + " rows="
                    + str(position)
                    + "/"
                    + str(len(shard_rows)),
                    flush=True,
                )
        write_backbone_shard_pair(
            global_path,
            head_path,
            shard_rows,
            np.stack(global_values, axis=0),
            np.stack(head_values, axis=0),
        )
        print(
            "wrote backbone shard "
            + str(shard_index)
            + " in "
            + format(time.time() - started, ".1f")
            + " seconds",
            flush=True,
        )


def prepare_semantic(
    output_root,
    dataset,
    split,
    model_path,
    device="cuda:0",
    shard_size=128,
    lane_count=1,
    lane_index=0,
    resume=False,
    limit_rows=0,
):
    rows = _selected_rows(output_root, dataset, split, limit_rows)
    slices = shard_slices(len(rows), shard_size, lane_count, lane_index)
    dataset_dir = _dataset_dir(output_root, dataset)
    pending = []
    for shard_index, start, end in slices:
        output = _semantic_path(dataset_dir, dataset, split, shard_index)
        if resume and output.exists():
            continue
        pending.append((shard_index, start, end, output))
    if not pending:
        print("No semantic shards require extraction")
        return
    extractor = SemanticExtractor(model_path, device)
    extractor.load()
    metadata = extractor.metadata(split)
    for shard_index, start, end, output in pending:
        shard_rows = rows[start:end]
        semantic_values = []
        audio_pool_values = []
        prompt_values = []
        started = time.time()
        for position, row in enumerate(shard_rows, start=1):
            semantic, audio_pool, prompt_valence = extractor.extract(row)
            semantic_values.append(semantic)
            audio_pool_values.append(audio_pool)
            prompt_values.append(prompt_valence)
            if position % 16 == 0 or position == len(shard_rows):
                print(
                    "semantic "
                    + split
                    + " shard="
                    + str(shard_index)
                    + " rows="
                    + str(position)
                    + "/"
                    + str(len(shard_rows)),
                    flush=True,
                )
        shard_metadata = dict(metadata)
        shard_metadata.update(
            {
                "row_count": len(shard_rows),
                "track_count": len({row["track_id"] for row in shard_rows}),
                "shard_index": shard_index,
                "shard_size": int(shard_size),
            }
        )
        write_semantic_shard(
            output,
            shard_rows,
            np.stack(semantic_values, axis=0),
            np.stack(audio_pool_values, axis=0),
            np.stack(prompt_values, axis=0),
            shard_metadata,
        )
        print(
            "wrote semantic shard "
            + str(shard_index)
            + " in "
            + format(time.time() - started, ".1f")
            + " seconds",
            flush=True,
        )


def prepare_acoustic(output_root, dataset, split, device="cuda:0"):
    rows = _selected_rows(output_root, dataset, split)
    by_track = {}
    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row)
    output_rows = []
    feature_blocks = []
    for track_id in sorted(by_track, key=_track_sort_key):
        track_rows = sorted(by_track[track_id], key=lambda row: row["target_time"])
        waveform = decode_audio(track_rows[0]["audio_path"], AUDIO_SAMPLE_RATE)
        times = np.asarray([row["target_time"] for row in track_rows], dtype=np.float32)
        required = round(float(np.max(times)) * AUDIO_SAMPLE_RATE)
        missing = required - len(waveform)
        if missing > round(0.5 * AUDIO_SAMPLE_RATE):
            raise ValueError("audio exceeds the terminal padding tolerance")
        if missing > 0:
            waveform = np.pad(waveform, (0, missing), mode="constant")
        features = extract_acoustic_features(
            waveform,
            AUDIO_SAMPLE_RATE,
            times,
            device=device,
        )
        feature_blocks.append(features)
        output_rows.extend(track_rows)
        print(
            "acoustic " + split + " track=" + track_id + " rows=" + str(len(track_rows)),
            flush=True,
        )
    output = _dataset_dir(output_root, dataset) / ("acoustic_" + split + ".npz")
    write_acoustic_cache(
        output,
        output_rows,
        np.concatenate(feature_blocks, axis=0),
        acoustic_feature_names(),
    )
    return output


def _release_cuda():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _stage_splits(value):
    if value == "all":
        return ("train", "test")
    return (str(value),)


# Command-line interface.
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare all HEVA feature caches from official audio and labels."
    )
    parser.add_argument("--dataset", choices=("deam", "pmemo"), required=True)
    parser.add_argument(
        "--stage",
        choices=("manifest", "backbone", "semantic", "acoustic", "verify", "all"),
        required=True,
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--backbone")
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--acoustic-device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--lane-count", type=int, default=1)
    parser.add_argument("--lane-index", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = _manifest_path(args.output_root, args.dataset)
    if args.stage in ("manifest", "all"):
        build_manifest(
            args.dataset,
            args.dataset_root,
            manifest_path,
            split_root=args.split_root,
        )
        rows = read_manifest(manifest_path)
        counts = {
            split: len([row for row in rows if row["split"] == split])
            for split in ("train", "test")
        }
        print("wrote manifest " + str(manifest_path) + " " + json.dumps(counts))
    if args.stage in ("backbone", "semantic", "acoustic", "verify") and not manifest_path.exists():
        raise FileNotFoundError("run the manifest stage first: " + str(manifest_path))
    if args.stage in ("backbone", "semantic", "all") and not args.backbone:
        raise ValueError("--backbone is required for Music Flamingo extraction")

    if args.stage in ("acoustic", "all"):
        for split in _stage_splits(args.split):
            prepare_acoustic(
                args.output_root,
                args.dataset,
                split,
                device=args.acoustic_device,
            )
    if args.stage in ("backbone", "all"):
        for split in _stage_splits(args.split):
            prepare_backbone(
                args.output_root,
                args.dataset,
                split,
                args.backbone,
                device=args.device,
                shard_size=args.shard_size,
                lane_count=args.lane_count,
                lane_index=args.lane_index,
                resume=args.resume,
                limit_rows=args.limit_rows,
            )
        _release_cuda()
    if args.stage in ("semantic", "all"):
        for split in _stage_splits(args.split):
            prepare_semantic(
                args.output_root,
                args.dataset,
                split,
                args.backbone,
                device=args.device,
                shard_size=args.shard_size,
                lane_count=args.lane_count,
                lane_index=args.lane_index,
                resume=args.resume,
                limit_rows=args.limit_rows,
            )
        _release_cuda()
    if args.stage in ("verify", "all"):
        summary = verify_prepared_dataset(args.output_root, args.dataset)
        audit_path = _dataset_dir(args.output_root, args.dataset) / "preparation.json"
        audit_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
