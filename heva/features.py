import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BackboneShardPair:
    global_path: Path
    head_path: Path


def resolve_backbone_shard_pairs(global_dir, head_dir):
    global_dir = Path(global_dir)
    head_dir = Path(head_dir)
    marker_paths = sorted(global_dir.glob("*.ready.json"))
    if not marker_paths:
        raise ValueError("no backbone ready markers found in " + str(global_dir))

    pairs = []
    for marker_path in marker_paths:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        global_path = global_dir / marker["global_filename"]
        head_path = head_dir / marker["head_filename"]
        if not global_path.exists() or not head_path.exists():
            raise RuntimeError("backbone shard pair is incomplete")
        pairs.append(
            BackboneShardPair(
                global_path=global_path,
                head_path=head_path,
            )
        )
    return pairs


@dataclass(frozen=True)
class PointFeatureMatrix:
    features: np.ndarray
    targets: np.ndarray
    track_ids: np.ndarray
    target_times: np.ndarray
    splits: np.ndarray


def _load_shard(path):
    with np.load(path, allow_pickle=False) as data:
        return PointFeatureMatrix(
            features=np.asarray(data["features"], dtype=np.float32),
            targets=np.asarray(data["targets"], dtype=np.float32),
            track_ids=np.asarray(data["track_ids"]).astype(str),
            target_times=np.asarray(
                data["target_times"],
                dtype=np.float32,
            ),
            splits=np.asarray(data["splits"]).astype(str),
        )


def _load_split(feature_dir, paths, split):
    features = []
    targets = []
    track_ids = []
    target_times = []
    splits = []

    for path in paths:
        shard = _load_shard(path)
        mask = shard.splits == split
        if not mask.any():
            continue
        features.append(shard.features[mask])
        targets.append(shard.targets[mask])
        track_ids.append(shard.track_ids[mask])
        target_times.append(shard.target_times[mask])
        splits.append(shard.splits[mask])

    if not features:
        raise ValueError(
            "no point feature rows found for split=" + split + " in " + str(feature_dir)
        )

    return PointFeatureMatrix(
        features=np.concatenate(features, axis=0).astype(np.float32),
        targets=np.concatenate(targets, axis=0).astype(np.float32),
        track_ids=np.concatenate(track_ids, axis=0).astype(str),
        target_times=np.concatenate(target_times, axis=0).astype(np.float32),
        splits=np.concatenate(splits, axis=0).astype(str),
    )


def load_named_split_point_feature_shards(
    feature_dir,
    split,
    counterpart_dir=None,
):
    feature_dir = Path(feature_dir)
    split = str(split)
    if counterpart_dir is None:
        paths = sorted(feature_dir.glob("*_" + split + "_*.npz"))
    else:
        pairs = resolve_backbone_shard_pairs(
            global_dir=feature_dir,
            head_dir=counterpart_dir,
        )
        paths = [pair.global_path for pair in pairs if "_" + split + "_" in pair.global_path.name]

    if not paths:
        raise ValueError(
            "no point feature shards found for split=" + split + " in " + str(feature_dir)
        )
    return _load_split(feature_dir, paths, split)


@dataclass(frozen=True)
class SemanticFeatureMatrix:
    semantic_features: np.ndarray
    audio_pool: np.ndarray
    prompt_valence: np.ndarray
    targets: np.ndarray
    track_ids: np.ndarray
    target_times: np.ndarray
    splits: np.ndarray
    metadata: dict


def load_semantic_feature_shard(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        return SemanticFeatureMatrix(
            semantic_features=np.asarray(
                data["semantic_features"],
                dtype=np.float32,
            ),
            audio_pool=np.asarray(data["audio_pool"], dtype=np.float32),
            prompt_valence=np.asarray(
                data["prompt_valence"],
                dtype=np.float32,
            ),
            targets=np.asarray(data["targets"], dtype=np.float32),
            track_ids=np.asarray(data["track_ids"]).astype(str),
            target_times=np.asarray(
                data["target_times"],
                dtype=np.float32,
            ),
            splits=np.asarray(data["splits"]).astype(str),
            metadata=metadata,
        )


@dataclass(frozen=True)
class AcousticFeatureMatrix:
    features: np.ndarray
    feature_names: tuple
    targets: np.ndarray
    track_ids: np.ndarray
    target_times: np.ndarray
    splits: np.ndarray


def _row_keys(track_ids, target_times):
    track_ids = np.asarray(track_ids).astype(str)
    target_times = np.asarray(target_times, dtype=np.float64)
    return list(zip(track_ids.tolist(), np.round(target_times, 6).tolist()))


def load_acoustic_features(path, split):
    with np.load(path, allow_pickle=False) as data:
        splits = np.asarray(data["splits"]).astype(str)
        mask = splits == str(split)
        if not mask.any():
            raise ValueError("no acoustic rows found for split=" + str(split))
        return AcousticFeatureMatrix(
            features=np.asarray(data["features"][mask], dtype=np.float32),
            feature_names=tuple(str(name) for name in data["feature_names"].tolist()),
            targets=np.asarray(data["targets"][mask], dtype=np.float32),
            track_ids=np.asarray(data["track_ids"][mask]).astype(str),
            target_times=np.asarray(
                data["target_times"][mask],
                dtype=np.float32,
            ),
            splits=splits[mask],
        )


# Align acoustic rows to the exact track-time order of the reference branch.
def align_acoustic_rows(
    reference_track_ids,
    reference_times,
    reference_targets,
    acoustic,
):
    reference_ids = np.asarray(reference_track_ids).astype(str)
    reference_times = np.asarray(reference_times, dtype=np.float32)
    reference_targets = np.asarray(reference_targets, dtype=np.float32)
    reference_keys = _row_keys(reference_ids, reference_times)

    cache_ids = np.asarray(acoustic.track_ids).astype(str)
    cache_times = np.asarray(acoustic.target_times, dtype=np.float32)
    cache_targets = np.asarray(acoustic.targets, dtype=np.float32)
    cache_keys = _row_keys(cache_ids, cache_times)
    cache_index = {key: index for index, key in enumerate(cache_keys)}
    if len(cache_index) != len(cache_keys):
        raise ValueError("acoustic cache contains duplicate track/time rows")

    missing = [key for key in reference_keys if key not in cache_index]
    if missing:
        raise ValueError("acoustic cache is missing track/time rows")

    indices = np.asarray(
        [cache_index[key] for key in reference_keys],
        dtype=np.int64,
    )
    aligned_targets = cache_targets[indices]
    if not np.allclose(
        aligned_targets,
        reference_targets,
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("acoustic targets do not match reference targets")

    return AcousticFeatureMatrix(
        features=np.asarray(acoustic.features[indices], dtype=np.float32),
        feature_names=tuple(acoustic.feature_names),
        targets=aligned_targets,
        track_ids=reference_ids,
        target_times=reference_times,
        splits=np.asarray(acoustic.splits)[indices].astype(str),
    )


@dataclass(frozen=True)
class PointHeadFeatureMatrix:
    head_features: np.ndarray
    targets: np.ndarray
    track_ids: np.ndarray
    target_times: np.ndarray
    splits: np.ndarray


@dataclass(frozen=True)
class SelectedPointHeadFeatureMatrix:
    features: np.ndarray
    targets: np.ndarray
    track_ids: np.ndarray
    target_times: np.ndarray
    splits: np.ndarray


def _head_paths(feature_dir, counterpart_dir):
    feature_dir = Path(feature_dir)
    if counterpart_dir is None:
        paths = sorted(feature_dir.glob("*.npz"))
        if not paths:
            raise ValueError("no head feature shards found in " + str(feature_dir))
        return paths

    pairs = resolve_backbone_shard_pairs(
        global_dir=counterpart_dir,
        head_dir=feature_dir,
    )
    return [pair.head_path for pair in pairs]


def _concat(parts, head_dtype):
    return PointHeadFeatureMatrix(
        head_features=np.concatenate(
            [part.head_features for part in parts],
            axis=0,
        ).astype(head_dtype, copy=False),
        targets=np.concatenate(
            [part.targets for part in parts],
            axis=0,
        ).astype(np.float32),
        track_ids=np.concatenate(
            [part.track_ids for part in parts],
            axis=0,
        ).astype(str),
        target_times=np.concatenate(
            [part.target_times for part in parts],
            axis=0,
        ).astype(np.float32),
        splits=np.concatenate(
            [part.splits for part in parts],
            axis=0,
        ).astype(str),
    )


def _select_evenly(indices, times, max_points):
    pairs = sorted(zip(indices, times), key=lambda item: item[1])
    ordered = [index for index, _time in pairs]
    if len(ordered) <= max_points:
        return ordered

    positions = (
        np.linspace(
            0,
            len(ordered) - 1,
            max_points,
        )
        .round()
        .astype(int)
    )
    return [ordered[int(position)] for position in np.unique(positions)]


def load_sampled_point_head_feature_shards(
    feature_dir,
    split,
    max_points_per_track,
    head_dtype=np.float16,
    counterpart_dir=None,
):
    paths = _head_paths(feature_dir, counterpart_dir)

    references = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            splits = np.asarray(data["splits"]).astype(str)
            track_ids = np.asarray(data["track_ids"]).astype(str)
            target_times = np.asarray(
                data["target_times"],
                dtype=np.float32,
            )
            for row in np.flatnonzero(splits == split):
                references.append(
                    (
                        path,
                        int(row),
                        str(track_ids[row]),
                        float(target_times[row]),
                    )
                )

    if not references:
        raise ValueError("no head feature rows found for split=" + str(split))

    by_track = {}
    for reference_index, reference in enumerate(references):
        track_id = reference[2]
        target_time = reference[3]
        if track_id not in by_track:
            by_track[track_id] = ([], [])
        by_track[track_id][0].append(reference_index)
        by_track[track_id][1].append(target_time)

    # Sample each track uniformly to avoid long-track dominance in head ranking.
    selected_reference_indices = []
    for track_id in sorted(by_track):
        indices, times = by_track[track_id]
        selected_reference_indices.extend(
            _select_evenly(
                indices,
                times,
                int(max_points_per_track),
            )
        )

    rows_by_path = defaultdict(list)
    for reference_index in selected_reference_indices:
        path, row, _track_id, _time = references[reference_index]
        rows_by_path[path].append(row)

    parts = []
    for path in sorted(rows_by_path):
        rows = np.asarray(rows_by_path[path], dtype=np.int64)
        with np.load(path, allow_pickle=False) as data:
            parts.append(
                PointHeadFeatureMatrix(
                    head_features=np.asarray(
                        data["head_features"][rows],
                        dtype=head_dtype,
                    ),
                    targets=np.asarray(
                        data["targets"][rows],
                        dtype=np.float32,
                    ),
                    track_ids=np.asarray(data["track_ids"][rows]).astype(str),
                    target_times=np.asarray(
                        data["target_times"][rows],
                        dtype=np.float32,
                    ),
                    splits=np.asarray(data["splits"][rows]).astype(str),
                )
            )
    return _concat(parts, head_dtype)


def _normalize_head_refs(heads):
    references = []
    for item in heads:
        if isinstance(item, dict):
            references.append((int(item["layer"]), int(item["head"])))
        else:
            layer, head = item
            references.append((int(layer), int(head)))
    if not references:
        raise ValueError("at least one selected head is required")
    return references


def load_selected_point_head_features(
    feature_dir,
    split,
    heads,
    counterpart_dir=None,
):
    paths = _head_paths(feature_dir, counterpart_dir)
    head_references = _normalize_head_refs(heads)

    features = []
    targets = []
    track_ids = []
    target_times = []
    splits = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            shard_splits = np.asarray(data["splits"]).astype(str)
            mask = shard_splits == split
            if not mask.any():
                continue

            head_features = np.asarray(
                data["head_features"][mask],
                dtype=np.float32,
            )
            # Preserve the frozen head order when concatenating 128-D vectors.
            selected = []
            for layer, head in head_references:
                selected.append(head_features[:, layer, head, :])

            features.append(np.concatenate(selected, axis=1).astype(np.float32))
            targets.append(np.asarray(data["targets"][mask], dtype=np.float32))
            track_ids.append(np.asarray(data["track_ids"][mask]).astype(str))
            target_times.append(
                np.asarray(
                    data["target_times"][mask],
                    dtype=np.float32,
                )
            )
            splits.append(shard_splits[mask])

    if not features:
        raise ValueError("no head feature rows found for split=" + str(split))

    return SelectedPointHeadFeatureMatrix(
        features=np.concatenate(features, axis=0).astype(np.float32),
        targets=np.concatenate(targets, axis=0).astype(np.float32),
        track_ids=np.concatenate(track_ids, axis=0).astype(str),
        target_times=np.concatenate(
            target_times,
            axis=0,
        ).astype(np.float32),
        splits=np.concatenate(splits, axis=0).astype(str),
    )
