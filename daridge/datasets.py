import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from daridge.features import (
    SemanticFeatureMatrix,
    align_acoustic_rows,
    load_acoustic_features,
    load_named_split_point_feature_shards,
    load_selected_point_head_features,
    load_semantic_feature_shard,
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_content_group_map(manifest_path):
    with Path(manifest_path).open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))

    track_paths = {}
    track_splits = {}
    for row in rows:
        track_id = str(row["track_id"])
        audio_path = str(Path(row["audio_path"]).resolve())
        split = str(row["split"])
        if track_id in track_paths and track_paths[track_id] != audio_path:
            raise RuntimeError("one track ID points to multiple audio files")
        if track_id in track_splits and track_splits[track_id] != split:
            raise RuntimeError("one track ID appears in multiple splits")
        track_paths[track_id] = audio_path
        track_splits[track_id] = split

    # Hash audio content so duplicate files share one group.
    path_hashes = {}
    track_to_group = {}
    for track_id, audio_path in sorted(track_paths.items()):
        if audio_path not in path_hashes:
            path_hashes[audio_path] = _sha256(audio_path)
        track_to_group[track_id] = path_hashes[audio_path]

    grouped_tracks = {}
    for track_id, digest in track_to_group.items():
        grouped_tracks.setdefault(digest, []).append(track_id)
    duplicate_groups = sorted(
        sorted(track_ids) for track_ids in grouped_tracks.values() if len(track_ids) > 1
    )
    train_groups = {
        track_to_group[track_id] for track_id, split in track_splits.items() if split == "train"
    }
    test_groups = {
        track_to_group[track_id] for track_id, split in track_splits.items() if split == "test"
    }
    # Reject duplicate audio content across official train and test splits.
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError("audio content overlap between train and test")

    audit = {
        "track_count": len(track_to_group),
        "content_group_count": len(grouped_tracks),
        "train_content_group_count": len(train_groups),
        "test_content_group_count": len(test_groups),
        "cross_split_content_overlap_count": 0,
        "duplicate_content_groups": duplicate_groups,
        "duplicate_content_group_count": len(duplicate_groups),
    }
    return track_to_group, audit


def groups_for_rows(track_ids, track_to_group):
    values = []
    for track_id in np.asarray(track_ids).astype(str):
        if track_id not in track_to_group:
            raise KeyError("track ID has no content group: " + track_id)
        values.append(track_to_group[track_id])
    return np.asarray(values).astype(str)


@dataclass(frozen=True)
class DatasetBundle:
    train_base: np.ndarray | None
    train_semantic: np.ndarray | None
    train_head: np.ndarray | None
    train_acoustic: np.ndarray | None
    train_targets: np.ndarray
    train_track_ids: np.ndarray
    train_group_ids: np.ndarray
    train_target_times: np.ndarray
    test_base: np.ndarray | None
    test_semantic: np.ndarray | None
    test_head: np.ndarray | None
    test_acoustic: np.ndarray | None
    test_targets: np.ndarray
    test_track_ids: np.ndarray
    test_target_times: np.ndarray
    grouping_audit: dict


def _row_keys(track_ids, target_times):
    ids = np.asarray(track_ids).astype(str).reshape(-1)
    times = np.asarray(target_times, dtype=np.float32).reshape(-1)
    if len(ids) != len(times):
        raise ValueError("track IDs and target times must align")
    return [(ids[index], float(times[index])) for index in range(len(ids))]


# Align cached modalities by the exact track-time key.
def align_semantic_to_global(global_track_ids, global_times, semantic):
    global_keys = _row_keys(global_track_ids, global_times)
    semantic_keys = _row_keys(
        semantic.track_ids,
        semantic.target_times,
    )
    if len(set(global_keys)) != len(global_keys):
        raise ValueError("global cache contains duplicate track-time keys")
    if len(set(semantic_keys)) != len(semantic_keys):
        raise ValueError("semantic cache contains duplicate track-time keys")
    if set(global_keys) != set(semantic_keys):
        raise ValueError("global and semantic row keys differ")
    positions = {key: index for index, key in enumerate(semantic_keys)}
    order = np.asarray(
        [positions[key] for key in global_keys],
        dtype=np.int64,
    )
    return SemanticFeatureMatrix(
        semantic_features=semantic.semantic_features[order],
        audio_pool=semantic.audio_pool[order],
        prompt_valence=semantic.prompt_valence[order],
        targets=semantic.targets[order],
        track_ids=semantic.track_ids[order],
        target_times=semantic.target_times[order],
        splits=semantic.splits[order],
        metadata=dict(semantic.metadata),
    )


def _semantic_signature(metadata):
    return (
        str(metadata.get("feature_protocol")),
        tuple(metadata.get("semantic_feature_names", [])),
        tuple(metadata.get("prompts", [])),
        str(metadata.get("null_logits_sha256")),
        int(metadata.get("audio_pool_dimension", 0)),
        int(metadata.get("trainable_parameters", -1)),
    )


def load_semantic_directories(feature_dirs, split):
    paths = []
    for directory in feature_dirs:
        directory_paths = sorted(Path(directory).glob("*.npz"))
        if not directory_paths:
            raise ValueError("no semantic shards found in " + str(directory))
        paths.extend(directory_paths)
    shards = [load_semantic_feature_shard(path) for path in paths]
    # Require one frozen semantic extraction protocol across all shards.
    signature = _semantic_signature(shards[0].metadata)
    for shard in shards:
        if _semantic_signature(shard.metadata) != signature:
            raise ValueError("semantic shard protocols differ")
        if set(shard.splits.astype(str)) != {str(split)}:
            raise ValueError("semantic shard rows have the wrong split")
        if int(shard.metadata.get("trainable_parameters", -1)) != 0:
            raise ValueError("semantic features came from a trainable backbone")
        if shard.metadata.get("uses_test_labels", False):
            raise ValueError("semantic extraction used test labels")

    return SemanticFeatureMatrix(
        semantic_features=np.concatenate(
            [item.semantic_features for item in shards],
            axis=0,
        ).astype(np.float32),
        audio_pool=np.concatenate(
            [item.audio_pool for item in shards],
            axis=0,
        ).astype(np.float32),
        prompt_valence=np.concatenate(
            [item.prompt_valence for item in shards],
            axis=0,
        ).astype(np.float32),
        targets=np.concatenate(
            [item.targets for item in shards],
            axis=0,
        ).astype(np.float32),
        track_ids=np.concatenate(
            [item.track_ids for item in shards],
            axis=0,
        ).astype(str),
        target_times=np.concatenate(
            [item.target_times for item in shards],
            axis=0,
        ).astype(np.float32),
        splits=np.concatenate(
            [item.splits for item in shards],
            axis=0,
        ).astype(str),
        metadata={
            **dict(shards[0].metadata),
            "loaded_shard_count": len(shards),
        },
    )


def load_aligned_split(
    global_feature_dir,
    semantic_feature_dirs,
    split,
    counterpart_dir=None,
):
    global_matrix = load_named_split_point_feature_shards(
        global_feature_dir,
        split,
        counterpart_dir=counterpart_dir,
    )
    semantic = load_semantic_directories(semantic_feature_dirs, split)
    semantic = align_semantic_to_global(
        global_matrix.track_ids,
        global_matrix.target_times,
        semantic,
    )
    if not np.allclose(
        global_matrix.targets,
        semantic.targets,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("global and semantic labels differ")
    return global_matrix, semantic


def assert_same_rows(reference, candidate, name):
    if not np.array_equal(
        np.asarray(reference.track_ids).astype(str),
        np.asarray(candidate.track_ids).astype(str),
    ):
        raise ValueError(name + " track IDs do not align")
    if not np.allclose(
        reference.target_times,
        candidate.target_times,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(name + " target times do not align")
    if not np.allclose(
        reference.targets,
        candidate.targets,
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError(name + " targets do not align")


def load_deam(dataset_config, heads, branch="both"):
    paths = dataset_config["paths"]
    need_valence = branch in ("both", "valence")
    need_arousal = branch in ("both", "arousal")

    train_global = test_global = None
    train_semantic = test_semantic = None
    train_head = test_head = None
    train_acoustic = test_acoustic = None

    if need_valence:
        train_global, train_semantic = load_aligned_split(
            paths["global_features"],
            paths["semantic_train"],
            "train",
        )
        test_global, test_semantic = load_aligned_split(
            paths["global_features"],
            paths["semantic_test"],
            "test",
        )

    if need_arousal:
        coordinates = [(int(item["layer"]), int(item["head"])) for item in heads]
        train_head = load_selected_point_head_features(
            paths["head_features"],
            split="train",
            heads=coordinates,
        )
        test_head = load_selected_point_head_features(
            paths["head_features"],
            split="test",
            heads=coordinates,
        )
        train_acoustic = align_acoustic_rows(
            train_head.track_ids,
            train_head.target_times,
            train_head.targets,
            load_acoustic_features(paths["acoustic_train"], split="train"),
        )
        test_acoustic = align_acoustic_rows(
            test_head.track_ids,
            test_head.target_times,
            test_head.targets,
            load_acoustic_features(paths["acoustic_test"], split="test"),
        )
    if need_valence and need_arousal:
        assert_same_rows(train_global, train_head, "DEAM train heads")
        assert_same_rows(test_global, test_head, "DEAM test heads")

    # DEAM uses song identifiers as leakage-control groups.
    train_reference = train_global if need_valence else train_head
    test_reference = test_global if need_valence else test_head
    train_tracks = np.asarray(train_reference.track_ids).astype(str)
    test_tracks = np.asarray(test_reference.track_ids).astype(str)
    track_overlap = set(train_tracks) & set(test_tracks)
    if track_overlap:
        raise RuntimeError("DEAM track overlap between train and test")
    return DatasetBundle(
        train_base=None if train_global is None else train_global.features,
        train_semantic=(None if train_semantic is None else train_semantic.semantic_features),
        train_head=None if train_head is None else train_head.features,
        train_acoustic=(None if train_acoustic is None else train_acoustic.features),
        train_targets=train_reference.targets,
        train_track_ids=train_tracks,
        train_group_ids=train_tracks.copy(),
        train_target_times=train_reference.target_times,
        test_base=None if test_global is None else test_global.features,
        test_semantic=(None if test_semantic is None else test_semantic.semantic_features),
        test_head=None if test_head is None else test_head.features,
        test_acoustic=(None if test_acoustic is None else test_acoustic.features),
        test_targets=test_reference.targets,
        test_track_ids=test_tracks,
        test_target_times=test_reference.target_times,
        grouping_audit={
            "method": "track_id",
            "train_group_count": len(np.unique(train_tracks)),
            "test_group_count": len(np.unique(test_tracks)),
            "cross_split_group_overlap_count": len(track_overlap),
        },
    )


def load_pmemo(dataset_config, heads, branch="both"):
    paths = dataset_config["paths"]
    need_valence = branch in ("both", "valence")
    need_arousal = branch in ("both", "arousal")
    train_global = test_global = None
    train_semantic = test_semantic = None
    train_head = test_head = None
    train_acoustic = test_acoustic = None

    if need_valence:
        train_global, train_semantic = load_aligned_split(
            paths["global_features"],
            paths["semantic_train"],
            "train",
            counterpart_dir=paths["head_features"],
        )
        test_global, test_semantic = load_aligned_split(
            paths["global_features"],
            paths["semantic_test"],
            "test",
            counterpart_dir=paths["head_features"],
        )
    if need_arousal:
        coordinates = [(int(item["layer"]), int(item["head"])) for item in heads]
        train_head = load_selected_point_head_features(
            paths["head_features"],
            split="train",
            heads=coordinates,
            counterpart_dir=paths["global_features"],
        )
        test_head = load_selected_point_head_features(
            paths["head_features"],
            split="test",
            heads=coordinates,
            counterpart_dir=paths["global_features"],
        )
        train_acoustic = align_acoustic_rows(
            train_head.track_ids,
            train_head.target_times,
            train_head.targets,
            load_acoustic_features(paths["acoustic_train"], split="train"),
        )
        test_acoustic = align_acoustic_rows(
            test_head.track_ids,
            test_head.target_times,
            test_head.targets,
            load_acoustic_features(paths["acoustic_test"], split="test"),
        )
    if need_valence and need_arousal:
        assert_same_rows(train_global, train_head, "PMEmo train heads")
        assert_same_rows(test_global, test_head, "PMEmo test heads")

    # PMEmo groups tracks by audio hash to catch duplicate files with new IDs.
    track_to_group, audit = build_content_group_map(paths["manifest"])
    train_reference = train_global if need_valence else train_head
    test_reference = test_global if need_valence else test_head
    train_tracks = np.asarray(train_reference.track_ids).astype(str)
    test_tracks = np.asarray(test_reference.track_ids).astype(str)
    train_groups = groups_for_rows(train_tracks, track_to_group)
    test_groups = groups_for_rows(test_tracks, track_to_group)
    content_overlap = set(train_groups) & set(test_groups)
    if content_overlap:
        raise RuntimeError("PMEmo content overlap between loaded train and test")
    expected = dataset_config["expected"]
    if int(audit["train_content_group_count"]) != int(expected["train_content_groups"]):
        raise RuntimeError("PMEmo train content-group count changed")
    if int(audit["test_content_group_count"]) != int(expected["test_content_groups"]):
        raise RuntimeError("PMEmo test content-group count changed")
    if len(np.unique(train_groups)) != int(expected["train_content_groups"]):
        raise RuntimeError("loaded PMEmo train content-group count changed")
    if len(np.unique(test_groups)) != int(expected["test_content_groups"]):
        raise RuntimeError("loaded PMEmo test content-group count changed")

    return DatasetBundle(
        train_base=None if train_global is None else train_global.features,
        train_semantic=(None if train_semantic is None else train_semantic.semantic_features),
        train_head=None if train_head is None else train_head.features,
        train_acoustic=(None if train_acoustic is None else train_acoustic.features),
        train_targets=train_reference.targets,
        train_track_ids=train_tracks,
        train_group_ids=train_groups,
        train_target_times=train_reference.target_times,
        test_base=None if test_global is None else test_global.features,
        test_semantic=(None if test_semantic is None else test_semantic.semantic_features),
        test_head=None if test_head is None else test_head.features,
        test_acoustic=(None if test_acoustic is None else test_acoustic.features),
        test_targets=test_reference.targets,
        test_track_ids=test_tracks,
        test_target_times=test_reference.target_times,
        grouping_audit={
            "method": "audio_file_sha256",
            **audit,
            "loaded_train_content_group_count": len(np.unique(train_groups)),
            "loaded_test_content_group_count": len(np.unique(test_groups)),
            "loaded_cross_split_content_overlap_count": len(content_overlap),
        },
    )
