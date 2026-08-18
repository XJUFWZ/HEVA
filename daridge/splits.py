from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupHoldout:
    fit_mask: np.ndarray
    validation_mask: np.ndarray
    fit_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]


# Split complete content groups to prevent track-level leakage.
def group_holdout(group_ids, val_fraction, seed):
    groups = np.asarray(group_ids).astype(str).reshape(-1)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("at least three groups are required")
    fraction = float(val_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("val_fraction must be inside (0, 1)")

    random = np.random.default_rng(int(seed))
    shuffled = unique_groups.copy()
    random.shuffle(shuffled)
    validation_count = int(round(len(shuffled) * fraction))
    validation_count = min(
        max(1, validation_count),
        len(shuffled) - 1,
    )
    validation_groups = set(shuffled[:validation_count].tolist())
    fit_groups = set(shuffled[validation_count:].tolist())
    fit_mask = np.asarray(
        [value in fit_groups for value in groups],
        dtype=bool,
    )
    validation_mask = np.asarray(
        [value in validation_groups for value in groups],
        dtype=bool,
    )
    return GroupHoldout(
        fit_mask=fit_mask,
        validation_mask=validation_mask,
        fit_groups=tuple(sorted(fit_groups)),
        validation_groups=tuple(sorted(validation_groups)),
    )


# Keep each content group inside exactly one OOF validation fold.
def grouped_folds(group_ids, max_splits=5):
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(group_ids).astype(str).reshape(-1)
    unique_count = len(np.unique(groups))
    split_count = min(int(max_splits), unique_count)
    if split_count < 2:
        raise ValueError("at least two groups are required for OOF")
    splitter = GroupKFold(n_splits=split_count)
    rows = np.arange(len(groups))
    return list(splitter.split(rows, groups=groups))
