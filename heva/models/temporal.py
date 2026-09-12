import numpy as np


def _ordered_rows(track_ids, target_times):
    groups = np.asarray(track_ids).astype(str).reshape(-1)
    times = np.asarray(target_times, dtype=np.float32).reshape(-1)
    if len(groups) != len(times):
        raise ValueError("track IDs and target times must align")
    for track_id in np.unique(groups):
        rows = np.flatnonzero(groups == track_id)
        order = np.argsort(times[rows], kind="stable")
        yield rows[order]


def _build_bidirectional(signal, track_ids, target_times, radius):
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    features = np.zeros((len(values), 3), dtype=np.float32)
    # Build each track independently in chronological order.
    for rows in _ordered_rows(track_ids, target_times):
        ordered_times = np.asarray(target_times, dtype=np.float32)[rows]
        if len(np.unique(ordered_times)) != len(ordered_times):
            raise ValueError("target times must be unique within each track")
        track_signal = values[rows].astype(np.float64, copy=False)
        cumulative = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                np.cumsum(track_signal),
            )
        )

        for position, row in enumerate(rows):
            current = float(track_signal[position])
            past_start = max(0, position - radius)
            if position == 0:
                past_mean = current
            else:
                past_mean = float(
                    (cumulative[position] - cumulative[past_start]) / (position - past_start)
                )
            future_stop = min(
                len(track_signal),
                position + radius + 1,
            )
            if position + 1 >= len(track_signal):
                future_mean = current
            else:
                future_mean = float(
                    (cumulative[future_stop] - cumulative[position + 1])
                    / (future_stop - position - 1)
                )
            # Exclude the current point from both context means.
            features[row] = (
                current,
                past_mean,
                future_mean,
            )
    return features


def build_temporal_features(
    signal,
    track_ids,
    target_times,
    radius=8,
):
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    groups = np.asarray(track_ids).astype(str).reshape(-1)
    times = np.asarray(target_times, dtype=np.float32).reshape(-1)
    radius_value = int(radius)
    if radius_value <= 0:
        raise ValueError("temporal radius must be positive")
    if not (len(values) == len(groups) == len(times)):
        raise ValueError("signal, track IDs and target times must align")
    if len(values) == 0:
        raise ValueError("at least one signal point is required")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)):
        raise ValueError("signal and target times must be finite")
    return _build_bidirectional(values, groups, times, radius_value)
