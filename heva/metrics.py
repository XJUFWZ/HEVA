import numpy as np


def one_dimension_metrics(prediction, target):
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    if len(pred) != len(truth):
        raise ValueError("prediction and target must have the same length")
    if len(pred) == 0:
        raise ValueError("at least one point is required")

    error = pred - truth
    pred_mean = float(np.mean(pred))
    target_mean = float(np.mean(truth))
    pred_centered = pred - pred_mean
    target_centered = truth - target_mean
    pred_var = float(np.mean(pred_centered**2))
    target_var = float(np.mean(target_centered**2))
    covariance = float(np.mean(pred_centered * target_centered))

    pcc_denominator = float(np.sqrt(pred_var * target_var))
    pcc = covariance / pcc_denominator if pcc_denominator > 1e-12 else 0.0
    ccc_denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
    ccc = 2.0 * covariance / ccc_denominator if ccc_denominator > 1e-12 else 0.0
    return {
        "pcc": float(pcc),
        "ccc": float(ccc),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


# Compute each song first so every song contributes equal metric weight.
def track_macro_metrics(prediction, target, track_ids):
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(target, dtype=np.float64).reshape(-1)
    groups = np.asarray(track_ids).astype(str).reshape(-1)
    if not (len(pred) == len(truth) == len(groups)):
        raise ValueError("prediction, target and track IDs must align")

    rows = []
    for track_id in np.unique(groups):
        mask = groups == track_id
        rows.append(one_dimension_metrics(pred[mask], truth[mask]))
    if not rows:
        raise ValueError("at least one track is required")
    return {
        "pcc": float(np.mean([row["pcc"] for row in rows])),
        "ccc": float(np.mean([row["ccc"] for row in rows])),
        "rmse": float(np.mean([row["rmse"] for row in rows])),
        "n_tracks": len(rows),
    }
