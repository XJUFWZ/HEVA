from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def require_compute_device(backend):
    backend_name = str(backend).strip().lower()
    if backend_name == "cpu":
        return torch.device("cpu")
    if backend_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("compute backend cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    raise ValueError("compute backend must be cpu or cuda")


def fit_standardized_ridge_state(
    train_x,
    train_y,
    alpha,
    device,
):
    train_array = np.asarray(train_x, dtype=np.float32)
    target_array = np.asarray(train_y, dtype=np.float32).reshape(-1)
    if train_array.ndim != 2:
        raise ValueError("Ridge features must have shape (n, d)")
    if len(train_array) != len(target_array):
        raise ValueError("Ridge features and targets must align")
    if float(alpha) <= 0.0:
        raise ValueError("Ridge alpha must be positive")

    train_tensor = torch.as_tensor(
        train_array,
        dtype=torch.float32,
        device=device,
    )
    target_tensor = torch.as_tensor(
        target_array,
        dtype=torch.float32,
        device=device,
    )
    feature_mean = train_tensor.mean(dim=0)
    centered_train = train_tensor - feature_mean
    feature_scale = torch.sqrt(torch.mean(centered_train.square(), dim=0))
    feature_scale = torch.where(
        feature_scale == 0.0,
        torch.ones_like(feature_scale),
        feature_scale,
    )
    standardized_train = centered_train / feature_scale
    target_mean = target_tensor.mean()
    centered_target = target_tensor - target_mean
    gram = standardized_train.T @ standardized_train
    right_hand_side = standardized_train.T @ centered_target
    # Solve the regularized normal equations with Cholesky factorization.
    system = gram + float(alpha) * torch.eye(
        gram.shape[0],
        dtype=torch.float32,
        device=device,
    )
    factor, info = torch.linalg.cholesky_ex(system)
    if int(info.item()) == 0:
        weights = torch.cholesky_solve(
            right_hand_side[:, None],
            factor,
        ).squeeze(1)
        return (
            feature_mean.detach().cpu().numpy(),
            feature_scale.detach().cpu().numpy(),
            weights.detach().cpu().numpy(),
            float(target_mean.item()),
        )

    # Retry in float64 only when the float32 system is unstable.
    train_tensor64 = train_tensor.to(torch.float64)
    target_tensor64 = target_tensor.to(torch.float64)
    feature_mean64 = train_tensor64.mean(dim=0)
    centered_train64 = train_tensor64 - feature_mean64
    feature_scale64 = torch.sqrt(torch.mean(centered_train64.square(), dim=0))
    feature_scale64 = torch.where(
        feature_scale64 == 0.0,
        torch.ones_like(feature_scale64),
        feature_scale64,
    )
    standardized_train64 = centered_train64 / feature_scale64
    target_mean64 = target_tensor64.mean()
    centered_target64 = target_tensor64 - target_mean64
    gram64 = standardized_train64.T @ standardized_train64
    right_hand_side64 = standardized_train64.T @ centered_target64
    system64 = gram64 + float(alpha) * torch.eye(
        gram64.shape[0],
        dtype=torch.float64,
        device=device,
    )
    factor64, info64 = torch.linalg.cholesky_ex(system64)
    if int(info64.item()) != 0:
        raise RuntimeError("Ridge linear system is not positive definite")
    weights64 = torch.cholesky_solve(
        right_hand_side64[:, None],
        factor64,
    ).squeeze(1)
    return (
        feature_mean64.detach().cpu().numpy().astype(np.float32),
        feature_scale64.detach().cpu().numpy().astype(np.float32),
        weights64.detach().cpu().numpy().astype(np.float32),
        float(target_mean64.item()),
    )


def predict_standardized_ridge(
    train_x,
    train_y,
    eval_x,
    alpha,
    device,
):
    return predict_standardized_ridge_many_alphas(
        train_x,
        train_y,
        eval_x,
        (alpha,),
        device,
    )[0]


def predict_standardized_ridge_many_alphas(
    train_x,
    train_y,
    eval_x,
    alphas,
    device,
):
    train_array = np.asarray(train_x, dtype=np.float32)
    target_array = np.asarray(train_y, dtype=np.float32).reshape(-1)
    eval_array = np.asarray(eval_x, dtype=np.float32)
    if train_array.ndim != 2 or eval_array.ndim != 2:
        raise ValueError("Ridge features must have shape (n, d)")
    if len(train_array) != len(target_array):
        raise ValueError("Ridge features and targets must align")
    if train_array.shape[1] != eval_array.shape[1]:
        raise ValueError("train and evaluation feature dimensions must match")
    alpha_values = tuple(float(value) for value in alphas)
    if not alpha_values or any(value <= 0.0 for value in alpha_values):
        raise ValueError("Ridge alphas must contain positive values")

    train_tensor = torch.as_tensor(
        train_array,
        dtype=torch.float32,
        device=device,
    )
    target_tensor = torch.as_tensor(
        target_array,
        dtype=torch.float32,
        device=device,
    )
    eval_tensor = torch.as_tensor(
        eval_array,
        dtype=torch.float32,
        device=device,
    )

    feature_mean = train_tensor.mean(dim=0)
    centered_train = train_tensor - feature_mean
    feature_scale = torch.sqrt(torch.mean(centered_train.square(), dim=0))
    feature_scale = torch.where(
        feature_scale == 0.0,
        torch.ones_like(feature_scale),
        feature_scale,
    )
    standardized_train = centered_train / feature_scale
    standardized_eval = (eval_tensor - feature_mean) / feature_scale

    target_mean = target_tensor.mean()
    centered_target = target_tensor - target_mean
    gram = standardized_train.T @ standardized_train
    right_hand_side = standardized_train.T @ centered_target

    alpha_tensor = torch.as_tensor(
        alpha_values,
        dtype=torch.float32,
        device=device,
    )
    identity = torch.eye(
        gram.shape[0],
        dtype=torch.float32,
        device=device,
    )
    # Solve all requested alpha values in one batched GPU operation.
    systems = gram.unsqueeze(0) + alpha_tensor[:, None, None] * identity.unsqueeze(0)

    factor, info = torch.linalg.cholesky_ex(systems)
    # Retry the full batch in float64 after a failed factorization.
    if int(info.max().item()) != 0:

        train_tensor64 = train_tensor.to(torch.float64)
        target_tensor64 = target_tensor.to(torch.float64)
        eval_tensor64 = eval_tensor.to(torch.float64)
        feature_mean64 = train_tensor64.mean(dim=0)
        centered_train64 = train_tensor64 - feature_mean64
        feature_scale64 = torch.sqrt(torch.mean(centered_train64.square(), dim=0))
        feature_scale64 = torch.where(
            feature_scale64 == 0.0,
            torch.ones_like(feature_scale64),
            feature_scale64,
        )
        standardized_train64 = centered_train64 / feature_scale64
        standardized_eval64 = (eval_tensor64 - feature_mean64) / feature_scale64
        target_mean64 = target_tensor64.mean()
        centered_target64 = target_tensor64 - target_mean64
        gram64 = standardized_train64.T @ standardized_train64
        right_hand_side64 = standardized_train64.T @ centered_target64
        alpha_tensor64 = torch.as_tensor(
            alpha_values,
            dtype=torch.float64,
            device=device,
        )
        identity64 = torch.eye(
            gram64.shape[0],
            dtype=torch.float64,
            device=device,
        )
        systems64 = gram64.unsqueeze(0) + alpha_tensor64[:, None, None] * identity64.unsqueeze(0)
        factor64, info64 = torch.linalg.cholesky_ex(systems64)
        if int(info64.max().item()) != 0:
            raise RuntimeError("Ridge linear system is not positive definite")
        batched_right_hand_side64 = right_hand_side64[None, :, None].expand(
            len(alpha_values),
            -1,
            -1,
        )
        weights64 = torch.cholesky_solve(
            batched_right_hand_side64,
            factor64,
        ).squeeze(2)
        prediction64 = (standardized_eval64 @ weights64.T).T + target_mean64
        return (
            prediction64.detach()
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )
    batched_right_hand_side = right_hand_side[None, :, None].expand(
        len(alpha_values),
        -1,
        -1,
    )
    weights = torch.cholesky_solve(
        batched_right_hand_side,
        factor,
    ).squeeze(2)
    prediction = (standardized_eval @ weights.T).T + target_mean
    return prediction.detach().cpu().numpy().astype(np.float32, copy=False)


RidgeSolver = Literal[
    "auto",
    "svd",
    "cholesky",
    "lsqr",
    "sparse_cg",
    "sag",
    "saga",
    "lbfgs",
]


@dataclass(frozen=True)
class RidgeState:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    intercept: float | np.ndarray
    alpha: float


def fit_ridge_model(features, targets, alpha, solver="lsqr"):
    solver_name = cast(RidgeSolver, str(solver))
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=float(alpha), solver=solver_name),
    )
    model.fit(features.astype(np.float32), targets.astype(np.float32))
    return model


# Fit standardization statistics on training rows only.
def fit_ridge_state(
    train_features,
    train_targets,
    alpha,
    solver="lsqr",
    compute_backend="cpu",
):
    backend_name = str(compute_backend).strip().lower()
    if backend_name == "cpu":
        model = fit_ridge_model(
            train_features,
            train_targets,
            alpha,
            solver,
        )
        scaler = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        return RidgeState(
            feature_mean=np.asarray(scaler.mean_, dtype=np.float32),
            feature_scale=np.asarray(scaler.scale_, dtype=np.float32),
            weights=np.asarray(ridge.coef_, dtype=np.float32),
            intercept=np.asarray(ridge.intercept_, dtype=np.float32),
            alpha=float(alpha),
        )
    # Use the same standardized Ridge objective on CUDA.
    device = require_compute_device(backend_name)
    if str(solver) != "lsqr":
        raise ValueError("CUDA Ridge requires solver=lsqr protocol")
    feature_mean, feature_scale, weights, intercept = fit_standardized_ridge_state(
        train_features,
        train_targets,
        alpha,
        device,
    )
    return RidgeState(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
        intercept=intercept,
        alpha=float(alpha),
    )


# Reuse stored training statistics during inference.
def predict_ridge_state(state, evaluation_features):
    evaluation_array = np.asarray(evaluation_features, dtype=np.float32)
    feature_mean = np.asarray(state.feature_mean, dtype=np.float32)
    feature_scale = np.asarray(state.feature_scale, dtype=np.float32)
    weights = np.asarray(state.weights, dtype=np.float32)
    if evaluation_array.ndim != 2:
        raise ValueError("Ridge features must have shape (n, d)")
    if feature_mean.ndim != 1 or feature_scale.ndim != 1:
        raise ValueError("Ridge state feature statistics must be one-dimensional")
    if evaluation_array.shape[1] != len(feature_mean):
        raise ValueError("Ridge state and evaluation feature dimensions must match")
    if feature_scale.shape != feature_mean.shape:
        raise ValueError("Ridge state feature statistics must have matching shapes")
    if np.any(feature_scale == 0.0):
        raise ValueError("Ridge state feature scales must be nonzero")

    standardized_features = (evaluation_array - feature_mean) / feature_scale
    if weights.ndim == 1:
        prediction = standardized_features @ weights + state.intercept
    elif weights.ndim == 2:
        prediction = standardized_features @ weights.T + state.intercept
    else:
        raise ValueError("Ridge state weights must be one- or two-dimensional")
    return np.asarray(prediction, dtype=np.float32)


def predict_ridge(
    train_features,
    train_targets,
    evaluation_features,
    alpha,
    solver="lsqr",
    compute_backend="cpu",
):
    backend_name = str(compute_backend).strip().lower()
    if backend_name == "cpu":
        model = fit_ridge_model(
            train_features,
            train_targets,
            alpha,
            solver,
        )
        return np.asarray(
            model.predict(evaluation_features),
            dtype=np.float32,
        )
    device = require_compute_device(backend_name)
    if str(solver) != "lsqr":
        raise ValueError("CUDA Ridge requires solver=lsqr protocol")
    return predict_standardized_ridge(
        train_features,
        train_targets,
        evaluation_features,
        alpha,
        device,
    )
