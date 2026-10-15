"""Small helpers: output folders, JSON, error metrics."""

import json
from pathlib import Path

import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_json(p: Path, obj: dict):
    def _conv(o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        if isinstance(o, Path):           return str(o)
        return str(o)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_conv)


def pos_error(true_xy: np.ndarray, pred_xy: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean position error (m)."""
    return np.linalg.norm(
        np.asarray(pred_xy, float) - np.asarray(true_xy, float), axis=1
    )


def compute_metrics(name: str, err: np.ndarray) -> dict:
    """Summary statistics of a position-error vector."""
    err = np.asarray(err, float)
    return {
        "model"        : name,
        "n"            : int(len(err)),
        "mean_m"       : float(np.mean(err)),
        "median_m"     : float(np.median(err)),
        "rmse_m"       : float(np.sqrt(np.mean(err**2))),
        "p75_m"        : float(np.percentile(err, 75)),
        "p90_m"        : float(np.percentile(err, 90)),
        "p95_m"        : float(np.percentile(err, 95)),
        "p99_m"        : float(np.percentile(err, 99)),
        "max_m"        : float(np.max(err)),
    }


def val_objective(err: np.ndarray) -> float:
    """Model-selection score on the validation set: mean + tail penalties."""
    e = np.asarray(err, float)
    return float(
        np.mean(e)
        + 0.10 * np.percentile(e, 90)
        + 0.10 * np.percentile(e, 95)
        + 0.05 * np.percentile(e, 99)
    )


def print_metrics(m: dict):
    print(f"\n{'-'*62}")
    print(f"  {m['model']}")
    print(f"{'-'*62}")
    print(f"  N       : {m['n']}")
    print(f"  Mean    : {m['mean_m']:.3f} m")
    print(f"  Median  : {m['median_m']:.3f} m")
    print(f"  RMSE    : {m['rmse_m']:.3f} m")
    print(f"  P90     : {m['p90_m']:.3f} m")
    print(f"  P95     : {m['p95_m']:.3f} m")
    print(f"  P99     : {m['p99_m']:.3f} m")
    print(f"  Max     : {m['max_m']:.3f} m")
