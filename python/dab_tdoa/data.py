"""Loading the AI-ready CSV and the scenario-wise train/val/test split."""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg


def find_latest_dataset(results_dir: Path = None) -> Path:
    """Return the AI-ready CSV of the newest MATLAB run in results/."""
    results_dir = Path(results_dir) if results_dir else cfg.REPO_ROOT / "results"
    candidates = sorted(results_dir.glob(f"dab_soop_tdoa_dataset_*/{cfg.DATA_FILE_NAME}"))
    if not candidates:
        raise FileNotFoundError(
            f"No {cfg.DATA_FILE_NAME} found under {results_dir}. Run "
            "matlab/generate_dab_tdoa_dataset.m first or pass --data."
        )
    return candidates[-1]


def load_and_clean(path: Path) -> pd.DataFrame:
    """Read the CSV and replace inf/NaN in numeric columns by the median."""
    print(f"\n[LOAD] {path}")
    df = pd.read_csv(path)
    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {df.shape[1]}")

    # Clean inf/NaN in numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    nan_count = int(df[num_cols].isna().sum().sum())
    if nan_count > 0:
        print(f"  [WARN] {nan_count} NaN values found, filling with the column median.")
        for c in num_cols:
            if df[c].isna().any():
                med = df[c].median()
                df[c] = df[c].fillna(med if np.isfinite(med) else 0.0)

    return df


def make_split(df: pd.DataFrame):
    """Scenario-wise split (prevents leakage between subsets).

    Uses the 'split' column written by MATLAB if available, otherwise a
    random 70 / 15 / 15 split of the scenario IDs.
    """
    if cfg.USE_EXISTING_SPLIT and "split" in df.columns:
        s = df["split"].astype(str).str.lower().str.strip()
        tr = df[s == "train"].copy()
        va = df[(s == "val") | (s == "validation")].copy()
        te = df[s == "test"].copy()
        if len(tr) > 0 and len(va) > 0 and len(te) > 0:
            print(f"  Using the existing split. "
                  f"train={len(tr)} val={len(va)} test={len(te)}")
            return tr, va, te

    scenarios = np.array(sorted(df["scenario_id"].unique()))
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    rng.shuffle(scenarios)
    n  = len(scenarios)
    n_tr = int(round(0.70 * n))
    n_va = int(round(0.15 * n))
    tr_sc = set(scenarios[:n_tr])
    va_sc = set(scenarios[n_tr:n_tr + n_va])
    te_sc = set(scenarios[n_tr + n_va:])
    tr = df[df["scenario_id"].isin(tr_sc)].copy()
    va = df[df["scenario_id"].isin(va_sc)].copy()
    te = df[df["scenario_id"].isin(te_sc)].copy()
    print(f"  New random split: train={len(tr)} val={len(va)} test={len(te)}")
    return tr, va, te
