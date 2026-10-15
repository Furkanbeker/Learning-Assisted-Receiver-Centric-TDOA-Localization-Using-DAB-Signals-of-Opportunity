"""
DAB SoOP TDOA - AI-assisted localization pipeline
=================================================

Trains and evaluates the learning-assisted correction methods on the
AI-ready dataset produced by matlab/generate_dab_tdoa_dataset.m.

Compared methods (test set):
  GN baseline     : multi-start weighted GN (from the dataset)
  CT-EKF baseline : adaptive CT-EKF on GN (from the dataset)
  Level 1 (L1)    : direct GN error correction (target gn_err_x/y)
  Level 2 (B1)    : per-channel TDOA error correction + weighted GN
                    (target delta_err_ch1-4)
  Level 3 (B3)    : B1 + per-channel R-scale model + tight TDOA CT-EKF

Leakage protection:
  - scenario-wise train / val / test split (GroupKFold for out-of-fold)
  - true positions, true TDOAs and errors are targets only, never features
  - simulation parameters (noise power, TX offsets) are not features
  - model selection and EKF tuning use the validation set only

Usage:
  pip install -r requirements.txt
  python run_ai_pipeline.py                       # newest MATLAB run in ../results
  python run_ai_pipeline.py --data path/to/dab_soop_tdoa_ai_ready_gn_correction.csv
  python run_ai_pipeline.py --out path/to/output_folder
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dab_tdoa import config as cfg
from dab_tdoa.analysis import (build_summary_table, detailed_analysis,
                               plot_cdf, plot_histogram, plot_scatter)
from dab_tdoa.data import find_latest_dataset, load_and_clean, make_split
from dab_tdoa.features import TARGET_B1, TARGET_L1, build_features, select_features
from dab_tdoa.levels import run_level1, run_level2, run_level3
from dab_tdoa.utils import compute_metrics, ensure_dir, pos_error, print_metrics, save_json

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description="DAB SoOP TDOA AI pipeline")
    p.add_argument("--data", type=Path, default=None,
                   help="AI-ready CSV (default: newest run in results/)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"output folder (default: {cfg.OUT_DIR})")
    return p.parse_args()


def main(data_path: Path = None, out_dir: Path = None):
    data_path = Path(data_path) if data_path else (cfg.DATA_PATH or find_latest_dataset())
    out_dir = Path(out_dir) if out_dir else cfg.OUT_DIR
    cfg.DATA_PATH, cfg.OUT_DIR = data_path, out_dir

    ensure_dir(out_dir)
    print("\n" + "="*68)
    print("DAB SoOP TDOA - AI Localization Pipeline")
    print("="*68)

    # -- Load data -------------------------------------------------------
    df_raw = load_and_clean(data_path)

    # Required columns
    required = (["scenario_id","meas_idx",
                 "gn_est_rel_x","gn_est_rel_y",
                 "uav_rel_x","uav_rel_y"] +
                [f"tx{k}_rel_x" for k in range(1,5)] +
                [f"tx{k}_rel_y" for k in range(1,5)] +
                [f"est_deltaTau_ch{k}" for k in range(1,5)])
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    has_ctekf = all(c in df_raw.columns
                    for c in ["ctekf_est_rel_x","ctekf_est_rel_y"])
    has_l1_target = all(c in df_raw.columns for c in TARGET_L1)
    has_b1_target = all(c in df_raw.columns for c in TARGET_B1)

    print(f"\n  CT-EKF baseline available : {has_ctekf}")
    print(f"  Level-1 target available  : {has_l1_target}")
    print(f"  B1 target available       : {has_b1_target}")

    # -- Feature engineering -----------------------------------------------
    print("\n[FEATURE ENGINEERING]")
    df = build_features(df_raw)

    feat_cols = select_features(df)
    print(f"  Number of features        : {len(feat_cols)}")

    pd.DataFrame({"feature": feat_cols}).to_csv(
        out_dir / "feature_list.csv", index=False)

    # -- Split ---------------------------------------------------------------
    print("\n[SPLIT]")
    tr, va, te = make_split(df)
    print(f"  Train: {len(tr)} rows | "
          f"{tr['scenario_id'].nunique()} scenarios")
    print(f"  Val  : {len(va)} rows | "
          f"{va['scenario_id'].nunique()} scenarios")
    print(f"  Test : {len(te)} rows | "
          f"{te['scenario_id'].nunique()} scenarios")

    # -- Baseline: GN --------------------------------------------------------
    true_te   = te[["uav_rel_x","uav_rel_y"]].values
    gn_te     = te[["gn_est_rel_x","gn_est_rel_y"]].values
    err_gn_te = pos_error(true_te, gn_te)
    print_metrics(compute_metrics("GN_baseline_test", err_gn_te))

    errors_all = {"GN_baseline": err_gn_te}

    # -- Baseline: CT-EKF (from the dataset) -------------------------------
    err_ct_te = None
    if has_ctekf:
        ct_te     = te[["ctekf_est_rel_x","ctekf_est_rel_y"]].values
        err_ct_te = pos_error(true_te, ct_te)
        print_metrics(compute_metrics("CT-EKF_baseline_test", err_ct_te))
        errors_all["CT-EKF_baseline"] = err_ct_te

    # -- Level 1 ---------------------------------------------------------------
    l1_res = None
    if has_l1_target:
        l1_res, best_l1 = run_level1(tr, va, te, feat_cols, out_dir)
        if l1_res:
            err_l1 = l1_res[best_l1]["err_test"]
            errors_all[f"L1_{best_l1}"] = err_l1
            print_metrics(compute_metrics(f"L1_{best_l1}_test", err_l1))
    else:
        print("\n[SKIP] Level 1: gn_err_x/y columns not found.")

    # -- Level 2 (B1) ----------------------------------------------------------
    b1_res = None
    if has_b1_target:
        b1_res, best_b1 = run_level2(tr, va, te, feat_cols, out_dir)
        if b1_res:
            err_b1 = b1_res[best_b1]["err_test"]
            errors_all[f"B1_{best_b1}"] = err_b1
            print_metrics(compute_metrics(f"B1_{best_b1}_test", err_b1))
    else:
        print("\n[SKIP] Level 2: delta_err_ch columns not found.")

    # -- Level 3 (B3 + tight TDOA EKF) ---------------------------------------
    b3_res = None
    if b1_res is not None:
        b3_res, best_b3 = run_level3(
            tr, va, te, feat_cols, b1_res, best_b1, out_dir)
        if b3_res:
            err_b3 = b3_res[best_b3]["err_test"]
            errors_all[f"B3_{best_b3}"] = err_b3
            print_metrics(compute_metrics(f"B3_{best_b3}_test", err_b3))
    else:
        print("\n[SKIP] Level 3: no B1 result.")

    # -- Summary table ---------------------------------------------------------
    print("\n" + "="*68)
    print("SUMMARY TABLE (test set, sorted by mean error)")
    print("="*68)
    df_summary = build_summary_table(errors_all)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(out_dir / "summary_table.csv", index=False)

    # -- Detailed analysis -----------------------------------------------------
    detailed_analysis(te, errors_all, out_dir)

    # -- Plots -----------------------------------------------------------------
    print("\n[PLOTS]")
    plot_cdf(errors_all, out_dir / "test_error_cdf.png")
    plot_histogram(errors_all, out_dir / "test_error_histogram.png")

    if l1_res:
        best_pred_l1 = l1_res[best_l1]["pred_test"]
        plot_scatter(true_te, best_pred_l1, f"L1_{best_l1}",
                     out_dir / "scatter_L1.png")
    if b3_res:
        best_pred_b3 = b3_res[best_b3]["pred_test"]
        plot_scatter(true_te, best_pred_b3, f"B3_{best_b3}",
                     out_dir / "scatter_B3.png")

    # -- JSON summary ----------------------------------------------------------
    # The validation objectives are stored so that make_paper_figures.py
    # can select the final method on the validation set (not on test).
    summary_json = {
        "data_path"       : str(data_path),
        "n_train"         : int(len(tr)),
        "n_val"           : int(len(va)),
        "n_test"          : int(len(te)),
        "n_features"      : int(len(feat_cols)),
        "gn_baseline_mean": float(np.mean(err_gn_te)),
        "gn_baseline_p95" : float(np.percentile(err_gn_te, 95)),
    }
    if err_ct_te is not None:
        summary_json["ctekf_baseline_mean"] = float(np.mean(err_ct_te))
        summary_json["ctekf_baseline_p95"]  = float(np.percentile(err_ct_te, 95))
    if l1_res:
        summary_json["L1_best"]      = best_l1
        summary_json["L1_mean_test"] = float(np.mean(errors_all[f"L1_{best_l1}"]))
        summary_json["L1_val_obj"]   = float(l1_res[best_l1]["val_obj"])
    if b1_res:
        summary_json["B1_best"]      = best_b1
        summary_json["B1_mean_test"] = float(np.mean(errors_all[f"B1_{best_b1}"]))
        summary_json["B1_val_obj"]   = float(b1_res[best_b1]["val_obj"])
    if b3_res:
        summary_json["B3_best"]      = best_b3
        summary_json["B3_mean_test"] = float(np.mean(errors_all[f"B3_{best_b3}"]))
        summary_json["B3_val_obj"]   = float(b3_res[best_b3]["val_obj"])

    save_json(out_dir / "final_summary.json", summary_json)

    print("\n" + "="*68)
    print(f"All outputs saved to: {out_dir}")
    print("="*68)
    print("\n[DONE]")


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.out)
