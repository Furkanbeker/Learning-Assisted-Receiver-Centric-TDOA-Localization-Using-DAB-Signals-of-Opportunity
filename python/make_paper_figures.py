"""
Paper figures: GN vs EKF vs proposed method (test set)
======================================================

Run after run_ai_pipeline.py. The script reads the test predictions
(l1/b1/b3_pred_test.npy) and final_summary.json from the AI output
folder and writes publication-style figures (PNG + PDF, English labels,
no titles) to <out>/paper_figures_gn_ekf_proposed/.

The "proposed method" is the level (L1, B1 or B3) with the lowest
validation objective. It is selected on the validation set only; the
test set is used for reporting.

Usage:
  python make_paper_figures.py
  python make_paper_figures.py --data path/to/ai_ready.csv --out path/to/ai_results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dab_tdoa import config as cfg
from dab_tdoa.data import find_latest_dataset, load_and_clean, make_split
from dab_tdoa.features import build_features
from dab_tdoa.utils import pos_error

# Output folder of the figures (set in main)
FIG_DIR = None

# Publication style
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 0.9,
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -------------------------------------------------------------------------
# Save helper: PNG + PDF
# -------------------------------------------------------------------------
def save_paper_figure_png_pdf(fig, filename_base):
    for ext in ["png", "pdf"]:
        fig.savefig(
            FIG_DIR / f"{filename_base}.{ext}",
            bbox_inches="tight",
            pad_inches=0.03
        )
    plt.close(fig)
    print(f"[SAVED] {filename_base}.png / .pdf")

# -------------------------------------------------------------------------
# Metrics table
# -------------------------------------------------------------------------
def metric_row(name, err):
    err = np.asarray(err, float)
    return {
        "method": name,
        "n": int(len(err)),
        "mean_m": float(np.mean(err)),
        "median_m": float(np.median(err)),
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "p75_m": float(np.percentile(err, 75)),
        "p90_m": float(np.percentile(err, 90)),
        "p95_m": float(np.percentile(err, 95)),
        "p99_m": float(np.percentile(err, 99)),
        "max_m": float(np.max(err)),
    }

# -------------------------------------------------------------------------
# CDF figure
# -------------------------------------------------------------------------
def plot_gn_ekf_proposed_cdf(errors_dict, filename_base="fig_gn_ekf_proposed_cdf"):
    fig, ax = plt.subplots(figsize=(3.55, 2.55))

    styles = {
        "GN": "-",
        "EKF": "--",
        "Proposed method": "-.",
    }

    for name, err in errors_dict.items():
        e = np.sort(np.asarray(err, float))
        y = np.arange(1, len(e) + 1) / len(e)

        ax.plot(
            e,
            y,
            linestyle=styles.get(name, "-"),
            linewidth=1.9,
            label=name
        )

    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="lower right")

    fig.tight_layout()
    save_paper_figure_png_pdf(fig, filename_base)


# -------------------------------------------------------------------------
# CCDF figure
# -------------------------------------------------------------------------
def plot_gn_ekf_proposed_ccdf(errors_dict, filename_base="fig_gn_ekf_proposed_ccdf"):
    fig, ax = plt.subplots(figsize=(3.55, 2.55))

    styles = {
        "GN": "-",
        "EKF": "--",
        "Proposed method": "-.",
    }

    for name, err in errors_dict.items():
        e = np.sort(np.asarray(err, float))
        ccdf = 1.0 - (np.arange(1, len(e) + 1) / len(e))
        ccdf = np.maximum(ccdf, 1.0 / len(e))

        ax.semilogy(
            e,
            ccdf,
            linestyle=styles.get(name, "-"),
            linewidth=1.9,
            label=name
        )

    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("CCDF")
    ax.set_xlim(left=0)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="upper right")

    fig.tight_layout()
    save_paper_figure_png_pdf(fig, filename_base)


# -------------------------------------------------------------------------
# Histogram / density figure
# -------------------------------------------------------------------------
def plot_gn_ekf_proposed_histogram(errors_dict, filename_base="fig_gn_ekf_proposed_histogram"):
    fig, ax = plt.subplots(figsize=(3.55, 2.55))

    all_err = np.concatenate([np.asarray(e, float) for e in errors_dict.values()])
    all_err = all_err[np.isfinite(all_err)]

    xmax = np.percentile(all_err, 99.5)

    # Moderate number of bins keeps the overlaid bars readable
    bins = np.linspace(0, xmax, 25)

    colors = {
        "GN": "tab:blue",
        "EKF": "tab:orange",
        "Proposed method": "tab:green",
    }

    for name, err in errors_dict.items():
        e = np.asarray(err, float)
        e = e[np.isfinite(e)]

        # Keep only the plotted range
        e = e[(e >= 0) & (e <= xmax)]

        ax.hist(
            e,
            bins=bins,
            density=True,
            alpha=0.45,
            histtype="bar",
            edgecolor="black",
            linewidth=0.35,
            color=colors.get(name, None),
            label=name
        )

    ax.set_xlabel("Position error (m)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, xmax)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    save_paper_figure_png_pdf(fig, filename_base)


# -------------------------------------------------------------------------
# Mean / P95 comparison bar figure
# -------------------------------------------------------------------------
def plot_gn_ekf_proposed_metric_bars(errors_dict, filename_base="fig_gn_ekf_proposed_metric_comparison"):
    names = list(errors_dict.keys())

    mean_vals = [np.mean(errors_dict[n]) for n in names]
    p95_vals = [np.percentile(errors_dict[n], 95) for n in names]

    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(3.55, 2.55))

    ax.bar(x - width / 2, mean_vals, width, label="Mean")
    ax.bar(x + width / 2, p95_vals, width, label="95th percentile")

    ax.set_ylabel("Position error (m)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    save_paper_figure_png_pdf(fig, filename_base)


# -------------------------------------------------------------------------
# Spatial error map for Proposed method
# -------------------------------------------------------------------------
def plot_proposed_spatial_error(true_xy, proposed_xy, filename_base="fig_proposed_method_spatial_error"):
    err = pos_error(true_xy, proposed_xy)

    fig, ax = plt.subplots(figsize=(3.25, 3.0))

    sc = ax.scatter(
        true_xy[:, 0],
        true_xy[:, 1],
        c=err,
        s=6,
        alpha=0.80,
        linewidths=0,
        rasterized=True
    )

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Position error (m)")

    ax.set_xlabel("x-position (m)")
    ax.set_ylabel("y-position (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    save_paper_figure_png_pdf(fig, filename_base)


# -------------------------------------------------------------------------
# Motion-wise mean error figure, if available
# -------------------------------------------------------------------------
def plot_gn_ekf_proposed_motionwise(errors_dict, te_df, filename_base="fig_gn_ekf_proposed_motionwise_mean_error"):
    if "motion_type" not in te_df.columns:
        print("[SKIP] motion_type column not found.")
        return

    motion_order = sorted(te_df["motion_type"].astype(str).dropna().unique())

    fig, ax = plt.subplots(figsize=(4.9, 2.75))

    models = list(errors_dict.keys())
    x = np.arange(len(motion_order))
    width = 0.78 / max(len(models), 1)

    for i, model in enumerate(models):
        vals = []

        for motion in motion_order:
            mask = te_df["motion_type"].astype(str).values == motion
            vals.append(float(np.mean(np.asarray(errors_dict[model])[mask])))

        ax.bar(
            x - 0.39 + width / 2 + i * width,
            vals,
            width,
            label=model
        )

    ax.set_ylabel("Mean position error (m)")
    ax.set_xlabel("Motion type")
    ax.set_xticks(x)
    ax.set_xticklabels(motion_order, rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", ncol=1)

    fig.tight_layout()
    save_paper_figure_png_pdf(fig, filename_base)



# -------------------------------------------------------------------------
# Selection of the proposed method (validation set)
# -------------------------------------------------------------------------
def select_proposed_method(out_dir, available):
    """Return the level (L1/B1/B3) with the lowest validation objective."""
    summary_path = out_dir / "final_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} not found. Run run_ai_pipeline.py first.")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    val_obj = {k: summary[f"{k}_val_obj"] for k in available
               if f"{k}_val_obj" in summary}
    if not val_obj:
        raise RuntimeError(
            "final_summary.json has no validation objectives. "
            "Re-run run_ai_pipeline.py with this version of the code.")
    for k, v in val_obj.items():
        print(f"  {k}: validation objective = {v:.4f}")
    return min(val_obj, key=val_obj.get)


def parse_args():
    p = argparse.ArgumentParser(description="GN / EKF / proposed-method paper figures")
    p.add_argument("--data", type=Path, default=None,
                   help="AI-ready CSV (default: newest run in results/)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"AI output folder (default: {cfg.OUT_DIR})")
    return p.parse_args()


def main(data_path=None, out_dir=None):
    global FIG_DIR
    data_path = Path(data_path) if data_path else (cfg.DATA_PATH or find_latest_dataset())
    out_dir = Path(out_dir) if out_dir else cfg.OUT_DIR

    FIG_DIR = out_dir / "paper_figures_gn_ekf_proposed"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # -- Test set (same features and split as the pipeline) ----------------
    df_raw = load_and_clean(data_path)
    df = build_features(df_raw)
    tr, va, te = make_split(df)

    true_te = te[["uav_rel_x", "uav_rel_y"]].values.astype(float)

    all_predictions = {}
    all_errors = {}

    # GN baseline
    gn_te = te[["gn_est_rel_x", "gn_est_rel_y"]].values.astype(float)
    all_predictions["GN"] = gn_te
    all_errors["GN"] = pos_error(true_te, gn_te)

    # EKF baseline
    if all(c in te.columns for c in ["ctekf_est_rel_x", "ctekf_est_rel_y"]):
        ekf_te = te[["ctekf_est_rel_x", "ctekf_est_rel_y"]].values.astype(float)
        all_predictions["EKF"] = ekf_te
        all_errors["EKF"] = pos_error(true_te, ekf_te)
    else:
        print("[WARN] CT-EKF columns were not found. EKF curve will not be plotted.")

    # Test predictions of the learning levels
    candidate_files = {
        "L1": out_dir / "l1_pred_test.npy",
        "B1": out_dir / "b1_pred_test.npy",
        "B3": out_dir / "b3_pred_test.npy",
    }

    candidate_predictions = {}
    for method_name, pred_path in candidate_files.items():
        if pred_path.exists():
            pred = np.load(pred_path)
            if pred.shape[0] == true_te.shape[0] and pred.shape[1] == 2:
                candidate_predictions[method_name] = pred
                print(f"[FOUND] {method_name}")
            else:
                print(
                    f"[SKIP] {method_name}: prediction shape mismatch "
                    f"{pred.shape} vs expected ({true_te.shape[0]}, 2)"
                )

    if len(candidate_predictions) == 0:
        raise RuntimeError(
            "No valid L1/B1/B3 prediction file was found. "
            "Run run_ai_pipeline.py first so l1_pred_test.npy, b1_pred_test.npy, "
            "or b3_pred_test.npy is generated."
        )

    # Proposed method = lowest validation objective
    print("\nSelecting the proposed method on the validation set:")
    best_proposed_key = select_proposed_method(out_dir, list(candidate_predictions))

    proposed_pred = candidate_predictions[best_proposed_key]
    proposed_err = pos_error(true_te, proposed_pred)

    all_predictions["Proposed method"] = proposed_pred
    all_errors["Proposed method"] = proposed_err

    # Keep GN, EKF and the proposed method in this order
    plot_errors = {}
    for key in ["GN", "EKF", "Proposed method"]:
        if key in all_errors:
            plot_errors[key] = all_errors[key]

    print("\nSelected final method for figures:")
    print(f"  Proposed method source : {best_proposed_key}")
    print(f"  Proposed mean error    : {np.mean(proposed_err):.3f} m")
    print(f"  Proposed median error  : {np.median(proposed_err):.3f} m")
    print(f"  Proposed p95 error     : {np.percentile(proposed_err, 95):.3f} m")

    # Metrics table
    df_metrics_3model = pd.DataFrame(
        [metric_row(name, err) for name, err in plot_errors.items()]
    )
    df_metrics_3model.to_csv(FIG_DIR / "metrics_gn_ekf_proposed.csv", index=False)
    print("\nGN / EKF / Proposed method metrics:")
    print(df_metrics_3model.to_string(index=False))

    # Figures
    plot_gn_ekf_proposed_cdf(plot_errors)
    plot_gn_ekf_proposed_ccdf(plot_errors)
    plot_gn_ekf_proposed_histogram(plot_errors)
    plot_gn_ekf_proposed_metric_bars(plot_errors)
    plot_proposed_spatial_error(true_te, proposed_pred)
    plot_gn_ekf_proposed_motionwise(plot_errors, te)

    print(f"\nFigures saved to: {FIG_DIR}")


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.out)
