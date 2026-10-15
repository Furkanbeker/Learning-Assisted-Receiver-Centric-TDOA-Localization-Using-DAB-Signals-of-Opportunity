"""Summary tables, per-group analysis and diagnostic plots."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import compute_metrics, pos_error


def detailed_analysis(te, errors_dict, out_dir):
    """
    Test errors per motion type, estimated geometry class and observation quality class.
    """
    print("\n" + "="*68)
    print("DETAILED ANALYSIS")
    print("="*68)
    rows = []

    # -- Motion-wise -------------------------------------------
    if "motion_type" in te.columns:
        for mt in sorted(te["motion_type"].dropna().unique()):
            mask = te["motion_type"].astype(str) == str(mt)
            r = {"group": "motion", "label": str(mt), "n": int(mask.sum())}
            for mname, err in errors_dict.items():
                e_sub = err[mask]
                r[f"{mname}_mean_m"]  = float(np.mean(e_sub))
                r[f"{mname}_p95_m"]   = float(np.percentile(e_sub, 95))
            rows.append(r)

    # -- Geometry-wise -----------------------------------------
    if "geometry_class_est" in te.columns:
        for gc in ["good", "medium", "bad"]:
            mask = te["geometry_class_est"].astype(str).str.lower() == gc
            if mask.sum() == 0:
                continue
            r = {"group": "geometry", "label": gc, "n": int(mask.sum())}
            for mname, err in errors_dict.items():
                e_sub = err[mask]
                r[f"{mname}_mean_m"] = float(np.mean(e_sub))
                r[f"{mname}_p95_m"]  = float(np.percentile(e_sub, 95))
            rows.append(r)

    # -- Observation quality wise ------------------------------
    if "observation_quality_class" in te.columns:
        for qc in ["good", "medium", "bad"]:
            mask = te["observation_quality_class"].astype(str).str.lower() == qc
            if mask.sum() == 0:
                continue
            r = {"group": "obs_quality", "label": qc, "n": int(mask.sum())}
            for mname, err in errors_dict.items():
                e_sub = err[mask]
                r[f"{mname}_mean_m"] = float(np.mean(e_sub))
                r[f"{mname}_p95_m"]  = float(np.percentile(e_sub, 95))
            rows.append(r)

    df_detail = pd.DataFrame(rows)
    df_detail.to_csv(out_dir / "detailed_analysis.csv", index=False)
    print("\n[Motion-wise / Geometry-wise / Quality-wise]")
    print(df_detail.to_string(index=False))
    return df_detail


def build_summary_table(errors_dict: dict) -> pd.DataFrame:
    rows = []
    for name, err in errors_dict.items():
        m = compute_metrics(name, err)
        rows.append(m)
    df = pd.DataFrame(rows).sort_values("mean_m")
    return df


def plot_cdf(errors_dict: dict, path: Path):
    plt.figure(figsize=(9, 5))
    styles = ["-", "--", "-.", ":", (0,(5,1)), (0,(3,1,1,1))]
    for i, (name, err) in enumerate(errors_dict.items()):
        e = np.sort(np.asarray(err, float))
        y = np.arange(1, len(e)+1) / len(e)
        plt.plot(e, y, label=name, linestyle=styles[i % len(styles)], linewidth=1.8)
    plt.xlabel("Position error [m]", fontsize=12)
    plt.ylabel("CDF",                fontsize=12)
    plt.title("Test set position error CDF", fontsize=13)
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"  [PLOT] {path.name}")


def plot_histogram(errors_dict: dict, path: Path):
    plt.figure(figsize=(9, 5))
    for name, err in errors_dict.items():
        plt.hist(np.asarray(err, float), bins=60,
                 alpha=0.40, label=name, density=True)
    plt.xlabel("Position error [m]", fontsize=12)
    plt.ylabel("Density",            fontsize=12)
    plt.title("Test set position error histogram", fontsize=13)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"  [PLOT] {path.name}")


def plot_scatter(true_xy, pred_xy, name, path):
    err = pos_error(true_xy, pred_xy)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(true_xy[:,0], true_xy[:,1],
                     c=err, cmap="hot_r", s=8, alpha=0.7)
    plt.colorbar(sc, label="Error [m]")
    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.title(f"{name} - test error map")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  [PLOT] {path.name}")
