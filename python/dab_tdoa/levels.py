"""
The three learning levels of the proposed method.

Level 1 : direct GN correction. Predicts the GN position error
          (gn_err_x, gn_err_y) and adds it to the GN estimate.
Level 2 : B1. Predicts the per-channel TDOA error (delta_err_ch1-4),
          corrects the TDOAs and re-solves the position with weighted GN.
Level 3 : B1 + B3. B3 predicts a per-channel measurement-noise scale
          (R-scale) that is used by a tightly coupled TDOA CT-EKF.

Each level trains several models, selects the best one (or the
ensemble) with the validation objective and reports test errors.
"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler

from . import config as cfg
from .ekf import apply_tight_ekf, tune_ekf
from .features import TARGET_B1, TARGET_L1, select_features
from .gn import gn_batch
from .models import ensemble_weights, make_ensemble, predict_all, train_all_models
from .utils import compute_metrics, pos_error, print_metrics, val_objective


def run_level1(tr, va, te, feat_cols, out_dir):
    """
    Target: gn_err_x, gn_err_y (GN position error)
    Output: corrected_pos = gn_est + predicted_err
    """
    print("\n" + "="*68)
    print("LEVEL 1 - Direct GN Residual Correction")
    print("="*68)

    X_tr  = tr[feat_cols].values.astype(float)
    X_val = va[feat_cols].values.astype(float)
    X_te  = te[feat_cols].values.astype(float)

    # Target
    tgt_cols_ok = [c for c in TARGET_L1 if c in tr.columns]
    if len(tgt_cols_ok) < 2:
        print("  [SKIP] gn_err_x / gn_err_y columns not found.")
        return None

    y_tr = tr[tgt_cols_ok].values.astype(float)

    # Scaler
    sx = RobustScaler(); X_tr_s  = sx.fit_transform(X_tr)
    X_val_s = sx.transform(X_val); X_te_s = sx.transform(X_te)

    sy = RobustScaler()
    trained = train_all_models(X_tr_s, y_tr, scaler_y=sy)

    preds_val = predict_all(trained, X_val_s, sy)
    preds_te  = predict_all(trained, X_te_s,  sy)

    true_val = va[["uav_rel_x","uav_rel_y"]].values
    true_te  = te[["uav_rel_x","uav_rel_y"]].values
    gn_val   = va[["gn_est_rel_x","gn_est_rel_y"]].values
    gn_te    = te[["gn_est_rel_x","gn_est_rel_y"]].values

    results = {}
    for name in trained:
        corr_val = gn_val + preds_val[name]
        corr_te  = gn_te  + preds_te[name]
        ev = pos_error(true_val, corr_val)
        et = pos_error(true_te,  corr_te)
        results[name] = {
            "val_obj": val_objective(ev),
            "err_val": ev, "err_test": et,
            "pred_val": corr_val, "pred_test": corr_te,
        }
        print_metrics(compute_metrics(f"L1_{name}_val", ev))

    # Ensemble (weights from the validation targets)
    y_val_true = va[tgt_cols_ok].values.astype(float)
    w = ensemble_weights(preds_val, y_val_true)
    ens_val = gn_val + make_ensemble(preds_val, w)
    ens_te  = gn_te  + make_ensemble(preds_te,  w)
    ev_ens  = pos_error(true_val, ens_val)
    et_ens  = pos_error(true_te,  ens_te)
    results["ensemble"] = {
        "val_obj": val_objective(ev_ens),
        "err_val": ev_ens, "err_test": et_ens,
        "pred_val": ens_val, "pred_test": ens_te,
    }
    print_metrics(compute_metrics("L1_ensemble_val", ev_ens))
    print_metrics(compute_metrics("L1_ensemble_test", et_ens))

    best_name = min(results, key=lambda n: results[n]["val_obj"])
    print(f"\n  [L1 BEST] {best_name} | val_obj={results[best_name]['val_obj']:.4f}")

    # Save the test predictions of the best model
    np.save(out_dir / "l1_pred_test.npy", results[best_name]["pred_test"])
    return results, best_name


def b1_oof_predictions(X_tr, y_tr_ns, groups):
    """Scenario-wise out-of-fold B1 predictions on the training set (B3 inputs)."""
    if cfg.OOF_MODEL == "lgbm":
        from lightgbm import LGBMRegressor
        print("  [OOF] Using LightGBM.")
        make_oof_model = lambda: MultiOutputRegressor(LGBMRegressor(
            n_estimators=600, learning_rate=0.04, num_leaves=48,
            subsample=0.85, colsample_bytree=0.85,
            random_state=cfg.RANDOM_SEED, n_jobs=-1, verbose=-1,
        ))
    else:
        print("  [OOF] Using ExtraTrees.")
        ModelCls = lambda: ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3,
            random_state=cfg.RANDOM_SEED, n_jobs=-1)
        make_oof_model = lambda: MultiOutputRegressor(ModelCls())

    unique_g = np.unique(groups)
    n_splits = min(cfg.B1_OOF_FOLDS, len(unique_g))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y_tr_ns, float)

    for fold, (ti, vi) in enumerate(gkf.split(X_tr, y_tr_ns, groups), 1):
        print(f"    OOF fold {fold}/{n_splits}: "
              f"train={len(ti)} val={len(vi)}")
        m = make_oof_model()
        m.fit(X_tr[ti], y_tr_ns[ti])
        oof[vi] = m.predict(X_tr[vi])

    mae = float(np.mean(np.abs(oof - y_tr_ns)))
    print(f"  OOF B1 MAE = {mae:.4f} ns")
    return oof


def run_level2(tr, va, te, feat_cols, out_dir):
    """
    B1: predict the per-channel TDOA error, correct the TDOAs, re-solve GN.
    Target: delta_err_ch1-4 (true minus estimated TDOA, in ns)
    """
    print("\n" + "="*68)
    print("LEVEL 2 - B1 Per-Channel TDOA Residual Correction")
    print("="*68)

    b1_tgt_ok = [c for c in TARGET_B1 if c in tr.columns]
    if len(b1_tgt_ok) < 4:
        print("  [SKIP] delta_err_ch columns not found.")
        return None

    X_tr  = tr[feat_cols].values.astype(float)
    X_val = va[feat_cols].values.astype(float)
    X_te  = te[feat_cols].values.astype(float)

    # Targets in ns
    y_tr_ns = tr[b1_tgt_ok].values.astype(float) * 1e9
    y_va_ns = va[b1_tgt_ok].values.astype(float) * 1e9

    sx = RobustScaler()
    X_tr_s  = sx.fit_transform(X_tr)
    X_val_s = sx.transform(X_val)
    X_te_s  = sx.transform(X_te)

    sy = RobustScaler()
    trained = train_all_models(X_tr_s, y_tr_ns, scaler_y=sy)
    preds_val_ns = predict_all(trained, X_val_s, sy)
    preds_te_ns  = predict_all(trained, X_te_s,  sy)

    # Out-of-fold predictions (training inputs of B3)
    print("\n  Computing B1 out-of-fold predictions (B3 training set)...")
    groups_tr = tr["scenario_id"].values
    oof_ns = b1_oof_predictions(X_tr_s, y_tr_ns, groups_tr)

    # Corrected TDOA -> GN
    raw_delta_cols   = [f"est_deltaTau_ch{k}" for k in range(1, 5)]
    tdoa_weight_cols = [f"tdoa_weight_ch{k}"  for k in range(1, 5)]

    true_val = va[["uav_rel_x","uav_rel_y"]].values
    true_te  = te[["uav_rel_x","uav_rel_y"]].values

    results_b1 = {}
    for name in trained:
        # Corrected TDOA = raw TDOA + predicted error (converted to s)
        va_corr = va.copy()
        te_corr = te.copy()
        tr_corr = tr.copy()

        for k in range(1, 5):
            va_corr[f"est_deltaTau_ch{k}"] = (
                va[f"est_deltaTau_ch{k}"].values +
                preds_val_ns[name][:, k-1] * 1e-9
            )
            te_corr[f"est_deltaTau_ch{k}"] = (
                te[f"est_deltaTau_ch{k}"].values +
                preds_te_ns[name][:, k-1] * 1e-9
            )
            tr_corr[f"est_deltaTau_ch{k}"] = (
                tr[f"est_deltaTau_ch{k}"].values +
                oof_ns[:, k-1] * 1e-9
            )

        gn_val = gn_batch(va_corr, raw_delta_cols, tdoa_weight_cols)
        gn_te  = gn_batch(te_corr, raw_delta_cols, tdoa_weight_cols)

        ev = pos_error(true_val, gn_val)
        et = pos_error(true_te,  gn_te)
        results_b1[name] = {
            "val_obj": val_objective(ev),
            "err_val": ev, "err_test": et,
            "pred_val": gn_val, "pred_test": gn_te,
            "pred_ns_val": preds_val_ns[name],
            "pred_ns_te" : preds_te_ns[name],
            "oof_ns"     : oof_ns,
            "tr_corr"    : tr_corr,
            "va_corr"    : va_corr,
            "te_corr"    : te_corr,
        }
        print_metrics(compute_metrics(f"B1_{name}_val", ev))

    # Ensemble (weights from the TDOA MAE)
    w_b1 = ensemble_weights(preds_val_ns, y_va_ns)
    ens_val_ns = make_ensemble(preds_val_ns, w_b1)
    ens_te_ns  = make_ensemble(preds_te_ns,  w_b1)
    ens_oof_ns = sum(w_b1.get(n, 0) * results_b1[n]["oof_ns"] for n in results_b1)

    va_corr_ens = va.copy(); te_corr_ens = te.copy(); tr_corr_ens = tr.copy()
    for k in range(1, 5):
        va_corr_ens[f"est_deltaTau_ch{k}"] = (
            va[f"est_deltaTau_ch{k}"].values + ens_val_ns[:, k-1] * 1e-9)
        te_corr_ens[f"est_deltaTau_ch{k}"] = (
            te[f"est_deltaTau_ch{k}"].values + ens_te_ns[:, k-1] * 1e-9)
        tr_corr_ens[f"est_deltaTau_ch{k}"] = (
            tr[f"est_deltaTau_ch{k}"].values + ens_oof_ns[:, k-1] * 1e-9)

    gn_val_ens = gn_batch(va_corr_ens, raw_delta_cols, tdoa_weight_cols)
    gn_te_ens  = gn_batch(te_corr_ens, raw_delta_cols, tdoa_weight_cols)
    ev_ens = pos_error(true_val, gn_val_ens)
    et_ens = pos_error(true_te,  gn_te_ens)
    results_b1["ensemble"] = {
        "val_obj": val_objective(ev_ens),
        "err_val": ev_ens, "err_test": et_ens,
        "pred_val": gn_val_ens, "pred_test": gn_te_ens,
        "pred_ns_val": ens_val_ns, "pred_ns_te": ens_te_ns,
        "oof_ns": ens_oof_ns,
        "tr_corr": tr_corr_ens, "va_corr": va_corr_ens, "te_corr": te_corr_ens,
    }
    print_metrics(compute_metrics("B1_ensemble_val",  ev_ens))
    print_metrics(compute_metrics("B1_ensemble_test", et_ens))

    best_b1 = min(results_b1, key=lambda n: results_b1[n]["val_obj"])
    print(f"\n  [B1 BEST] {best_b1} | val_obj={results_b1[best_b1]['val_obj']:.4f}")

    np.save(out_dir / "b1_pred_test.npy",
            results_b1[best_b1]["pred_test"])
    return results_b1, best_b1


def run_level3(tr, va, te, feat_cols, b1_results, best_b1, out_dir):
    """
    B3: predict a per-channel R scale, then run the tight TDOA CT-EKF.
    """
    print("\n" + "="*68)
    print("LEVEL 3 - B3 Per-Channel R-Scale + Tight TDOA CT-EKF")
    print("="*68)

    b1 = b1_results[best_b1]
    tr_corr = b1["tr_corr"]
    va_corr = b1["va_corr"]
    te_corr = b1["te_corr"]
    oof_ns  = b1["oof_ns"]
    val_ns  = b1["pred_ns_val"]
    te_ns   = b1["pred_ns_te"]

    corr_delta_cols = [f"est_deltaTau_ch{k}" for k in range(1, 5)]

    # B1-corrected GN positions (EKF initialization)
    tdoa_w = [f"tdoa_weight_ch{k}" for k in range(1, 5)]
    b1_gn_tr  = gn_batch(tr_corr, corr_delta_cols, tdoa_w)
    b1_gn_val = gn_batch(va_corr, corr_delta_cols, tdoa_w)
    b1_gn_te  = gn_batch(te_corr, corr_delta_cols, tdoa_w)

    true_val = va[["uav_rel_x","uav_rel_y"]].values
    true_te  = te[["uav_rel_x","uav_rel_y"]].values

    # -- B3 target ---------------------------------------------
    # Corrected residual = true_delta_err - B1_pred
    true_err_tr_ns = tr[[f"delta_err_ch{k}" for k in range(1, 5)]].values * 1e9
    resid_tr_ns    = true_err_tr_ns - oof_ns
    sigma_ref_ns   = max(float(np.median(np.abs(resid_tr_ns))),
                         cfg.B3_RESIDUAL_FLOOR_NS)
    base_sigma_s   = sigma_ref_ns * 1e-9
    print(f"  B3 sigma_ref_ns = {sigma_ref_ns:.4f} ns")

    # log R-scale per channel
    r_scale_tr = ((np.abs(resid_tr_ns) + cfg.B3_RESIDUAL_FLOOR_NS) / sigma_ref_ns)**2
    r_scale_tr = np.clip(r_scale_tr, cfg.MIN_R_SCALE, cfg.MAX_R_SCALE)
    y_b3_tr    = np.log(r_scale_tr)

    # B3 features: B1-corrected data + innovations of the corrected GN
    def add_b3_features(df_part, gn_xy, pred_ns):
        d = df_part.copy()
        d["b1gn_x"] = gn_xy[:, 0]
        d["b1gn_y"] = gn_xy[:, 1]
        for k in range(1, 5):
            d[f"b1_corr_tdoa_ch{k}_ns"] = d[f"est_deltaTau_ch{k}"].values * 1e9
            d[f"b1_pred_ns_ch{k}"]      = pred_ns[:, k-1]
        # innovation (B1-corrected GN)
        gx = gn_xy[:, 0]; gy = gn_xy[:, 1]
        for k in range(1, 5):
            txk_x = d[f"tx{k}_rel_x"].values
            txk_y = d[f"tx{k}_rel_y"].values
            du  = np.sqrt((gx - txk_x)**2 + (gy - txk_y)**2)
            dr  = np.sqrt(txk_x**2 + txk_y**2)
            inn = d[f"est_deltaTau_ch{k}"].values * 1e9 - (du - dr)/cfg.C_LIGHT*1e9
            d[f"b3_innov_ch{k}_ns"]     = inn
            d[f"b3_abs_innov_ch{k}_ns"] = np.abs(inn)
        return d

    tr_b3 = add_b3_features(tr_corr, b1_gn_tr,  oof_ns)
    va_b3 = add_b3_features(va_corr, b1_gn_val,  val_ns)
    te_b3 = add_b3_features(te_corr, b1_gn_te,   te_ns)

    # Select the B3 feature columns
    b3_extra = (
        ["b1gn_x","b1gn_y"] +
        [f"b1_corr_tdoa_ch{k}_ns" for k in range(1, 5)] +
        [f"b1_pred_ns_ch{k}"      for k in range(1, 5)] +
        [f"b3_innov_ch{k}_ns"     for k in range(1, 5)] +
        [f"b3_abs_innov_ch{k}_ns" for k in range(1, 5)]
    )
    b3_cols = select_features(tr_b3) +               [c for c in b3_extra if c in tr_b3.columns]
    b3_cols = list(dict.fromkeys(b3_cols))

    X_tr_b3  = tr_b3[b3_cols].values.astype(float)
    X_val_b3 = va_b3[b3_cols].values.astype(float)
    X_te_b3  = te_b3[b3_cols].values.astype(float)

    sx_b3 = RobustScaler()
    X_tr_b3_s  = sx_b3.fit_transform(X_tr_b3)
    X_val_b3_s = sx_b3.transform(X_val_b3)
    X_te_b3_s  = sx_b3.transform(X_te_b3)

    print("\n  Training the B3 models...")
    b3_trained = train_all_models(X_tr_b3_s, y_b3_tr)
    b3_preds_val = predict_all(b3_trained, X_val_b3_s)
    b3_preds_te  = predict_all(b3_trained, X_te_b3_s)

    def to_rscale(log_arr):
        r = np.exp(np.asarray(log_arr, float))
        r[~np.isfinite(r)] = 1.0
        return np.clip(r, cfg.MIN_R_SCALE, cfg.MAX_R_SCALE)

    results_b3 = {}
    for name in b3_trained:
        r_val = to_rscale(b3_preds_val[name])
        r_te  = to_rscale(b3_preds_te[name])

        # EKF hyper-parameters tuned on the validation set with uniform R
        uniform = np.ones((len(va_b3), 4))
        tune_res = tune_ekf(va_b3, true_val, b1_gn_val,
                            uniform, corr_delta_cols, base_sigma_s)

        pred_val = apply_tight_ekf(va_b3, corr_delta_cols, b1_gn_val,
                                   r_val, base_sigma_s,
                                   tune_res["r_mult"], tune_res["q"])
        pred_te  = apply_tight_ekf(te_b3, corr_delta_cols, b1_gn_te,
                                   r_te,  base_sigma_s,
                                   tune_res["r_mult"], tune_res["q"])

        ev = pos_error(true_val, pred_val)
        et = pos_error(true_te,  pred_te)
        results_b3[name] = {
            "val_obj"  : val_objective(ev),
            "err_val"  : ev, "err_test": et,
            "pred_val" : pred_val, "pred_test": pred_te,
            "r_val"    : r_val, "r_te": r_te,
            "tune"     : tune_res,
        }
        print_metrics(compute_metrics(f"B3_{name}_val", ev))
        print_metrics(compute_metrics(f"B3_{name}_test", et))

    # Ensemble: arithmetic mean of the predicted R-scales
    r_val_ens = np.mean([to_rscale(b3_preds_val[n])
                         for n in b3_trained], axis=0)
    r_te_ens  = np.mean([to_rscale(b3_preds_te[n])
                         for n in b3_trained], axis=0)
    tune_ens = tune_ekf(va_b3, true_val, b1_gn_val,
                        r_val_ens, corr_delta_cols, base_sigma_s)
    pred_val_ens = apply_tight_ekf(va_b3, corr_delta_cols, b1_gn_val,
                                   r_val_ens, base_sigma_s,
                                   tune_ens["r_mult"], tune_ens["q"])
    pred_te_ens  = apply_tight_ekf(te_b3, corr_delta_cols, b1_gn_te,
                                   r_te_ens, base_sigma_s,
                                   tune_ens["r_mult"], tune_ens["q"])
    ev_ens = pos_error(true_val, pred_val_ens)
    et_ens = pos_error(true_te,  pred_te_ens)
    results_b3["ensemble"] = {
        "val_obj": val_objective(ev_ens),
        "err_val": ev_ens, "err_test": et_ens,
        "pred_val": pred_val_ens, "pred_test": pred_te_ens,
    }
    print_metrics(compute_metrics("B3_ensemble_val",  ev_ens))
    print_metrics(compute_metrics("B3_ensemble_test", et_ens))

    best_b3 = min(results_b3, key=lambda n: results_b3[n]["val_obj"])
    print(f"\n  [B3 BEST] {best_b3} | val_obj={results_b3[best_b3]['val_obj']:.4f}")

    np.save(out_dir / "b3_pred_test.npy",
            results_b3[best_b3]["pred_test"])
    return results_b3, best_b3
