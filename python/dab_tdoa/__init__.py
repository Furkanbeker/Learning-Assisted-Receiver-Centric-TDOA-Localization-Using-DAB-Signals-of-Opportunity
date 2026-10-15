"""
dab_tdoa: learning-assisted receiver-centric TDOA localization with DAB
signals of opportunity (training and evaluation code).

Modules
-------
config    : all user settings (paths, seeds, model switches, EKF grids)
utils     : small helpers (metrics, JSON output)
data      : CSV loading, cleaning and the scenario-wise split
features  : leakage-free feature engineering and feature selection
models    : regression models, training, prediction and ensembling
gn        : weighted Gauss-Newton TDOA solver (Python version)
ekf       : tightly coupled TDOA CT-EKF used in Level 3
levels    : Level 1 / Level 2 (B1) / Level 3 (B3) pipelines
analysis  : summary tables, per-group analysis and plots
"""
