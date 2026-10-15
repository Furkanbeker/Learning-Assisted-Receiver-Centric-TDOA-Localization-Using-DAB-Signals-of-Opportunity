"""
User settings of the AI pipeline.

The command-line scripts (run_ai_pipeline.py, make_paper_figures.py) set
DATA_PATH and OUT_DIR from their arguments. All other values can be
changed here.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# AI-ready CSV written by matlab/generate_dab_tdoa_dataset.m.
# None = use the newest results/dab_soop_tdoa_dataset_*/ folder.
DATA_PATH = None
DATA_FILE_NAME = "dab_soop_tdoa_ai_ready_gn_correction.csv"

# Output folder of the AI pipeline (models' test predictions, tables, plots).
OUT_DIR = REPO_ROOT / "results" / "ai_results"

# -----------------------------------------------------------------------------
# Physics and reproducibility
# -----------------------------------------------------------------------------
C_LIGHT = 3e8          # Propagation speed (m/s)
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Data split and features
# -----------------------------------------------------------------------------
# Use the scenario-wise 'split' column written by MATLAB (recommended).
USE_EXISTING_SPLIT = True

# Use the normalized measurement index as a feature. It is always known
# in a real system, but it carries information about the time step.
USE_MEAS_IDX = True

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
# The results reported in the paper were obtained with XGBoost, ExtraTrees
# and HistGradientBoosting. LightGBM is optional and off by default, so
# that installing it does not change the results.
TRAIN_LGBM = False
TRAIN_XGB = True
TRAIN_ET = True
TRAIN_HGB = True

# Model used for the B1 out-of-fold predictions ("et" or "lgbm").
# The paper results use "et" (ExtraTrees).
OOF_MODEL = "et"

# Number of scenario-wise folds for the B1 out-of-fold predictions. The
# OOF predictions are the B3 training inputs, so B3 is not trained on
# B1 predictions of samples that B1 has already seen.
B1_OOF_FOLDS = 5

# -----------------------------------------------------------------------------
# Level 3 (B3 R-scale model + tight TDOA CT-EKF)
# -----------------------------------------------------------------------------
B3_RESIDUAL_FLOOR_NS = 5.0
MIN_R_SCALE = 0.05
MAX_R_SCALE = 200.0

# CT-EKF tuning grid, searched on the validation set.
EKF_Q_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
EKF_RMULT_GRID = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
