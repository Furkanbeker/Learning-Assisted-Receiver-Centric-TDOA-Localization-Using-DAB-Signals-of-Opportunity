#!/usr/bin/env bash
# =============================================================================
# tx_random.sh  ("Random" mode of the GUI)
#
# Transmits the currently active ETI file (eti/myfirst.eti) with the
# PlutoSDR. If no ETI is active yet, the first file in eti/ is used.
# No audio file is needed.
#
# Usage:
#   ./tx_random.sh URI FREQ_HZ GAIN_DB [LOOP] [DURATION_S]
# =============================================================================
set -euo pipefail

URI="${1:?uri}"
FREQ="${2:?freq_hz}"
GAIN="${3:?gain_db}"
LOOP="${4:-1}"
DUR="${5:-0}"

# Work in the dab_tools folder (parent of scripts/)
cd "$(dirname "$0")/.."

if [[ ! -e eti/myfirst.eti ]]; then
  CAND="$(ls -1 eti/*.eti 2>/dev/null | head -n 1 || true)"
  if [[ -z "$CAND" ]]; then
    echo "[ERROR] No eti/*.eti file for Random mode. Create one first with File mode (tx_file.sh) or copy an ETI file into eti/." >&2
    exit 1
  fi
  ln -sf "$(basename "$CAND")" eti/myfirst.eti
  echo "[INFO] myfirst.eti missing -> linked to $(basename "$CAND")" >&2
fi

./run_dabmod_tx.sh "$URI" "$FREQ" "$GAIN" "$LOOP" "$DUR"
