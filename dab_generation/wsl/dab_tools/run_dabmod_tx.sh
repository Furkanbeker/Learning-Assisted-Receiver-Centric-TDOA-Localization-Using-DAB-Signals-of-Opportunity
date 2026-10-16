#!/usr/bin/env bash
# =============================================================================
# run_dabmod_tx.sh
#
# Transmits eti/myfirst.eti with ODR-DabMod through an ADALM-PLUTO
# (SoapySDR output). The base configuration configs/dabmod_pluto_tx.ini is
# copied to dabmod_runtime.ini and the Pluto URI, the centre frequency and
# the TX gain are filled in. The ETI file is played in a loop until the
# process is stopped (Stop button in the GUI or Ctrl+C).
#
# Usage:
#   ./run_dabmod_tx.sh URI FREQ_HZ GAIN_DB [LOOP] [DURATION_S]
#
#   URI         Pluto URI, e.g. ip:192.168.2.1
#   FREQ_HZ     centre frequency in Hz, e.g. 195936000 (channel 8A)
#   GAIN_DB     PlutoSDR TX gain
#   LOOP, DURATION_S  accepted for compatibility with the GUI; not used
#                     here (the ETI loop is set in the configuration)
#
# Called by scripts/tx_file.sh and scripts/tx_random.sh.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"

URI="$1"
FREQ="$2"
GAIN="$3"
LOOP="${4:-0}"
DUR="${5:-0}"

CONFIG="configs/dabmod_pluto_tx.ini"
TMP_CONFIG="dabmod_runtime.ini"

cp "$CONFIG" "$TMP_CONFIG"

# Fill in the runtime parameters
sed -i "s|uri=.*|uri=${URI}|" "$TMP_CONFIG"
sed -i "s|frequency=.*|frequency=${FREQ}|" "$TMP_CONFIG"
sed -i "s|txgain=.*|txgain=${GAIN}|" "$TMP_CONFIG"

echo "[TX] Using ETI source: $(grep '^source=' "$TMP_CONFIG")"
echo "[TX] Starting ODR-DabMod..."
odr-dabmod "$TMP_CONFIG"
