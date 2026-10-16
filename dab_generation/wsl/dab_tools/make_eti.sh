#!/usr/bin/env bash
# =============================================================================
# make_eti.sh
#
# Converts an audio file into a DAB+ ETI file (one ensemble, one service):
#   audio --ffmpeg--> WAV --odr-audioenc--> .dabp --odr-dabmux--> .eti
#
# Usage:
#   ./make_eti.sh AUDIO OUT_ETI [LABEL] [BITRATE_KBPS] [SR] [CH]
#
#   AUDIO         input audio file (Windows paths such as C:\... are accepted)
#   OUT_ETI       output ETI file
#   LABEL         service label             (default: MediComm-DAB)
#   BITRATE_KBPS  DAB+ sub-channel bitrate  (default: 120)
#   SR, CH        sample rate / channels of the intermediate WAV (48000, 2)
#
# The ETI file contains 2000 DAB frames (24 ms each, about 48 s).
# Called by scripts/tx_file.sh.
# =============================================================================
set -euo pipefail

AUDIO_FILE="${1:-}"
OUT_ETI="${2:-}"
LABEL="${3:-MediComm-DAB}"
BITRATE="${4:-120}"     # kbps
SR="${5:-48000}"
CH="${6:-2}"

if [[ -z "$AUDIO_FILE" || -z "$OUT_ETI" ]]; then
  echo "Usage: ./make_eti.sh AUDIO OUT_ETI [LABEL] [BITRATE_KBPS] [SR] [CH]" >&2
  exit 1
fi

# Windows path -> WSL path
if [[ "$AUDIO_FILE" =~ ^[A-Za-z]:\\ ]]; then
  AUDIO_FILE="$(wslpath -u "$AUDIO_FILE")"
elif [[ "$AUDIO_FILE" =~ ^[A-Za-z]:/ ]]; then
  AUDIO_WIN="${AUDIO_FILE/\//\\}"
  AUDIO_WIN="${AUDIO_WIN//\//\\}"
  AUDIO_FILE="$(wslpath -u "$AUDIO_WIN")"
fi

if [[ ! -f "$AUDIO_FILE" ]]; then
  echo "[ERROR] Audio not found: $AUDIO_FILE" >&2
  exit 1
fi

command -v ffmpeg >/dev/null || { echo "[ERROR] ffmpeg missing" >&2; exit 1; }
command -v odr-audioenc >/dev/null || { echo "[ERROR] odr-audioenc missing" >&2; exit 1; }
command -v odr-dabmux >/dev/null || { echo "[ERROR] odr-dabmux missing" >&2; exit 1; }

OUT_ETI_ABS="$(readlink -f "$OUT_ETI" 2>/dev/null || echo "$OUT_ETI")"
mkdir -p "$(dirname "$OUT_ETI_ABS")"
rm -f "$OUT_ETI_ABS"

# Temporary working folder, removed on exit
TMPD="$(mktemp -d /tmp/make_eti_XXXXXX)"
MUX_CFG="$TMPD/dabmux.conf"
TMP_WAV="$TMPD/input.wav"
TMP_DABP="$TMPD/audio.dabp"

cleanup() {
  rm -rf "$TMPD" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[ETI] Generating ETI..." >&2
echo "[ETI] audio=$AUDIO_FILE" >&2
echo "[ETI] out=$OUT_ETI_ABS" >&2
echo "[ETI] label=$LABEL bitrate=${BITRATE}kbps sr=$SR ch=$CH" >&2

# 1) Convert the input to a clean 16-bit PCM WAV
ffmpeg -hide_banner -loglevel error -y -i "$AUDIO_FILE" \
  -ar "$SR" -ac "$CH" -c:a pcm_s16le "$TMP_WAV"

# 2) WAV -> DAB+ audio payload (.dabp)
odr-audioenc -i "$TMP_WAV" -b "$BITRATE" -o "$TMP_DABP"

# 3) File-based ODR-DabMux configuration (DAB Mode I, one service)
cat > "$MUX_CFG" <<EOMUX
general {
  dabmode 1
  nbframes 2000
  tist false
}
remotecontrol { telnetport 0 }

ensemble {
  id 0x4fff
  ecc 0xec
  local-time-offset auto
  international-table 1
  label "DAB+ GUI"
  shortlabel "DABGUI"
}

services {
  srv1 {
    label "$LABEL"
  }
}

subchannels {
  sub1 {
    type dabplus
    bitrate $BITRATE
    id 1
    protection 3
    inputproto file
    inputuri "$TMP_DABP"
  }
}

components {
  comp1 {
    service srv1
    subchannel sub1
  }
}

outputs {
  output "file://$OUT_ETI_ABS"
}
EOMUX

# 4) Generate the ETI file
odr-dabmux "$MUX_CFG" >/tmp/make_eti_mux.log 2>&1

if [[ ! -s "$OUT_ETI_ABS" ]]; then
  echo "[ERROR] ETI empty: $OUT_ETI_ABS" >&2
  echo "Mux log: /tmp/make_eti_mux.log" >&2
  exit 1
fi

echo "[OK] ETI ready: $OUT_ETI_ABS" >&2
