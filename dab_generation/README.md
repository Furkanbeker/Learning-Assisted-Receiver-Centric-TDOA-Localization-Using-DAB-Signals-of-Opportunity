# DAB waveform generation and over-the-air test

Tools used to create the DAB waveform of this repository
(`data/dab_iq/basic.cf32`) and to transmit it with an ADALM-PLUTO SDR for
the over-the-air test. They are **not needed** to run the MATLAB simulator
or the Python pipeline; they document how the waveform was produced.

```
 Windows                          WSL (Ubuntu)
 ┌──────────────────┐   wsl bash   ┌──────────────────────────────────────────────┐
 │ PySide6 GUI      │ ───────────► │ audio ─ffmpeg─► WAV ─odr-audioenc─► DAB+     │
 │ (main.py)        │              │       ─odr-dabmux─► ETI ─odr-dabmod─┬─► Pluto │
 │ log window       │ ◄─────────── │                                     └─► .cf32 │
 └──────────────────┘   output     └──────────────────────────────────────────────┘
```

All signal processing is done by the Opendigitalradio (ODR) tools; the
GUI only starts the scripts and shows their output.

## Folder structure

```
dab_generation/
├── main.py              GUI entry point (Windows)
├── ui_main.py           GUI widgets
├── tx_manager.py        Builds and runs the WSL commands
├── ui_style.qss         Dark theme
├── run_gui.bat / .ps1   Launchers
├── requirements.txt     PySide6
├── audio/basic.wav      Source audio of data/dab_iq/basic.cf32 (5.16 s)
└── wsl/
    ├── setup_wsl.sh     Installs ffmpeg, ODR tools and SoapySDR in WSL
    └── dab_tools/
        ├── make_eti.sh          audio -> ETI
        ├── run_dabmod_tx.sh     ETI -> PlutoSDR (ODR-DabMod)
        ├── configs/dabmod_pluto_tx.ini
        └── scripts/
            ├── tx_record.sh     Record mode: audio -> IQ file (.cf32), no SDR
            ├── tx_file.sh       File mode:   audio -> ETI -> PlutoSDR
            └── tx_random.sh     Random mode: active ETI -> PlutoSDR
```

## Requirements

- Windows 10/11 with WSL 2 (Ubuntu) for the GUI. The scripts in
  `wsl/dab_tools` also run on any Linux machine without the GUI.
- In WSL / Linux: `ffmpeg`, `odr-audioenc`, `odr-dabmux`, `odr-dabmod`
  (built with SoapySDR support), the SoapySDR PlutoSDR module and
  `python3`. `wsl/setup_wsl.sh` installs the available packages.
- On Windows: Python 3 with PySide6 (`pip install -r requirements.txt`).
- An ADALM-PLUTO, only for transmission (File and Random modes).

The shell scripts must keep Unix (LF) line endings; the repository's
`.gitattributes` takes care of this on Windows.

## Usage

1. Install the tools once inside WSL: `bash wsl/setup_wsl.sh`
2. Start the GUI on Windows: `run_gui.bat` (or `python main.py`)
3. Select the mode and press **Start**:

| Mode | What it does | SDR needed |
|---|---|---|
| Record | Converts the selected audio file into a baseband IQ file (complex float32, 2.048 MS/s) and a JSON metadata file | No |
| File | Builds an ETI file from the audio file and transmits it on the selected DAB channel | Yes |
| Random | Transmits the active ETI file (`eti/myfirst.eti`) again | Yes |

### Reproducing the waveform of the paper

`data/dab_iq/basic.cf32` was produced in **Record** mode with
`audio/basic.wav`, label `MediComm-DAB` and 120 kbps. Without the GUI
(inside WSL or Linux, from `wsl/dab_tools`):

```bash
bash scripts/tx_record.sh ../../audio/basic.wav MediComm-DAB 120 48000 2 195936000 ./basic.cf32
```

The DAB ensemble carries the local time, so a regenerated file is
equivalent to the one in the repository but not byte-identical.

### Generated files

`eti/`, `offline_record/` and `dabmod_runtime.ini` are created in
`wsl/dab_tools` while the tools run and are not tracked by git.

## Notes

- The **Duration** and **Loop** settings of the GUI are passed to the
  scripts but not applied: in File and Random modes the ETI file is
  played in a loop until **Stop** is pressed.
- At start-up the GUI pings 192.168.0.1 ... 192.168.20.1 to look for a
  PlutoSDR. The Pluto URI is entered in the GUI (default `ip:192.168.2.1`).
- The WSL folder used by the GUI is derived from the location of
  `main.py`. Set the environment variable `DAB_WSL_DIR` to use another one.

## Regulatory note

Transmitting in the DAB band (Band III, 174-240 MHz) requires an
authorization in most countries. Use a cable connection with attenuators
or a shielded environment and keep the TX gain low. The GUI also contains
a test preset at 860 MHz, which is not a DAB channel.
