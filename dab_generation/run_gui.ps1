# Start the DAB TX GUI (Windows PowerShell).
# Requires Python with PySide6 (pip install -r requirements.txt).
# If Qt reports a missing "windows" platform plugin (e.g. with Anaconda), set:
#   $env:QT_QPA_PLATFORM_PLUGIN_PATH = "<python folder>\Lib\site-packages\PySide6\plugins\platforms"
Set-Location $PSScriptRoot
python .\main.py
