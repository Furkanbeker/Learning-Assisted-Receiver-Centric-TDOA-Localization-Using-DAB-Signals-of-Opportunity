@echo off
REM Start the DAB TX GUI (Windows). Requires Python with PySide6 (pip install -r requirements.txt).
cd /d "%~dp0"
REM If Qt reports a missing "windows" platform plugin (e.g. with Anaconda), set:
REM set QT_QPA_PLATFORM_PLUGIN_PATH=<python folder>\Lib\site-packages\PySide6\plugins\platforms
python main.py
pause
