@echo off
cd /d "%~dp0"
echo =========================================================
echo   LLVC Ultra-Fast Voice Conversion Studio (ROCm Optimized)
echo =========================================================
echo.
echo Starting LLVC Studio WebUI with Python 3.12 (ROCm)...
echo Opening browser automatically...
echo.

py -3.12 gui.py

echo.
echo Application finished.
pause
