@echo off
title Fast-LLVC - PTH to iOS CoreML Converter & AirTransfer
cd /d "%~dp0"

echo ============================================================
echo   Fast-LLVC: .pth to iOS CoreML Converter
echo   Easily convert PyTorch weights and send to iPhone / iPad!
echo ============================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "venv_cpu\Scripts\activate.bat" (
    call venv_cpu\Scripts\activate.bat
)

python pth_to_ios_converter.py
if errorlevel 1 (
    echo.
    echo [!] Process ended. Press any key to exit...
    pause > nul
)
