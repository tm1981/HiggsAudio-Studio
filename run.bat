@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   Higgs Audio Studio
echo ========================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if not exist "python\python.exe" (
    echo ERROR: Python not found! Please run install.bat
    pause
    exit /b 1
)
if not exist "app.py" (
    echo ERROR: app.py not found!
    pause
    exit /b 1
)

if exist "cuda_version.txt" (
    set /p CUDA_VERSION=<cuda_version.txt
    echo Configuration: !CUDA_VERSION!
)

REM === ISOLATION: all cache/models/temp directories inside the application folder ===
set "TEMP=%SCRIPT_DIR%temp"
set "TMP=%SCRIPT_DIR%temp"
set "GRADIO_TEMP_DIR=%SCRIPT_DIR%temp"
if not exist "%TEMP%" mkdir "%TEMP%"

set "HF_HOME=%SCRIPT_DIR%models"
set "HUGGINGFACE_HUB_CACHE=%SCRIPT_DIR%models"
set "TRANSFORMERS_CACHE=%SCRIPT_DIR%models"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

set "TORCH_HOME=%SCRIPT_DIR%models\torch"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"

set "XDG_CACHE_HOME=%SCRIPT_DIR%cache"
if not exist "%XDG_CACHE_HOME%" mkdir "%XDG_CACHE_HOME%"

if exist "%SCRIPT_DIR%ffmpeg\ffmpeg.exe" (
    set "PATH=%SCRIPT_DIR%ffmpeg;%PATH%"
)

set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set GGML_CUDA_NO_PINNED=1

echo Starting application...
python\python.exe app.py

if errorlevel 1 (
    echo.
    echo ERROR during startup! Possible reasons:
    echo  1. Dependencies not installed - run install.bat
    echo  2. Insufficient VRAM - select a smaller director model in the UI
    echo  3. CUDA driver issues
    pause
    exit /b 1
)
pause
