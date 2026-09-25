@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ========================================
echo   Higgs Audio Studio - Installation
echo ========================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
set "TEMP=%SCRIPT_DIR%temp"
set "TMP=%SCRIPT_DIR%temp"

if not exist "downloads" mkdir downloads
if not exist "temp" mkdir temp
if not exist "models" mkdir models
if not exist "cache" mkdir cache
if not exist "output" mkdir output
if not exist "voices" mkdir voices

REM ============================================================
REM  Step 1: GPU Selection (torch 2.7.1 in all branches - prebuilt accelerators available)
REM ============================================================
echo.
echo Select GPU:
echo.
echo   1. NVIDIA GTX 10xx (Pascal)
echo   2. NVIDIA RTX 20xx (Turing)
echo   3. NVIDIA RTX 30xx (Ampere)
echo   4. NVIDIA RTX 40xx (Ada Lovelace)
echo   5. NVIDIA RTX 50xx (Blackwell)
echo   6. CPU only (no GPU)
echo.
set /p GPU_CHOICE="Enter choice (1-6): "

if "%GPU_CHOICE%"=="1" goto :gpu_10xx
if "%GPU_CHOICE%"=="2" goto :gpu_20xx
if "%GPU_CHOICE%"=="3" goto :gpu_30xx
if "%GPU_CHOICE%"=="4" goto :gpu_40xx
if "%GPU_CHOICE%"=="5" goto :gpu_50xx
if "%GPU_CHOICE%"=="6" goto :gpu_cpu
echo Invalid choice!
pause
exit /b 1

:gpu_10xx
set "CUDA_VERSION=cu118"
set "CUDA_NAME=CUDA 11.8 (GTX 10xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_20xx
set "CUDA_VERSION=cu126"
set "CUDA_NAME=CUDA 12.6 (RTX 20xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_30xx
set "CUDA_VERSION=cu126"
set "CUDA_NAME=CUDA 12.6 (RTX 30xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_40xx
set "CUDA_VERSION=cu128"
set "CUDA_NAME=CUDA 12.8 (RTX 40xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_50xx
set "CUDA_VERSION=cu128"
set "CUDA_NAME=CUDA 12.8 (RTX 50xx)"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done
:gpu_cpu
set "CUDA_VERSION=cpu"
set "CUDA_NAME=CPU only"
set "TORCH_VERSION=2.7.1"
set "TORCHAUDIO_VERSION=2.7.1"
goto :gpu_done

:gpu_done
echo.
echo Selected: %CUDA_NAME%
echo.

REM ============================================================
REM  Step 2: Python 3.12.9 embed
REM ============================================================
if exist "python\python.exe" (
    echo [OK] Python is already installed
) else (
    echo [1/7] Downloading Python 3.12.9...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/python-3.12.9-embed-amd64.zip' -OutFile 'downloads\python.zip'}"
    powershell -Command "& {Expand-Archive -Path 'downloads\python.zip' -DestinationPath 'python' -Force}"
    cd python
    if exist "python312._pth" (
        echo python312.zip> python312._pth
        echo .>> python312._pth
        echo Lib\site-packages>> python312._pth
        echo ..\Lib\site-packages>> python312._pth
        echo import site>> python312._pth
    )
    cd ..
    echo [OK] Python 3.12.9 installed
)

REM ============================================================
REM  Step 3: pip
REM ============================================================
if exist "python\Scripts\pip.exe" (
    echo [OK] pip is already installed
) else (
    echo [2/7] Installing pip...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'downloads\get-pip.py'}"
    python\python.exe downloads\get-pip.py --no-warn-script-location
)
python\python.exe -m pip install --upgrade pip setuptools wheel --no-warn-script-location

REM ============================================================
REM  Step 4: PyTorch 2.7.1
REM ============================================================
echo [3/7] Installing PyTorch %TORCH_VERSION% (%CUDA_NAME%)...
if "%CUDA_VERSION%"=="cpu" (
    python\python.exe -m pip install torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --no-warn-script-location
) else (
    python\python.exe -m pip install torch==%TORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/%CUDA_VERSION% --no-warn-script-location
)

REM ============================================================
REM  Step 5: Dependencies
REM ============================================================
echo [4/7] Installing dependencies...
python\python.exe -m pip install -r requirements.txt --no-warn-script-location
REM Cloud voices are downloaded via huggingface_hub (httpx). Remove urllib3-future/niquests
REM if brought in transitively - their broken HTTP/2 (hface) broke voice downloads (issue #2).
python\python.exe -m pip uninstall -y urllib3-future niquests 2>nul

REM ============================================================
REM  Step 6: Triton for torch.compile (~2x). Parentheses/for-blocks omitted (goto-flow).
REM  Higgs uses SDPA (built-in flash kernels) - external Flash-Attention 2 is NOT needed (model rejects it).
REM ============================================================
if "%CUDA_VERSION%"=="cpu" goto :after_accel
echo [5/7] Installing Triton for torch.compile...
python\python.exe -m pip install "triton-windows>=3.0.0,<3.4" --no-warn-script-location
if exist "python\Include\Python.h" goto :after_accel
echo Downloading Python headers for Triton...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/amd64/dev.msi' -OutFile 'downloads\pydev.msi'}"
if not exist "downloads\pydev.msi" goto :after_accel
msiexec /a "downloads\pydev.msi" /qn TARGETDIR="%SCRIPT_DIR%downloads\pydev_extract"
if not exist "python\Include" mkdir "python\Include"
if not exist "python\libs" mkdir "python\libs"
xcopy /E /Y "downloads\pydev_extract\include\*" "python\Include\" >nul 2>&1
xcopy /E /Y "downloads\pydev_extract\libs\*" "python\libs\" >nul 2>&1
if exist "downloads\pydev_extract" rmdir /s /q "downloads\pydev_extract"
echo [OK] Python headers installed
:after_accel

REM ============================================================
REM  Step 7: llama-cpp-python (GGUF director, GPU) + ITS OWN CUDA 12.4 runtime.
REM  The abetlen wheel is built for cu124 and does not bundle cudart/cublas. Taking them from torch (12.6/12.8)
REM  is NOT allowed: ggml-cuda(12.4)+cuBLAS(12.8) breaks shared CUDA context -> torch "invalid argument".
REM  Install llama's native 12.4 runtime (nvidia packages), NOT from torch.
REM ============================================================
echo [6/7] Installing llama-cpp-python (AI director, GGUF)...
if "%CUDA_VERSION%"=="cpu" goto :llama_cpu
python\python.exe -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --no-warn-script-location
echo Installing CUDA 12.4 runtime for llama (matches ggml-cuda build)...
python\python.exe -m pip install nvidia-cuda-runtime-cu12==12.4.127 nvidia-cublas-cu12==12.4.5.8 --no-warn-script-location
echo Placing 12.4 cudart/cublas next to llama.dll...
for %%P in (cuda_runtime cublas) do for %%D in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll) do if exist "python\Lib\site-packages\nvidia\%%P\bin\%%D" copy /y "python\Lib\site-packages\nvidia\%%P\bin\%%D" "python\Lib\site-packages\llama_cpp\lib\%%D" >nul
if not exist "python\Lib\site-packages\llama_cpp\lib\cublasLt64_12.dll" echo [WARNING] cublasLt64_12.dll was not copied - AI director may fail to start!
goto :after_llama
:llama_cpu
python\python.exe -m pip install llama-cpp-python --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-warn-script-location
:after_llama

REM ============================================================
REM  Step 8: Starter voice-pack (voice presets downloaded as zip from HF)
REM ============================================================
echo [+] Downloading starter voice-pack...
if exist "voices\*.mp3" (
    echo [OK] Voice presets are already present
) else (
    curl -L -o downloads\voice-pack.zip https://huggingface.co/datasets/nerualdreming/VibeVoice/resolve/main/voice-pack.zip
    if exist "downloads\voice-pack.zip" (
        powershell -Command "& {Expand-Archive -Path 'downloads\voice-pack.zip' -DestinationPath 'downloads\vp' -Force}"
        if exist "downloads\vp\voice-pack" (
            xcopy /E /Y /Q "downloads\vp\voice-pack\*" "voices\" >nul
        ) else (
            xcopy /E /Y /Q "downloads\vp\*" "voices\" >nul
        )
        echo [OK] Voice-pack installed
    )
)
echo [7/7] Finalizing...
echo %CUDA_VERSION%> cuda_version.txt

echo ========================================
echo   Installation complete!
echo   To start: run.bat
echo ========================================
pause
