# DenseCap Setup Script for Windows 11
# Run this in PowerShell from the project root directory
# This script recreates the full environment from scratch

param(
    [switch]$SkipPython,
    [switch]$SkipDetectron2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  DenseCap Environment Setup (Windows 11)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# Step 1: Check Python 3.11
if (-not $SkipPython) {
    Write-Host "`n[1/5] Checking Python 3.11..." -ForegroundColor Yellow
    try {
        $pyVersion = py -3.11 --version 2>&1
        Write-Host "  Found: $pyVersion" -ForegroundColor Green
    } catch {
        Write-Host "  Python 3.11 not found. Installing via winget..." -ForegroundColor Red
        winget install Python.Python.3.11 --silent --accept-source-agreements --accept-package-agreements
        Write-Host "  Python 3.11 installed." -ForegroundColor Green
    }
}

# Step 2: Create venv
Write-Host "`n[2/5] Creating virtual environment..." -ForegroundColor Yellow
$venvPath = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path $venvPath)) {
    py -3.11 -m venv $venvPath
    Write-Host "  Created .venv/" -ForegroundColor Green
} else {
    Write-Host "  .venv/ already exists" -ForegroundColor Green
}

$pip = Join-Path $venvPath "Scripts\pip.exe"
$python = Join-Path $venvPath "Scripts\python.exe"

& $pip install --upgrade pip setuptools wheel

# Step 3: Install PyTorch with CUDA 12.8
Write-Host "`n[3/5] Installing PyTorch nightly (CUDA 12.8)..." -ForegroundColor Yellow
& $pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cu128

# Step 4: Install remaining dependencies
Write-Host "`n[4/5] Installing dependencies..." -ForegroundColor Yellow
& $pip install -r (Join-Path $ProjectRoot "requirements.txt")

# Step 5: Build detectron2
if (-not $SkipDetectron2) {
    Write-Host "`n[5/5] Building detectron2 from source..." -ForegroundColor Yellow
    $d2path = Join-Path $ProjectRoot "_detectron2_build"
    if (-not (Test-Path $d2path)) {
        git clone https://github.com/facebookresearch/detectron2.git $d2path
    }
    & cmd /c (Join-Path $ProjectRoot "build_detectron2.bat")
}

# Register Jupyter kernel
Write-Host "`nRegistering Jupyter kernel..." -ForegroundColor Yellow
& $python -m ipykernel install --user --name densecap-gpu --display-name "DenseCap GPU (Python 3.11)"

# Verify
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Verification" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
& $python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
& $python -c "import detectron2; print('detectron2: OK')"

Write-Host "`n✓ Setup complete! Open the notebook and select 'DenseCap GPU (Python 3.11)' kernel." -ForegroundColor Green
