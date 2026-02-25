@echo off
echo ============================================
echo Building detectron2 from source with MSVC
echo ============================================

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo FAILED: Could not activate Visual Studio environment
    exit /b 1
)

echo MSVC environment activated.

set FORCE_CUDA=0
set DISTUTILS_USE_SDK=1

echo Installing detectron2...
"y:\Research_Windows\.venv\Scripts\pip.exe" install -e "y:\Research_Windows\_detectron2_build" --no-build-isolation
if errorlevel 1 (
    echo ============================================
    echo Build with pip failed. Trying setup.py...
    echo ============================================
    cd /d "y:\Research_Windows\_detectron2_build"
    "y:\Research_Windows\.venv\Scripts\python.exe" setup.py build develop
    if errorlevel 1 (
        echo FAILED: detectron2 build failed
        exit /b 1
    )
)

echo ============================================
echo detectron2 build SUCCEEDED
echo ============================================
