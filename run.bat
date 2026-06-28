@echo off
REM ===========================================================================
REM  run.bat - build (if needed) and run the ncnn-in-wazero demo on Windows.
REM
REM  Usage:
REM     run.bat                 classify tests\1.jpg (default)
REM     run.bat tests\2.jpg     classify a specific image
REM ===========================================================================
setlocal
cd /d "%~dp0"

REM --- Build the wasm module if it doesn't exist yet ---------------------------
if not exist "mod\ncnn\ncnn_classify.wasm" (
    echo [run] mod\ncnn\ncnn_classify.wasm not found - building it...
    where bash >nul 2>nul
    if errorlevel 1 (
        echo [run] ERROR: 'bash' not found. Install Git for Windows, or run:
        echo        bash build/build.sh
        exit /b 1
    )
    bash build/build.sh
    if errorlevel 1 (
        echo [run] build failed.
        exit /b 1
    )
)

REM --- Run the pure-Go host (wazero) ------------------------------------------
echo [run] running classifier via wazero...
go run . %*
endlocal
