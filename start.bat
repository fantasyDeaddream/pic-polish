@echo off
setlocal EnableExtensions
rem =====================================================================
rem  PicPolish launcher (Windows) - ASCII only, double-click friendly.
rem  Requires PowerShell 7+ (pwsh).
rem  Thin wrapper around scripts\server.ps1 -Action start
rem  Extra arguments are forwarded, e.g.: start.bat -NoBrowser -Port 8795
rem  First run is fully automatic: it may prepare the Python env
rem  (setup_env.ps1, incl. installing torch if needed) and download the
rem  model weights (fetch_models.py) before the server starts.
rem  Set PIC_POLISH_NO_PAUSE=1 to skip the final pause (for automation).
rem =====================================================================
set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%scripts\server.ps1"

where.exe pwsh >nul 2>nul
if errorlevel 1 (
    echo [ERROR] PowerShell 7+ ^(pwsh^) was not found in PATH.
    echo         PicPolish needs pwsh 7 or newer.
    echo         Install it from https://aka.ms/powershell then run this file again.
    if not defined PIC_POLISH_NO_PAUSE pause
    exit /b 1
)

if not exist "%PS1%" (
    echo [ERROR] Missing script: "%PS1%"
    if not defined PIC_POLISH_NO_PAUSE pause
    exit /b 1
)

pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -Action start %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo PicPolish is running. This window can be closed - the server keeps running.
    echo To stop it and clean the image cache, double-click stop.bat
) else (
    echo [FAILED] PicPolish failed to start ^(exit code %RC%^).
    echo          Check logs\server.log and logs\server.err.log for details.
)
echo.
if not defined PIC_POLISH_NO_PAUSE pause
exit /b %RC%
