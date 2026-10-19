@echo off
setlocal EnableExtensions
rem =====================================================================
rem  PicPolish stopper (Windows) - ASCII only, double-click friendly.
rem  Requires PowerShell 7+ (pwsh).
rem  Thin wrapper around scripts\server.ps1 -Action stop
rem  It also clears cache\uploads, cache\results and cache\tmp.
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

pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -Action stop %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo PicPolish has been stopped and the image cache has been cleared.
) else (
    echo [WARN] stop finished with exit code %RC%. Check the log above.
    echo        You can retry, or inspect logs\server.err.log / logs\server.log.
)
echo.
if not defined PIC_POLISH_NO_PAUSE pause
rem stop is idempotent: always exit 0 so a double-click never reports failure.
exit /b 0
