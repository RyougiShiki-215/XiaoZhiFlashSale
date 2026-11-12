@echo off
REM ============================================================
REM  Windows one-click launcher for start.ps1
REM  Runs the PowerShell script with ExecutionPolicy Bypass so it
REM  works even when the system blocks .ps1 scripts (default: Restricted)
REM
REM  Usage (double-click = start / pass a word for other actions):
REM     start.cmd            -> start all services (ES + app, background)
REM     start.cmd stop       -> stop app + ES
REM     start.cmd status     -> show running status
REM     start.cmd es-only    -> start ES only
REM ============================================================
setlocal
set "DIR=%~dp0"
set "ACTION=up"
if /i "%~1"=="stop"    set "ACTION=stop"
if /i "%~1"=="status"  set "ACTION=status"
if /i "%~1"=="es-only" set "ACTION=es-only"
if /i "%~1"=="logs"    set "ACTION=logs"

echo Calling: start.ps1 -Action %ACTION%
powershell -NoProfile -ExecutionPolicy Bypass -File "%DIR%start.ps1" -Action %ACTION%

echo.
echo [done] Press any key to close this window...
pause >nul
endlocal