@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto ready
call setup_windows.bat
if errorlevel 1 exit /b 1
:ready
.venv\Scripts\python.exe run.py
if errorlevel 1 pause
