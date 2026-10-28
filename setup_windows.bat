@echo off
cd /d "%~dp0"
echo Creating a Python 3.12 environment for Mozhi...
where py >nul 2>&1
if errorlevel 1 goto use_python
py -3.12 -m venv .venv
if errorlevel 1 goto failed
goto install
:use_python
python -m venv .venv
if errorlevel 1 goto failed
:install
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Run start_windows.bat to open the local backend.
exit /b 0
:failed
echo.
echo Setup failed. Install Python 3.12 from python.org and enable the Python launcher.
echo Then run this file again.
pause
exit /b 1
