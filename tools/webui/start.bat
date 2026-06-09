@echo off
REM Launch the F2K web UI on Windows: worker (if down) + Flask, via the venv python.
REM   set F2K_WEB_PASSWORD=yourpw & tools\webui\start.bat
setlocal
set "HERE=%~dp0"
set "ROOT=%HERE%..\.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%HERE%start.py" %*
endlocal
