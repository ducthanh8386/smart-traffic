@echo off
cd /d %~dp0
set "PYTHON_EXE="

for /f "tokens=1,* delims==" %%A in ('findstr /b "executable =" .venv\pyvenv.cfg 2^>nul') do set "PYTHON_EXE=%%B"
if defined PYTHON_EXE set "PYTHON_EXE=%PYTHON_EXE:~1%"

if not defined PYTHON_EXE (
  for /f "tokens=1,* delims==" %%A in ('findstr /b "home =" .venv\pyvenv.cfg 2^>nul') do set "PYTHON_HOME=%%B"
)
if defined PYTHON_HOME (
  set "PYTHON_HOME=%PYTHON_HOME:~1%"
  if exist "%PYTHON_HOME%\python.exe" set "PYTHON_EXE=%PYTHON_HOME%\python.exe"
)

if not defined PYTHON_EXE (
  if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
)

set "PYTHONPATH=%CD%\.venv\Lib\site-packages;%CD%"
if defined PYTHON_EXE (
  "%PYTHON_EXE%" -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
) else (
  python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
)
pause

