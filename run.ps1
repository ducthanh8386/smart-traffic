$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvConfig = Join-Path $ProjectRoot ".venv\pyvenv.cfg"
$PythonExe = $null

if (Test-Path $VenvConfig) {
  $execLine = Get-Content $VenvConfig | Where-Object { $_ -like "executable =*" } | Select-Object -First 1
  if ($execLine) {
    $candidate = ($execLine -replace "^executable =\s*", "").Trim()
    if (Test-Path $candidate) { $PythonExe = $candidate }
  }
  if (-not $PythonExe) {
    $homeLine = Get-Content $VenvConfig | Where-Object { $_ -like "home =*" } | Select-Object -First 1
    if ($homeLine) {
      $homeDir = ($homeLine -replace "^home =\s*", "").Trim()
      $candidate = Join-Path $homeDir "python.exe"
      if (Test-Path $candidate) { $PythonExe = $candidate }
    }
  }
}

if (-not $PythonExe) {
  $candidate = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
  if (Test-Path $candidate) { $PythonExe = $candidate }
}

if (-not $PythonExe) {
  throw "Cannot find a valid Python executable for this project."
}

$env:PYTHONPATH = "$ProjectRoot\.venv\Lib\site-packages;$ProjectRoot"
& $PythonExe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

