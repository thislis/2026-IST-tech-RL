$ErrorActionPreference = 'Stop'
$V12ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$V12Python = Join-Path $V12ProjectRoot '.venv/Scripts/python.exe'
if ($env:V12_PYTHON) { $V12Python = $env:V12_PYTHON }
if (-not (Test-Path -LiteralPath $V12Python -PathType Leaf)) {
    throw "Python environment missing: $V12Python. Set V12_PYTHON to an existing interpreter."
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONHASHSEED = '0'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
& $V12Python (Join-Path $V12ProjectRoot 'code/v12/scripts/v12_experiments.py') @args
exit $LASTEXITCODE
