$ErrorActionPreference = 'Stop'
$V13ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$V13Python = Join-Path $V13ProjectRoot '.venv/Scripts/python.exe'
if ($env:V13_PYTHON) { $V13Python = $env:V13_PYTHON }
if (-not (Test-Path -LiteralPath $V13Python -PathType Leaf)) {
    throw "Python environment missing: $V13Python. Set V13_PYTHON to an existing interpreter."
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONHASHSEED = '0'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& $V13Python (Join-Path $V13ProjectRoot 'code/v13/scripts/v13_experiments.py') @args
exit $LASTEXITCODE
