$ErrorActionPreference = 'Stop'
$V12LeagueRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$V12LeaguePython = Join-Path $V12LeagueRoot '.venv/Scripts/python.exe'
if ($env:V12_PYTHON) { $V12LeaguePython = $env:V12_PYTHON }
if (-not (Test-Path -LiteralPath $V12LeaguePython -PathType Leaf)) {
    throw "Python environment missing: $V12LeaguePython. Set V12_PYTHON to an existing interpreter."
}
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONHASHSEED = '0'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:VECLIB_MAXIMUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
& $V12LeaguePython (Join-Path $V12LeagueRoot 'code/run.py') -m agent_league.windows @args
exit $LASTEXITCODE
