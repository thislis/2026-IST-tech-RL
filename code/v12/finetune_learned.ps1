$ErrorActionPreference = 'Stop'
$V12LearnedConfig = Join-Path $PSScriptRoot 'configs/finetune_learned.json'
& (Join-Path $PSScriptRoot 'finetune_scripted.ps1') --config $V12LearnedConfig @args
exit $LASTEXITCODE
