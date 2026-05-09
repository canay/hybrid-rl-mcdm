$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$started = Get-Date
python -u code\validation_extensions.py --runs 30 --size-runs 30 --sizes 200 400 800 2>&1 | Tee-Object -FilePath logs\validation_extensions.log
if ($LASTEXITCODE -ne 0) { throw "validation_extensions failed with exit code $LASTEXITCODE" }
$ended = Get-Date
[PSCustomObject]@{
  started = $started.ToString("s")
  ended = $ended.ToString("s")
  elapsed_seconds = [Math]::Round(($ended - $started).TotalSeconds, 3)
} | ConvertTo-Json | Set-Content -Encoding UTF8 logs\validation_extensions_runtime.json
