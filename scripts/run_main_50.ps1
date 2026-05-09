$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$start = Get-Date
python -u code\run_amazon_experiments.py --runs 50 --drift-runs 50 2>&1 | Tee-Object -FilePath logs\main_50run.log
if ($LASTEXITCODE -ne 0) { throw "main_50run failed with exit code $LASTEXITCODE" }
$end = Get-Date
[PSCustomObject]@{
  started = $start.ToString("s")
  ended = $end.ToString("s")
  elapsed_seconds = [Math]::Round(($end - $start).TotalSeconds, 3)
} | ConvertTo-Json | Set-Content -Encoding UTF8 logs\main_50run_runtime.json
Get-Content results\run_summary.json
