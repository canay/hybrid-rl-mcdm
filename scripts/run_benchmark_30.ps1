$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$start = Get-Date
python -u code\benchmark_recommenders.py --runs 30 --min-items 3 --factors 32 --epochs 80 2>&1 | Tee-Object -FilePath logs\benchmark_30run.log
if ($LASTEXITCODE -ne 0) { throw "benchmark_30run failed with exit code $LASTEXITCODE" }
$end = Get-Date
[PSCustomObject]@{
  started = $start.ToString("s")
  ended = $end.ToString("s")
  elapsed_seconds = [Math]::Round(($end - $start).TotalSeconds, 3)
} | ConvertTo-Json | Set-Content -Encoding UTF8 logs\benchmark_30run_runtime.json
Get-Content results\recommender_benchmarks_summary.csv
