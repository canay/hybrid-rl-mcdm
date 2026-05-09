$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$start = Get-Date
python -u code\deep_recommender_benchmarks.py --runs 10 --epochs 30 --factors 64 --batch-size 1024 2>&1 | Tee-Object -FilePath logs\deep_benchmark_10run.log
if ($LASTEXITCODE -ne 0) { throw "deep_benchmark failed with exit code $LASTEXITCODE" }
$end = Get-Date
[PSCustomObject]@{
  started = $start.ToString("s")
  ended = $end.ToString("s")
  elapsed_seconds = [Math]::Round(($end - $start).TotalSeconds, 3)
} | ConvertTo-Json | Set-Content -Encoding UTF8 logs\deep_benchmark_10run_runtime.json
Get-Content results\deep_recommender_benchmarks_summary.csv
