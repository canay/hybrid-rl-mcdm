$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$start = Get-Date
python -u code\mccauley_home_data.py --max-users 1000 --min-unique-items 25 2>&1 | Tee-Object -FilePath logs\mccauley_home_build.log
if ($LASTEXITCODE -ne 0) { throw "mccauley_home_data failed with exit code $LASTEXITCODE" }
python -u code\mccauley_home_experiment.py 2>&1 | Tee-Object -FilePath logs\mccauley_home_experiment.log
if ($LASTEXITCODE -ne 0) { throw "mccauley_home_experiment failed with exit code $LASTEXITCODE" }
$end = Get-Date
[PSCustomObject]@{
  started = $start.ToString("s")
  ended = $end.ToString("s")
  elapsed_seconds = [Math]::Round(($end - $start).TotalSeconds, 3)
} | ConvertTo-Json | Set-Content -Encoding UTF8 logs\mccauley_home_runtime.json
Get-Content results\mccauley_home_real_results.json -TotalCount 80
