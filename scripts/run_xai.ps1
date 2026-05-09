$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$start = Get-Date
python -u code\xai_analysis.py --max-runs 50 --shap-sample 5000 2>&1 | Tee-Object -FilePath logs\xai_analysis.log
if ($LASTEXITCODE -ne 0) { throw "xai_analysis failed with exit code $LASTEXITCODE" }
$end = Get-Date
[PSCustomObject]@{
  started = $start.ToString("s")
  ended = $end.ToString("s")
  elapsed_seconds = [Math]::Round(($end - $start).TotalSeconds, 3)
} | ConvertTo-Json | Set-Content -Encoding UTF8 logs\xai_analysis_runtime.json
Get-Content results\xai\xai_report.json -TotalCount 120
