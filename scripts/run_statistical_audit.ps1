$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
python -u code\statistical_audit.py 2>&1 | Tee-Object -FilePath logs\statistical_audit.log
if ($LASTEXITCODE -ne 0) { throw "statistical_audit failed with exit code $LASTEXITCODE" }
