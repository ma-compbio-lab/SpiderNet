# Dot-source from the repository root: . ./scripts/activate_reproduction.ps1
$releaseScript = Join-Path $PSScriptRoot 'release_environment.py'
$releaseLines = & python -B $releaseScript --configure --shell powershell
if ($LASTEXITCODE -ne 0) { throw 'Could not configure reproduction paths' }
$releaseLines | ForEach-Object { Invoke-Expression $_ }
Write-Host 'Saved-result paths now point to this checkout. Existing analysis commands are unchanged.'
