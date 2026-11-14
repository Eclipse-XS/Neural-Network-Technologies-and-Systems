param([string]$Python = 'py', [switch]$DownloadEmbedding, [switch]$SkipEmbedding, [switch]$SkipDocker)
$ErrorActionPreference = 'Stop'
$labPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $labPython)) {
    & $Python -c 'import sys; assert (3,12) <= sys.version_info[:2] < (3,14), "Python 3.12-3.13 required"'
    if ($LASTEXITCODE -ne 0) { throw 'Select Python 3.12-3.13 with -Python <executable>.' }
    & $Python -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $labPython -c 'import sys; assert (3,12) <= sys.version_info[:2] < (3,14), "Existing Lab 7 environment requires Python 3.12-3.13"'
if ($LASTEXITCODE -ne 0) { throw 'The existing Lab 7 environment has an incompatible Python version.' }
& $labPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Inspect the pip error above.' }
& $labPython -m pip install --no-deps --editable $PSScriptRoot
if ($LASTEXITCODE -ne 0) { throw 'Editable package installation failed.' }
if (-not $SkipEmbedding) {
    & $labPython (Join-Path $PSScriptRoot '../../scripts/fetch_models.py') e5
    if ($LASTEXITCODE -ne 0) { throw 'Embedding download failed. Check network access to Hugging Face.' }
}
& $labPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts remain. Setup failed; inspect pip check output.' }
if (-not $SkipDocker) {
    & docker info --format '{{.ServerVersion}}'
    if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop Linux engine, then rerun setup; use -SkipDocker only for offline component tests.' }
    & docker pull redis/redis-stack:7.4.0-v8
    if ($LASTEXITCODE -ne 0) { throw 'Redis Stack image download failed.' }
    $expected = (Get-Content -Raw (Join-Path $PSScriptRoot '../../services.lock.json') | ConvertFrom-Json).redis.digest
    $digests = & docker image inspect redis/redis-stack:7.4.0-v8 --format '{{json .RepoDigests}}'
    if ($LASTEXITCODE -ne 0 -or ($digests | ConvertFrom-Json) -notcontains $expected) { throw 'Redis image digest differs from services.lock.json' }
}
Write-Output "Lab interpreter: $labPython"
Write-Output 'Setup complete. Start the application with .\run.ps1.'
