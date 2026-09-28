param([string]$Python = ".\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
$labPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$sourcePackages = & $Python -c "import sysconfig, torch; print(sysconfig.get_paths()['purelib'])"
if ($LASTEXITCODE -ne 0) { throw "Supply an existing Python environment with working torch." }
if (-not (Test-Path -LiteralPath $labPython)) {
    & $Python -m venv --system-site-packages (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create isolated Lab 7 environment" }
}
$siteDirectory = Join-Path $PSScriptRoot ".venv\Lib\site-packages"
[System.IO.File]::WriteAllText((Join-Path $siteDirectory "repository_environment.pth"), "$sourcePackages`n")
$metadataDirectory = Join-Path $PSScriptRoot "outputs\metadata"
New-Item -ItemType Directory -Force -Path $metadataDirectory | Out-Null
$constraintPath = Join-Path $metadataDirectory "protected_constraints.txt"
& $Python -m pip freeze | Where-Object { $_ -match '^(torch|transformers|openai|openai-agents)==' } | Set-Content $constraintPath
& $labPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt") -c $constraintPath
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed; inspect the conflict without upgrading the parent environment" }
Write-Output "Lab interpreter: $labPython"
