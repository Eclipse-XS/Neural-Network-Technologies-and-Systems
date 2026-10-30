param([string]$Python = ".\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
$labPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$sourcePackages = & $Python -c "import sysconfig, torch; print(sysconfig.get_paths()['purelib'])"
if ($LASTEXITCODE -ne 0) { throw "Supply an existing Python environment with working torch." }
$torchBefore = & $Python -c "import torch; print(torch.__version__)"
if (-not (Test-Path -LiteralPath $labPython)) {
    & $Python -m venv --system-site-packages (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create isolated Lab 7 environment" }
}
$siteDirectory = Join-Path $PSScriptRoot ".venv\Lib\site-packages"
[System.IO.File]::WriteAllText((Join-Path $siteDirectory "repository_environment.pth"), "$sourcePackages`n")
$metadataDirectory = Join-Path $PSScriptRoot "outputs\metadata"
New-Item -ItemType Directory -Force -Path $metadataDirectory | Out-Null
$constraintPath = Join-Path $metadataDirectory "protected_constraints.txt"
& $labPython -m ensurepip --upgrade | Out-Null
$requirements = Get-Content (Join-Path $PSScriptRoot "requirements.txt") |
    Where-Object { $_ -and -not $_.StartsWith('#') }
$ordinary = $requirements | Where-Object { -not $_.StartsWith('sentence-transformers') }
& $labPython -m pip install @ordinary
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed; inspect the conflict without upgrading the parent environment" }
& $labPython -m pip install --no-deps ($requirements | Where-Object { $_.StartsWith('sentence-transformers') })
if ($LASTEXITCODE -ne 0) { throw "sentence-transformers installation failed" }
$torchAfter = & $Python -c "import torch; print(torch.__version__)"
if ($torchAfter -ne $torchBefore) { throw "Protected torch changed from $torchBefore to $torchAfter" }
@("torch==$torchBefore", "source_site_packages=$sourcePackages") | Set-Content $constraintPath
Write-Output "Lab interpreter: $labPython"
