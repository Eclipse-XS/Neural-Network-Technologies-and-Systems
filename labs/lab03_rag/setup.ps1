param([string]$Python = ".\.venv\Scripts\python.exe")
$ErrorActionPreference = "Stop"
$labDirectory = $PSScriptRoot
$labPython = Join-Path $labDirectory ".venv\Scripts\python.exe"
$sourcePackages = & $Python -c "import sysconfig, torch; print(sysconfig.get_paths()['purelib'])"
if ($LASTEXITCODE -ne 0) { throw "Use the existing Python environment with working torch. CUDA will not be installed by this script." }
$torchVersion = & $Python -c "import importlib.metadata; print(importlib.metadata.version('torch'))"
if (-not (Test-Path -LiteralPath $labPython)) {
    & $Python -m venv --system-site-packages (Join-Path $labDirectory ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Lab virtual environment creation failed" }
}
$siteDirectory = Join-Path $labDirectory ".venv\Lib\site-packages"
[System.IO.File]::WriteAllText((Join-Path $siteDirectory "repository_environment.pth"), "$sourcePackages`n")
$outputDirectory = Join-Path $labDirectory "outputs"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$constraintPath = Join-Path $outputDirectory "torch-constraint.txt"
[System.IO.File]::WriteAllText($constraintPath, "torch==$torchVersion`n")
& $labPython -m pip install -r (Join-Path $labDirectory "requirements.txt") -c $constraintPath
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed. The source environment has not been modified." }
Write-Output "Lab interpreter: $labPython"
