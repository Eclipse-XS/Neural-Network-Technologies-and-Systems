param([Parameter(Mandatory=$true)][ValidateRange(1,7)][int]$Lab, [switch]$Smoke)
$ErrorActionPreference = 'Stop'
$names = @('labs/lab01_transformer', 'labs/lab02_llm_api', 'labs/lab03_rag', 'labs/lab04_stable_diffusion', 'labs/lab05_multimodal', 'labs/lab06_agents', 'labs/lab07_final_project')
$labRoot = Join-Path $PSScriptRoot $names[$Lab - 1]
$python = Join-Path $labRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "Run ./setup.ps1 -Lab $Lab first" }
if ($Smoke) {
    & $python (Join-Path $PSScriptRoot 'scripts/smoke.py') --lab $Lab
} elseif ($Lab -eq 7) {
    & $python (Join-Path $labRoot 'scripts/startup.py')
} elseif ($Lab -eq 2) {
    Push-Location $PSScriptRoot
    try { & $python -m labs.lab02_llm_api.src.cli } finally { Pop-Location }
} else {
    & $python (Join-Path $PSScriptRoot 'scripts/run_notebooks.py') $labRoot
}
if ($LASTEXITCODE -ne 0) { throw "Lab $Lab failed with exit code $LASTEXITCODE" }
