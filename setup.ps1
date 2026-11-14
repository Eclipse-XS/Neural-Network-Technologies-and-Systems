param([Parameter(Mandatory=$true)][ValidateRange(1,7)][int]$Lab, [string]$Python, [switch]$Cuda)
$ErrorActionPreference = 'Stop'
$names = @('labs/lab01_transformer', 'labs/lab02_llm_api', 'labs/lab03_rag', 'labs/lab04_stable_diffusion', 'labs/lab05_multimodal', 'labs/lab06_agents', 'labs/lab07_final_project')
$labRoot = Join-Path $PSScriptRoot $names[$Lab - 1]
if ($Lab -eq 7) {
    if ($Python) { & (Join-Path $labRoot 'setup.ps1') -Python $Python } else { & (Join-Path $labRoot 'setup.ps1') }
    return
}
if (-not $Python) {
    $Python = (& py -3.13 -c 'import sys; print(sys.executable)')
    if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.13 or pass -Python <executable>.' }
}
& $Python -c 'import sys; assert sys.version_info[:2] == (3,13), "Python 3.13 required"'
if ($LASTEXITCODE -ne 0) { throw 'Use Python 3.13.' }
$venvPython = Join-Path $labRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv (Join-Path $labRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
}
$requirements = Join-Path $labRoot 'requirements.lock.txt'
if (-not (Test-Path -LiteralPath $requirements)) { $requirements = Join-Path $labRoot 'requirements.txt' }
& $venvPython -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed' }
& $venvPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
if ($Cuda) {
    & $venvPython -m pip install --no-deps torch==2.14.1+cu130 --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) { throw 'CUDA wheel installation failed' }
    & $venvPython -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA validation failed' }
}
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency conflicts' }
if ($Lab -eq 7) { & $venvPython -m pip install --no-deps --editable $labRoot; if ($LASTEXITCODE -ne 0) { throw 'Editable install failed' } }
if ($Lab -eq 1 -or $Lab -eq 3) {
    & $venvPython (Join-Path $PSScriptRoot 'scripts/fetch_data.py')
    if ($LASTEXITCODE -ne 0) { throw 'Dataset reconstruction failed' }
}
$model = switch ($Lab) { 3 { 'minilm' }; 4 { 'diffusion' }; 5 { 'llava' } }
if ($model) {
    $modelArgs = @((Join-Path $PSScriptRoot 'scripts/fetch_models.py'), $model)
    if ($Lab -eq 4 -and $Cuda) { $modelArgs += '--fp16' }
    & $venvPython @modelArgs
    if ($LASTEXITCODE -ne 0) { throw 'Model download/checksum verification failed' }
}
Write-Output "Interpreter: $venvPython"
