param([string]$Python = 'py', [switch]$DownloadEmbedding)
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
if ($DownloadEmbedding) {
    & $labPython -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("intfloat/multilingual-e5-small", revision="614241f622f53c4eeff9890bdc4f31cfecc418b3", device="cpu")'
    if ($LASTEXITCODE -ne 0) { throw 'Embedding download failed. Check network access to Hugging Face.' }
}
& $labPython -m pip check
if ($LASTEXITCODE -ne 0) { Write-Warning 'Dependency conflicts remain in this environment. Resolve the reported conflicts before using affected packages.' }
Write-Output "Lab interpreter: $labPython"
Write-Output 'Setup complete. Start the application with .\run.ps1.'
