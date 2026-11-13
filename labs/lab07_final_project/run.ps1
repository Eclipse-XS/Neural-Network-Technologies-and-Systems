param([string]$Python, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $candidates = @(
        (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
        (Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe')
        if ($env:VIRTUAL_ENV) { Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe' }
    )
    $Python = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
}
if (-not $Python) { throw 'No project virtual environment found. Run .\setup.ps1 -Python <Python 3.12-3.13 executable> first.' }
$arguments = @((Join-Path $PSScriptRoot 'scripts\startup.py'))
if ($NoBrowser) { $arguments += '--no-browser' }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Lab 7 stopped with exit code $LASTEXITCODE. See the diagnostic above." }
