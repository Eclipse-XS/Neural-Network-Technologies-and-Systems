<#
.SYNOPSIS
    Idempotent script to register and verify Jupyter kernels across all labs.
.DESCRIPTION
    Maps each lab to its validated Python environment and registers unique,
    descriptive Jupyter kernels in %APPDATA%\jupyter\kernels.
    Also normalizes notebook kernelspec metadata across all lab notebooks.
#>
param (
    [switch]$SkipNotebookNormalization
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "=== Registering Project-Wide Jupyter Kernels ===" -ForegroundColor Cyan
Write-Host "Repository root: $RepoRoot`n"

$Kernels = @(
    @{
        Name        = "lab01-transformer"
        DisplayName = "Lab 01 Transformer (CUDA RTX 3050)"
        PythonPath  = "$RepoRoot\.venv-lab04\Scripts\python.exe"
    },
    @{
        Name        = "lab03-rag"
        DisplayName = "Lab 03 Redis CSV RAG (CPU)"
        PythonPath  = "$RepoRoot\labs\lab03_rag\.venv\Scripts\python.exe"
    },
    @{
        Name        = "lab04-cuda"
        DisplayName = "Lab 04 CUDA RTX 3050 (Stable Diffusion)"
        PythonPath  = "$RepoRoot\.venv-lab04\Scripts\python.exe"
    },
    @{
        Name        = "lab05-cuda"
        DisplayName = "Lab 05 CUDA RTX 3050 (LLaVA)"
        PythonPath  = "$RepoRoot\labs\lab05_multimodal\.venv\Scripts\python.exe"
    },
    @{
        Name        = "lab06-agents"
        DisplayName = "Lab 06 OpenAI Agents SDK"
        PythonPath  = "$RepoRoot\.venv-lab06\Scripts\python.exe"
    },
    @{
        Name        = "lab07"
        DisplayName = "Lab 07 Final Project Agentic RAG (CUDA)"
        PythonPath  = "$RepoRoot\labs\lab07_final_project\.venv\Scripts\python.exe"
    },
    @{
        Name        = "lab07-cuda"
        DisplayName = "Lab 07 Final Project Agentic RAG (CUDA)"
        PythonPath  = "$RepoRoot\labs\lab07_final_project\.venv\Scripts\python.exe"
    }
)

foreach ($k in $Kernels) {
    if (-not (Test-Path $k.PythonPath)) {
        Write-Warning "Interpreter not found for $($k.Name): $($k.PythonPath)"
        continue
    }

    Write-Host "Registering kernel: $($k.Name) -> $($k.DisplayName)" -ForegroundColor Yellow
    & $k.PythonPath -m ipykernel install --user --name $k.Name --display-name $k.DisplayName | Out-Null

    # Verify kernel.json
    $kernelJson = "$env:APPDATA\jupyter\kernels\$($k.Name)\kernel.json"
    if (Test-Path $kernelJson) {
        Write-Host "  [OK] Verified $kernelJson" -ForegroundColor Green
    } else {
        Write-Error "Failed to install kernel: $($k.Name)"
    }
}

if (-not $SkipNotebookNormalization) {
    Write-Host "`n=== Normalizing Notebook Kernelspec Metadata ===" -ForegroundColor Cyan
    $NormScript = @"
import json, os

repo = r'$RepoRoot'
mappings = {
    'labs/lab01_transformer/notebooks/01_spam_classification.ipynb': ('lab01-transformer', 'Lab 01 Transformer (CUDA RTX 3050)'),
    'labs/lab01_transformer/notebooks/02_text_generation.ipynb': ('lab01-transformer', 'Lab 01 Transformer (CUDA RTX 3050)'),
    'labs/lab01_transformer/notebooks/reference/CausalLM.ipynb': ('lab01-transformer', 'Lab 01 Transformer (CUDA RTX 3050)'),
    'labs/lab01_transformer/notebooks/reference/TEncoderClassifier.ipynb': ('lab01-transformer', 'Lab 01 Transformer (CUDA RTX 3050)'),
    'labs/lab01_transformer/notebooks/reference/TransformerEncoderExample.ipynb': ('lab01-transformer', 'Lab 01 Transformer (CUDA RTX 3050)'),
    'labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb': ('lab03-rag', 'Lab 03 Redis CSV RAG (CPU)'),
    'labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb': ('lab04-cuda', 'Lab 04 CUDA RTX 3050 (Stable Diffusion)'),
    'labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb': ('lab05-cuda', 'Lab 05 CUDA RTX 3050 (LLaVA)'),
    'labs/lab06_agents/notebooks/01_openai_agents_sdk.ipynb': ('lab06-agents', 'Lab 06 OpenAI Agents SDK'),
    'labs/lab07_final_project/notebooks/01_final_project.ipynb': ('lab07', 'Lab 07 Final Project Agentic RAG (CUDA)')
}

for rel_path, (kname, kdisp) in mappings.items():
    full_path = os.path.join(repo, rel_path.replace('/', os.sep))
    if not os.path.exists(full_path):
        continue
    with open(full_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    if 'metadata' not in nb:
        nb['metadata'] = {}
    nb['metadata']['kernelspec'] = {
        'name': kname,
        'display_name': kdisp,
        'language': 'python'
    }
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'  [OK] {rel_path} -> {kname}')
"@
    & "$RepoRoot\.venv-lab04\Scripts\python.exe" -c $NormScript
}

Write-Host "`nAll Jupyter kernels registered and verified successfully!" -ForegroundColor Green
