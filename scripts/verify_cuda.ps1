<#
.SYNOPSIS
    Idempotent script to verify CUDA availability, device name, and real GPU tensor operations.
.DESCRIPTION
    Runs nvidia-smi diagnostics, then executes actual tensor arithmetic on GPU
    for each CUDA-enabled environment in the repository.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "=== CUDA & GPU Hardware Verification ===" -ForegroundColor Cyan
Write-Host "Repository root: $RepoRoot`n"

# 1. nvidia-smi check
Write-Host "--- nvidia-smi Check ---" -ForegroundColor Yellow
try {
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used --format=csv,noheader
} catch {
    Write-Error "nvidia-smi failed or NVIDIA driver is not installed."
}

Write-Host "`n--- CUDA Environments Verification ---" -ForegroundColor Yellow

$CudaEnvs = @(
    @{
        Lab        = "Lab 01 & Lab 04 (Transformer / Stable Diffusion)"
        Kernel     = "lab01-transformer / lab04-cuda"
        PythonPath = "$RepoRoot\.venv-lab04\Scripts\python.exe"
    },
    @{
        Lab        = "Lab 05 (Multimodal LLaVA)"
        Kernel     = "lab05-cuda"
        PythonPath = "$RepoRoot\labs\lab05_multimodal\.venv\Scripts\python.exe"
    },
    @{
        Lab        = "Lab 07 (Agentic RAG Final Project)"
        Kernel     = "lab07 / lab07-cuda"
        PythonPath = "$RepoRoot\labs\lab07_final_project\.venv\Scripts\python.exe"
    }
)

$AllPassed = $true

foreach ($env in $CudaEnvs) {
    Write-Host "`nTesting: $($env.Lab)" -ForegroundColor Cyan
    Write-Host "Interpreter: $($env.PythonPath)"

    if (-not (Test-Path $env.PythonPath)) {
        Write-Host "  [FAIL] Interpreter not found: $($env.PythonPath)" -ForegroundColor Red
        $AllPassed = $false
        continue
    }

    $CudaTest = @"
import sys
try:
    import torch
    py_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    cuda_build = getattr(torch.version, 'cuda', 'None')
    if not cuda_avail:
        print(f'FAIL|No CUDA available|torch={torch_ver}|py={py_ver}')
        sys.exit(1)
    
    device_name = torch.cuda.get_device_name(0)
    # Real tensor compute
    a = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    b = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Simple reduction check
    s = float(c.sum().cpu().item())
    allocated = round(torch.cuda.memory_allocated(0) / (1024*1024), 2)
    reserved = round(torch.cuda.memory_reserved(0) / (1024*1024), 2)
    print(f'PASS|{torch_ver}|CUDA {cuda_build}|{device_name}|alloc={allocated}MB,res={reserved}MB')
except Exception as e:
    print(f'FAIL|Exception: {e}')
    sys.exit(1)
"@

    $out = & $env.PythonPath -c $CudaTest 2>&1
    if ($LASTEXITCODE -eq 0 -and $out -like "PASS*") {
        $parts = $out.Split("|")
        $torchVer = $parts[1]
        $cudaVer = $parts[2]
        $devName = $parts[3]
        $memInfo = $parts[4]
        Write-Host "  $($env.Lab) | Kernel $($env.Kernel) | $torchVer | $devName | GPU COMPUTE PASS ($memInfo)" -ForegroundColor Green
    } else {
        Write-Host "  [FAIL] $out" -ForegroundColor Red
        $AllPassed = $false
    }
}

Write-Host "`n----------------------------------------"
if ($AllPassed) {
    Write-Host "ALL CUDA ENVIRONMENTS PASSED REAL GPU COMPUTE TESTS!" -ForegroundColor Green
    exit 0
} else {
    Write-Host "ONE OR MORE CUDA CHECKS FAILED." -ForegroundColor Red
    exit 1
}
