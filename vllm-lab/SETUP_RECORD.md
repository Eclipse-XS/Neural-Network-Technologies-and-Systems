# vLLM setup record

Recorded on 2026-10-09. This is an installation record, not the laboratory report.

## Verified platform

- Windows build: 10.0.26300.9550 (Windows 11; some legacy APIs identify it as Windows 10)
- WSL: 2.6.3.0, kernel 6.6.87.2-microsoft-standard-WSL2
- Distribution: Ubuntu 24.04.5 LTS
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4096 MiB, compute capability 8.6
- Windows NVIDIA driver: 610.88
- WSL NVIDIA-SMI: 610.57.01; CUDA driver compatibility: 13.3
- Host RAM: 15.19 GiB
- Python: 3.12.3
- uv: 0.12.24
- vLLM: 0.31.0
- PyTorch: 2.13.0+cu132
- Triton: 3.7.1
- Transformers: 5.17.0
- GCC/G++: 13.3.0
- Model: Qwen/Qwen2.5-1.5B-Instruct-AWQ

No system CUDA Toolkit was installed. CUDA runtime libraries are dependencies inside the isolated Python environment.

## Commands used

Run from an elevated or ordinary PowerShell session as appropriate:

```powershell
wsl --install -d Ubuntu-24.04 --no-launch
wsl -d Ubuntu-24.04 -u root -- useradd --create-home --shell /bin/bash vllm-lab
```

Run inside Ubuntu as root:

```bash
install -d -o vllm-lab -g vllm-lab /home/vllm-lab/lab2
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential python3.12-dev
```

Run as `vllm-lab`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /home/vllm-lab/lab2
~/.local/bin/uv venv --python 3.12 .venv
~/.local/bin/uv pip install --python .venv/bin/python vllm --torch-backend=auto
```

Start the server from PowerShell using the saved configuration:

```powershell
wsl -d Ubuntu-24.04 -u vllm-lab -- bash /mnt/c/Users/Eclipse/PycharmProjects/neural-network-technologies-and-systems/vllm-lab/start-server.sh
```

The verified persistent Windows launch uses a hidden `wsl.exe` process, because a Linux background process alone does not keep the WSL VM alive:

```powershell
$args = @(
  '-d','Ubuntu-24.04','-u','vllm-lab','--',
  '/home/vllm-lab/lab2/.venv/bin/vllm','serve','Qwen/Qwen2.5-1.5B-Instruct-AWQ',
  '--host','0.0.0.0','--port','8000','--max-model-len','2048',
  '--gpu-memory-utilization','0.80','--max-num-seqs','1','--enforce-eager'
)
Start-Process -FilePath wsl.exe -ArgumentList $args -WindowStyle Hidden `
  -RedirectStandardOutput "$PWD/vllm-lab/server.stdout.log" `
  -RedirectStandardError "$PWD/vllm-lab/server.stderr.log"
```

To stop this instance, identify the exact `wsl.exe` process whose command line contains the model ID, then stop that PID. Do not terminate every WSL process indiscriminately.

## Runtime configuration and measurements

```text
max_model_len: 2048
gpu_memory_utilization: 0.80
max_num_seqs: 1
enforce_eager: true
quantization selected by vLLM: auto_awq
linear kernel: Marlin
attention backend: FlashAttention 2
checkpoint size: 1.50 GiB
model load GPU memory: 1.1 GiB
weights plus non-Torch memory: 1.18 GiB
peak activation memory: 0.12 GiB
CUDA Graph memory: 0 GiB
KV cache: approximately 1.9 GiB
KV cache capacity: 71,296 tokens
measured after chat request: 3555 MiB used, 407 MiB free
final persistent-server measurement: 3567 MiB used, 396 MiB free
```

## API verification

`GET http://127.0.0.1:8000/v1/models` returned HTTP success and:

```json
{
  "id": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
  "owned_by": "vllm",
  "max_model_len": 2048
}
```

`POST http://127.0.0.1:8000/v1/chat/completions` returned HTTP success. The actual generated content was:

```text
vLLM is an open-source library for generating high-quality text using large language models, making it easier to integrate AI-powered text generation into applications.
```

Token usage: 32 prompt, 31 completion, 63 total. Run `test-api.ps1` to repeat both checks.

A second request against the persistent background instance returned exactly `vLLM API is working`; the Windows listener was confirmed on `127.0.0.1:8000`.

Final observed free space on C: was 11.35 GiB. This is low; do not install another model or duplicate the environment without first increasing free space.

## Reproduce package versions

```bash
cd /home/vllm-lab/lab2
~/.local/bin/uv pip freeze --python .venv/bin/python
```

Core resolved versions are recorded above. The full environment is reproducible from the `uv pip install ... vllm --torch-backend=auto` command, but CUDA package versions are platform-specific and must not be reused on Apple Silicon.

## Apple Silicon transfer boundary

Transfer the model ID, API parameters, prompts, and test requests. Do not transfer the Linux `.venv`, CUDA wheels, or NVIDIA package pins to macOS. On the M3 Max, create a new isolated environment and install vLLM-Metal according to its current documentation, then start the same model with `--max-model-len 2048`. Hardware-specific quantization support must be revalidated on the Mac rather than assuming the CUDA AWQ path is portable.
