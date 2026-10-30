#!/usr/bin/env python3
"""Comprehensive environment and Jupyter kernel verification tool.

Checks:
1. Environment health (Python, PyTorch, CUDA runtime, GPU device name, real CUDA tensor compute).
2. Jupyter kernel registration (%APPDATA%/jupyter/kernels/...).
3. Notebook kernelspec binding (Notebook -> kernelspec.name -> kernel.json -> python.exe).
4. GPU requirement alignment (GPU labs use CUDA PyTorch on RTX 3050; CPU labs remain clean).
"""

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Expected environment requirements per lab
LAB_SPECS = [
    {
        "lab": "Lab 01",
        "title": "Transformer Classification & Generation",
        "notebooks": [
            "labs/lab01_transformer/notebooks/01_spam_classification.ipynb",
            "labs/lab01_transformer/notebooks/02_text_generation.ipynb",
        ],
        "expected_kernel": "lab01-transformer",
        "expected_env": ".venv-lab04",
        "backend": "CUDA",
        "gpu_required": True,
    },
    {
        "lab": "Lab 02",
        "title": "LLM API & vLLM Client",
        "notebooks": [],
        "expected_kernel": "lab06-agents",
        "expected_env": ".venv-lab06",
        "backend": "CPU / REST (vLLM on WSL2)",
        "gpu_required": False,
    },
    {
        "lab": "Lab 03",
        "title": "Redis CSV RAG",
        "notebooks": [
            "labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb",
        ],
        "expected_kernel": "lab03-rag",
        "expected_env": "labs/lab03_rag/.venv",
        "backend": "CPU",
        "gpu_required": False,
    },
    {
        "lab": "Lab 04",
        "title": "Stable Diffusion Text-to-Image",
        "notebooks": [
            "labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb",
        ],
        "expected_kernel": "lab04-cuda",
        "expected_env": ".venv-lab04",
        "backend": "CUDA",
        "gpu_required": True,
    },
    {
        "lab": "Lab 05",
        "title": "LLaVA Visual Question Answering",
        "notebooks": [
            "labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb",
        ],
        "expected_kernel": "lab05-cuda",
        "expected_env": "labs/lab05_multimodal/.venv",
        "backend": "CUDA",
        "gpu_required": True,
    },
    {
        "lab": "Lab 06",
        "title": "OpenAI Agents SDK",
        "notebooks": [
            "labs/lab06_agents/notebooks/01_openai_agents_sdk.ipynb",
        ],
        "expected_kernel": "lab06-agents",
        "expected_env": ".venv-lab06",
        "backend": "CPU / API (LM Studio)",
        "gpu_required": False,
    },
    {
        "lab": "Lab 07",
        "title": "Agentic RAG Showcase Final Project",
        "notebooks": [
            "labs/lab07_final_project/notebooks/01_final_project.ipynb",
        ],
        "expected_kernel": "lab07",
        "expected_env": "labs/lab07_final_project/.venv",
        "backend": "CUDA",
        "gpu_required": True,
    },
]


def probe_python_interpreter(py_path: Path) -> dict:
    """Run an isolated probe inside the specified interpreter."""
    if not py_path.exists():
        return {"exists": False, "error": f"Path not found: {py_path}"}

    probe_code = """
import sys, json
info = {'python': sys.version.split()[0], 'executable': sys.executable}
try:
    import torch
    info['torch'] = torch.__version__
    info['cuda_available'] = torch.cuda.is_available()
    info['cuda_version'] = getattr(torch.version, 'cuda', None)
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        try:
            x = torch.randn(256, 256, device='cuda')
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            info['cuda_compute'] = True
        except Exception as e:
            info['cuda_compute'] = str(e)
    else:
        info['cuda_compute'] = False
except ImportError:
    info['torch'] = None
    info['cuda_available'] = False
    info['cuda_version'] = None
    info['cuda_compute'] = False

print(json.dumps(info))
"""
    try:
        out = subprocess.check_output(
            [str(py_path), "-c", probe_code],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        data = json.loads(out.strip().splitlines()[-1])
        data["exists"] = True
        return data
    except Exception as exc:
        return {"exists": True, "error": str(exc)}


def get_registered_jupyter_kernels() -> dict:
    """Return dictionary of kernel_name -> kernel_info from Jupyter."""
    appdata = os.environ.get("APPDATA", "")
    kernels_dir = Path(appdata) / "jupyter" / "kernels"
    registered = {}
    if not kernels_dir.exists():
        return registered

    for item in kernels_dir.iterdir():
        kjson = item / "kernel.json"
        if kjson.exists():
            try:
                data = json.loads(kjson.read_text(encoding="utf-8"))
                argv = data.get("argv", [])
                py_exe = argv[0] if argv else ""
                registered[item.name] = {
                    "name": item.name,
                    "display_name": data.get("display_name", item.name),
                    "executable": py_exe,
                    "path": str(kjson),
                }
            except Exception as e:
                registered[item.name] = {"error": str(e)}
    return registered


def check_all():
    print("=" * 80)
    print("PROJECT-WIDE ENVIRONMENT & JUPYTER KERNEL AUDIT")
    print("=" * 80)

    registered_kernels = get_registered_jupyter_kernels()
    print(f"Discovered {len(registered_kernels)} registered Jupyter kernels:")
    for kname, kinfo in registered_kernels.items():
        print(f"  * {kname}: {kinfo.get('display_name')} -> {kinfo.get('executable')}")
    print()

    # Cache interpreter probes
    probed_envs = {}
    for spec in LAB_SPECS:
        env_rel = spec["expected_env"]
        py_exe = REPO_ROOT / env_rel / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if str(py_exe) not in probed_envs:
            probed_envs[str(py_exe)] = probe_python_interpreter(py_exe)

    # Verification table rows
    rows = []
    overall_ok = True

    for spec in LAB_SPECS:
        lab_name = spec["lab"]
        expected_kname = spec["expected_kernel"]
        env_rel = spec["expected_env"]
        py_exe = REPO_ROOT / env_rel / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env_info = probed_envs.get(str(py_exe), {})

        # Check kernel registration
        kernel_registered = expected_kname in registered_kernels
        kernel_exe_match = False
        if kernel_registered:
            k_exe = Path(registered_kernels[expected_kname].get("executable", "")).resolve()
            kernel_exe_match = (k_exe == py_exe.resolve())

        # Check notebooks
        notebook_list = spec["notebooks"]
        if not notebook_list:
            # Standalone lab (like Lab 2)
            nb_display = "(CLI / tests)"
            status = "PASS" if env_info.get("exists") else "FAIL"
            if not status == "PASS":
                overall_ok = False
            rows.append({
                "lab": lab_name,
                "notebook": nb_display,
                "kernel": expected_kname,
                "python": str(env_info.get("python") or "N/A"),
                "torch": str(env_info.get("torch") or "None"),
                "cuda": str(env_info.get("cuda_version") or "None"),
                "gpu_compute": "PASS" if env_info.get("cuda_compute") is True else ("N/A (CPU)" if not spec["gpu_required"] else "FAIL"),
                "status": status,
            })
            continue

        for nb_rel in notebook_list:
            nb_path = REPO_ROOT / nb_rel
            nb_short = Path(nb_rel).name
            if not nb_path.exists():
                rows.append({
                    "lab": lab_name,
                    "notebook": nb_short,
                    "kernel": expected_kname,
                    "python": "MISSING",
                    "torch": "N/A",
                    "cuda": "N/A",
                    "gpu_compute": "N/A",
                    "status": "FAIL (Notebook missing)",
                })
                overall_ok = False
                continue

            with open(nb_path, "r", encoding="utf-8") as fp:
                nb_data = json.load(fp)
            nb_ks = nb_data.get("metadata", {}).get("kernelspec", {})
            actual_kname = nb_ks.get("name")

            # Validate binding
            nb_kernel_ok = (actual_kname == expected_kname)
            gpu_ok = True
            if spec["gpu_required"]:
                gpu_ok = bool(env_info.get("cuda_available") and env_info.get("cuda_compute") is True)

            status = "PASS"
            if not nb_kernel_ok:
                status = f"FAIL (Notebook has {actual_kname}, expected {expected_kname})"
                overall_ok = False
            elif not kernel_registered:
                status = f"FAIL (Kernel {expected_kname} not registered in Jupyter)"
                overall_ok = False
            elif not kernel_exe_match:
                status = f"FAIL (Kernel exe mismatch)"
                overall_ok = False
            elif spec["gpu_required"] and not gpu_ok:
                status = f"FAIL (CUDA compute failed on GPU lab)"
                overall_ok = False

            compute_str = "PASS" if env_info.get("cuda_compute") is True else ("N/A (CPU)" if not spec["gpu_required"] else "FAIL")
            rows.append({
                "lab": lab_name,
                "notebook": nb_short,
                "kernel": actual_kname or "UNSET",
                "python": str(env_info.get("python") or "N/A"),
                "torch": str(env_info.get("torch") or "None"),
                "cuda": str(env_info.get("cuda_version") or "None"),
                "gpu_compute": compute_str,
                "status": status,
            })

    # Print Table
    print(f"{'Lab':<8} | {'Notebook':<32} | {'Kernel':<18} | {'Python':<8} | {'Torch':<14} | {'CUDA':<8} | {'Compute':<10} | {'Status'}")
    print("-" * 125)
    for r in rows:
        print(f"{r['lab']:<8} | {r['notebook']:<32} | {r['kernel']:<18} | {r['python']:<8} | {r['torch']:<14} | {r['cuda']:<8} | {r['gpu_compute']:<10} | {r['status']}")

    print("-" * 125)
    if overall_ok:
        print("\n[SUCCESS] ALL LABS, NOTEBOOKS, INTERPRETERS, AND CUDA VERIFICATIONS PASSED!")
        return 0
    else:
        print("\n[FAILURE] ONE OR MORE INTEGRITY CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(check_all())
