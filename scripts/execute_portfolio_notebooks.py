"""Execute notebooks whose declared runtime dependencies match the selected mode."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb",
    ROOT / "labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb",
)


def execute(path: Path, mode: str = "replay") -> tuple[int, int]:
    notebook = nbformat.read(path, as_version=4)
    os.environ["PORTFOLIO_MODE"] = mode
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute(cwd=str(ROOT))
    notebook.metadata["portfolio_execution"] = {
        "status": "EXECUTED", "mode": mode,
        "tool": "nbclient.NotebookClient", "cwd": ".",
    }
    nbformat.write(notebook, path)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [out for cell in code_cells for out in cell.outputs if out.output_type == "error"]
    if errors or any(cell.execution_count is None for cell in code_cells):
        raise RuntimeError(f"Notebook verification failed: {path}")
    if not any(cell.outputs for cell in code_cells):
        raise RuntimeError(f"Notebook produced no outputs: {path}")
    return len(code_cells), len(errors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("replay", "full"), default="replay")
    parser.add_argument("--lab", choices=("3", "4", "all"), default="all")
    args = parser.parse_args()
    targets = NOTEBOOKS
    if args.lab == "3":
        targets = [NOTEBOOKS[0]]
    elif args.lab == "4":
        targets = [NOTEBOOKS[1]]
    elif args.mode == "replay":
        # Lab 3 is deliberately live-only: executing it without Redis and LM
        # Studio would either fail honestly or require fabricated replay paths.
        targets = [NOTEBOOKS[1]]
        print(f"{NOTEBOOKS[0].relative_to(ROOT)}: SKIPPED mode=replay reason=live-only")
    if args.mode == "replay" and args.lab == "3":
        raise SystemExit("Lab 3 has no replay mode; start Redis and LM Studio, then use --mode full --lab 3")
    for path in targets:
        cells, errors = execute(path, args.mode)
        print(f"{path.relative_to(ROOT)}: EXECUTED mode={args.mode} code_cells={cells} errors={errors}")


if __name__ == "__main__":
    main()
