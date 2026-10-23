"""Execute and verify both portfolio notebooks with a real Jupyter kernel."""
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
    args = parser.parse_args()
    for path in NOTEBOOKS:
        cells, errors = execute(path, args.mode)
        print(f"{path.relative_to(ROOT)}: EXECUTED mode={args.mode} code_cells={cells} errors={errors}")


if __name__ == "__main__":
    main()
