import ast
from pathlib import Path
import nbformat


def test_executed_notebook():
    path = Path(__file__).resolve().parents[1]/'notebooks/01_text_to_image.ipynb'
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is not None
            assert all(o.output_type != 'error' for o in cell.outputs)
