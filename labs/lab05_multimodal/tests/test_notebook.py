import ast
import nbformat
from labs.lab05_multimodal.src.config import LAB_ROOT


def test_executed_notebook():
    notebook = nbformat.read(LAB_ROOT / 'notebooks/01_llava_vqa.ipynb', as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert cells
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is not None
        assert all(output.output_type != 'error' for output in cell.outputs)
    assert any('image/png' in o.get('data', {}) for c in cells for o in c.outputs)
