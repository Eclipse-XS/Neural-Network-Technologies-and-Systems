import ast
import nbformat
from labs.lab06_agents.src.config import LAB_ROOT


def test_executed_notebook():
    notebook = nbformat.read(LAB_ROOT / 'notebooks/01_openai_agents_sdk.ipynb', as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(cells) >= 10
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is not None
        assert all(o.output_type != 'error' for o in cell.outputs)
    assert any('load_verified' in c.source for c in cells)
    assert any('baseline' in c.source and 'comparison_tools' in c.source for c in cells)
    assert any('numeric_extended' in c.source for c in cells)
