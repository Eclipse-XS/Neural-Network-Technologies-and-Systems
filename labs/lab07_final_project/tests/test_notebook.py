import ast
import nbformat
from labs.lab07_final_project.src.config import LAB_ROOT


def test_executed_notebook():
    notebook=nbformat.read(LAB_ROOT/'notebooks/01_final_project.ipynb',as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            # ast.parse accepts top-level await; IPython executes it in the kernel.
            ast.parse(cell.source)
            assert cell.execution_count is not None
            assert not any(output.output_type=='error' for output in cell.outputs)
