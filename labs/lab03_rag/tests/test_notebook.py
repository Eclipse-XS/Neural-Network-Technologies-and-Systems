import ast
from pathlib import Path
import nbformat


def test_notebook_schema_syntax_and_outputs():
    p=Path(__file__).resolve().parents[1]/'notebooks/01_redis_csv_rag.ipynb'
    notebook=nbformat.read(p,as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            assert 'RUN_' not in cell.source
            assert all(output.output_type!='error' for output in cell.get('outputs',[]))
