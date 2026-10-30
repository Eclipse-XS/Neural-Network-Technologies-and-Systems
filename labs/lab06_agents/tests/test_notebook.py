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
    source = '\n'.join(c.source for c in cells)
    assert 'Agent(' in source
    assert 'await Runner.run(' in source
    assert 'SQLiteSession(' in source
    assert 'RUN_FULL_LIVE' in source
    assert 'BLOCKED_BY_ENVIRONMENT' in source
    assert 'provenance' in source
    assert 'load_verified' in source
    assert 'baseline' in source and 'comparison_tools' in source


def test_live_and_replay_are_not_conflated():
    notebook = nbformat.read(LAB_ROOT / 'notebooks/01_openai_agents_sdk.ipynb', as_version=4)
    source = '\n'.join(c.source for c in notebook.cells)
    assert '"provenance": "LIVE"' in source
    assert 'REPLAY_UNVERIFIED_WORKSPACE_MISMATCH' in source
    assert 'outputs/notebook_live/' in source
