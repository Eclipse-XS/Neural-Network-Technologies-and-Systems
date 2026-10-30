import ast
import nbformat
from labs.lab05_multimodal.src.config import LAB_ROOT


def test_executed_notebook():
    notebook = nbformat.read(LAB_ROOT / 'notebooks/01_llava_vqa.ipynb', as_version=4)
    nbformat.validate(notebook)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert cells
    assert [cell.execution_count for cell in cells] == list(range(1, len(cells) + 1))
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is not None
        assert all(output.output_type != 'error' for output in cell.outputs)
    assert any('image/png' in o.get('data', {}) for c in cells for o in c.outputs)
    stream_text = ''.join(o.get('text', '') for c in cells for o in c.outputs
                          if o.output_type == 'stream')
    assert 'REPLAY evidence: 19 experiment rows, 17 unique generations' in stream_text


def test_notebook_has_distinct_live_and_replay_paths():
    notebook = nbformat.read(LAB_ROOT / 'notebooks/01_llava_vqa.ipynb', as_version=4)
    sources = '\n'.join(cell.source for cell in notebook.cells)
    assert 'LAB05_MODE' in sources
    assert 'MODE == "LIVE"' in sources
    assert 'HISTORICAL_REPLAY' in sources
    assert 'evaluation_label": "NOT_REVIEWED"' in sources
    assert 'output = pipe(' in sources
    assert 'return_full_text=False' in sources
    assert 'generate_kwargs=smoke_generation.kwargs()' in sources


def test_notebook_exposes_required_experiment_loops():
    notebook = nbformat.read(LAB_ROOT / 'notebooks/01_llava_vqa.ipynb', as_version=4)
    sources = '\n'.join(cell.source for cell in notebook.cells)
    assert 'for temperature in [0.2, 0.7, 1.0]' in sources
    assert 'for max_new_tokens in [24, 64, 128]' in sources
    assert 'for index, wording in enumerate(paraphrases, 1)' in sources
    assert 'len(historical_rows) == 19' in sources
    assert 'len({row["run_id"] for row in historical_rows}) == 17' in sources
