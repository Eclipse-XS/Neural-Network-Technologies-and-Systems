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
    source = '\n'.join(c.source for c in notebook.cells)
    for concept in ['SigLIP', 'Qwen2', 'visual embeddings', 'continuation',
                    'temperature', 'max_new_tokens', 'casefold']:
        assert concept in source
    assert sum('image/png' in o.get('data', {}) for c in cells for o in c.outputs) >= 4


def test_live_notebook_is_static_but_contains_real_pipeline_call():
    path = LAB_ROOT / 'notebooks/02_llava_live_inference.ipynb'
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata['lab5'] == {
        'purpose': 'live_gpu_inference', 'execution_status': 'NOT_VERIFIED'}
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert cells
    for cell in cells:
        ast.parse(cell.source)
        assert cell.execution_count is None
        assert not cell.outputs
    source = '\n'.join(cell.source for cell in cells)
    assert 'pipe(' in source
    assert 'text=messages' in source
    assert 'return_full_text=False' in source
    assert 'run_experiments(' not in source
    assert 'verified_runs.json").read' not in source
    assert "verified_runs.json').read" not in source
    assert 'save_live_result(live_record)' in source
    assert 'NOT_REVIEWED' in source
    assert 'torch.cuda.synchronize()' in source
