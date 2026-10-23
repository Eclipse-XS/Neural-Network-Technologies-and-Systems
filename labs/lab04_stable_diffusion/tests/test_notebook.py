import ast
from pathlib import Path
import nbformat


def test_executed_notebook():
    path = Path(__file__).resolve().parents[1]/'notebooks/01_text_to_image.ipynb'
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert nb.metadata.portfolio_execution.status == 'EXECUTED'
    assert nb.metadata.portfolio_execution.mode == 'replay'
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is not None
            assert cell.outputs
            assert all(o.output_type != 'error' for o in cell.outputs)


def test_experiment_series_are_resolved_and_plotted_programmatically():
    path = Path(__file__).resolve().parents[1] / 'notebooks/01_text_to_image.ipynb'
    nb = nbformat.read(path, as_version=4)
    sources = [cell.source for cell in nb.cells if cell.cell_type == 'code']
    joined = '\n'.join(sources)
    for name in ('seed', 'steps', 'cfg', 'negative_prompt_seed42',
                 'negative_prompt_seed123', 'styles'):
        assert name in joined
    assert joined.count('resolve_group(') >= 5
    assert joined.count('plot_generation_gallery(') >= 5
    assert 'display(Image(filename=' not in joined
    assert 'sha256_verified' in joined
    assert "pipe, pipeline_context=load_pipeline(model_config)" in joined
