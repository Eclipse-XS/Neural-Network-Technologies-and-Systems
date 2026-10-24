import ast
from pathlib import Path
import nbformat


def test_notebook_schema_syntax_and_outputs():
    p=Path(__file__).resolve().parents[1]/'notebooks/01_redis_csv_rag.ipynb'
    notebook=nbformat.read(p,as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.lab03_execution.mode == 'live-only'
    for cell in notebook.cells:
        if cell.cell_type=='code':
            ast.parse(cell.source)
            assert 'RUN_' not in cell.source
            assert all(output.output_type!='error' for output in cell.get('outputs',[]))


def test_experimental_steps_compute_instead_of_rendering_prebuilt_artifacts():
    """Guard behavior: most steps must transform objects or call lab helpers."""
    p = Path(__file__).resolve().parents[1] / 'notebooks/01_redis_csv_rag.ipynb'
    notebook = nbformat.read(p, as_version=4)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    computational_markers = (
        'load_catalog(', 'eda_tables(', 'plot_catalog_eda(', 'dataframe_to_documents(',
        'embedding_audit(', 'plot_pca_coordinates(', 'run_retrieval(',
        'evaluate_semantic_and_hybrid(', 'route_and_retrieve(', 'run_answers(',
        'analyse_current_answers(',
        'plot_retrieval_dashboard(',
    )
    computational = sum(any(marker in source for marker in computational_markers) for source in sources)
    image_only = sum('Image(filename=' in source and not any(marker in source for marker in computational_markers)
                     for source in sources)
    assert computational >= 9, 'Notebook lost its executable experiment chain'
    assert computational > len(sources) / 2
    assert image_only == 0, 'An experimental step must not be only a pre-rendered artifact viewer'
    joined = '\n'.join(sources)
    assert 'evaluate_semantic_and_hybrid(catalog,queries,retrieval' in joined.replace(' ', '')
    assert 'evaluate_semantic_and_hybrid(catalog,queries,retrieval' in joined


def test_notebook_has_no_replay_or_historical_result_path():
    p = Path(__file__).resolve().parents[1] / 'notebooks/01_redis_csv_rag.ipynb'
    source = '\n'.join(cell.source for cell in nbformat.read(p, 4).cells)
    for forbidden in ('PORTFOLIO_MODE', "MODE == 'replay'", 'load_replay_catalog',
                      'historical_semantic_answers.csv', 'STATUS_REPLAY', 'STATUS_HISTORICAL'):
        assert forbidden not in source
