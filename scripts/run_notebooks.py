"""Execute a lab's notebooks in order in one kernel, including notebook dependencies.

Outputs go to an ignored run artifact; source notebooks and historical results stay intact.
"""
import sys, os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
lab = Path(sys.argv[1]).resolve()
files = sorted((lab / 'notebooks').glob('*.ipynb'))
if not files:
    raise SystemExit('No executable notebooks; use the CLI documented in the lab README')
merged = nbformat.v4.new_notebook()
for path in files:
    notebook = nbformat.read(path, as_version=4)
    merged.cells.extend(notebook.cells)
output = lab / 'outputs' / 'reproducibility-run.ipynb'
output.parent.mkdir(parents=True, exist_ok=True)
# A private kernel spec guarantees the selected lab interpreter, independent of global Jupyter.
import tempfile, json
with tempfile.TemporaryDirectory() as temp:
    spec = Path(temp) / 'kernels' / 'lab'
    spec.mkdir(parents=True)
    (spec / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': 'Lab', 'language': 'python'}))
    os.environ['JUPYTER_PATH'] = temp + os.pathsep + os.environ.get('JUPYTER_PATH', '')
    client = NotebookClient(merged, timeout=3600, kernel_name='lab', resources={'metadata': {'path': str(lab / 'notebooks')}})
    try:
        client.execute()
    finally:
        nbformat.write(merged, output)
print(output)
