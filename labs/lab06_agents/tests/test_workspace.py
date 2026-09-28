import hashlib
import json
from labs.lab06_agents.src.config import WORKSPACE, LAB_ROOT


def test_hashes():
    manifest = json.loads((WORKSPACE / 'manifest.json').read_text())
    assert len(manifest) == 3
    for item in manifest:
        data = (WORKSPACE / item['workspace_filename']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        assert len(data) == item['size_bytes']
        assert data == (LAB_ROOT.parents[1] / item['source_repository_path']).read_bytes()
