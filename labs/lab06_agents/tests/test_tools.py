import json
import pytest
from labs.lab06_agents.src.tools import WorkspaceTools, safe_calculate, build_tools
from labs.lab06_agents.src.config import Settings


def test_tools_real_reports():
    store = WorkspaceTools()
    assert len(store.list_files()) == 3
    assert store.search('STABLE DIFFUSION', 'lab04_report.md')['matches']
    assert store.read('lab05_report.md')['lines'][0]['line'] == 1
    assert len(store.search('а')['matches']) <= 8
    assert len(json.dumps(store.read('lab05_report.md'), ensure_ascii=False)) < 13000
    assert store.search('NO_SUCH_LITERAL_12814')['matches'] == []


@pytest.mark.parametrize('name', ['../README.md','../../.env','C:\\Users\\private.txt','/etc/passwd','unknown.md','sub/../lab04_report.md','lab04_report.md:stream','manifest.json'])
def test_reject_paths(name):
    with pytest.raises(ValueError):
        WorkspaceTools().read(name)


@pytest.mark.parametrize('expression', ['__import__("os")','open("x")','1 .__class__','[1][0]','2**100000','True','1/0','1e309','x+1','[x for x in []]'])
def test_reject_code(expression):
    with pytest.raises((ValueError, SyntaxError, ZeroDivisionError)):
        safe_calculate(expression)


def test_arithmetic():
    assert safe_calculate('(17 - 13) * 2 / 4')['result'] == 2
    assert safe_calculate('-3 + +5')['result'] == 2


def test_bounds_and_schema():
    with pytest.raises(ValueError): WorkspaceTools().read('lab04_report.md', 1, 100)
    with pytest.raises(ValueError): WorkspaceTools().search('')
    tools = build_tools()
    assert len(tools) == 4
    for tool in tools:
        assert len(tool.description) > 40
        assert tool.params_json_schema['type'] == 'object'


@pytest.mark.parametrize('url', ['https://api.openai.com/v1','http://example.com/v1','http://localhost:1234/other'])
def test_only_local(url):
    with pytest.raises(ValueError): Settings(base_url=url)


def test_resolved_path_escape(tmp_path):
    outside = tmp_path / 'outside.md'
    outside.write_text('private')
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    (workspace / 'manifest.json').write_text('[{"workspace_filename":"report.md"}]')
    try:
        (workspace / 'report.md').symlink_to(outside)
    except OSError:
        pytest.skip('Windows does not grant symlink creation')
    with pytest.raises(ValueError):
        WorkspaceTools(workspace).read('report.md')
