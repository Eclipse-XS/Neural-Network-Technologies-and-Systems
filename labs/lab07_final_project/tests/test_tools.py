import asyncio
import json
import pytest
from agents.tool_context import ToolContext
from labs.lab07_final_project.src.tools import safe_calculate,build_tools
from labs.lab07_final_project.src.knowledge_base import RetrievedChunk
from labs.lab07_final_project.src.agent import validate_citations


@pytest.mark.parametrize('expression,result',[('17 - 13',4),('(13 + 17) / 2',15),('-2 * (3 + 4)',-14)])
def test_arithmetic(expression,result):
    assert safe_calculate(expression)['result']==result


@pytest.mark.parametrize('expression',['__import__("os")','open("file")','(1).__class__','x+1','True','2**100','[1]','1/0','1e309','1e16','1; print(1)'])
def test_reject_unsafe_arithmetic(expression):
    with pytest.raises((ValueError,SyntaxError,ZeroDivisionError)):
        safe_calculate(expression)


class FakeKB:
    def search(self,query,k,source_id):
        return [RetrievedChunk('lab04','lab04_report.md','4','Model','chunk-ab','Evidence',.3)]
    def sources(self):
        return [dict(source_id='lab04',lab_number=4,title='Lab 4',workspace_filename='lab04_report.md')]


def test_real_tool_schema_and_output():
    tools=build_tools(FakeKB())
    assert [t.name for t in tools]==['search_knowledge','list_knowledge_sources','calculate']
    for tool in tools:
        assert tool.description and tool.params_json_schema['type']=='object'
    arguments=json.dumps(dict(query='model',k=5,source_id='lab04'))
    context=ToolContext(context=None,tool_name='search_knowledge',tool_call_id='unit-1',tool_arguments=arguments)
    result=asyncio.run(tools[0].on_invoke_tool(context,arguments))
    parsed=json.loads(result) if isinstance(result,str) else result
    assert parsed['chunks'][0]['citation']=='[lab04#chunk-ab]'
    assert parsed['chunks'][0]['cosine_distance']==.3


def test_citations_require_evidence_from_current_run():
    events=[dict(event_type='tool_result',tool_name='search_knowledge',result={'chunks':[{'citation':'[lab04#chunk-ab]'}]})]
    assert validate_citations('Claim [lab04#chunk-ab]',events)['citation']=='PASS'
    assert validate_citations('Claim [lab04#chunk-fake]',events)['citation']=='FAIL'
    assert validate_citations('Claim [lab04#chunk-ab]',[])['citation']=='FAIL'
    assert validate_citations('Claim without citation',events)['citation']=='FAIL'
    assert validate_citations('Hello',[])['citation']=='NOT_APPLICABLE'
