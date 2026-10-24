import io
import json
from urllib.error import URLError
from unittest.mock import Mock
import pytest
from langchain_core.documents import Document
from labs.lab03_rag.src.rag import answer_query, build_prompt, generate_answer, validate_citations
from labs.lab03_rag.tests.test_analysis import catalog
from labs.lab03_rag.src.config import Settings
from labs.lab03_rag.src.llm import list_models, create_llm
from labs.lab03_rag.src.evaluation import field_present, verify_answer, run_answers


def test_prompt_cites_sources_and_separates_untrusted_records():
    doc=Document(page_content='title: sample',metadata=dict(record_id='a.csv:2',source_file='a.csv',row_id=2))
    prompt=build_prompt('Which item?', [doc])
    assert 'Insufficient context' in prompt[0].content
    assert 'untrusted data' in prompt[0].content
    assert all(x in prompt[1].content for x in ['Which item?', 'a.csv:2', 'title: sample'])
    assert 'No records retrieved' in build_prompt('Question',[])[1].content
    llm=Mock(); llm.invoke.return_value.content='Insufficient context'
    assert generate_answer(llm,'Question',[])[0]=='Insufficient context'


def test_lexical_checks_are_not_substring_numeric_matches():
    assert field_present('Price ₹1,299.00.', '1299')
    assert not field_present('Price 1799', '799')
    assert not field_present('Rating 4.8', '4')
    assert not field_present('Price 1299.50', '1299')
    assert field_present('Rating 4.0', '4')


def test_citation_and_field_checked_separately():
    d=Document(page_content='x',metadata=dict(record_id='a.csv:0',source_file='a.csv',row_id=0))
    expected=[dict(product_id='1',record_id='a.csv:0',field='price',value='799')]
    check=verify_answer('799 [b.csv:9]',[d],expected)
    assert check['unknown_citations']==['b.csv:9']
    assert check['field_checks'][0]['value_present']
    assert not check['field_checks'][0]['expected_citation_present']


def test_citation_validation_requires_evidence_for_factual_answers():
    document = Document(page_content='x', metadata={
        'record_id': 'a.csv:0', 'source_file': 'a.csv', 'row_id': 0})
    assert not validate_citations('A factual answer without a citation.', [document])['valid']
    assert validate_citations('Supported [a.csv:0].', [document])['valid']
    assert validate_citations('Insufficient context: missing field.', [document])['valid']


def test_local_url_and_offline_error():
    with pytest.raises(ValueError):
        Settings(base_url='https://example.com/v1')
    with pytest.raises(ValueError):
        Settings(base_url='http://localhost:1234')
    with pytest.raises(RuntimeError,match='Local Server'):
        list_models(Settings(), opener=Mock(side_effect=URLError('offline')))
    opener=Mock(return_value=io.BytesIO(json.dumps({'data':[{'id':'local-chat'}]}).encode()))
    assert list_models(Settings(),opener)==['local-chat']
    assert opener.call_args.args[0].full_url.endswith('/v1/models')


def test_configured_model_must_exist(monkeypatch):
    monkeypatch.setattr('labs.lab03_rag.src.llm.list_models',lambda settings:['chat'])
    with pytest.raises(ValueError,match='LM_STUDIO_MODEL'):
        create_llm(Settings(model='missing'))


def test_generation_failure_is_saved(tmp_path):
    llm=Mock();llm.invoke.side_effect=RuntimeError('offline')
    query=dict(id='q',question='question',expected_fields=[])
    table=run_answers(llm,[query],{'q':[]},tmp_path,'mock-only')
    assert table.iloc[0]['status']=='failed'
    assert 'offline' in table.iloc[0]['error']
    assert (tmp_path/'rag/final_answers.csv').exists()


def test_hybrid_entrypoint_exact_id_uses_catalog_and_preserves_provenance():
    store = Mock()
    result = answer_query(store, Mock(), "Compare 17745816 and 15720652", catalog=catalog())
    store.similarity_search_with_score.assert_not_called()
    assert result["strategy"] == "exact_id"
    assert result["answer_mode"] == "deterministic_catalog"
    assert [d.metadata["product_id"] for d in result["documents"]] == ["17745816", "15720652"]
    assert all(d.metadata["record_id"] in result["answer"] for d in result["documents"])
    assert result["citations"]["valid"]


def test_hybrid_entrypoint_numeric_and_unsupported_do_not_call_llm():
    store, llm = Mock(), Mock()
    numeric = answer_query(store, llm,
        "Find the cheapest sports shoes rated at least 4 with discount > 0", catalog=catalog())
    assert numeric["documents"][0].metadata["product_id"] == "11960808"
    assert "659 INR" in numeric["answer"] and numeric["citations"]["valid"]
    unsupported = answer_query(store, llm, "Can 17745816 be delivered to Kyiv tomorrow?", catalog=catalog())
    assert unsupported["answer_mode"] == "deterministic_refusal"
    assert "Insufficient context" in unsupported["answer"]
    llm.invoke.assert_not_called()


def test_hybrid_nonexistent_id_abstains_and_mixed_lookup_reports_missing_id():
    store, llm = Mock(), Mock()
    missing = answer_query(store, llm, 'Look up product 99999999', catalog=catalog())
    assert missing['answer_mode'] == 'deterministic_catalog'
    assert missing['answer'].startswith('Insufficient context:')
    assert missing['citations']['valid']

    mixed = answer_query(store, llm, 'Compare 17745816 and 99999999', catalog=catalog())
    assert '17745816' in mixed['answer']
    assert '99999999' in mixed['answer'] and 'absent from the catalog' in mixed['answer']
    assert mixed['citations']['valid']


def test_hybrid_semantic_intent_uses_redis_then_llm():
    frame = catalog()
    from labs.lab03_rag.src.documents import dataframe_to_documents
    document = dataframe_to_documents(frame.iloc[[0]])[0]
    store = Mock(); store.similarity_search_with_score.return_value = [(document, 0.12)]
    llm = Mock(); llm.invoke.return_value.content = "Lancer [sports-shoes.csv:0]"
    result = answer_query(store, llm, "comfortable footwear", catalog=frame)
    store.similarity_search_with_score.assert_called_once()
    llm.invoke.assert_called_once()
    assert result["strategy"] == "semantic" and result["answer_mode"] == "llm"
    assert result["citations"]["valid"]


def test_semantic_baseline_remains_available():
    frame = catalog()
    from labs.lab03_rag.src.documents import dataframe_to_documents
    document = dataframe_to_documents(frame.iloc[[0]])[0]
    store = Mock(); store.similarity_search_with_score.return_value = [(document, 0.2)]
    llm = Mock(); llm.invoke.return_value.content = "answer"
    result = answer_query(store, llm, "17745816", catalog=frame, mode="semantic_baseline")
    assert result["strategy"] == "semantic" and result["answer_mode"] == "llm"
