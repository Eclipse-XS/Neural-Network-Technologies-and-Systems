import hashlib
import json
import shutil
from pathlib import Path
import pytest
from labs.lab07_final_project.src.config import KNOWLEDGE, LAB_ROOT, Settings
from labs.lab07_final_project.src.knowledge_base import validate_corpus, load_chunks, KnowledgeBase, RetrievedChunk


def test_corpus_integrity_and_provenance():
    manifest=validate_corpus()
    assert len({s['source_id'] for s in manifest})==4
    for source in manifest:
        original=LAB_ROOT.parents[1]/source['original_repository_path']
        if original.exists():
            assert hashlib.sha256(original.read_bytes()).hexdigest()==source['sha256']


def test_modified_report_rejected(tmp_path):
    shutil.copytree(KNOWLEDGE,tmp_path/'corpus')
    manifest=json.loads((tmp_path/'corpus/manifest.json').read_text(encoding='utf-8'))
    (tmp_path/'corpus'/manifest[0]['workspace_filename']).write_text('modified',encoding='utf-8')
    with pytest.raises(ValueError,match='integrity'):
        validate_corpus(tmp_path/'corpus')


def test_chunk_provenance_and_determinism():
    first=load_chunks(Settings());second=load_chunks(Settings())
    assert len(first)>4
    assert [d.metadata['chunk_id'] for d in first]==[d.metadata['chunk_id'] for d in second]
    assert len({d.metadata['chunk_id'] for d in first})==len(first)
    for d in first:
        assert d.page_content.strip()
        assert len(d.page_content) <= 1000
        assert {'source_id','filename','lab_number','section','chunk_id','source_sha256','evidence_quality'} <= d.metadata.keys()


def test_token_budget_is_enforced_and_error_output_is_classified():
    # A deterministic character counter simulates a tokenizer without model access.
    chunks=load_chunks(Settings(embedding_max_tokens=80),token_counter=len)
    assert chunks and max(len(d.page_content) for d in chunks) <= 80
    assert any(d.metadata['evidence_quality']=='quoted_model_output' for d in chunks)
    assert any(d.metadata['evidence_quality']=='canonical_report' for d in chunks)
    smaller=load_chunks(Settings(embedding_max_tokens=240),token_counter=len,max_tokens=64)
    assert max(len(d.page_content) for d in smaller) <= 64


def test_committed_ingestion_audit_schema():
    artifact=json.loads((LAB_ROOT/'artifacts/portfolio/ingestion_audit.json').read_text(encoding='utf-8'))
    assert artifact['chunks_over_budget']==0
    assert artifact['maximum_observed_tokens'] <= artifact['configured_chunk_token_budget']
    assert artifact['configured_chunk_token_budget'] < artifact['model_max_sequence_length']
    assert len(artifact['embedding_revision'])==40


def test_search_bounds_without_live_services():
    kb=KnowledgeBase(Settings())
    for args in [('',5,None),('q',0,None),('q',9,None),('q',True,None),('q',5,'unknown')]:
        with pytest.raises(ValueError):
            kb.search(*args)


def test_explicit_lab_reference_can_be_detected():
    assert KnowledgeBase._terms('Which fact was reported in Lab 5?')


def test_local_configuration():
    for kwargs in [dict(base_url='https://api.openai.com/v1'),dict(redis_url='redis://example.com'),dict(index_name='lab03_existing')]:
        with pytest.raises(ValueError):
            Settings(**kwargs)
    assert Settings().session_db.is_absolute()
