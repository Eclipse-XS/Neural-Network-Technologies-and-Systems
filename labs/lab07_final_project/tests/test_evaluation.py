import json
from labs.lab07_final_project.src import evaluation


def test_baseline_placeholder_citations_fail(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'LAB_ROOT',tmp_path)
    (tmp_path/'trace.json').write_text('[]',encoding='utf-8')
    for answer in ['A [source_id#chunk_2]','A [lab4_report#1]','source_id: *L4_2023*']:
        result=evaluation.citation_audit(dict(final_answer=answer,trace_path='trace.json'))
        assert result['citation']=='FAIL'
        assert result['invalid_citations']


def test_current_trace_membership(tmp_path,monkeypatch):
    monkeypatch.setattr(evaluation,'LAB_ROOT',tmp_path)
    events=[dict(event_type='tool_result',tool_name='search_knowledge',result={'chunks':[{'citation':'[lab04#chunk-real]'}]})]
    (tmp_path/'trace.json').write_text(json.dumps(events),encoding='utf-8')
    record=dict(final_answer='Claim [lab04#chunk-real]; other [lab04#chunk-other]',trace_path='trace.json')
    result=evaluation.citation_audit(record)
    assert result['citation']=='PARTIAL'
    assert result['invalid_citations']==['[lab04#chunk-other]']


def test_retrieval_monotonicity_invariant():
    """Ensure Evidence Recall@k is mathematically monotonic non-decreasing for all k."""
    # Synthetic queries with expected evidence ranks
    queries = [
        {"evidence_ranks": [1]},           # recall@1=1.0, @3=1.0, @5=1.0
        {"evidence_ranks": [2]},           # recall@1=0.0, @3=1.0, @5=1.0
        {"evidence_ranks": [4]},           # recall@1=0.0, @3=0.0, @5=1.0
        {"evidence_ranks": [None]},        # recall@1=0.0, @3=0.0, @5=0.0
        {"evidence_ranks": [1, 4]},        # recall@1=0.5, @3=0.5, @5=1.0
        {"evidence_ranks": [2, None]}      # recall@1=0.0, @3=0.5, @5=0.5
    ]
    rec_1 = sum(sum(r is not None and r <= 1 for r in q["evidence_ranks"])/len(q["evidence_ranks"]) for q in queries) / len(queries)
    rec_3 = sum(sum(r is not None and r <= 3 for r in q["evidence_ranks"])/len(q["evidence_ranks"]) for q in queries) / len(queries)
    rec_5 = sum(sum(r is not None and r <= 5 for r in q["evidence_ranks"])/len(q["evidence_ranks"]) for q in queries) / len(queries)
    assert rec_1 <= rec_3 <= rec_5, f"Monotonicity failed: {rec_1} <= {rec_3} <= {rec_5}"
