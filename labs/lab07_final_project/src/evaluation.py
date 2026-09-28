"""Combine transparent human claim review with mechanical trace/citation checks."""
import json
import re
import pandas as pd
from .config import LAB_ROOT, OUTPUTS
from .experiments import scenarios, load_batch

RUBRIC = {
    'answer_correctness': {'CORRECT','PARTIAL','INCORRECT','NOT_SCORABLE'},
    'grounding': {'GROUNDED','PARTIAL','UNSUPPORTED'},
    'retrieval': {'RELEVANT','PARTIAL','FAILED','NOT_APPLICABLE'},
    'tool_behavior': {'APPROPRIATE','UNNECESSARY','MISSING','FAILED'},
    'memory': {'PASS','FAIL','NOT_APPLICABLE'},
    'planning': {'PASS','PARTIAL','FAIL','NOT_APPLICABLE'},
}


def citation_audit(record: dict) -> dict:
    """Also detect malformed report citations, including baseline fabrications."""
    trace=json.loads((LAB_ROOT/record['trace_path']).read_text(encoding='utf-8'))
    available={c['citation'] for e in trace if e['event_type']=='tool_result' and e['tool_name']=='search_knowledge'
        and isinstance(e['result'],dict) for c in e['result'].get('chunks',[])}
    claimed=re.findall(r'\[[^\]\n]*(?:lab\d|chunk[-_]|_report)[^\]\n]*\]',record['final_answer'],re.I)
    claimed += re.findall(r'source_id:\s*\*[^*]+\*',record['final_answer'])
    invalid=[c for c in claimed if c not in available]
    valid=[c for c in claimed if c in available]
    status='PARTIAL' if valid and invalid else 'FAIL' if invalid else 'PASS' if valid else 'FAIL' if available else 'NOT_APPLICABLE'
    return dict(citation=status,citation_claims=claimed,invalid_citations=invalid,
        note='Identifier membership only. Claim support is reviewed separately.')


def summarize(records=None):
    records=load_batch() if records is None else records
    reviews=json.loads((LAB_ROOT/'experiments/reviews.json').read_text(encoding='utf-8'))
    cases={c['scenario_id']:c for c in scenarios()}
    rows=[]
    for r in records:
        review=reviews[r['run_id']]
        for field,allowed in RUBRIC.items():
            if review[field] not in allowed:
                raise ValueError(f'Invalid rubric label: {field}')
        case=cases[r['scenario_id']]
        rows.append(dict(run_id=r['run_id'],scenario_id=r['scenario_id'],category=case['category'],variant=r['agent_variant'],
            session_id=r['session_id'],requires_rag=case['requires_rag'],requires_memory=case['requires_memory'],
            tools_used=' → '.join(r['tools_used']),tool_call_count=r['tool_call_count'],sources_retrieved=','.join(r['sources_retrieved']),
            **review,citation=citation_audit(r)['citation'],runtime_seconds=r['runtime_seconds'],status=r['status']))
    table=pd.DataFrame(rows)
    table.to_csv(OUTPUTS/'runs/summary.csv',index=False)
    comparisons=[]
    for case in cases.values():
        if not case['baseline_comparison']:
            continue
        pair={r['agent_variant']:r for r in records if r['scenario_id']==case['scenario_id']}
        a,b=pair['plain'],pair['assistant']
        comparisons.append(dict(scenario_id=case['scenario_id'],question=a['user_input'],plain_answer=a['final_answer'],assistant_answer=b['final_answer'],
            plain_correctness=reviews[a['run_id']]['answer_correctness'],assistant_correctness=reviews[b['run_id']]['answer_correctness'],
            plain_grounding=reviews[a['run_id']]['grounding'],assistant_grounding=reviews[b['run_id']]['grounding'],
            sources_cited=' '.join(citation_audit(b)['citation_claims']),tools_used=' → '.join(b['tools_used']),
            plain_runtime=a['runtime_seconds'],assistant_runtime=b['runtime_seconds'],
            observation=reviews[b['run_id']]['notes'],plain_hallucination=reviews[a['run_id']]['hallucination'],assistant_hallucination=reviews[b['run_id']]['hallucination']))
    pd.DataFrame(comparisons).to_csv(OUTPUTS/'runs/baseline_comparison.csv',index=False)
    return table,pd.DataFrame(comparisons)


if __name__=='__main__':
    summary,_=summarize()
    print(summary[['scenario_id','variant','answer_correctness','citation','tool_behavior']].to_string(index=False))
