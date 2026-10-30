import json
from pathlib import Path

batch = json.loads(Path("labs/lab07_final_project/outputs/runs/batch.json").read_text(encoding="utf-8"))
run_ids = set(batch["run_ids"])

runs = []
with open("labs/lab07_final_project/outputs/runs/runs.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("run_id") in run_ids:
            runs.append(r)

print(f"Total runs analyzed: {len(runs)}")
scenarios = {}
for r in runs:
    sid = r["scenario_id"]
    scenarios.setdefault(sid, []).append(r)

for sid, rlist in scenarios.items():
    print(f"\n### Scenario: {sid} ({len(rlist)} runs)")
    for r in rlist:
        rid = r["run_id"]
        trace_file = Path(f"labs/lab07_final_project/outputs/traces/{rid}.json")
        trace_events = json.loads(trace_file.read_text(encoding="utf-8")) if trace_file.exists() else []
        policy_calls = [e for e in trace_events if e.get("policy_enforced")]
        auto_tool_calls = [e for e in trace_events if e.get("event_type") == "tool_call" and not e.get("policy_enforced")]
        print(f"- Run: {rid[:8]} | Variant: {r.get('agent_variant')} | Status: {r.get('status')}")
        print(f"  Input: {r.get('user_input')}")
        print(f"  Auto tools: {[e.get('tool_name') for e in auto_tool_calls]} | Policy enforced: {[e.get('tool_name') for e in policy_calls]}")
        print(f"  Citations: {r.get('citations')} | Invalid: {r.get('invalid_citations')} | Citation status: {r.get('citation')}")
        ans = (r.get("final_answer") or "").replace("\n", " ")
        print(f"  Answer: {ans[:200]}...")
