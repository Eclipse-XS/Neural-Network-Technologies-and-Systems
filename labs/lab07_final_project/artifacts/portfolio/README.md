# Portfolio artifact provenance

`ingestion_audit.json` records an offline run on 2026-10-30 using the exact cached historical MiniLM revision. It verifies tokenizer input lengths after the current deterministic chunking change. It is not a Redis, retrieval, agent, or LLM end-to-end result.

Historical generated answers and traces remain in ignored local `outputs/`; they were not copied or relabeled as current evidence.
