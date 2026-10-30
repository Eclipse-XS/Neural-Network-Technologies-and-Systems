import json
from pathlib import Path

index = json.loads(Path('labs/lab07_final_project/outputs/metadata/index.json').read_text(encoding='utf-8'))
emb = json.loads(Path('labs/lab07_final_project/outputs/metadata/embedding.json').read_text(encoding='utf-8'))
chunks = json.loads(Path('labs/lab07_final_project/outputs/metadata/chunks.json').read_text(encoding='utf-8'))
manifest = json.loads(Path('labs/lab07_final_project/assets/knowledge/manifest.json').read_text(encoding='utf-8'))

print('=== KNOWLEDGE BASE AUDIT SUMMARY ===')
print('Index name:', index['index_name'])
print('Redis indexed vectors:', index['indexed_count'])
print('Corpus fingerprint:', index['corpus_fingerprint'])
print('Total document count:', len(manifest))
print('Total chunk count:', chunks['chunk_count'])
print('Chunks per source:', chunks['chunks_per_source'])
print('Embedding model:', emb['model'])
print('Embedding revision:', emb['revision'])
print('Embedding dimension:', emb['dimension'])
print('Embedding max sequence length:', emb['max_seq_length'])
print('Effective chunk token budget:', index['embedding_max_tokens'])
print('Maximum observed tokens:', max(emb['token_lengths']))
print('Chunks over budget (truncation violations):', emb['truncated_chunks'])
print('--- Sample Chunks ---')
for s in chunks['samples'][:3]:
    print(f"[{s['source_id']}#{s['chunk_id']}] ({s['section']}) -> {s['text'][:120]}...")
