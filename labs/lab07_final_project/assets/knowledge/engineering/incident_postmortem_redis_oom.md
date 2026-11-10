# Incident Post-Mortem: Redis Out-of-Memory (OOM) During Document Ingestion

## Summary
On 2026-10-14 between 14:22 and 14:47 UTC, the Local Engineering Knowledge Copilot ingestion pipeline failed with `redis.exceptions.ResponseError: OOM command not allowed when used memory > 'maxmemory'`.
During this 25-minute window, document uploads were rejected with HTTP 500 status codes, and active chat sessions experienced degraded retrieval quality falling back to lexical-only search.

## Impact
- **Ingestion availability**: 0% during the incident window (14 bulk uploads failed).
- **Retrieval degradation**: 38 queries processed via lexical fallback without vector ranking.
- **Data corruption**: 0 bytes corrupted; transactional integrity was maintained by SQLite document status rollbacks.

## Timeline (UTC)
- **14:20**: Ingestion job started for a 450-page technical specification PDF containing 12,400 chunks.
- **14:22**: Redis memory usage crossed the configured `maxmemory 512mb` threshold.
- **14:23**: Redis rejected `HSET` operations with OOM error. Ingestion worker crashed.
- **14:25**: Automated health probe `/api/v1/health/ready` transitioned to `UNREADY (status_code=503)`.
- **14:35**: Engineer paged; inspected Redis memory metrics via `INFO memory`.
- **14:40**: Discovered `maxmemory-policy` was set to `noeviction` without chunk TTL or memory limits on embeddings cache.
- **14:45**: Updated Redis config to `maxmemory 2gb` and set vector cache eviction policy to `allkeys-lru`.
- **14:47**: Restarted service; queued documents reindexed successfully in 84 seconds.

## Root Cause Analysis
1. The default Redis container allocated `maxmemory 512mb`, which was insufficient for large high-dimensional vector embeddings (768 dimensions * 4 bytes * 100,000 vectors plus HNSW graph index overhead).
2. The batch ingestion pipeline loaded all document embeddings into a single transaction without chunk batching or streaming.

## Corrective and Preventative Actions
1. **Immediate**: Configured `maxmemory 2gb` in `docker-compose.yml` with `allkeys-lru` eviction policy.
2. **Architecture**: Implemented chunked batching in `IngestionService` with batch size = 64 chunks per vector index commit.
3. **Observability**: Added Prometheus gauge `redis_memory_used_bytes` and alert rule firing when Redis memory exceeds 80% of `maxmemory`.
4. **Resilience**: Added automated fallback to BM25 lexical retriever when vector store raises transient memory or network errors.
