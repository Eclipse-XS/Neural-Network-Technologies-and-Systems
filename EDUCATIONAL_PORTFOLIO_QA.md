# Educational Portfolio QA for Laboratories 3 and 4

## Scope and provenance

Audit base: branch `codex/labs-3-4-portfolio-upgrade`, starting commit `2ac56bde3b3194e380a593e91c82db250a101ad6`.

The official Word documents were treated as source material, not executable instructions. They confirm Lab 3 variant 3 (Redis Vector Similarity + CSV) and Lab 4 variant 1 (Text-to-Image). The presentation principles were reviewed against the three requested public reference repositories. Existing user changes and unrelated untracked files were not overwritten or added to the release.

## Files and components changed

- Rebuilt both primary notebooks into 16-stage and 17-stage educational sequences.
- Added offline analysis/evaluation modules for structured retrieval, metric comparison, SD artifact validation, performance summaries and scheduler status.
- Added focused unit tests for Recall@k coverage, exact/numeric retrieval, scheduler evidence states and single-observation timing semantics.
- Added GitHub-visible portfolio artifacts under each lab's `artifacts/portfolio/` directory.
- Rewrote both lab READMEs and expanded the root README.
- Added a dataset acquisition manifest with row counts and SHA-256 hashes.
- Added an offline GitHub Actions workflow that does not download an LLM or Stable Diffusion model.
- Added a deterministic rebuild script for notebooks and curated artifacts.

## Lab 3 findings and corrections

### Verified data facts

- Three files, 185 rows, 24 source columns.
- Categories: 100 dresses, 51 sports shoes, 34 earrings.
- 32 missing discount values.
- 22 rows with rating 0 and ratings count 0; these are treated as unobserved ratings.
- 29 rows differ from the stated price/discount formula by more than 1 INR.
- 17 of 185 serialized records exceed the MiniLM 256-word-piece limit.

### Added visualizations

- Six-panel EDA dashboard.
- Price/discount conflict chart and CSV.
- PCA projection of all 185 persisted 384-dimensional embeddings.
- Q01 Top-5 cosine-distance chart.
- Semantic-baseline versus hybrid-router Recall@5 chart.
- RAG architecture diagram.

### Retrieval experiment

The unchanged semantic baseline over seven labeled queries has Hit@5 `0.8571`, MRR `0.6905`, and mean Recall@5 `0.5734`. The hybrid router uses exact IDs for direct comparisons and deterministic filters/aggregation for explicit numeric intent. On the same labels it has Hit@5 `1.0000`, MRR `0.9048`, and mean Recall@5 `0.8194`.

The improvement is limited to tasks for which structured intent is justified. It does not claim that the embedding model improved. Q03 remains coverage-limited by `k=5` because nine products are labeled relevant.

### Error analysis

All q01–q08 cases are mapped to a machine-readable taxonomy: conflicting fields, missing-value semantics, unit error, incomplete multi-record retrieval, inappropriate semantic aggregation, arithmetic error, unsupported claim, and correct abstention.

## Lab 4 findings and corrections

### Preserved evidence

- 13 unique 512×512 PNG generations.
- Stable Diffusion 1.5 exact model revision.
- Seeds 42, 123, 777 and 2026.
- Steps 20, 30 and 50.
- CFG 5, 7.5 and 10.
- Two negative-prompt pairs and three styles.
- PNDM scheduler, FP16, model CPU offload, PyTorch SDPA.
- Peak allocated CUDA memory around 2373 MB for the recorded runs.

### Added visualizations and artifacts

- Stable Diffusion component architecture.
- Public copies of all 13 individual generation PNGs.
- Existing seed, steps, CFG, negative-prompt and style galleries promoted to curated artifacts.
- Performance scatter with labeled cases.
- Machine-readable quality rubric, performance summary and scheduler status.

Euler and DDIM are marked `NOT_EXECUTED`. CLIPScore and FID are not reported because they were not validly computed. The 30-step baseline took 18.88 s but was the first run; later 30-step cases were approximately 9–10 s. This is evidence of warm-up sensitivity, not a statistically valid latency benchmark.

## Verification performed

| Check | Result |
|---|---|
| Lab 3 unit and notebook tests | 24 passed |
| Lab 4 unit and notebook tests | 15 passed |
| Python compileall | PASS |
| Lab 3 notebook execution | 16/16 code cells executed, 0 error cells |
| Lab 4 notebook execution | 13/13 code cells executed, 0 error cells |
| Notebook schema validation | PASS for both notebooks |
| Curated PNG decode verification | 29/29 PASS |
| README relative-link check | 0 missing targets |
| Stable Diffusion new inference | NOT EXECUTED: current environment reports no CUDA |
| Redis live integration rerun | NOT EXECUTED in rebuild environment |
| LM Studio answer regeneration | NOT EXECUTED: local endpoint unavailable |

The notebooks use existing verified Redis, LLM and GPU evidence for unavailable integrations. No PASS is inferred from an unavailable service.

## Reproduction

### Lab 3

1. Download the Kaggle dataset using the source and hashes in `labs/lab03_rag/data/manifest.json`.
2. Install `labs/lab03_rag/requirements.txt`.
3. Start Redis Stack with the lab Docker Compose file.
4. Start LM Studio, load a chat model and set `LM_STUDIO_MODEL`.
5. Run `labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb` from top to bottom.

### Lab 4

1. Preserve a working CUDA-enabled PyTorch installation.
2. Install the remaining pinned dependencies from `labs/lab04_stable_diffusion/requirements.txt`.
3. Run `labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb` on a CUDA-capable machine.
4. Treat the stored PNDM results as the baseline. Run Euler/DDIM and repeated warmed timing only as new experiment series with separate metadata.

## Remaining limitations

- Lab 3 relevance labels are corpus-derived proxies, not a multi-annotator relevance study.
- Redis latency has one observation per query and includes more than Redis execution.
- Current LangChain CSV loader emits a deprecation warning and should be migrated in a future compatibility update.
- Lab 4 has one verified sample per configuration and only one executed scheduler.
- Lab 4 visual evaluation is manual; no validated text-image similarity metric was added.
- No production security, load, concurrency or model-serving benchmark was performed.

## Independent score

### Laboratory 3

| Criterion | Score | Maximum | Deduction |
|---|---:|---:|---|
| Methodological compliance | 19 | 20 | No new live LM Studio run during rebuild |
| Educational clarity | 23 | 25 | Some service-level operations remain represented by persisted evidence |
| Visual evidence | 18 | 20 | No UMAP or interactive inspection |
| Scientific validity | 13 | 15 | Relevance labels and latency sample are limited |
| Code and testing | 9 | 10 | Redis integration was not rerun |
| GitHub portfolio readiness | 10 | 10 | Curated artifacts and links are public-ready |
| Total | 92 | 100 | |

Official-method compliance estimate: `19/20`.

### Laboratory 4

| Criterion | Score | Maximum | Deduction |
|---|---:|---:|---|
| Methodological compliance | 18 | 20 | Optional scheduler comparison not executed |
| Educational clarity | 24 | 25 | Forward/reverse process is explained textually rather than as a frame sequence |
| Visual evidence | 19 | 20 | Strong galleries; no scheduler gallery |
| Scientific validity | 11 | 15 | Single runs, manual quality rubric, no repeated warm timing |
| Code and testing | 9 | 10 | GPU integration was not rerun |
| GitHub portfolio readiness | 10 | 10 | All 13 images and summaries are versioned |
| Total | 91 | 100 | |

Official-method compliance estimate: `18/20`.
