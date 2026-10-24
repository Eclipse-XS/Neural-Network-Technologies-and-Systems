# Final acceptance audit: Labs 3 and 4

Audit date: 2026-10-24
Scope: PR #1, branch `codex/labs-3-4-portfolio-upgrade`

## 1. Executive verdict

**MERGE READY WITH DOCUMENTED LIMITATIONS.**

- `PASS`: implementation, committed evidence, CPU test suite, Lab 4 replay, notebook structure, HTML export, and artifact integrity were independently checked.
- `NOT_VERIFIED`: live Lab 3 inference. Redis at `localhost:6379` and the OpenAI-compatible LM Studio endpoint at `localhost:1234` were unavailable during the audit.
- `NOT_APPLICABLE`: new Stable Diffusion GPU inference. The audit found no defect requiring an expensive regeneration; it validated the committed historical run instead.
- `PASS`: no critical defect, failing applicable test, fabricated metric, or merge conflict remains.

## 2. Verified architecture

### Lab 3: RAG

- `PASS`: the implemented path is CSV ingestion and normalization (`labs/lab03_rag/src/data.py`) → provenance-bearing documents (`documents.py`) → MiniLM embeddings and Redis FLAT/COSINE storage (`embeddings.py`, `redis_store.py`) → intent routing and deterministic catalog retrieval (`hybrid.py`) → semantic fallback and grounded answer generation (`rag.py`).
- `PASS`: `source_file`, `row_id`, `product_id`, and `record_id` provenance is retained and covered by the Lab 3 tests.
- `PASS`: `answer_query` actually routes exact-ID, structured/numeric, semantic, and unsupported requests. Exact and numeric answers do not depend on an LLM.
- `PASS`: absent IDs and empty structured results now abstain explicitly; mixed known/unknown ID requests identify the missing ID.
- `PASS`: factual generated answers require an allowed citation. Unknown citations invalidate the answer; explicit insufficient-context responses remain valid abstentions.

### Lab 4: Stable Diffusion

- `PASS`: `labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb` and `report/report.md` correctly distinguish forward noising, reverse denoising, and scheduler inference steps, and explain CLIP conditioning, U-Net noise prediction, classifier-free guidance, PNDM scheduling, VAE decoding, FP16, and SDPA.
- `PASS`: the experiment manifest in `artifacts/portfolio/generations.csv` contains 13 unique artifact IDs and 13 unique controlled configurations: baseline (1), CFG (2), negative prompt (2), seed (3), steps (2), and style (3), with the baseline shared where appropriate.
- `PASS`: no claim was found that Euler/DDIM, LoRA, CLIPScore, or FID had been executed.

## 3. Test results

- `PASS`: combined applicable suite: **65 passed**, 0 failed, with one third-party `langchain-community` deprecation warning.
- `PASS`: regression coverage was added in `labs/lab03_rag/tests/test_rag.py` for citation-free factual answers, nonexistent IDs, and mixed known/unknown ID lookup.
- `PASS`: `compileall` completed for the maintained Python sources.
- `PASS`: `git diff --check` reported no whitespace errors.
- `PASS`: tests are CPU-compatible; they do not require Redis, LM Studio, or GPU inference.

## 4. Notebook execution results

- `PASS`: both notebooks parse with `nbformat`; execution is performed by `nbclient.NotebookClient` in `scripts/execute_portfolio_notebooks.py`.
- `PASS`: replay mode deliberately reports Lab 3 as `SKIPPED mode=replay reason=live-only`. Its 16 code cells retain historical live outputs and are labeled `HISTORICAL_LIVE_OUTPUTS`, not replay output.
- `PASS`: Lab 4 replay executed **21/21 code cells** with 0 execution errors and refreshed `portfolio_execution` metadata.
- `PASS`: both notebooks exported to HTML successfully (Lab 3: 550,600 bytes; Lab 4: 10,119,073 bytes).
- `PASS`: repository paths are relative; no author-specific absolute path is required by the notebook execution path.
- `PASS`: README and audit documentation now match the executor. Earlier counts of 13 Lab 4 cells described an older notebook revision; the current notebook contains 21 executable cells.

## 5. Evaluation integrity

- `PASS`: metrics were recomputed from `labs/lab03_rag/artifacts/portfolio/baseline_vs_hybrid.csv`, not copied from prose:
  - semantic baseline: Hit@5 `0.857143`, MRR `0.690476`, Recall@5 `0.573413`;
  - hybrid routing: Hit@5 `1.000000`, MRR `1.000000`, Recall@5 `0.882937`;
  - post-rule wording set: Hit@5 `1.000000`, MRR `1.000000`, Recall@5 `0.911111`.
- `PASS`: retrieval metrics are not presented as answer-quality metrics.
- `PASS`: documentation now calls `holdout_queries.json` a post-rule wording set. It is not represented as an independent external benchmark; doing so would overstate generalization.
- `PASS`: all 13 SHA-256 values in the Stable Diffusion generation manifest match the committed image files; artifact IDs and controlled configurations are unique.

## 6. Visual QA

- `PASS`: the RAG architecture/evaluation diagrams and Lab 4 architecture, steps, CFG, seed, negative-prompt, and style comparisons were inspected for readable titles, axes, labels, and controlled-variable interpretation.
- `PASS`: committed PNG files decode successfully, and comparison grids correspond to manifest configurations.
- `PASS`: qualitative observations are labeled as observations or rubric judgments, not standardized perceptual metrics.
- `NOT_VERIFIED`: three Lab 3 and thirteen Lab 4 rendered notebook output images lack HTML alternative text. This is an accessibility limitation, not a correctness failure.

## 7. Live execution status

- `NOT_VERIFIED`: Redis was unreachable at `localhost:6379`; therefore the requested live exact-ID, structured, numeric, semantic, and unsupported-query smoke tests were not fabricated.
- `NOT_VERIFIED`: LM Studio was unreachable at `localhost:1234`; no live generated-answer claim is made.
- `NOT_APPLICABLE`: Stable Diffusion GPU generation was not repeated. Historical environment data and generation evidence are recorded in `labs/lab04_stable_diffusion/artifacts/portfolio/historical_environment.json` and `generations.csv`.

## 8. Defects found and fixed

- `PASS`: fixed vague catalog-empty behavior in `labs/lab03_rag/src/hybrid.py`; it now distinguishes nonexistent products from filters with no matching rows.
- `PASS`: fixed citation validation in `labs/lab03_rag/src/rag.py`; a factual response without evidence no longer passes validation.
- `PASS`: fixed contradictory Lab 3 notebook metadata and ensured a future full execution writes `EXECUTED_LIVE`.
- `PASS`: corrected README/audit claims that incorrectly described Lab 3 as offline-replayed and Lab 4 as having 13 current code cells.
- `PASS`: corrected the overstated independent-holdout claim and stale test/result counts.
- `PASS`: repaired Lab 4 report links so they target committed `artifacts/portfolio` files.
- `PASS`: removed four redundant, stale Lab 4 audit documents that duplicated claims and were introduced only on this PR; this combined audit replaces them. Experimental evidence was not removed.

## 9. Remaining limitations

- `NOT_VERIFIED`: Lab 3 end-to-end live behavior remains environment-dependent on Redis, MiniLM availability, and an LM Studio-compatible endpoint.
- `NOT_VERIFIED`: the post-rule wording set is useful regression evidence but not an independently designed external benchmark.
- `NOT_VERIFIED`: HTML output images do not yet have complete alternative text.
- `NOT_APPLICABLE`: no new GPU timing, memory, or image-quality measurements were produced in this audit.
- `PASS`: these limitations are explicit and do not invalidate the committed deterministic tests, retrieval evidence, or historical generation integrity.

## 10. Merge readiness

- `PASS`: PR scope is limited to Labs 3–4 and shared CI/documentation/execution files required by them. Local `NNTS_VARIANT_2/`, `vllm-lab/`, `scripts/fix_lab1_report_layout.py`, and the unrelated `.gitignore` edit are excluded.
- `PASS`: important experiment artifacts remain committed and internally consistent.
- `PASS`: PR #1 is conflict-free and mergeable at the audited head.
- `PASS`: merge may proceed after the final GitHub Actions run for the audit commit is green. Automatic merge is intentionally not performed.
