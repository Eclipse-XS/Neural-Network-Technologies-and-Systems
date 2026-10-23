# Neural Network Technologies and Systems

This repository contains seven independent laboratory workspaces for the university course **Neural Network Technologies and Systems**. Laboratories 3 and 4 are complete educational and portfolio projects with executed notebooks, tested source modules, machine-readable evaluation, and curated public artifacts. The status of Labs 5–7 is not inferred from that work.

## Featured completed projects

### Lab 3: RAG with Redis and CSV

An auditable product-catalog RAG system over 185 records. It compares a Redis semantic baseline with a leakage-free hybrid router for exact, attribute and numeric questions, separates inference from ground-truth evaluation, preserves row provenance, and documents eight concrete failure cases.

[Open Lab 3](labs/lab03_rag/README.md)

![Lab 3 evaluation](labs/lab03_rag/artifacts/portfolio/baseline_vs_hybrid.png)

### Lab 4: Stable Diffusion Text-to-Image

A controlled SD 1.5 experiment with 13 verified generations covering seeds, inference steps, classifier-free guidance, negative prompts and prompt-based styles. Each result is linked to controls, runtime metadata and a content hash.

[Open Lab 4](labs/lab04_stable_diffusion/README.md)

![Lab 4 seed comparison](labs/lab04_stable_diffusion/artifacts/portfolio/seed_comparison.png)

## Laboratory works

1. **Simple Transformer Model** — study and implement the core components of a transformer.
2. **LLM Integration** — use a large language model through an API or local inference.
3. **RAG System** — implemented: Redis Vector Similarity + CSV, grounded local LLM, evaluation and hybrid retrieval.
4. **Stable Diffusion** — implemented: SD 1.5 Text-to-Image with controlled experiments and verified artifacts.
5. **Multimodal Models / LLaVA** — work with combined visual and textual inputs.
6. **Agent-Based Assistant** — develop an assistant that uses agent-based behavior.
7. **Final Integrated Project** — combine at least two technologies from earlier labs.

## Repository navigation

- `labs/` contains one isolated workspace per laboratory, including source code, notebooks, local data, generated outputs, and a lab README.
- `shared/` is reserved for reusable configuration, LLM integration, utility, and evaluation code when duplication emerges across labs.
- `tests/` contains repository-level and shared-code tests; lab-specific tests may be added within the relevant lab when needed.
- `docs/lab_guidelines/` contains course-wide laboratory guidance and supporting documentation.
- `scripts/` contains narrowly scoped development or repository automation scripts.

Large datasets, model weights, checkpoints, vector databases, full generated outputs, secrets, virtual environments, and IDE metadata are excluded from version control. Small reviewed artifacts under each lab's `artifacts/portfolio/` directory are intentionally versioned for GitHub rendering.

## CPU-only portfolio replay

Labs 3 and 4 have a versioned replay mode that does not require Redis, LM Studio, CUDA or model downloads. After installing the lightweight CI dependencies, run `python scripts/execute_portfolio_notebooks.py --mode replay` from the repository root. This executes both notebooks through a real Jupyter kernel and marks their outputs `REPLAYED FROM VERIFIED EVIDENCE`. Full experimental execution remains separate and requires the services and hardware documented in each lab README.
