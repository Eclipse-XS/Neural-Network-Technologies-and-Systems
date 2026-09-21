# Neural Network Technologies and Systems

This repository contains all seven laboratory works for the university course **Neural Network Technologies and Systems**. The labs are kept in one repository while remaining independent educational units. Implementations, experiments, and results will be added incrementally as the course progresses.

## Laboratory works

1. **Simple Transformer Model** — study and implement the core components of a transformer.
2. **LLM Integration** — use a large language model through an API or local inference.
3. **RAG System** — build a retrieval-augmented generation workflow.
4. **Stable Diffusion** — generate images with a diffusion model.
5. **Multimodal Models / LLaVA** — work with combined visual and textual inputs.
6. **Agent-Based Assistant** — develop an assistant that uses agent-based behavior.
7. **Final Integrated Project** — combine at least two technologies from earlier labs.

## Repository navigation

- `labs/` contains one isolated workspace per laboratory, including source code, notebooks, local data, generated outputs, and a lab README.
- `shared/` is reserved for reusable configuration, LLM integration, utility, and evaluation code when duplication emerges across labs.
- `tests/` contains repository-level and shared-code tests; lab-specific tests may be added within the relevant lab when needed.
- `docs/lab_guidelines/` contains course-wide laboratory guidance and supporting documentation.
- `scripts/` contains narrowly scoped development or repository automation scripts.

Large datasets, model weights, checkpoints, vector databases, generated outputs, secrets, virtual environments, and IDE metadata are excluded from version control.
