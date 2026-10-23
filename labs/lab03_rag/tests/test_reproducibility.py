import re
from pathlib import Path

import nbformat
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]


def test_generator_does_not_forge_execution_counts():
    source = (ROOT / "scripts/build_labs34_portfolio.py").read_text(encoding="utf-8")
    assert ".execution_count =" not in source


def test_replay_evidence_is_published_and_not_local_output_dependent():
    required = [
        "labs/lab03_rag/artifacts/portfolio/embedding_audit.json",
        "labs/lab03_rag/artifacts/portfolio/index_info.json",
        "labs/lab03_rag/artifacts/portfolio/semantic_results.csv",
        "labs/lab03_rag/artifacts/portfolio/historical_semantic_answers.csv",
        "labs/lab04_stable_diffusion/artifacts/portfolio/historical_environment.json",
        "labs/lab04_stable_diffusion/artifacts/portfolio/generations.csv",
    ]
    assert not [path for path in required if not (ROOT / path).is_file()]
    for notebook_path in (
        ROOT / "labs/lab03_rag/notebooks/01_redis_csv_rag.ipynb",
        ROOT / "labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb",
    ):
        notebook = nbformat.read(notebook_path, 4)
        source = "\n".join(cell.source for cell in notebook.cells)
        assert "C:\\Users\\Eclipse" not in source
        assert "REPLAYED FROM VERIFIED EVIDENCE" in source


def test_all_portfolio_images_decode_and_markdown_links_resolve():
    images = list((ROOT / "labs/lab03_rag/artifacts/portfolio").rglob("*.png"))
    images += list((ROOT / "labs/lab04_stable_diffusion/artifacts/portfolio").rglob("*.png"))
    assert len(images) == 30
    for path in images:
        with Image.open(path) as image:
            image.verify()
    missing = []
    for markdown in (ROOT / "README.md", ROOT / "labs/lab03_rag/README.md",
                     ROOT / "labs/lab04_stable_diffusion/README.md"):
        text = markdown.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]*\]\(([^)]+)\)", text):
            if "://" not in target and not (markdown.parent / target).resolve().exists():
                missing.append(f"{markdown}:{target}")
    assert not missing
