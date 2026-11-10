"""Factual integrity and documentation consistency tests for Labs 4-7.
Ensures primary evidence, READMEs, reports, and code artifacts match perfectly.
"""
from pathlib import Path
import csv
import json
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_lab04_baseline_integrity():
    csv_path = REPO_ROOT / "labs/lab04_stable_diffusion/artifacts/portfolio/generations.csv"
    assert csv_path.exists(), f"Missing {csv_path}"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        baseline_rows = [row for row in reader if row.get("experiment") == "baseline"]

    assert len(baseline_rows) == 1, "Expected exactly 1 baseline row in generations.csv"
    baseline = baseline_rows[0]
    expected_prompt = (
        "a futuristic city street at sunset, neon signs, "
        "wet pavement reflections, detailed architecture, "
        "cinematic lighting, pedestrians in the distance"
    )
    assert baseline["prompt"] == expected_prompt
    assert baseline["seed"] == "42"
    assert baseline["num_inference_steps"] == "30"
    assert baseline["guidance_scale"] == "7.5"
    assert baseline["scheduler"] == "PNDMScheduler"

    # Verify README doesn't contain fabricated library scene
    readme_path = REPO_ROOT / "labs/lab04_stable_diffusion/README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "ancient stone library overgrown with ivy" not in content.lower()
    assert expected_prompt in content


def test_lab05_evidence_and_image_paths():
    canonical_img = REPO_ROOT / "labs/lab05_multimodal/assets/input/primary_image.png"
    assert canonical_img.exists(), f"Canonical input image missing: {canonical_img}"

    runs_path = REPO_ROOT / "labs/lab05_multimodal/evidence/verified_runs.json"
    with open(runs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs", [])
    assert len(runs) == 17, f"Expected 17 verified runs, got {len(runs)}"

    readme_path = REPO_ROOT / "labs/lab05_multimodal/README.md"
    content = readme_path.read_text(encoding="utf-8")
    assert "yellow chair" not in content.lower()
    assert "assets/input_image.png" not in content
    assert "assets/architecture_diagram.png" not in content
    assert "assets/input/primary_image.png" in content


def test_lab06_registered_tools():
    from labs.lab06_agents.src.tools import build_tools

    tools = build_tools()
    tool_names = sorted(t.name for t in tools)
    expected_names = sorted([
        "calculate",
        "list_workspace_files",
        "read_workspace_file",
        "search_workspace",
    ])
    assert tool_names == expected_names

    # Check root README doesn't misattribute lab 7 tools to lab 6
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "labs/lab06_agents" in root_readme
    assert "list_workspace_files" in root_readme
    assert "calculate" in root_readme


def test_lab07_api_import_and_command():
    from labs.lab07_final_project.src.knowledge_copilot.api.app import app
    assert app is not None

    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "labs.lab07_final_project.src.knowledge_copilot.api.app:app" in root_readme
    assert "labs.lab07_final_project.src.knowledge_copilot.api.main:app" not in root_readme


def test_lab07_benchmark_numbers_in_docs():
    bench_path = REPO_ROOT / "labs/lab07_final_project/evaluation/baselines/benchmark_results.json"
    with open(bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)

    # Hybrid
    assert round(bench["modes"]["hybrid"]["evidence_recall_1"], 4) == 0.5721
    assert round(bench["modes"]["hybrid"]["evidence_recall_3"], 4) == 0.8500
    assert round(bench["modes"]["hybrid"]["evidence_recall_5"], 4) == 0.9029
    assert round(bench["modes"]["hybrid"]["hit_at_5"], 4) == 0.9647
    assert round(bench["modes"]["hybrid"]["mrr"], 4) == 0.8363
    assert round(bench["modes"]["hybrid"]["latency"]["mean_ms"], 2) == 37.92

    # Lexical / BM25
    assert round(bench["modes"]["lexical"]["evidence_recall_3"], 4) == 0.8588
    assert round(bench["modes"]["lexical"]["hit_at_5"], 4) == 0.9765
    assert round(bench["modes"]["lexical"]["latency"]["mean_ms"], 2) == 2.27

    # Dense
    assert round(bench["modes"]["dense"]["latency"]["mean_ms"], 2) == 47.61

    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "0.5721" in root_readme  # Hybrid Recall@1
    assert "0.8588" in root_readme  # BM25 Recall@3
    assert "0.9765" in root_readme  # BM25 Hit@5
    assert "47.61" in root_readme   # Dense mean latency
    assert "2.27" in root_readme    # BM25 mean latency
    assert "37.92" in root_readme   # Hybrid mean latency
    assert "0.6941" not in root_readme  # Fabricated old BM25 recall@3


def test_lab07_generation_results_status():
    gen_path = REPO_ROOT / "labs/lab07_final_project/evaluation/baselines/generation_results.json"
    with open(gen_path, "r", encoding="utf-8") as f:
        gen = json.load(f)

    if gen.get("status") == "VERIFIED_LIVE":
        assert gen["total_queries"] == 50
        assert gen["successful_turns"] == 45
        assert gen["model_id"] == "mistralai/ministral-3-3b"
    else:
        assert gen["status"] == "NOT_VERIFIED"
        assert gen["gate_status"].startswith("BLOCKED")
        assert gen["executed_live_turns"] == 0

    report_path = REPO_ROOT / "labs/lab07_final_project/report/report.md"
    report_text = report_path.read_text(encoding="utf-8")
    assert "VERIFIED_LIVE" in report_text or "NOT_VERIFIED" in report_text
