import os
import subprocess
from pathlib import Path

EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
NBCONVERT_EXE = Path("../NNTS_VARIANT_2/.venv/Scripts/jupyter-nbconvert.exe").resolve()

notebooks_to_capture = [
    # Lab 3
    {
        "nb_path": Path("../NNTS_VARIANT_2/lab03_chroma_rag/notebooks/01_corpus_preparation.ipynb").resolve(),
        "out_png": Path("../NNTS_VARIANT_2/lab03_chroma_rag/outputs/figures/screenshot_nb01_corpus.png").resolve()
    },
    {
        "nb_path": Path("../NNTS_VARIANT_2/lab03_chroma_rag/notebooks/02_chroma_retrieval.ipynb").resolve(),
        "out_png": Path("../NNTS_VARIANT_2/lab03_chroma_rag/outputs/figures/screenshot_nb02_retrieval.png").resolve()
    },
    {
        "nb_path": Path("../NNTS_VARIANT_2/lab03_chroma_rag/notebooks/03_rag_evaluation.ipynb").resolve(),
        "out_png": Path("../NNTS_VARIANT_2/lab03_chroma_rag/outputs/figures/screenshot_nb03_rag.png").resolve()
    },
    # Lab 4
    {
        "nb_path": Path("../NNTS_VARIANT_2/lab04_img2img/notebooks/01_input_images_and_pipeline.ipynb").resolve(),
        "out_png": Path("../NNTS_VARIANT_2/lab04_img2img/outputs/figures/screenshot_nb01_pipeline.png").resolve()
    },
    {
        "nb_path": Path("../NNTS_VARIANT_2/lab04_img2img/notebooks/02_parameter_experiments.ipynb").resolve(),
        "out_png": Path("../NNTS_VARIANT_2/lab04_img2img/outputs/figures/screenshot_nb02_experiments.png").resolve()
    },
    {
        "nb_path": Path("../NNTS_VARIANT_2/lab04_img2img/notebooks/03_results_analysis.ipynb").resolve(),
        "out_png": Path("../NNTS_VARIANT_2/lab04_img2img/outputs/figures/screenshot_nb03_analysis.png").resolve()
    }
]

for item in notebooks_to_capture:
    nb_path = item["nb_path"]
    out_png = item["out_png"]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    
    html_path = nb_path.with_suffix(".html")
    print(f"Конвертація {nb_path.name} в HTML...")
    cmd_nbconvert = [
        str(NBCONVERT_EXE),
        "--to", "html",
        str(nb_path),
        "--output", str(html_path)
    ]
    subprocess.run(cmd_nbconvert, check=True)
    
    print(f"Зняття скріншоту для {nb_path.name} -> {out_png.name}...")
    file_url = html_path.as_uri()
    cmd_edge = [
        EDGE_EXE,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1280,960",
        f"--screenshot={str(out_png)}",
        file_url
    ]
    subprocess.run(cmd_edge, check=True)
    assert out_png.exists() and out_png.stat().st_size > 10000, f"Помилка створення скріншоту: {out_png}"
    print(f"Скріншот успішно збережено: {out_png} ({out_png.stat().st_size} байт)")

print("\nВсі 6 скріншотів блокнотів успішно згенеровано!")
