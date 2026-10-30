import os
import shutil
from pathlib import Path

SRC_DIR = Path("../NNTS_VARIANT_2").resolve()
DEST_DIR = Path(r"C:\Users\Eclipse\PycharmProjects\Neural-Network-Labs-Max-Deploy").resolve()

assert SRC_DIR.exists(), f"Source dir {SRC_DIR} does not exist"
assert DEST_DIR.exists(), f"Dest dir {DEST_DIR} does not exist"

# Ensure lab1 and lab2_vllm are untouched
assert (DEST_DIR / "lab1").exists(), "lab1 is missing in dest!"
assert (DEST_DIR / "lab2_vllm").exists(), "lab2_vllm is missing in dest!"

def ignore_patterns(path, names):
    ignored = set()
    for name in names:
        if name in ("__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv", "venv", ".git"):
            ignored.add(name)
        elif name.endswith(".pyc") or name.endswith(".tmp") or name.endswith(".log"):
            ignored.add(name)
    return ignored

# 1. Copy Lab 3
src_lab3 = SRC_DIR / "lab03_chroma_rag"
dest_lab3 = DEST_DIR / "lab3_chroma_rag"
if dest_lab3.exists():
    shutil.rmtree(dest_lab3)
print(f"Копіювання {src_lab3} -> {dest_lab3}...")
shutil.copytree(src_lab3, dest_lab3, ignore=ignore_patterns)

# 2. Copy Lab 4
src_lab4 = SRC_DIR / "lab04_img2img"
dest_lab4 = DEST_DIR / "lab4_img2img"
if dest_lab4.exists():
    shutil.rmtree(dest_lab4)
print(f"Копіювання {src_lab4} -> {dest_lab4}...")
shutil.copytree(src_lab4, dest_lab4, ignore=ignore_patterns)

# 3. Create Root README.md
readme_content = """# Нейромережеві технології і системи — Лабораторні роботи

**Національний технічний університет України «КПІ ім. Ігоря Сікорського»**  
**Кафедра обчислювального інтелекту та смарт-систем**  
**Спеціальність:** 122 «Комп’ютерні науки»  

---

## Огляд репозиторію

Даний репозиторій містить повний комплекс програмних реалізацій, виконаних Jupyter-ноутбуків, датасетів, згенерованих результатів та академічних звітів з дисципліни «Нейромережеві технології і системи».

| Лабораторна | Тема | Варіант | Стек технологій | Статус |
|---|---|---|---|---|
| **Лабораторна №1** | Дослідження нейромережевих архітектур | Варіант 1 | PyTorch, torchvision, CNN | ✅ Завершено |
| **Лабораторна №2** | Високопродуктивний інференс LLM через vLLM | Варіант 1 | vLLM, OpenAI API, CLI | ✅ Завершено |
| **Лабораторна №3** | Реалізація RAG-системи для витягування знань | **Варіант №2** (ChromaDB + Markdown/Текст) | LangChain, ChromaDB, all-MiniLM-L6-v2, Ollama API (`qwen3:4b-instruct`) | ✅ Завершено (10/10 PASS) |
| **Лабораторна №4** | Генерування зображень за допомогою Stable Diffusion | **Варіант №2** (Image-to-Image, 3 сюжети) | Diffusers, StableDiffusionImg2ImgPipeline, SD 1.5 FP16, CPU Offload | ✅ Завершено (10/10 PASS) |

---

## Структура нових лабораторних робіт

### [lab3_chroma_rag/](./lab3_chroma_rag/) — Лабораторна робота №3 (Варіант №2)
- **`data/knowledge_base/`** — 8 авторських технічних документів з життєвого циклу MLOps (6 `.md`, 2 `.txt`).
- **`notebooks/`** — 3 повністю виконані блокноти з багатим візуальним виводом:
  - `01_corpus_preparation.ipynb` — завантаження через `DirectoryLoader`, порівняння чанкінгу, аудит метаданих (`source`, `type`).
  - `02_chroma_retrieval.ipynb` — створення та верифікація персистентної ChromaDB (55 векторів), підбір гіперпараметрів MMR, фільтрація за метаданими.
  - `03_rag_evaluation.ipynb` — побудова RAG-ланцюга з локальною LLM `qwen3:4b-instruct`, перевірка стійкості до галюцинацій та оцінка заземленості.
- **`outputs/`** — персистентне сховище ChromaDB, згенеровані таблиці, відповіді LLM та порівняльні графіки.
- **`report/`** — офіційні звіти у форматах `report.docx`, `report.pdf`, Markdown-звіт `report.md`, матеріали до захисту `defense_notes.md` та аудит методички `methodology_audit.md`.
- **`tests/test_lab03.py`** — набір автоматичних тестів (10/10 PASS).

### [lab4_img2img/](./lab4_img2img/) — Лабораторна робота №4 (Варіант №2)
- **`inputs/`** — 3 оригінальні вхідні цифрові зображення $512 \\times 512$ RGB (Маяк, Кафе з велосипедом, Бібліотека).
- **`notebooks/`** — 3 повністю виконані блокноти:
  - `01_input_images_and_pipeline.ipynb` — діагностика середовища CUDA, ініціалізація `StableDiffusionImg2ImgPipeline` FP16 + CPU offload, базові стилізації.
  - `02_parameter_experiments.ipynb` — контрольовані параметричні серії (`strength`, `CFG`, `steps`, `negative prompt`) при фіксованому `seed=42`.
  - `03_results_analysis.ipynb` — поглиблений візуальний аналіз, бенчмарк продуктивності GPU (RTX 3050) та теорія LoRA.
- **`outputs/`** — 14 згенерованих PNG-файлів, 13 детальних файлів JSON метаданих, CSV-реєстр та аналітичні графіки.
- **`report/`** — офіційні звіти у форматах `report.docx`, `report.pdf`, Markdown-звіт `report.md`, матеріали до захисту `defense_notes.md` та матриця відповідності методичці `methodology_audit.md`.
- **`tests/test_lab04.py`** — набір автоматичних тестів (10/10 PASS).

---

## Відтворюваність та запуск

1. Встановлення залежностей:
   ```bash
   pip install -r requirements.txt
   ```
2. Запуск тестів:
   ```bash
   pytest lab3_chroma_rag/tests/ lab4_img2img/tests/ -v
   ```
3. Локальний LLM-сервер:
   Підтримує будь-який OpenAI-сумісний інтерфейс (Ollama, vLLM, LM Studio) на `http://127.0.0.1:11434/v1`.
"""

readme_path = DEST_DIR / "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

print(f"Створено кореневий README.md: {readme_path}")
print("Копіювання завершено успішно!")
