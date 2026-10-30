"""Build the pedagogical Lab 5 notebook. Execute it separately; do not fake outputs."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "notebooks/01_llava_vqa.ipynb"
nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbf.v4.new_code_cell(source))


md("""# Лабораторна робота №5: мультимодальна VQA з LLaVA

**Варіант 1 — Visual Question Answering.** Мета: пройти повний практичний шлях від PNG і текстового запитання до мультимодального prompt, генерації, вимірювання та критичної оцінки відповіді.

Мультимодальна модель спільно обробляє дані різних модальностей. LLaVA поєднує візуальний encoder, projector і мовну модель. VQA відрізняється від captioning: captioning просить загальний опис, а VQA вимагає відповіді на конкретне питання за видимими ознаками. Тут використано `llava-hf/llava-interleave-qwen-0.5b-hf` на зафіксованій revision; fine-tuning не виконується.

Notebook має два несумісні за походженням режими:

- `LIVE`: завантажує модель і створює **нові** відповіді. За замовчуванням на сумісному CUDA GPU виконує один smoke inference. Повна матриця запускається лише після явної зміни `RUN_FULL_MATRIX=True`.
- `REPLAY`: читає 17 перевірених історичних генерацій/19 експериментальних рядків і виконує тільки аналіз. REPLAY ніколи не називається LIVE.

Для CPU-перевірки задайте `LAB05_MODE=REPLAY`. Новим LIVE-відповідям старі ручні оцінки не призначаються.""")

md("## 2. Імпорт бібліотек і конфігурація\n\nІмпорти та діагностика середовища залишені видимими. Модель тут ще не завантажується.")
code("""from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from IPython.display import display
try:
    import torch
    import transformers
except ImportError:
    torch = None
    transformers = None

repo_root = next(p for p in (Path.cwd(), *Path.cwd().parents)
                 if (p / "labs/lab05_multimodal").is_dir())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from labs.lab05_multimodal.src.config import (
    LAB_ROOT, INPUT_IMAGE_PATH, OUTPUT_ROOT, MODEL_ID, MODEL_REVISION,
    GROUNDING_INSTRUCTION, GenerationConfig,
)
from labs.lab05_multimodal.src.experiments import Case, experiment_plan, load_questions
from labs.lab05_multimodal.src.inference import answer_question, extract_answer, save_json
from labs.lab05_multimodal.src.evaluation import evaluated_results, factual_summary
from labs.lab05_multimodal.src.model import load_pipeline

requested_mode = os.getenv("LAB05_MODE", "AUTO").upper()
if requested_mode not in {"AUTO", "LIVE", "REPLAY"}:
    raise ValueError("LAB05_MODE must be AUTO, LIVE, or REPLAY")
cuda_available = bool(torch is not None and torch.cuda.is_available())
MODE = "LIVE" if requested_mode == "LIVE" or (requested_mode == "AUTO" and cuda_available) else "REPLAY"
RUN_FULL_MATRIX = False  # Change explicitly to True only for a deliberate GPU experiment.

def installed_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"

versions = {name: installed_version(name) for name in
            ("torch", "transformers", "Pillow", "pandas", "matplotlib")}
device_info = {
    "mode": MODE, "python": platform.python_version(), "versions": versions,
    "cuda_available": cuda_available, "cuda_runtime": torch.version.cuda if torch else None,
    "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 3)
                if cuda_available else None,
    "device": "cuda:0" if cuda_available else "cpu",
}
execution_context = {
    "python": device_info["python"], "versions": versions,
    "cuda_runtime": device_info["cuda_runtime"], "gpu_name": device_info["gpu"],
    "dtype": "torch.float16", "device_strategy": "cuda:0",
    "attention": "sdpa", "protocol_version": 2,
}
display(pd.DataFrame([device_info]))
print("LIVE creates new generations; REPLAY reads historical evidence only.")""")

md("## 3. Підготовка вхідного зображення\n\nPillow декодує PNG, після чого зображення явно переводиться в RGB. SHA-256 зв'язує кожну відповідь із точними байтами вхідного файла. Image processor моделі пізніше змінить розмір, нормалізує пікселі й сформує tensor `pixel_values`; ручне масштабування до inference не потрібне.")
code("""source_metadata = json.loads((LAB_ROOT / "assets/input/source.json").read_text(encoding="utf-8"))
input_bytes = INPUT_IMAGE_PATH.read_bytes()
actual_sha256 = hashlib.sha256(input_bytes).hexdigest()
image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
assert actual_sha256 == source_metadata["sha256"], "Input image SHA-256 mismatch"
display(image)
print({"path": str(INPUT_IMAGE_PATH), "dimensions": image.size, "mode": image.mode,
       "sha256": actual_sha256})""")

md("""## 4. Архітектура LLaVA

```text
Image → SigLIP Vision → visual features → MLP Projector ┐
                                                       ├→ Qwen2 → Answer
Question → tokenizer → text embeddings ────────────────┘
```

У LIVE фактичний config береться з завантаженої моделі. У REPLAY нижче читається історичний `model.json`; це позначено окремо і не є поточним завантаженням.""")

md("## 5. Завантаження LLaVA\n\nPipeline створюється один раз за kernel session. Перевірені налаштування: FP16 на CUDA, SDPA, `batch_size=1`; на CPU LIVE навмисно блокується через неприйнятну вартість. Повідомлення про OOM, відсутню CUDA, ваги або несумісність не маскуються.")
code("""pipe = None
model_info = None
if MODE == "LIVE":
    if torch is None or transformers is None:
        raise RuntimeError("LIVE BLOCKED: PyTorch/Transformers are not installed; use REPLAY for analysis.")
    if not cuda_available:
        raise RuntimeError("LIVE BLOCKED: CUDA GPU is unavailable; use LAB05_MODE=REPLAY for analysis.")
    try:
        pipe, model_info = load_pipeline()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("LIVE BLOCKED: insufficient CUDA VRAM for the pinned FP16 checkpoint.") from exc
    except (OSError, ImportError, ValueError) as exc:
        raise RuntimeError(f"LIVE BLOCKED: weights or environment are incompatible: {exc}") from exc
    config = pipe.model.config
    architecture = {
        "provenance": "live model.config",
        "vision_backbone": config.vision_config.model_type,
        "language_backbone": config.text_config.model_type,
        "image_size": config.vision_config.image_size,
        "vision_hidden_size": config.vision_config.hidden_size,
        "text_hidden_size": config.text_config.hidden_size,
        "parameter_count": sum(p.numel() for p in pipe.model.parameters()),
        "processor": type(pipe.processor.image_processor).__name__,
    }
else:
    recorded_model = json.loads((OUTPUT_ROOT / "metadata/model.json").read_text(encoding="utf-8"))
    cfg = recorded_model["model_config"]
    architecture = {
        "provenance": "historical outputs/metadata/model.json",
        "vision_backbone": recorded_model["vision_backbone"],
        "language_backbone": recorded_model["language_backbone"],
        "image_size": recorded_model["image_size"],
        "vision_hidden_size": cfg["vision_config"]["hidden_size"],
        "text_hidden_size": cfg["text_config"]["hidden_size"],
        "parameter_count": recorded_model["parameter_count"],
        "processor": recorded_model["image_processor"]["image_processor_type"],
    }
display(pd.DataFrame([architecture]))""")

md("## 6. Перший реальний VQA inference\n\nЦя комірка містить прямий виклик pipeline. Вона виконується лише в LIVE. Новий результат записується в `outputs/live/` і не отримує історичної ручної оцінки.")
code("""smoke_question = "Is the cabin on the left, in the center, or on the right side of the image?"
smoke_messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": GROUNDING_INSTRUCTION + "\\n\\nQuestion: " + smoke_question},
]}]
smoke_generation = GenerationConfig(do_sample=False, max_new_tokens=64)
live_results = []

if MODE == "LIVE":
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = pipe(
        text=smoke_messages,
        return_full_text=False,
        generate_kwargs=smoke_generation.kwargs(),
    )
    torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - started
    smoke_record = {
        "provenance": "LIVE", "evaluation_label": "NOT_REVIEWED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": smoke_question, "answer": extract_answer(output),
        "generation_config": smoke_generation.kwargs(),
        "runtime_seconds": runtime_seconds,
        "generated_token_count": output[0].get("generated_token_count"),
        "ended_with_eos": output[0].get("ended_with_eos"),
        "reached_token_limit": (
            output[0].get("generated_token_count") == smoke_generation.max_new_tokens
            and not output[0].get("ended_with_eos")
        ),
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "image_sha256": actual_sha256,
    }
    live_path = OUTPUT_ROOT / "live" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    save_json(live_path, smoke_record)
    live_results.append(smoke_record)
    display(pd.DataFrame([smoke_record]))
    print("Saved new unreviewed LIVE result:", live_path)
else:
    print("REPLAY: direct pipe(...) call intentionally not executed; no new answer was generated.")""")

md("## 7. Описові запитання\n\nВидимий код відбирає питання, будує `Case`, викликає спільну inference-функцію в повному LIVE-запуску і формує DataFrame. У звичайному smoke запуску та REPLAY дорогий цикл не виконується.")
code("""questions = load_questions()
questions_by_type = {kind: [q for q in questions if q["question_type"] == kind]
                     for kind in {q["question_type"] for q in questions}}

def run_question_group(question_type, experiment_name):
    group_records = []
    for question in questions_by_type[question_type]:
        case = Case(experiment_name, question["question_id"], question, GenerationConfig())
        record = answer_question(pipe, image, case, execution_context)
        record = {**record, "provenance": "LIVE", "evaluation_label": "NOT_REVIEWED"}
        group_records.append(record)
    return pd.DataFrame(group_records)

print(pd.DataFrame(questions_by_type["descriptive"])[["question_id", "question"]].to_string(index=False))
if MODE == "LIVE" and RUN_FULL_MATRIX:
    descriptive_live = run_question_group("descriptive", "live_descriptive")""")

md("## 8. Фактологічні запитання\n\nНові відповіді мають оцінюватися заново. Старі labels не копіюються за `question_id`.")
code("""print(pd.DataFrame(questions_by_type["factual"])[["question_id", "question", "expected_facts"]].to_string(index=False))
if MODE == "LIVE" and RUN_FULL_MATRIX:
    factual_live = run_question_group("factual", "live_factual")
    factual_live["evaluation_label"] = "NOT_REVIEWED"
    display(factual_live[["question", "answer", "evaluation_label"]])""")

md("## 9. Аналітичні запитання\n\nОцінюється не лише висновок, а й наявність посилання на видимі ознаки. Автоматична перевірка ключових слів може бути лише підказкою, не ручним verdict.")
code("""visible_evidence_terms = ("reflection", "tree", "water", "shore", "inverted", "wave", "cabin")
print(pd.DataFrame(questions_by_type["analytical"])[["question_id", "question"]].to_string(index=False))
if MODE == "LIVE" and RUN_FULL_MATRIX:
    analytical_live = run_question_group("analytical", "live_analytical")
    analytical_live["mentions_visible_evidence"] = analytical_live.answer.str.lower().apply(
        lambda text: any(term in text for term in visible_evidence_terms))
    display(analytical_live[["question", "answer", "mentions_visible_evidence", "evaluation_label"]])""")

md("## 10. Дослідження temperature\n\nЗмінюється тільки `temperature`; `top_p=0.9`, `max_new_tokens=64`, seed=42 фіксовані. Один seed на умову не оцінює розподіл якості.")
code("""temperature_results = []
temperature_question = next(q for q in questions if q["question_id"] == "analysis_01")
if MODE == "LIVE" and RUN_FULL_MATRIX:
    for temperature in [0.2, 0.7, 1.0]:
        generation = GenerationConfig(True, 64, temperature, 0.9, 42)
        case = Case("live_temperature", f"temperature_{temperature}", temperature_question, generation)
        row = answer_question(pipe, image, case, execution_context)
        temperature_results.append({**row, "provenance": "LIVE", "evaluation_label": "NOT_REVIEWED"})
temperature_live_df = pd.DataFrame(temperature_results)
print("LIVE temperature configurations:", [0.2, 0.7, 1.0])""")

md("## 11. Дослідження max_new_tokens\n\nЛіміт — верхня межа, а не бажана довжина. EOS і факт досягнення ліміту вимірюються з фактичного continuation.")
code("""token_results = []
token_question = next(q for q in questions if q["question_id"] == "desc_01")
if MODE == "LIVE" and RUN_FULL_MATRIX:
    for max_new_tokens in [24, 64, 128]:
        generation = GenerationConfig(max_new_tokens=max_new_tokens)
        case = Case("live_max_tokens", f"tokens_{max_new_tokens}", token_question, generation)
        row = answer_question(pipe, image, case, execution_context)
        token_results.append({**row, "provenance": "LIVE", "evaluation_label": "NOT_REVIEWED"})
token_live_df = pd.DataFrame(token_results)
print("LIVE token-limit configurations:", [24, 64, 128])""")

md("## 12. Дослідження узгодженості\n\nТри семантично близькі питання запускаються з однаковим greedy config. Суперечність визначається після нормалізації змісту, а не регістру.")
code("""paraphrases = [
    "Is the cabin on the left, in the center, or on the right side of the image?",
    "Where is the cabin located horizontally in the image: left, center, or right?",
    "Which horizontal part of the image contains the cabin: left, center, or right?",
]
paraphrase_results = []
if MODE == "LIVE" and RUN_FULL_MATRIX:
    template = next(q for q in questions if q["question_id"] == "fact_01")
    for index, wording in enumerate(paraphrases, 1):
        question = {**template, "question": wording}
        case = Case("live_paraphrase", f"phrasing_{index}", question, GenerationConfig())
        row = answer_question(pipe, image, case, execution_context)
        paraphrase_results.append({**row, "provenance": "LIVE", "evaluation_label": "NOT_REVIEWED"})
paraphrase_live_df = pd.DataFrame(paraphrase_results)
print("Paraphrases:", *paraphrases, sep="\\n- ")""")

md("## 13. Галюцинації та обмеження\n\nКонтрольне питання не має визначуваної з пікселів відповіді. Очікується явне визнання невизначеності, але один приклад не доводить, що модель завжди галюцинує або завжди відмовляється.")
code("""unanswerable_question = next(q for q in questions if not q["answerable"])
print("Question:", unanswerable_question["question"])
print("Expected behavior: explicitly state that the answer cannot be determined from the image.")
if MODE == "LIVE" and RUN_FULL_MATRIX:
    control_case = Case("live_control", "unanswerable", unanswerable_question, GenerationConfig())
    control_live = answer_question(pipe, image, control_case, execution_context)
    control_live["evaluation_label"] = "NOT_REVIEWED"
    display(pd.DataFrame([control_live]))""")

md("## 14. Аналіз історичних результатів (REPLAY)\n\nДалі читається перевірений manifest: 19 рядків, 17 унікальних `run_id`. Це історичні GPU-відповіді, не результат поточного запуску notebook. Ручні оцінки зв'язані з `run_id`, exact answer та image SHA.")
code("""historical_rows = json.loads((OUTPUT_ROOT / "metadata/manifest.json").read_text(encoding="utf-8"))
assert len(historical_rows) == 19
assert len({row["run_id"] for row in historical_rows}) == 17
historical = evaluated_results(historical_rows)
historical["provenance"] = "HISTORICAL_REPLAY"
assert not (historical.evaluation_label == "NOT_REVIEWED").any()
print(f"REPLAY evidence: {len(historical)} experiment rows, {historical.run_id.nunique()} unique generations")
display(historical.groupby(["experiment", "question_type"]).size().rename("rows").reset_index())""")
code("""baseline = historical[historical.experiment == "baseline"]
for question_type in ["descriptive", "factual", "analytical", "unanswerable"]:
    print("\\n", question_type.upper())
    display(baseline.loc[baseline.question_type == question_type,
                         ["question", "answer", "evaluation_label", "notes", "provenance"]])""")
code("""temperature_history = historical[historical.experiment == "temperature"].sort_values("temperature")
token_history = historical[historical.experiment == "max_tokens"].sort_values("max_new_tokens")
paraphrase_history = historical[historical.experiment == "paraphrase"]
display(temperature_history[["temperature", "answer", "runtime_seconds", "generated_token_count", "evaluation_label"]])
display(token_history[["max_new_tokens", "answer", "generated_token_count", "ended_with_eos", "reached_token_limit"]])
display(paraphrase_history[["question", "answer", "evaluation_label"]])""")
code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
axes[0].plot(temperature_history.temperature, temperature_history.generated_token_count, marker="o")
axes[0].set(xlabel="temperature", ylabel="generated tokens", title="Historical temperature sweep")
axes[1].bar(token_history.max_new_tokens.astype(str), token_history.generated_token_count,
            color=["#b44949", "#256d85", "#256d85"])
axes[1].set(xlabel="max_new_tokens", ylabel="generated tokens", title="Historical token limits")
for axis in axes:
    axis.grid(axis="y", alpha=.2)
plt.show()""")
code("""display(factual_summary(historical))
error_analysis = baseline[["question_id", "question_type", "evaluation_label", "notes"]].copy()
error_analysis["needs_attention"] = ~error_analysis.evaluation_label.isin(
    ["CORRECT", "GROUNDED", "SUPPORTED_BY_VISIBLE_EVIDENCE", "CORRECT_UNCERTAINTY"])
display(error_analysis[error_analysis.needs_attention])""")

md("""## 15. Висновки

Історичний експеримент показав 3/3 правильні прості фактологічні baseline-відповіді, але це лише один синтетичний кадр. Описові відповіді містили непідтверджені або неточні деталі; аналітичні відповіді інколи давали правильну категорію без достатнього видимого обґрунтування. Контроль координат не визнав невизначеність. Три перефразування дали `left`, `center`, `Left`, тому узгодженість неповна.

За temperature 0.2/0.7/1.0 висновок про спокійну воду залишався семантично стабільним у цій одиничній серії. `max_new_tokens=24` обірвав відповідь без EOS; 64 і 128 дали однакову завершену відповідь. Більший бюджет не гарантував кращого змісту.

Вибірка не дає підстав для загальної оцінки LLaVA. VQA відповідає на цільове питання, тоді як captioning створює загальний опис. Поточний режим виконання надруковано у розділі 2; збережені outputs цього notebook є REPLAY і не доводять новий GPU inference.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
TARGET.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, TARGET)
print(f"Built {TARGET} with {len(cells)} cells")
