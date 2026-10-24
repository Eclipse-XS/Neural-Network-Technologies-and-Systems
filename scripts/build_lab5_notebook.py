"""Build the educational Lab 5 notebook from source-controlled evidence."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'labs/lab05_multimodal/notebooks/01_llava_vqa.ipynb'
nb = nbf.v4.new_notebook(metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13'},
})
m = nbf.v4.new_markdown_cell
c = nbf.v4.new_code_cell
nb.cells = [
m('''# Лабораторна робота №5: Visual Question Answering із LLaVA

Це відтворюване навчальне дослідження checkpoint `llava-hf/llava-interleave-qwen-0.5b-hf`. Воно розділяє три різні дії:

1. **Історичний GPU inference**: 17 реальних генерацій, збережених разом із параметрами, SHA-256 і середовищем.
2. **Повторне використання результату**: два результати входять до кількох експериментальних зрізів, тому маємо 19 рядків.
3. **CPU-аналітика**: перевірка доказів, таблиці, підрахунки та графіки, які виконуються заново.

Run All не видає старий час за новий inference і не завантажує модель. Для нового inference використовується CLI з `src/run.py`.'''),
c('''from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display, Markdown

ROOT = Path.cwd().resolve()
while not (ROOT / "labs/lab05_multimodal").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / "labs/lab05_multimodal").exists():
    raise RuntimeError("Запустіть notebook усередині репозиторію")
sys.path.insert(0, str(ROOT))
LAB = ROOT / "labs/lab05_multimodal"
from labs.lab05_multimodal.src.evidence import historical_experiment_rows, file_sha256
from labs.lab05_multimodal.src.evaluation import evaluated_results
from labs.lab05_multimodal.src.experiments import experiment_plan
from labs.lab05_multimodal.src.visualization import token_budget_figure, temperature_figure, error_taxonomy_figure

rows = historical_experiment_rows()
results = evaluated_results(rows)
print(f"Перевірено: {len(results)} рядків, {results.run_id.nunique()} унікальних GPU-генерацій")'''),
m('''## 1. Задача і межі висновків

**VQA** відповідає на конкретне запитання про зображення; **image captioning** створює загальний опис без цільового запиту. Тут зображення обмежує доступні факти, а текст визначає, які ознаки потрібно використати. Один синтетичний кадр і десять baseline-питань не вимірюють загальну точність LLaVA. Вони демонструють механізм inference та характерні відмови.

Питання поділено на описові, фактологічні, аналітичні й контрольне, на яке пікселі не дають відповіді.'''),
c('''questions = pd.read_json(LAB / "questions.json")
taxonomy = questions.groupby("question_type").agg(
    questions=("question_id", "count"), answerable=("answerable", "all"))
display(taxonomy)
display(questions[["question_id", "question_type", "question", "expected_facts"]])'''),
m('''## 2. Вхід: байти, пікселі та provenance

До моделі надходить RGB-зображення, а не назва файлу чи prompt, яким його колись створили. SHA-256 фіксує точні байти. Розмір 512×512 описує оригінал; vision encoder працює з 384×384 після штатного preprocessing.'''),
c('''image_path = LAB / "assets/input/primary_image.png"
source = json.loads(image_path.with_name("source.json").read_text(encoding="utf-8"))
image = Image.open(image_path).convert("RGB")
actual_sha = file_sha256(image_path)
assert actual_sha == source["sha256"]
pixels = np.asarray(image)
inspection = pd.Series({
    "формат": Image.open(image_path).format,
    "початковий розмір": f"{image.width}×{image.height}",
    "режим після convert": image.mode,
    "масив": str(pixels.shape),
    "dtype": str(pixels.dtype),
    "мін./макс. каналів": f"{pixels.min()} / {pixels.max()}",
    "SHA-256": actual_sha,
})
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axes[0].imshow(image); axes[0].set_title("Фактичний вхід 512×512 RGB"); axes[0].axis("off")
for channel, color, label in zip(range(3), ["#b24a44", "#3d7a57", "#3d638a"], "RGB"):
    axes[1].hist(pixels[..., channel].ravel(), bins=32, alpha=.55, color=color, label=label)
axes[1].set(title="Розподіл інтенсивностей", xlabel="Значення 0–255", ylabel="Пікселі")
axes[1].legend()
display(inspection.to_frame("значення")); plt.show()'''),
m('''## 3. Як саме LLaVA поєднує модальності

За збереженим config цього checkpoint:

- `SiglipVisionModel`: 26 transformer-шарів, hidden size 1152, patch 14, вхід 384×384.
- `image_seq_length=576`: vision-гілка формує послідовність візуальних позицій. Це не RGB-пікселі й не слова, а вектори ознак.
- MLP projector із GELU переводить ознаки 1152-вимірного vision-простору в 1024-вимірний простір мовної моделі.
- `Qwen2ForCausalLM`: 24 шари, hidden size 1024. Tokenizer кодує chat template та питання; спеціальні image-token positions замінюються спроєктованими visual embeddings.

Модель далі оцінює розподіл наступного токена $P(y_t\\mid y_{<t}, x_{text}, x_{image})$. Згенерований токен додається до контексту, і цикл повторюється до EOS або ліміту. Vision encoder не «відповідає» сам: відповідь формує Qwen із уже вирівняних представлень.'''),
c('''model_meta = json.loads((LAB / "evidence/model_metadata.json").read_text(encoding="utf-8"))
cfg = model_meta["model_config"]
architecture = pd.DataFrame([
    ["Vision encoder", cfg["vision_config"]["model_type"], cfg["vision_config"]["num_hidden_layers"], cfg["vision_config"]["hidden_size"]],
    ["Projector", f"MLP + {cfg['projector_hidden_act']}", "—", cfg["text_config"]["hidden_size"]],
    ["Language model", cfg["text_config"]["model_type"], cfg["text_config"]["num_hidden_layers"], cfg["text_config"]["hidden_size"]],
], columns=["компонент", "тип", "шари", "hidden size"])
display(architecture)
print("image_size =", cfg["vision_config"]["image_size"], "| patch_size =", cfg["vision_config"]["patch_size"], "| image_seq_length =", cfg["image_seq_length"])'''),
c('''fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")
boxes = [(0.3, 4.2, 1.8, .9, "Image\\n512×512 RGB"), (2.7, 4.2, 2.1, .9, "SigLIP\\n384×384"),
         (5.4, 4.2, 2.1, .9, "Visual features\\n576 × 1152"), (8.1, 4.2, 1.8, .9, "MLP projector\\n→ 1024"),
         (0.3, 1.2, 1.8, .9, "Question"), (2.7, 1.2, 2.1, .9, "Tokenizer +\\nchat template"),
         (8.1, 1.2, 1.8, .9, "Text embeddings\\n1024"), (10.3, 2.7, 1.5, 1.0, "Qwen2\\n24 layers")]
for x,y,w,h,label in boxes:
    ax.add_patch(plt.Rectangle((x,y),w,h,fill=False,linewidth=1.4,edgecolor="#263746"))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
for a,b in [((2.1,4.65),(2.7,4.65)),((4.8,4.65),(5.4,4.65)),((7.5,4.65),(8.1,4.65)),
            ((2.1,1.65),(2.7,1.65)),((4.8,1.65),(8.1,1.65)),((9.9,4.65),(10.55,3.7)),((9.9,1.65),(10.55,2.7))]:
    ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=1.3,color="#263746"))
ax.annotate("наступний токен → цикл",xy=(11.05,2.65),xytext=(11.05,.45),ha="center",arrowprops=dict(arrowstyle="->",lw=1.2))
ax.set_title("Фактичний multimodal data flow checkpoint LLaVA Interleave Qwen 0.5B")
plt.show()'''),
m('''### Processor, tokenizer і структура повідомлення

`LlavaProcessor` об’єднує `SiglipImageProcessor` та tokenizer. Записаний processor config вимагає resize до 384×384, rescale $x/255$ і normalize $(x-0.5)/0.5$. Наступна комірка показує структуру реального pipeline-входу. Вона не називає ручний resize «викликом processor»: фактичний processor викликається в `src/model.py` через pipeline; тут без ваг перевіряються його зафіксовані параметри.'''),
c('''from labs.lab05_multimodal.src.config import GROUNDING_INSTRUCTION
sample_case = experiment_plan()[0]
prompt = GROUNDING_INSTRUCTION + "\\n\\nQuestion: " + sample_case.question["question"]
messages = [{"role": "user", "content": [
    {"type": "image", "image": "<PIL.Image RGB 512×512>"},
    {"type": "text", "text": prompt},
]}]
display(messages)
processor_cfg = model_meta["image_processor"]
display(pd.Series({"processor": processor_cfg["image_processor_type"],
                   "resize": str(processor_cfg["size"]), "rescale_factor": processor_cfg["rescale_factor"],
                   "mean": processor_cfg["image_mean"], "std": processor_cfg["image_std"]}).to_frame("config"))'''),
m('''## 4. Generation config і inference boundary

Greedy decoding (`do_sample=False`) обирає найімовірніший токен. Sampling випадково вибирає з розподілу після temperature/top-p перетворень; seed фіксує генератор, але не гарантує однакові біти між різними GPU й версіями бібліотек. `max_new_tokens` — верхня межа continuation, не бажана довжина.

Pipeline викликається з `return_full_text=False`. Підклас у `src/model.py` відділяє continuation від input IDs, рахує continuation tokens і перевіряє останній token ID на EOS. `extract_answer` приймає текст або assistant chat payload і відкидає порожню/невідому схему.'''),
c('''plan = experiment_plan()
generation_matrix = pd.DataFrame([{
    "experiment": case.experiment, "case_id": case.case_id,
    **case.generation.__dict__} for case in plan])
display(generation_matrix.drop_duplicates().reset_index(drop=True))
print("19 experiment rows ->", len({row["run_id"] for row in rows}), "physical generations")'''),
m('''## 5. Baseline: відповіді, оцінка, конкретний наслідок

Ручні labels прив’язані одночасно до `run_id`, точного тексту відповіді та SHA зображення. Новий текст не успадковує стару оцінку. Шкали різні для різних типів питання, тому їх не зводимо до одного беззмістовного відсотка.'''),
c('''baseline = results.loc[results.experiment.eq("baseline")].copy()
baseline["висновок"] = baseline.apply(lambda r: r.notes, axis=1)
display(baseline[["question_type", "question", "answer", "evaluation_label", "висновок"]])
factual = baseline.loc[baseline.question_type.eq("factual")]
print("Знаменник фактологічної перевірки:", len(factual))
display(factual.evaluation_label.value_counts().to_frame("кількість"))'''),
m('''Факти 3/3 правильні, але це лише три прості питання до одного кадру. Контроль координат провалено: чотири числа не є обґрунтованою географічною відповіддю. Опис містить непідтверджений матеріал даху; аналітичні відповіді іноді повторюють висновок замість візуального доказу. Візуально правильний клас відповіді тому не гарантує правильного reasoning.'''),
c('''error_taxonomy_figure(results); plt.show()'''),
m('''## 6. Temperature: три контрольовані спостереження

Змінюється лише temperature; `do_sample=True`, `top_p=0.9`, `seed=42`, питання й token budget однакові. Три одиничні генерації не дають оцінки дисперсії або частоти галюцинацій.'''),
c('''temperature = results.loc[results.experiment.eq("temperature"),
    ["temperature", "answer", "generated_token_count", "runtime_seconds", "evaluation_label"]]
display(temperature)
temperature_figure(results); plt.show()'''),
m('''Усі три відповіді семантично узгоджені: вода спокійна, доказом названі відбиття. Формулювання й довжина різняться. Графік часу описує історичний GPU run; ця комірка inference не виконувала, тому час не є швидкістю поточного Run All.'''),
m('''## 7. Token budget: межа проти фактичної довжини

Перевіряємо `generated_token_count`, EOS і умову truncation. Якщо count дорівнює бюджету й EOS немає, відповідь обірвана. Якщо EOS з’явився раніше, невикористаний бюджет нічого не додає.'''),
c('''budget = results.loc[results.experiment.eq("max_tokens"),
    ["max_new_tokens", "generated_token_count", "ended_with_eos", "reached_token_limit", "answer"]]
assert list(budget.generated_token_count) == [24, 48, 48]
assert list(budget.reached_token_limit) == [True, False, False]
display(budget)
token_budget_figure(results); plt.show()'''),
m('''За 24 токени текст закінчується на `The cabin` без EOS: це реальне усічення. За 64 і 128 отримано той самий завершений текст із 48 токенів. Більший бюджет не виправив непідтверджений матеріал даху, бо довжина і factual grounding — різні проблеми.'''),
m('''## 8. Перефразування і два види «однаковості»

`left` та `Left` різняться лише регістром і є семантично однаковими. `center` лексично теж коротка відповідь того самого формату, але суперечить двом іншим відповідям і зображенню.'''),
c('''paraphrase = results.loc[results.experiment.eq("paraphrase"),
    ["case_id", "question", "answer", "evaluation_label", "notes"]].copy()
paraphrase["normalized"] = paraphrase.answer.str.strip().str.casefold()
paraphrase["case_insensitive_match_to_first"] = paraphrase.normalized.eq(paraphrase.normalized.iloc[0])
display(paraphrase)
print("Унікальні нормалізовані відповіді:", paraphrase.normalized.nunique())'''),
m('''## 9. Чому виникають галюцинації

Авторегресивна модель оптимізує правдоподібність наступного токена, а не перевіряє кожне твердження окремим модулем істинності. Grounding instruction зменшує ризик, але не створює гарантії. Помилка може виникнути через слабке візуальне розрізнення, втрату деталей при 384×384, projector alignment, мовні prior або декодування. Контрольна відповідь із числами показує, що впевнена форма не є доказом опори на пікселі.'''),
c('''checks = pd.Series({
    "image SHA matches": file_sha256(image_path) == source["sha256"],
    "all reviews attached": not results.evaluation_label.eq("NOT_REVIEWED").any(),
    "19 matrix rows": len(results) == 19,
    "17 unique generations": results.run_id.nunique() == 17,
    "no empty answers": results.answer.str.strip().ne("").all(),
    "one truncated row": results.drop_duplicates("run_id").reached_token_limit.sum() == 1,
})
display(checks.to_frame("passed")); assert checks.all()'''),
m('''## 10. Висновки та обмеження

Checkpoint правильно відповів на три прості факти, але дав непідтверджені координати, нестійку просторову відповідь після перефразування та слабкі пояснення у двох аналітичних випадках. Temperature sweep не показав погіршення, проте трьох одиничних seed-fixed запусків недостатньо для статистичного висновку. Token budget 24 спричинив truncation; 64 і 128 дали однакові 48 токенів з EOS.

Дослідження обмежене одним синтетичним зображенням, малою кількістю питань і ручною оцінкою однією стороною. Воно придатне як case study механіки LLaVA та provenance, але не як benchmark моделі. Повний GPU inference у цьому Run All не виконувався. Для нового запуску: `python -m labs.lab05_multimodal.src.run`; нові відповіді отримають `NOT_REVIEWED`, доки їх не буде звірено із зображенням.'''),
]
nbf.write(nb, TARGET)
print(TARGET)
