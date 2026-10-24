"""Build the unexecuted GPU practical for Lab 5."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'labs/lab05_multimodal/notebooks/02_llava_live_inference.ipynb'
nb = nbf.v4.new_notebook(metadata={
    'kernelspec': {'display_name': 'Python 3 (CUDA)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.13'},
    'lab5': {'purpose': 'live_gpu_inference', 'execution_status': 'VERIFIED'},
})
m, c = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
m('''# Практикум: live inference LLaVA

**Статус практикуму: LIVE GPU INFERENCE — VERIFIED 24 жовтня 2026 року.**

Контрольний запуск на RTX 3050 Laptop GPU виконав усі 8 Python-комірок без помилок:
відповідь `left`, 2 continuation tokens, EOS `true`, truncation `false`, runtime
10.529 с. Новий запис має `NOT_REVIEWED` і live run ID
`4b58bcb6e6150990dc733c77654de010a97ace5caaaaabdb4c94c08af0c3026f`.
Числа наведено як provenance виконаного контрольного запуску, а не як підставлений
output: новий Run All повторно виконає модель і може створити інший live run.

Цей notebook виконує нову генерацію `llava-hf/llava-interleave-qwen-0.5b-hf`. Він не читає історичні відповіді й не запускається в CPU-only CI. Виконувати послідовно в CUDA-enabled kernel. Ваги завантажуються один раз у кроці 3, один inference виконується у кроці 6. Новий результат записується лише до ignored `outputs/live/`; `evidence/verified_runs.json` не змінюється.'''),
m('''## Крок 1. Середовище до завантаження моделі

FP16 приблизно вдвічі зменшує пам’ять параметрів проти FP32, але VRAM також потрібна для activation і KV cache. Перевірка CUDA до завантаження не дозволяє помилково назвати CPU-виконання GPU inference. Не встановлюйте іншу збірку PyTorch поверх працездатного CUDA-оточення.'''),
c('''from pathlib import Path
import hashlib, importlib.metadata, json, platform, sys, time
import numpy as np
import pandas as pd
from PIL import Image
from IPython.display import display
import torch

ROOT = Path.cwd().resolve()
while not (ROOT / "labs/lab05_multimodal").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if not (ROOT / "labs/lab05_multimodal").exists():
    raise RuntimeError("Запустіть notebook усередині репозиторію")
sys.path.insert(0, str(ROOT))
LAB = ROOT / "labs/lab05_multimodal"

from labs.lab05_multimodal.src.config import MODEL_ID, MODEL_REVISION, GenerationConfig
from labs.lab05_multimodal.src.evidence import file_sha256
from labs.lab05_multimodal.src.inference import extract_answer, image_sha
from labs.lab05_multimodal.src.live import build_multimodal_message, require_cuda, save_live_result
from labs.lab05_multimodal.src.model import load_pipeline

cuda_available = torch.cuda.is_available()
environment = {
    "Python": platform.python_version(),
    "PyTorch": torch.__version__,
    "Transformers": importlib.metadata.version("transformers"),
    "CUDA available": cuda_available,
    "CUDA runtime": torch.version.cuda,
    "GPU": torch.cuda.get_device_name(0) if cuda_available else None,
    "VRAM GiB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if cuda_available else None,
    "precision": "float16" if cuda_available else "unavailable",
    "device": "cuda:0" if cuda_available else "unavailable",
}
display(pd.Series(environment, name="значення").to_frame())
require_cuda(torch)'''),
m('''## Крок 2. RGB-зображення і provenance

PIL зберігає семантику зображення та режим RGB; NumPy-подання має форму `(height, width, channels)`. Це ще не `pixel_values`: processor пізніше виконає resize, rescale і normalize відповідно до SigLIP config.'''),
c('''image_path = LAB / "assets/input/primary_image.png"
source = json.loads(image_path.with_name("source.json").read_text(encoding="utf-8"))
image = Image.open(image_path).convert("RGB")
rgb = np.asarray(image)
sha256 = file_sha256(image_path)
assert sha256 == source["sha256"] == image_sha()
display(pd.Series({
    "PIL size": image.size, "PIL mode": image.mode,
    "NumPy shape": rgb.shape, "NumPy dtype": str(rgb.dtype), "SHA-256": sha256,
}, name="значення").to_frame())
display(image)'''),
m('''## Крок 3. Один processor і одна модель

`image-text-to-text` означає: pipeline приймає зображення разом із текстом і повертає текстове продовження. `load_pipeline()` фіксує revision, FP16/CUDA, SDPA та вимірювальний підклас. Підклас лише додає `generated_token_count` і `ended_with_eos` після стандартного decoding; forward і вибір токенів не змінюються.

Vision encoder SigLIP формує visual features; MLP projector переводить їх у hidden space Qwen2; Qwen2 авторегресивно прогнозує наступні токени.'''),
c('''load_started = time.perf_counter()
pipe, model_info = load_pipeline()  # єдине завантаження моделі в notebook
model_load_seconds = time.perf_counter() - load_started
cfg = pipe.model.config
model_table = pd.DataFrame([
    ["checkpoint", model_info["model_id"]],
    ["revision", model_info["revision"]],
    ["pipeline", model_info["pipeline_base_class"]],
    ["vision encoder", f"{cfg.vision_config.model_type}; {cfg.vision_config.num_hidden_layers} layers"],
    ["projector", f"MLP, {cfg.vision_config.hidden_size} → {cfg.text_config.hidden_size}"],
    ["language backbone", f"{cfg.text_config.model_type}; {cfg.text_config.num_hidden_layers} layers"],
    ["dtype / device", f"{model_info['dtype']} / {model_info['device_strategy']}"],
    ["model load", f"{model_load_seconds:.3f} s"],
], columns=["параметр", "фактичне значення"])
display(model_table)
display(pd.Series({
    "do_sample": pipe.generation_config.do_sample,
    "max_new_tokens": pipe.generation_config.max_new_tokens,
    "eos_token_id": pipe.generation_config.eos_token_id,
    "pad_token_id": pipe.generation_config.pad_token_id,
}, name="generation default").to_frame())'''),
m('''## Крок 4. Реальний multimodal payload і processor outputs

Payload має роль `user` і два content elements: PIL image та текст із grounding instruction і питанням. Chat template додає службові токени діалогу. Processor кодує текст у `input_ids` і зображення у `pixel_values`; повні tensors не друкуються. Image-token positions — позиції-заповнювачі для visual embeddings, а не кількість RGB-пікселів.'''),
c('''question = "Is the cabin on the left, in the center, or on the right side of the image?"
messages, prompt = build_multimodal_message(image, question)
payload_view = [{"role": messages[0]["role"], "content": [
    {"type": "image", "value": f"PIL.Image {image.size} {image.mode}"},
    {"type": "text", "value": prompt},
]}]
display(payload_view)'''),
c('''chat_text = pipe.processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
prepared = pipe.processor(text=chat_text, images=image, return_tensors="pt")
input_ids = prepared["input_ids"]
pixel_values = prepared["pixel_values"]
image_token_positions = int((input_ids == cfg.image_token_index).sum().item())
display(pd.Series({
    "input_ids.shape": tuple(input_ids.shape),
    "text sequence positions": int(input_ids.shape[-1]),
    "image-token positions": image_token_positions,
    "pixel_values.shape": tuple(pixel_values.shape),
    "pixel_values.dtype": str(pixel_values.dtype),
}, name="processor output").to_frame())
del prepared, input_ids, pixel_values  # не тримаємо зайві tensor copies перед generation'''),
m('''## Крок 5. Безпечна базова генерація

Greedy decoding (`do_sample=False`) кожного разу обирає найімовірніший токен; `temperature`, `top_p` і seed тут не застосовуються. `max_new_tokens=64` є верхньою межею continuation. EOS означає природне завершення; досягнення 64 без EOS означало б truncation.'''),
c('''generation = GenerationConfig(do_sample=False, max_new_tokens=64)
generation_kwargs = generation.kwargs()
display(pd.Series({**generation.__dict__, "eos_token_id": pipe.generation_config.eos_token_id},
                  name="значення").to_frame())'''),
m('''## Крок 6. Справжній pipeline inference

Наступна комірка безпосередньо викликає pipeline. Вона не читає JSON і не викликає `run_experiments()`. CUDA synchronization робить вимір часу коректним для асинхронних kernels. `return_full_text=False` залишає assistant continuation без повторення prompt.'''),
c('''torch.cuda.synchronize()
inference_started = time.perf_counter()
raw_output = pipe(
    text=messages,
    return_full_text=False,
    generate_kwargs=generation_kwargs,
)
torch.cuda.synchronize()
runtime_seconds = time.perf_counter() - inference_started

answer = extract_answer(raw_output)
continuation_tokens = raw_output[0].get("generated_token_count")
ended_with_eos = raw_output[0].get("ended_with_eos")
reached_token_limit = continuation_tokens == generation.max_new_tokens and not ended_with_eos

live_record = {
    "question": question,
    "answer": answer,
    "model_id": MODEL_ID,
    "revision": MODEL_REVISION,
    "image_sha256": sha256,
    "generation_config": generation.__dict__,
    "generated_token_count": continuation_tokens,
    "ended_with_eos": ended_with_eos,
    "reached_token_limit": reached_token_limit,
    "runtime_seconds": runtime_seconds,
    "device": model_info["device_strategy"],
    "dtype": model_info["dtype"],
}
live_path = save_live_result(live_record)  # outputs/live/, не evidence/
display(pd.Series({
    "question": question, "answer": answer,
    "continuation tokens": continuation_tokens, "EOS": ended_with_eos,
    "token limit reached": reached_token_limit, "runtime, s": round(runtime_seconds, 3),
    "review status": "NOT_REVIEWED", "saved to": live_path.relative_to(LAB),
}, name="live inference").to_frame())'''),
m('''## Крок 7. Інтерпретація нового результату

Нова відповідь показується поруч із тим самим RGB-входом. Перевірка має спиратися на пікселі: будинок розташований у лівій частині кадру. Проте notebook навмисно не присвоює старий manual label: навіть дослівний збіг потребує окремого provenance-запису, а інший текст може містити нові unsupported details.'''),
c('''import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
ax.imshow(image); ax.axis("off"); ax.set_title("Вхід для нового live inference")
plt.show()
print("Запитання:", question)
print("Нова відповідь:", answer)
print("Статус оцінювання: NOT_REVIEWED")
print("Ручна перевірка: зіставити просторове твердження з будинком у лівій частині кадру;")
print("не переносити labels з evidence/verified_runs.json.")'''),
m('''## Що доведено цим запуском

Якщо всі комірки виконані без помилок, отримано повний шлях RGB → processor → SigLIP features → projector → Qwen2 decoding → assistant continuation. Успішний запуск не доводить загальну надійність моделі: Qwen2 прогнозує правдоподібні токени, а не виконує незалежну перевірку істинності. Unsupported details можливі навіть із grounding instruction. Для temperature, token-budget і paraphrase експериментів використовуйте notebook 01; тут додатковий sweep навмисно відсутній.'''),
]
nbf.write(nb, TARGET)
print(TARGET)
