# Лабораторна робота №5: мультимодальний VQA

Варіант 1 реалізовано через `llava-hf/llava-interleave-qwen-0.5b-hf`, revision
`1090956dd1c79bc93ae98dcf395590369435ec91`. Лабораторна розділяє новий inference,
runtime cache і версіоновані історичні докази. Відкриття notebook не видається за
повторний запуск GPU.

## Що досліджується

Матриця містить 10 baseline-питань, три temperature-умови, три token-budget умови й
три перефразування. Два однакові набори умов повторно використовують baseline, тому
19 експериментальних рядків відповідають 17 фізичним генераціям.

Checkpoint поєднує SigLIP Vision (384×384, 26 шарів, hidden size 1152), MLP projector
і Qwen2 (24 шари, hidden size 1024). Повний зафіксований config доступний у
[`evidence/model_metadata.json`](evidence/model_metadata.json), а notebook пояснює
processor, chat message, visual embeddings, continuation tokens, EOS, greedy decoding
і sampling на конкретних даних.

## Швидка перевірка без GPU

```powershell
py -3.13 -m pytest labs/lab05_multimodal/tests -q
py -3.13 -m compileall -q labs/lab05_multimodal/src
```

### Notebook 01: аналіз історичних результатів

Відкрити [`notebooks/01_llava_vqa.ipynb`](notebooks/01_llava_vqa.ipynb) і виконати
Run All. Notebook читає `evidence/verified_runs.json`, перевіряє image SHA, будує
таблиці й графіки заново. Він не потребує PyTorch, Transformers, ваг або GPU.

### Notebook 02: новий live inference

[`notebooks/02_llava_live_inference.ipynb`](notebooks/02_llava_live_inference.ipynb)
показує весь практичний шлях: CUDA preflight, RGB/provenance, одноразове завантаження
processor і pipeline, chat template, `input_ids`/`pixel_values`, generation config,
прямий виклик `pipe(...)`, continuation tokens, EOS, runtime і окремий unreviewed output.
Він не читає historical evidence і не виконується в CPU CI.

Контрольний live Run All перевірено 24 жовтня 2026 року на RTX 3050 Laptop GPU:
8/8 code cells, відповідь `left`, 2 continuation tokens, EOS, без truncation,
10.529 с. Запис збережено окремо як `NOT_REVIEWED`; historical evidence не змінено.

## Новий inference

```powershell
python -m pip install -r labs/lab05_multimodal/requirements.txt
python -m labs.lab05_multimodal.src.run
```

У Windows/Jupyter вибрати kernel того середовища, де `torch.cuda.is_available()`
повертає `True`, відкрити notebook 02 та виконувати згори вниз. Не перевстановлювати
PyTorch командою з `requirements.txt`: CUDA-збірка має бути встановлена й перевірена
окремо. Зафіксований запуск використовував RTX 3050 Laptop із 4 ГБ VRAM, FP16,
batch size 1 та SDPA. 4 ГБ є перевіреною конфігурацією цього checkpoint, але не
універсальною гарантією: доступна VRAM залежить від інших процесів і версій kernels.

CLI показує етап, режим `INFERENCE`, `CACHE HIT` або `REUSED RESULT`, параметри,
питання, відповідь, token count і EOS. Для cache/reuse історичний час маркується як
історичний, а не як час поточного виконання. Runtime-файли пишуться до `outputs/` і
не версіонуються. Новий текст без точної відповідності `run_id`, answer та image SHA
отримує `NOT_REVIEWED`.

## Фактичні результати

- Три baseline-факти оцінені як CORRECT; знаменник дорівнює трьом, а не всім рядкам.
- Контроль координат дав необґрунтовані числа і позначений HALLUCINATED.
- Temperature 0.2/0.7/1.0 дала семантично узгоджені відповіді 19/23/23 токени; один
  запуск на умову не є статистичним дослідженням.
- Бюджети 24/64/128 дали 24/48/48 токенів. Лише 24 досяг межі без EOS.
- Перефразування дало `left / center / Left`: регістр не змінює зміст, `center` змінює.

## Артефакти

- `evidence/verified_runs.json`: 17 реальних відповідей, параметри, token counts,
  історичний runtime, execution context і provenance.
- `report/manual_evaluation.json`: ручні оцінки, прив’язані до run ID, answer та SHA.
- `src/evidence.py`: сувора schema/image/model validation.
- `src/visualization.py`: графіки тільки з перевірених рядків.
- `src/live.py`: payload, CUDA guard та ізольоване збереження нового `NOT_REVIEWED` run.
- `.github/workflows/lab5-ci.yml`: чиста CPU-перевірка evidence, tests, notebook 01 та HTML.
- `report/report.md`, `methodology_audit.md`, `defense_notes.md`: письмовий звіт,
  аудит методики й тези для захисту.

Вхід: 512×512 RGB PNG, SHA-256
`1fc881e5282f86a05f3abb98201985881a331a5241cd404bf9e2b560b5bb08c5`.
GUI-скріншоти не фабрикуються; перелік ручних знімків наведено в
[`report/screenshot_checklist.md`](report/screenshot_checklist.md).
