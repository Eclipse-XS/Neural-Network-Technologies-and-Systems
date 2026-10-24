# Аудит відповідності методиці ЛР5

Статус перевірено за кодом і версіонованими доказами, а не прийнято з попереднього
аудиту. PASS означає наявність виконаного етапу, а не безпомилковість моделі.

| Вимога | Статус | Перевірений доказ |
|---|---|---|
| LLaVA через Hugging Face pipeline | PASS | `src/model.py`; model/revision у `evidence/model_metadata.json` |
| VQA для заданого зображення | PASS | 17 унікальних записів із тим самим image SHA |
| Описові, фактологічні, аналітичні питання | PASS | 3+3+3 та 1 контроль у `questions.json` |
| Вплив параметрів | PASS | temperature 0.2/0.7/1.0; budget 24/64/128 |
| Узгодженість | PASS | три перефразування: left/center/Left |
| Аналіз якості | PASS | 17 ручних reviews, прив’язаних до answer і SHA |
| Код, результати, звіт | PASS | `src/`, виконаний notebook, `evidence/`, `report/` |
| GUI-скріншоти | MANUAL | Не створювалися; точний перелік у `screenshot_checklist.md` |

## Перевірені інваріанти

- 19 експериментальних рядків, 17 фізичних генерацій.
- Cache signature включає model ID/revision, image SHA, question, grounding,
  generation config і execution context.
- Пошкоджений, порожній або несумісний runtime cache відкидається.
- Історичні GPU runtimes не називаються часом cache replay чи notebook execution.
- Manual evaluation не переноситься на змінену відповідь або зображення.
- Notebook відновлює аналітику без torch/transformers і не симулює inference.
- Графіки будуються з validated evidence, не з ручно введених чисел.

## Обмеження аудиту

Повтор усіх 17 GPU-генерацій не був технічно необхідним: вихідні JSON містять текст,
generation config, continuation count, EOS, timestamps, environment і точний image SHA.
Новий CPU Run All перевірено окремо. Відсутність GUI-скріншотів лишається формальним
ручним етапом, якщо викладач вимагає саме знімки інтерфейсу.

## Фінальне приймання live notebook і CI

| Перевірка | Статус | Фактичний результат |
|---|---|---|
| Notebook 01 CPU Run All | PASS | 13 code cells, 0 error outputs |
| Notebook 01 HTML export | PASS | `nbconvert` створив непорожній HTML; 5 відрендерених фігур без перекривань |
| Notebook 02 schema і Python syntax | PASS | 17 cells, 8 code cells; `nbformat.validate` та `ast.parse` |
| Notebook 02 real pipeline path | PASS | Явний `pipe(text=messages, return_full_text=False, ...)`, processor inputs, token/EOS/runtime extraction |
| Notebook 02 live GPU execution | NOT_VERIFIED | У середовищі acceptance немає torch, Transformers, CUDA та GPU; outputs відсутні |
| Історичний evidence | PASS | Git object hash `8ad5682aec1e9cf68a2878a778abb8db121770d0`; 17 unique runs / 19 rows |
| CPU pytest | PASS | 14 tests; модель і torch не завантажуються |
| Python compilation | PASS | `src/` і три Lab 5 build scripts |
| GitHub Actions final HEAD | NOT_VERIFIED | Оновити після виконання workflow `Lab 5 CPU CI` |

Notebook 02 зберігає новий результат лише в ignored `outputs/live/` зі статусом
`NOT_REVIEWED` і provenance `live_gpu_inference`. Це не змінює історичні 17 запусків.
Успішна статична перевірка notebook 02 не називається успішним live inference.
