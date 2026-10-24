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
