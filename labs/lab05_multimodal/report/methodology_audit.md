# Перевірка відповідності методиці

Перевірено після фактичного GPU-виконання, ручного зіставлення відповідей із зображенням та виконання notebook. PASS означає виконання експериментальної вимоги, а не правильність усіх відповідей моделі.

| Вимога | Статус | Доказ |
|---|---|---|
| Використано LLaVA | PASS | llava-hf/llava-interleave-qwen-0.5b-hf, revision 1090956dd1c79bc93ae98dcf395590369435ec91; outputs/metadata/model.json |
| Hugging Face pipeline/API | PASS | ImageTextToTextPipeline, task image-text-to-text; src/model.py, execution.log |
| VQA зі зображенням і текстом | PASS | RGB + structured chat message; src/inference.py; 17 реальних генерацій |
| Кілька запитань | PASS | 10 baseline-запитань і відповідей; questions.json, manifest.json |
| Описові питання | PASS | desc_01–desc_03 виконано |
| Фактологічні питання | PASS | fact_01–fact_03 виконано; 3 CORRECT |
| Аналітичні питання | PASS | analysis_01–analysis_03 виконано; 1 supported, 2 partly supported |
| Параметри генерації | PASS | temperature 0.2/0.7/1.0 та max_new_tokens 24/64/128; контрольовані матриці |
| Аналіз якості | PASS | report/manual_evaluation.json: 17 ручних оцінок, точні відповіді, image SHA, обґрунтування |
| Аналіз узгодженості | PASS | left/center/Left; семантична стабільність трьох температурних відповідей |
| Приклади у звіті | PASS | report.md, розділи 11–17: дослівні питання/відповіді, параметри та оцінки |
| П’ять контрольних запитань | PASS | defense_notes.md; відповіді підготовлено, усний захист виконує студент |
| Код і результати | PASS | src/, notebook із 15 виконаними code cells, реальне зображення, таблиці та графік |
| Саме GUI-скріншоти | FAIL — ручний крок | GUI-знімки не створювалися; screenshot_checklist.md указує потрібні блоки. Таблиці й графік не названо скріншотами |

## Додаткові вимоги запиту

| Вимога | Статус | Доказ |
|---|---|---|
| Первинний аудит без змін Lab 1–4 | PASS | repository_audit.md, git_status_before.txt; 113 зовнішніх неігнорованих файлів перевірено SHA-256 |
| Реальне середовище | PASS | Python 3.13.5, torch 2.14.0+cu130, Transformers 5.17.0, CUDA 13.0, RTX 3050 4 ГБ; environment.json |
| Збережено CUDA/PyTorch | PASS | Пакети не встановлювали й не оновлювали |
| Вхід скопійовано та перевірено | PASS | assets/input/primary_image.png, source.json; 512×512, SHA-256 збігається |
| Відсутня runtime-залежність Lab 4 | PASS | Власна PNG-копія; походження — metadata, не імпорт |
| Smoke test без зайвого inference | PASS | Перший fact_01 збережено в baseline; execution.log |
| Контроль відсутньої інформації | PASS | control_01 виконано, результат HALLUCINATED; поганий результат не приховано |
| Параметри без змішування змінних | PASS | CPU-тести матриці; seed=42 перед кожним sampling; greedy без temperature |
| Відновлення після переривання | PASS | Атомарний JSON на кожний успішний inference; підпис включає image/model/revision/prompt/config/environment |
| Без зайвих повторів | PASS | 19 рядків експериментів, 17 унікальних run_id; повтор notebook використав кеш |
| Реальна довжина та затримка | PASS | Continuation tokens, EOS, perf_counter + CUDA sync; loading окремо |
| Notebook реально виконано | PASS | 15 code cells, execution_count заданий, 0 error outputs, nbformat validation |
| Ручна оцінка без псевдометрик | PASS | Рубрика окрема для кожного типу; accuracy лише для 3 об’єктивних baseline-фактів |
| Немає абсолютних шляхів у коді/звітах | PASS | Перевірено текстові deliverables; локальні журнали можуть містити шляхи середовища |
| Зміни тільки в Lab 5 | PASS | Усі 113 файлів поза Lab 5 незмінні; нових неігнорованих файлів поза Lab 5 немає |
| Без commit/push та force-add | PASS | Git використано лише для читання/перевірки; outputs залишаються ignored |

## Валідація

| Перевірка | Результат |
|---|---|
| pytest labs/lab05_multimodal/tests -q | PASS: 7 passed in 5.05s; без inference/download |
| python -m compileall -q labs/lab05_multimodal/src | PASS |
| nbformat.validate | PASS |
| Notebook execution | PASS: 15 code cells; немає error outputs |
| Image SHA + повнота manifest + відповідність question IDs | PASS |
| Manual review ↔ run_id ↔ answer ↔ image SHA | PASS |
| git diff --check | PASS; лише інформаційне попередження Windows про LF/CRLF |
| Порівняння файлів поза Lab 5 | PASS: 113/113 SHA-256 збігаються |

Машинний підсумок: outputs/metadata/validation.json. Обчислювальну частину завершено. Для формальної здачі з вимогою GUI-скріншотів залишається ручне створення цих знімків та усний захист.
