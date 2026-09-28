"""Build the Ukrainian report and an executable artifact-review notebook."""
import json
from pathlib import Path
import nbformat
from .config import LAB_ROOT, OUTPUTS, INSTRUCTIONS
from .evaluation import evaluated_rows, load_verified


def markdown_table(rows, fields):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |'] +
                     ['| ' + ' | '.join(cell(row.get(f, '')) for f in fields) + ' |' for row in rows])


def build_report():
    rows = evaluated_rows()
    by_id = {r['scenario_id']: r for r in rows}
    environment = json.loads((OUTPUTS / 'metadata/environment.json').read_text())
    initial = json.loads((OUTPUTS / 'metadata/initial_batch.json').read_text())
    all_runs = [json.loads(x) for x in (OUTPUTS / 'runs/runs.jsonl').read_text(encoding='utf-8').splitlines()]
    old = [r for r in all_runs if r['run_id'] in initial['run_ids']]
    fields = ['scenario_id','status','tool_call_count','answer_correctness','tool_behavior','grounding','memory_behavior','planning_behavior','runtime_seconds']
    text = f'''# Лабораторна робота №6

## Тема, мета та варіант

Розробка agent-based асистента. Мета — реалізувати локального LLM-агента з пам'яттю, інструментами та багатокроковими рішеннями. **Варіант 3 — OpenAI Agents SDK.**

Практичний сценарій — аналіз завершених лабораторних звітів. Використано один агент, окремо створюється контрольний екземпляр без інструментів. Це контрольовані запуски однієї архітектури, не мультиагентна система.

## Теоретичні відомості та архітектура

LLM формує текст або запит функції на основі system instructions, user input, історії та JSON-схем інструментів. Runner виконує функцію, додає результат і викликає модель знову. Сама модель Python не виконує. Звичайний виклик LLM не отримує даних локального диска без явної передачі. Агентна програма доповнює модель зовнішніми діями та станом.

```text
Запит → Agent / Runner → локальна LLM у LM Studio
            ├── SQLiteSession → локальна історія
            └── function_tool → list / search / read / calculate
                                    ↓
                        assets/workspace: три звіти
```

Окремого Planner немає. Планування спостерігається через послідовність вибраних дій. Журнал не містить прихованих міркувань. Завершення — фінальна відповідь, помилка, max_turns або timeout.

## Середовище та модель

Python {environment['python'].split()[0]}; openai-agents {environment['packages']['openai-agents']}; openai {environment['packages']['openai']}; pydantic {environment['packages']['pydantic']}; transport httpx2 {environment['packages']['httpx2']}.

Endpoint: `{rows[0]['base_url']}`. Фактично виявлена модель: `{rows[0]['model_id']}`. API — Chat Completions, temperature={rows[0]['temperature']}, tool_choice=auto, max_turns={rows[0]['max_turns']}, max_tokens={rows[0]['max_tokens']}. Початкова відповідь /v1/models також містила embedding-модель nomic; вона не використовується для агента. До першого запиту жодна модель не була завантажена; LM Studio завантажив локальну Ministral на запит. Контекст у lms ps — 8192.

AsyncOpenAI використовує явний loopback base_url та placeholder key. OpenAIChatCompletionsModel отримує саме цей клієнт. Cloud tracing вимкнено через set_tracing_disabled(True), use_for_tracing=False та RunConfig. Проксі середовища і HTTP redirects вимкнені. Хмарні сесії, hosted tools, веб-пошук та shell відсутні. Офіційна документація: [Agents SDK](https://developers.openai.com/api/docs/guides/agents-sdk); фактичні сигнатури встановленої версії збережено в environment.json.

Preflight: прямий локальний запит повернув READY; Agent попросив add_numbers(17,25), Python повернув 42, Runner передав результат моделі й отримав фінальну відповідь. Справжні SDK items збережено в outputs/metadata/tool_preflight.json. Службовий __fake_id__ у Chat Completions adapter є ID адаптера SDK, не ознакою синтетичного запуску.

## Робочий простір і інструменти

Копії report.md Lab 3, 4, 5 байтово ідентичні джерелам, SHA-256 і size_bytes наведені в assets/workspace/manifest.json. Runtime читає тільки копії. Оригінали залишені без змін.

| Інструмент | Аргументи | Результат та обмеження |
|---|---|---|
| list_workspace_files | немає | Тільки три дозволені звіти, імена та розміри |
| search_workspace | query, filename? | Literal case-insensitive пошук, до 8 збігів по 650 символів |
| read_workspace_file | filename, start_line?, end_line? | Нумеровані рядки; до 80 рядків, 6500 символів тексту; next_line |
| calculate | expression | AST для чисел, + - * / і дужок; без eval, імен, викликів чи атрибутів |

Типізовані функції декоровані function_tool; SDK створює схеми. Перевіряються allowlist, absolute paths, traversal і resolved path. Числові вирази обмежені 200 символами, 64 AST-вузлами та модулем результату 1e15. Цілі операції точні; дробова арифметика має обмеження IEEE floating point. Помилки інструментів повертаються коротким структурованим результатом і потрапляють у trace.

## System instructions

```text
{INSTRUCTIONS.strip()}
```

Очікувані факти зі scenarios.json читає лише evaluator. В агент передається виключно user_input; відповіді не вбудовані в інструкції.

## Дизайн експерименту

Початкова серія — 11 запусків. Після виявлених помилок виконано ще {len(rows)} з уточненими загальними інструкціями та відновлюваними помилками інструментів. Додаткові запити про відсоток та явний виклик calculate використано для діагностики пропущеного калькулятора. calculator_probe є керованою пробою й не рахується доказом автономного вибору калькулятора. Початкові результати збережені; їх не замінено успішними. Перша невдала спроба журналювання окремо описана нижче. Підсумкова таблиця стосується другої серії, без змішування різних конфігурацій.

Порівняння baseline/comparison_tools використовує однакові запит, модель, system, temperature, max_tokens і порожню історію; відрізняється тільки доступність функцій. Сесії пам'яті мають нові ID для кожної серії. Латентність виміряно perf_counter навколо Runner; перше завантаження моделі було в preflight, поза основною серією. Кеш LM Studio й довжина контексту впливають на час; один запуск не дає статистичної оцінки продуктивності.

## Підсумкова таблиця

{markdown_table(rows, fields)}

SUCCESS означає завершення Runner, а не автоматично правильну відповідь. CORRECT/PARTIAL/INCORRECT — ручна перевірка тверджень і джерел. NOT_OBJECTIVELY_SCORABLE використовується для стилістичних відповідей. Grounding оцінюється за прочитаними джерелами та цитатами. Автоматичні перевірки наявності рядків/чисел є допоміжними, не семантичною метрикою.

## Багатокрокове виконання

Еталон, отриманий з джерел: Lab 4 — 13 унікальних генерацій (lab04_report.md:41), Lab 5 — 17 унікальних inference, 19 рядків через повторне використання (lab05_report.md:13). Правильна різниця 17−13=4 на користь Lab 5. Еталон не передавався моделі.

Фактична траєкторія multi: {' → '.join(by_id['multi']['tools_used'])}.

{by_id['multi']['final_output']}

Оцінка: {by_id['multi']['planning_behavior']}. {by_id['multi']['notes']}

## Додаткова арифметична перевірка

Траєкторія: {' → '.join(by_id['numeric_extended']['tools_used'])}.

{by_id['numeric_extended']['final_output']}

{by_id['numeric_extended']['notes']}

## Керована діагностика калькулятора

Запит прямо називає calculate, тому цей запуск не є доказом автономного вибору цього інструмента. Дані з файлів не передані у запиті; пошук обирає агент.

Траєкторія: {' → '.join(by_id['calculator_probe']['tools_used'])}.

{by_id['calculator_probe']['final_output']}

{by_id['calculator_probe']['notes']}

## Пам'ять та ізоляція

SQLiteSession зберігає conversation items в outputs/sessions/agent_memory.sqlite. Це контекст діалогу, не семантична довготривала пам'ять. Перед другим запитом Runner отримує попередні items. Перевірено збережений текст alias у A та його відсутність у B. Повні items після серії наведені в metadata/memory_<batch>.json та notebook.

Session A, встановлення: {by_id['memory_set']['final_output']}

Session A, запит моделі: {by_id['memory_recall']['final_output']}

Session B, той самий запит: {by_id['memory_isolated']['final_output']}

Оцінка пам'яті: {by_id['memory_recall']['memory_behavior']}; ізоляції: {by_id['memory_isolated']['memory_behavior']}.

## Без інструментів і з інструментами

{markdown_table([by_id['baseline'],by_id['comparison_tools']], ['agent_variant','answer_correctness','grounding','tool_call_count','runtime_seconds'])}

Без інструментів:

{by_id['baseline']['final_output']}

З інструментами:

{by_id['comparison_tools']['final_output']}

Без інструментів у початковій серії модель відмовилася вигадувати дані; у фінальній серії вона натомість вигадала CSV-файли та числа. Така нестабільність є фактичним результатом. Інструменти дають доступ до джерел і калькулятора, проте сам факт їх виклику не гарантує правильного синтезу. Для запиту про роль вони зайві. Детальні ручні висновки: {by_id['baseline']['notes']} {by_id['comparison_tools']['notes']}

## Невідома інформація

{by_id['unsupported']['final_output']}

{by_id['unsupported']['notes']}

## Ефективність рішень

Запитів з потрібним інструментом: {sum(r['tool_required'] for r in rows)}, з них один baseline навмисно без інструментів. APPROPRIATE: {sum(r['tool_behavior']=='APPROPRIATE' for r in rows)}; UNNECESSARY: {sum(r['tool_behavior']=='UNNECESSARY' for r in rows)}; MISSING: {sum(r['tool_behavior']=='MISSING' for r in rows)}; FAILED: {sum(r['tool_behavior']=='FAILED' for r in rows)}. Калькулятор використано в {sum(r['calculator_used'] for r in rows)} запусках. Повторених ідентичних викликів: {sum(r['repeated_identical_calls'] for r in rows)}. Це опис {len(rows)} контрольованих запусків, не універсальний agent score.

## Фактичні помилки та застрягання

Під час розробки перший run завершив модельну відповідь, але журналювання впало з TypeError: InputTokensDetails не JSON-serializable. Додано serializer для вкладених Pydantic-об'єктів і regression test. Ця спроба не входить у дві повні серії; її trace збережено. Не приховуємо її як успішний експеримент.

Початкова серія:

{markdown_table(old, ['scenario_id','status','tools_used','error_type','runtime_seconds'])}

Початкова memory_recall передала filename="Lab 5", захист відхилив шлях, Runner завершився UserError. Початкові multi та comparison_tools не знайшли загальне число Lab 4, зробили непідтверджений висновок із частини таблиці та пропустили calculate. Початкова lab5 не назвала точний checkpoint і не процитувала файл. Початкова cross додала непідтверджене LLaVA-v1.6 і переплутала роздільність вихідного зображення з латентним простором. Це помилки моделі, не відсутність функцій.

Зміни після цієї серії: правило точних filenames; читання початку звіту при невдалому literal search; заборона виводити загальні кількості з часткової таблиці; перевірка checkpoint, цитат і калькулятора; короткі відновлювані помилки функцій. Значення 13, 17, 4 або checkpoint не додавались в system.

Застрягання визначено як повторення однакових дій без прогресу або MaxTurnsExceeded. Фактичних MaxTurnsExceeded в усіх двох серіях: {sum(r.get('error_type')=='MaxTurnsExceeded' for r in all_runs)}. Timeout і max_turns залишаються захисними межами. Відсутність зациклення в цьому наборі не доводить його неможливість.

## Зауваження до кожної відповіді фінальної серії

''' + '\n\n'.join(f"- **{r['scenario_id']}**: {r['notes']}" for r in rows) + '''

## Обмеження та висновки

Невеликий локальний checkpoint і literal search чутливі до мови формулювання. Бounded читання може відрізати потрібний фрагмент; next_line дає змогу продовжити, але рішення залежить від моделі. Схема auto допускає і пропущені, і зайві виклики. Пам'ять ізольована за ID, однак довга історія може перевищити контекст. Низька temperature не є доказом побітової відтворюваності. Чотири інструменти не мають shell, вебу та прав модифікації звітів; журнали пише runtime.

Робота демонструє справжній SDK-цикл і показує, чому правильне отримання даних слід оцінювати окремо від якості остаточної відповіді. Повні фінальні відповіді, observable trajectories, token usage, історія SQLiteSession і порівняння доступні у виконаному notebook. Формальна відповідність вимогам наведена в methodology_audit.md. Скріншоти GUI залишаються ручним пунктом.
'''
    (LAB_ROOT / 'report/report.md').write_text(text, encoding='utf-8')


def build_notebook():
    md = nbformat.v4.new_markdown_cell; code = nbformat.v4.new_code_cell
    cells = [md('# Лабораторна робота №6\n\nРозробка agent-based асистента. **Варіант 3 — OpenAI Agents SDK.**\n\nМета: дослідити автономні інструменти, пам\'ять та багатокрокове виконання локально.'),
      md('## 1. Теорія та архітектура\n\nLLM отримує system, запит, історію та схеми функцій. Runner виконує запитані функції й повертає результати моделі. SQLiteSession зберігає діалог. Планування проявляється у послідовності дій; окремого Planner немає.\n\nUser → Agent/Runner → LM Studio; Runner ↔ SQLiteSession; Runner → list/search/read/calculate → три локальні звіти.'),
      md('## 2. Режим виконання\n\nЦей notebook **завантажує перевірені артефакти справжніх локальних запусків**, не повторює всі LLM-запити. Код нижче перевіряє hash інструкцій, workspace та batch. Для нової live-серії з кореня: `python -m labs.lab06_agents.src.experiments`. Потім потрібна нова ручна оцінка конкретних run_id; старі оцінки не переносяться автоматично.'),
      code("from pathlib import Path\nimport sys, json, hashlib\nimport pandas as pd\nfrom IPython.display import display, Markdown\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())\nif str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))\nfrom labs.lab06_agents.src.config import LAB_ROOT, OUTPUTS, WORKSPACE, INSTRUCTIONS, Settings\nfrom labs.lab06_agents.src.evaluation import load_verified, evaluated_rows\nfrom labs.lab06_agents.src.tools import build_tools\nruns, batch = load_verified()\nrows = evaluated_rows()\nby_id = {r['scenario_id']: r for r in rows}\nprint('Verified real batch:', batch['batch_id'], 'runs:', len(runs))"),
      md('## 3. Середовище, endpoint і preflight'),
      code("environment = json.loads((OUTPUTS / 'metadata/environment.json').read_text())\nprint(environment['python'])\ndisplay(environment['packages'])\ndisplay(json.loads((OUTPUTS / 'metadata/model.json').read_text()))\npreflight = json.loads((OUTPUTS / 'metadata/tool_preflight.json').read_text())\ndisplay(preflight)\nprint('Actual model:', runs[0]['model_id'], 'API:', runs[0]['api_mode'], 'tracing disabled:', runs[0]['tracing_disabled'])"),
      md('## 4. Робочий простір, SHA-256 та інструменти'),
      code("display(pd.DataFrame(json.loads((WORKSPACE / 'manifest.json').read_text())))\nfor tool in build_tools():\n    print(tool.name, tool.description)\n    display(tool.params_json_schema)"),
      md('## 5. Agent, інструкції та обмеження\n\nRunner.run, tool_choice=auto, max_turns=10. Agent сам обирає дії; notebook не викликає файлові функції замість нього.'),
      code("print(INSTRUCTIONS)\nprint('temperature:', runs[0]['temperature'], 'max_tokens:', runs[0]['max_tokens'])\ndef show_run(scenario_id):\n    r = by_id[scenario_id]\n    display(Markdown('**Запит:** ' + r['user_input']))\n    display(Markdown(r['final_output'] or str(r.get('error_message'))))\n    trace = json.loads((LAB_ROOT / r['trace_path']).read_text(encoding='utf-8'))\n    display(pd.DataFrame([{k: e.get(k) for k in ['step_index','event_type','tool_name','tool_arguments','tool_output_preview','error_type']} for e in trace]))\n    print('Status:', r['status'], 'Runtime:', r['runtime_seconds'], 'seconds')\n    print('Оцінка:', r['answer_correctness'], r['notes'])"),
      md('## 6. Пряма відповідь без інструментів'),code("show_run('role')"),
      md('## 7. Фактологічний пошук'),code("show_run('lab4')\nshow_run('lab5')"),
      md('## 8. Синтез двох файлів'),code("show_run('cross')"),
      md('## 9. Багатокрокова задача\n\nПоказано observable trajectory; приховані міркування не записуються.'),code("show_run('multi')\nshow_run('numeric_extended')\nshow_run('calculator_probe')"),
      md('## 10. Пам\'ять SQLiteSession та ізоляція\n\nПерший запит містить лише псевдонім. Інформацію про модель слід отримати зі звіту.'),
      code("show_run('memory_set')\nshow_run('memory_recall')\nshow_run('memory_isolated')\nmemory = json.loads((LAB_ROOT / batch['memory_path']).read_text(encoding='utf-8'))\nassert memory['A']['session_id'] != memory['B']['session_id']\nassert any(i.get('role') == 'user' and 'For this conversation' in str(i.get('content')) for i in memory['A']['items'])\nassert not any('For this conversation' in str(i.get('content')) for i in memory['B']['items'])\ndisplay(memory)"),
      md('## 11. Невідома інформація'),code("show_run('unsupported')"),
      md('## 12. Без інструментів і з інструментами\n\nУ парі змінено лише доступність функцій. Обидві історії порожні.'),
      code("a, b = by_id['baseline'], by_id['comparison_tools']\nfor key in ['user_input','model_id','temperature','max_tokens','instructions_sha256','session_id']:\n    assert a[key] == b[key]\nshow_run('baseline')\nshow_run('comparison_tools')"),
      md('## 13. Ефективність рішень, підсумкова таблиця'),
      code("columns = ['scenario_id','category','agent_variant','session_id','tool_required','tools_used','tool_call_count','answer_correctness','tool_behavior','grounding','memory_behavior','planning_behavior','runtime_seconds','status']\ndisplay(pd.DataFrame(rows)[columns])\nprint('Tool decisions:', pd.Series([r['tool_behavior'] for r in rows]).value_counts().to_dict())\nprint('Repeated identical calls:', sum(r['repeated_identical_calls'] for r in rows))"),
      md('## 14. Фактичні помилки, застрягання, обмеження та висновки\n\nНижче наведено звіт з аналізом обох серій. SUCCESS не гарантує правильності змісту; оцінки прив\'язані до run_id.'),
      code("display(Markdown((LAB_ROOT / 'report/report.md').read_text(encoding='utf-8')))"),
      md('## 15. Контрольні запитання'),code("display(Markdown((LAB_ROOT / 'report/defense_notes.md').read_text(encoding='utf-8')))"),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.write(notebook, LAB_ROOT / 'notebooks/01_openai_agents_sdk.ipynb')


if __name__ == '__main__':
    build_report()
    build_notebook()
