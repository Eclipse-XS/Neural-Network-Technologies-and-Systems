"""Build readable Ukrainian notebook/report from integrity-checked real runs."""
import json
from pathlib import Path
import nbformat
from .config import LAB_ROOT, OUTPUTS
from .experiments import load_batch
from .evaluation import summarize

ARCHITECTURE = '''```mermaid
flowchart TD
    U[Користувач CLI] --> A[OpenAI Agents SDK Agent / Runner]
    A <--> M[SQLiteSession: історія діалогу]
    A <--> L[LM Studio: локальна LLM / Chat Completions]
    A --> T[Function tools]
    T --> S[search_knowledge]
    T --> C[calculate: AST]
    T --> F[list_knowledge_sources]
    S --> R[LangChain RedisVectorStore]
    R --> E[MiniLM: query embedding на CPU]
    E --> V[Redis Stack: COSINE / FLAT]
    V --> K[Top-k фрагменти + metadata]
    K --> A
```'''


def read_json(relative):
    return json.loads((OUTPUTS/relative).read_text(encoding='utf-8'))


def markdown_table(frame):
    def cell(value):
        return str(value).replace('|','/').replace('\n','<br>')
    return '\n'.join(['| '+' | '.join(map(cell,frame.columns))+' |',
        '| '+' | '.join('---' for _ in frame.columns)+' |']+
        ['| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None)])


def build_notebook():
    md=nbformat.v4.new_markdown_cell;code=nbformat.v4.new_code_cell
    cells=[md('# Лабораторна робота №7\n## Фінальний проєкт\n\n**Варіант 1 — асистент із базою знань (RAG + Agent).**\n\nМета: інтегрувати Redis RAG лабораторної 3 та інструменти й пам’ять Agents SDK лабораторної 6 у локальний застосунок. Notebook показує збережені реальні експерименти, перевіряє їхні хеші та виконує новий пошуковий запит. Для нової повної серії використовуйте `python -m src.experiments --force` із каталогу Lab 7.'),
        code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display\nlab = next(p for p in [Path.cwd(), Path.cwd().parent, Path.cwd()/'labs/lab07_final_project'] if (p/'assets/knowledge/manifest.json').exists())\nsys.path.insert(0, str(lab))\nfrom src.config import Settings, OUTPUTS\nfrom src.knowledge_base import KnowledgeBase, validate_corpus\nfrom src.agent import connect\nfrom src.experiments import load_batch, experiment_fingerprint\nfrom src.evaluation import summarize\nread = lambda name: json.loads((OUTPUTS/name).read_text(encoding='utf-8'))\nrecords = load_batch()\nprint('Реальні збережені результати, SHA-256 перевірено:', len(records))"),
        md('## 1. Концепція застосунку\n\nАсистент відповідає на питання про власні завершені лабораторні. Джерела — чотири незмінені звіти; модель не отримує еталонні відповіді. Користувач працює через CLI з командами `/help`, `/sources`, `/new`, `/session`, `/exit`.'),
        md('## 2. Інтегровані попередні лабораторні\n\n| Компонент | Роль |\n|---|---|\n| Lab 3: LangChain, MiniLM, Redis | Пошук фрагментів звітів |\n| Lab 6: OpenAI Agents SDK, function tools, SQLiteSession | Вибір дій та історія діалогу |\n| LM Studio | Локальна генерація через Chat Completions |'),
        md('## 3. Архітектура\n\n'+ARCHITECTURE+'\n\nПошуковий інструмент повертає фрагменти. Окремої LLM для переказу retrieval немає. Runner може зробити кілька модельних викликів для інструментів; фінальну відповідь формує цей самий агент.'),
        md('## 4. Середовище\n\nНаведено виміряні версії. Пакет `langchain` як метапакет не потрібний: використано `langchain-core`, splitters та окремі інтеграції.'),
        code("environment = read('metadata/environment.json')\ndisplay(pd.DataFrame(environment['packages'].items(), columns=['package','version']))\nprint({k:environment[k] for k in ['python','cuda','gpu','model_id','lm_studio_url']})"),
        md('## 5. База знань\n\nSHA-256 перевіряється за локальними копіями. Оригінальні лабораторні не є runtime-залежностями.'),
        code("display(pd.DataFrame(validate_corpus())[['source_id','workspace_filename','size_bytes','sha256']])"),
        md('## 6. Підготовка документів\n\nMarkdownHeaderTextSplitter зберігає розділи, RecursiveCharacterTextSplitter обмежує фрагмент 900 символами з overlap 120. ID залежить від SHA джерела, розділу, нормалізованого тексту й порядкового номера. Короткі завершені секції збережено; окремий фрагмент має лише 74 символи.'),
        code("chunks = read('metadata/chunks.json')\nprint({k:v for k,v in chunks.items() if k!='samples'})\ndisplay(pd.DataFrame(chunks['samples'])[['source_id','section','text']].head(3))"),
        md('## 7. Embeddings\n\nMiniLM на CPU, нормалізовані FLOAT32-вектори. Розмірність перевірено на реальному embedding. Ліміт MiniLM — 256 токенів; український текст токенізується неекономно, тому частина фрагментів обрізається лише для embedding. Повний фрагмент залишається у Redis та результаті інструмента. Це суттєве обмеження цього експерименту.'),
        code("embedding = read('metadata/embedding.json')\nprint({k:v for k,v in embedding.items() if k!='token_lengths'})"),
        md('## 8. Redis vector index\n\nПовторний запуск має додати нуль документів. Перевіряються справжній FT.INFO, кількість і схема, а не лише наявність маркера.'),
        code("settings = Settings()\nkb = KnowledgeBase(settings)\nindex = kb.initialize()\nprint(index)\nassert index['inserted'] == 0\nclient, settings = await connect(settings)\ntry:\n    assert experiment_fingerprint(settings, kb) == read('runs/batch.json')['fingerprint']\nfinally:\n    await client.close()\nprint('Поточна конфігурація відповідає збереженій серії.')"),
        md('## 9. Перевірка retrieval\n\nSource Hit@5 означає наявність потрібного звіту серед п’яти сусідів. Не доводить, що знайдено потрібний факт. Невідомий запит не входить у знаменник.'),
        code("metrics = read('retrieval/metrics.json')\nprint('Source Hit@5:', metrics['source_hits'], '/', metrics['eligible'])\ndisplay(pd.DataFrame(metrics['queries']))\nlive = kb.search('Which Stable Diffusion model was used in Lab 4?')\ndisplay(pd.DataFrame([c.to_dict() for c in live])[['source_id','section','chunk_id','cosine_distance']])"),
        md('## 10. Agent\n\n`tool_choice=auto`, temperature=0, max_turns=10, max_tokens=900. Три функції: `search_knowledge`, `list_knowledge_sources`, `calculate`. Еталони відокремлені у scenarios.json і не передаються Runner. Cloud tracing вимкнено; клієнт дозволяє лише loopback endpoint.'),
        code("preflight = read('metadata/tool_preflight.json')\nprint(preflight['user_input'], preflight['tools_used'], preflight['final_answer'])\nsummary, comparison = summarize(records)\ndisplay(summary[['scenario_id','variant','tool_call_count','status']])"),
        md('## 11. Структурована пам’ять\n\nSQLiteSession зберігає повідомлення й результати функцій між ходами. RAG зберігає зовнішні документи. Це різні види інформації. Кожний незалежний сценарій та baseline мають новий session ID.'),
        code("display(summary[summary.category.isin(['memory','isolation'])][['scenario_id','session_id','memory','answer_correctness']])"),
        md('## 12. Простий RAG-запит'),
        code("r = next(r for r in records if r['scenario_id']=='lab4' and r['agent_variant']=='assistant')\nprint(r['user_input']); print(r['final_answer']); print('Tools:',r['tools_used'])"),
        md('## 13. Cross-document query'),
        code("r = next(r for r in records if r['scenario_id']=='cross' and r['agent_variant']=='assistant')\nprint(r['final_answer']); print('Sources:',r['sources_retrieved'])"),
        md('## 14. Multi-step scenario\n\nНижче лише спостережувані function calls та відповідь. Правильний калькулятор не гарантує правильності отриманих із retrieval операндів.'),
        code("r = next(r for r in records if r['scenario_id']=='multi' and r['agent_variant']=='assistant')\ntrace = json.loads((lab/r['trace_path']).read_text(encoding='utf-8'))\ndisplay(pd.DataFrame([{'step':e['step'],'tool':e['tool_name'],'arguments':e['arguments']} for e in trace if e['event_type']=='tool_call']))\nprint(r['final_answer'])"),
        md('## 15. Memory scenario\n\nДругий запит не повторює назву лабораторної. Окремо наведено той самий follow-up у свіжій сесії.'),
        code("for r in records:\n    if r['scenario_id'] in ['memory','isolation']:\n        print(r['session_id'], '\\nUser:',r['user_input'],'\\nAssistant:',r['final_answer'],'\\n')"),
        md('## 16. Unsupported information'),
        code("r = next(r for r in records if r['scenario_id']=='unsupported' and r['agent_variant']=='assistant')\nprint(r['user_input']); print(r['final_answer'])"),
        md('## 17. Простий LLM vs RAG + Agent\n\nЧотири однакові запитання, одна модель, system instructions, temperature та token limit, порожня історія. Відрізняється доступ до інструментів. Пряме звертання інструкції до неіснуючих tools у baseline може сприяти вигаданим викликам; це обмеження контрольованого дизайну, а не привід зараховувати їх як справжні.'),
        code("display(comparison[['scenario_id','plain_correctness','assistant_correctness','tools_used','plain_runtime','assistant_runtime']])\nwith pd.option_context('display.max_colwidth', 400):\n    display(comparison[['scenario_id','plain_answer','assistant_answer']])"),
        md('## 18. Зведення експериментів\n\nSUCCESS — лише завершення Runner. Оцінки фактів і підтримки тверджень отримано ручним переглядом відповідей та відповідних retrieval traces, не LLM-суддею.'),
        code("display(summary[['scenario_id','variant','answer_correctness','grounding','retrieval','tool_behavior','citation','memory','planning','runtime_seconds']])"),
        md('## 19. Помилки та обмеження\n\nАнгломовний MiniLM на українських звітах; обрізання embedding-входів; top-k не гарантує потрібного розділу. Lab 6 цитує хибні відповіді попереднього агента — їх не можна приймати за факти. Валідний chunk ID не доводить, що твердження випливає з chunk. Один запуск на питання не оцінює стабільність.'),
        code("display(summary[['scenario_id','variant','notes']])"),
        md('## 20. Висновки\n\nСистема реально об’єднує локальну LLM, Redis retrieval та агентні функції, але доступ до джерел не усуває хибного синтезу. Остаточні кількості нижче отримано зі збереженої серії.'),
        code("assistant = summary[summary.variant=='assistant']\nprint('Завершені запуски:', int((summary.status=='SUCCESS').sum()), '/', len(summary))\nprint('Оцінки асистента:', assistant.answer_correctness.value_counts().to_dict())\nprint('Ідентифікатори цитат:', assistant.citation.value_counts().to_dict())\nprint('Порівнянь із baseline:', len(comparison))")]
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python (Lab 7)','language':'python','name':'lab07'},'language_info':{'name':'python'}})
    path=LAB_ROOT/'notebooks/01_final_project.ipynb'
    nbformat.write(nb,path)
    return path


def build_report():
    records=load_batch();summary,comparison=summarize(records)
    env=read_json('metadata/environment.json');index=read_json('metadata/index.json')
    embedding=read_json('metadata/embedding.json');metrics=read_json('retrieval/metrics.json')
    def result(sid):
        return next(r for r in records if r['scenario_id']==sid and r['agent_variant']=='assistant')
    multi=result('multi')
    sections=[
        ('Тема','Фінальний проєкт: Neural Network Technologies Coursework Knowledge Assistant.'),
        ('Мета','Об’єднати пошук локальних знань, агентні функції та структуровану пам’ять. Перевірити користь інтеграції контрольованим порівнянням із тією самою LLM без доступу до корпусу.'),
        ('Варіант','Варіант 1 — асистент із доступом до бази знань (RAG + Agent). Методичка «ЛАБ 7.docx», НУ «Львівська політехніка», 2025. ПІБ і група не надані; їх слід додати під час оформлення титульної сторінки.'),
        ('Постановка задачі','Інтерактивно відповідати на текстові запити, знаходити докази у векторному індексі, цитувати фрагменти, виконувати арифметику та зберігати контекст діалогу. Runtime не імпортує попередні лабораторні й не має інструмента прямого читання цілих файлів.'),
        ('Практичний сценарій','Студент уточнює модель, кількість генерацій і спостережені недоліки у власних лабораторних звітах. Наступне питання може посилатися на попередній предмет без повторення його назви.'),
        ('Інтегровані компоненти попередніх лабораторних','Lab 3: LangChain, all-MiniLM-L6-v2 і RedisVectorStore. Lab 6: OpenAI Agents SDK, function_tool і SQLiteSession. Перенесено підходи в окремі модулі Lab 7; прямі runtime-імпорти попередніх лабораторних відсутні. Локальний inference забезпечує LM Studio.'),
        ('Архітектура системи',ARCHITECTURE+'\n\nІнструмент повертає первинні фрагменти. Немає другого LLM, який спершу генерує RAG-відповідь для переказу агентом.'),
        ('Локальна LLM та LM Studio',f"Виявлено `{env['model_id']}` на `{env['lm_studio_url']}`. OpenAIChatCompletionsModel + AsyncOpenAI; temperature=0, max_tokens=900, max_turns=10. Вимкнено tracing, проксі середовища та redirects. Placeholder key — lm-studio. Preflight перевірив звичайну відповідь та справжній calculate із поверненням результату моделі. Python {env['python'].split()[0]}, openai {env['packages']['openai']}, openai-agents {env['packages']['openai-agents']}, langchain-core {env['packages']['langchain-core']}, langchain-redis {env['packages']['langchain-redis']}. Метапакет langchain не потрібний.\n\nRAG-пакети встановлено в окрему .venv Lab 7, яка бачить вже наявний torch із кореневого середовища через .pth. Конфлікт hf-xet вирішено локально версією 1.5.2; батьківські пакети не змінено."),
        ('Формування бази знань','Чотири canonical report.md лабораторних 3, 4, 5, 6 скопійовано без зміни байтів у assets/knowledge. Manifest містить source_id, lab_number, title, workspace_filename, original_repository_path, SHA-256 та size_bytes. Перед індексацією перевіряються хеші; в початковому аудиті також перевірено рівність оригіналам. Посилання на картинки в копіях — частина незміненого тексту, зображення не завантажуються та не індексуються.'),
        ('Підготовка та chunking документів',f"MarkdownHeaderTextSplitter зберігає розділи; RecursiveCharacterTextSplitter: 900 символів, overlap 120 у великих розділах. {index['chunk_count']} фрагментів: Lab 3 — 32, Lab 4 — 22, Lab 5 — 29, Lab 6 — 38. Мінімум 74, медіана 572, максимум 898 символів. Одну коротку завершену секцію збережено. ID — SHA від хешу джерела, секції, нормалізованого тексту та ordinal. Metadata містить джерело, filename, lab_number, section, chunk_id, source_sha256. Реальні приклади — metadata/chunks.json."),
        ('Формування embeddings',f"Текст → `{embedding['model']}`, revision `{embedding['revision']}` → {embedding['dimension']} координати → unit normalization → FLOAT32. Обчислення на CPU, GPU залишено LM Studio. Dimension виміряно, норми й скінченність перевірено. Ліміт encoder — {embedding['max_seq_length']} токенів: **{embedding['truncated_chunks']} із {index['chunk_count']} фрагментів обрізаються при embedding**. Повний текст залишається в Redis. Це реальна втрата coverage, а не лише попередження бібліотеки; для наступної версії доцільні tokenizer-aware splitting та мультимовний encoder з окремим контрольованим експериментом."),
        ('Redis Vector Store та індексація',f"Контейнер lab07-redis, redis://localhost:6380, Redis {env['redis_server']}. `{index['index_name']}`, prefix lab07:kb:, HASH, COSINE, FLAT, FLOAT32, dim={index['embedding_dimension']}. FT.INFO підтвердив {index['indexed_count']} записів і відсутність indexing failures. Fingerprint охоплює звіти, embedding revision, chunking та schema. Незмінний індекс повторно використовується, відсутні ключі відновлюються. Перебудова обмежена префіксом Lab 7; FLUSHALL/FLUSHDB відсутні. Lab 3 на порту 6379 збережено."),
        ('RAG retrieval','Query embedding → Redis KNN → top-k Document + cosine distance → структурований tool result. Менша distance означає ближчого сусіда; це не ймовірність правильності. k=5 за замовчуванням, дозволено 1–8. Необов’язковий source_id — TAG filter, який обирає модель. Cutoff не застосовано, оскільки він не відкалібрований. Навіть нерелевантний запит має сусідів, тому агент повинен перевіряти текстову підтримку.'),
        ('Agent architecture','Один Agent і Runner, окремий контрольний екземпляр тієї самої конфігурації без tools. System instructions відокремлюють conversational memory від evidence, вимагають пошуку звітних фактів, exact citations та арифметичного інструмента. Фрагменти звітів трактуються як дані, включно з цитованими помилками попередніх моделей.'),
        ('Function tools','search_knowledge повертає source, section, chunk_id, citation, distance та text. list_knowledge_sources повертає лише каталог джерел. calculate використовує AST для чисел, + − × / і дужок; не дозволяє імена, calls, attributes, exponentiation чи shell. Ліміт 200 символів, 64 AST-вузли, модуль числа до 1e15. Цілочислові операції точні; дробові мають стандартне floating-point округлення.'),
        ('Структурована пам’ять','SQLiteSession зберігає conversation items у outputs/sessions/assistant_memory.sqlite. Ідентифікатор визначає діалог; /new створює новий ID, не видаляючи старий. Структурований журнал retrieval і tool messages є історією, а не окремою базою знань. Offline test повторно відкриває сесію та перевіряє ізоляцію.'),
        ('Логіка прийняття рішення агентом','LLM отримує інструкції, текст користувача, історію та JSON-схеми. Вона повертає фінальний текст або function call. Runner виконує Python-функцію, передає результат і знову запитує ту саму LLM. Немає if scenario → scripted actions. max_turns=10, HTTP timeout 180 с, весь run обмежено 600 с. Trace містить лише input, function arguments/results, final answer та помилки.'),
        ('Інтерактивний застосунок','app.py перевіряє endpoints, корпус та індекс до циклу введення. Команди /help, /sources, /new, /session, /exit. --session відновлює історію, --debug показує імена інструментів, час та перевірку ID. Реальний транскрипт: outputs/runs/cli_demo.txt.'),
        ('Експериментальні сценарії','10 категоризованих сценаріїв, один має два ходи: 11 assistant runs. Для 4 запитів додатково виконано plain baseline: 15 основних запусків загалом. Preflight і CLI не включені до цього знаменника. Всі результати збережено одразу; очікувані факти доступні лише evaluator. У цій серії поведінкові невдачі не перезапускались заради кращого результату.'),
        ('Retrieval evaluation',f"8 запитів; Source Hit@5 = **{metrics['source_hits']}/{metrics['eligible']}**. Для unsupported denominator не визначений. q06 не знайшов Lab 6. q05 знаходить Lab 5, але не потрібний контрольний факт: source-hit переоцінює повноту доказів. Повна таблиця: outputs/retrieval/retrieval_results.csv.\n\n"+markdown_table(__import__('pandas').DataFrame(metrics['queries'])[['query_id','hit_at_k','first_expected_rank','runtime_ms']])),
        ('Cross-document queries',result('cross')['final_answer']+'\n\nОцінка і пояснення наведені у підсумковій таблиці нижче; факт двох пошуків сам по собі не гарантує правильного порівняння.'),
        ('Multi-step execution','Фактична послідовність: **'+' → '.join(multi['tools_used'])+'**.\n\n'+multi['final_answer']+'\n\nЕталон з оригіналів: Lab 4 — 13, Lab 5 — 17 унікальних inference, різниця 4. 19 у Lab 5 означає рядки експерименту, не унікальні генерації. Ці значення не передавались у system prompt.'),
        ('Memory experiment','\n\n'.join('**'+r['user_input']+'**\n\n'+r['final_answer'] for r in records if r['scenario_id'] in ['memory','isolation'])+'\n\nДва memory-turns мають один session_id; isolation має інший. Структурні items збережено у metadata/memory_*.json.'),
        ('Unsupported-information experiment',result('unsupported')['final_answer']+'\n\nЦіль: не вигадати outdoor temperature. Наявність retrieved сусідів не означає наявність цього факту.'),
        ('Простий LLM vs RAG + Agent','Одна модель, temperature=0, max_tokens=900, те саме питання та system instructions; для кожного порівняння порожня історія. Єдина керована різниця — доступність tools і відповідних даних. System усе ще згадує tools у baseline: це зберігає контроль, але може підштовхувати модель симулювати виклики; симуляції не зараховано як tool calls.\n\n'+markdown_table(comparison[['scenario_id','plain_correctness','assistant_correctness','plain_runtime','assistant_runtime']])+'\n\nПовні тексти обох відповідей, джерела, інструменти, grounding і hallucination збережено у baseline_comparison.csv та виконаному notebook.'),
        ('Аналіз правильності та grounding','Ручна оцінка кожного твердження за збереженими chunks і корпусом. Автоматично перевіряється лише членство citation ID у retrieval поточного run, включно з неправильно оформленими посиланнями baseline. PASS для ID не доводить entailment.\n\n'+markdown_table(summary[['scenario_id','variant','answer_correctness','grounding','tool_behavior','citation','memory','planning']])+'\n\n'+markdown_table(summary[['scenario_id','variant','notes']])),
        ('Помилки та обмеження','MiniLM слабко переносить англомовні запити на український корпус; 76 embedding-входів обрізано. Retrieval може знайти звіт, але пропустити потрібний розділ. У Lab 6 є цитовані помилки попереднього агента, що збільшує ризик контамінації відповіді. Локальна 3B-модель інколи ігнорує цитування та додає непідтверджені причинні пояснення. Перший offline тест tool schema використовував застарілий RunContextWrapper; його виправлено на фактичний ToolContext SDK. Жодну відповідь експерименту не відредаговано. Один run на питання та provider cache не дають оцінки статистичної стабільності чи production latency.'),
        ('Практичні переваги інтегрованої системи','Асистент отримує доступ до приватних звітів, залишає перевірюваний шлях до фрагментів і виконує справжні детерміновані дії. Plain LLM такого доступу не має. Перевага полягає у доступності доказів і контрольованих функцій; якість їх використання потрібно оцінювати окремо. Retrieval додає latency та інфраструктуру й не гарантує правильності.'),
        ('Висновки',f"Source retrieval знайшов потрібний звіт у {metrics['source_hits']} із {metrics['eligible']} оцінюваних запитів. Усі підсумкові оцінки отримано з реальних відповідей: "+str(summary[summary.variant=='assistant'].answer_correctness.value_counts().to_dict())+'. Агент сам обирав retrieval; роль пояснив без tools. Пам’ять, ізоляція, багатокрокова арифметика й чотири baseline-пари перевірені в окремих сценаріях вище. Реалізація інтеграції завершена, але твердження про безпомилковий асистент не підтверджене.')]
    text='# Лабораторна робота №7. Фінальний проєкт\n\n'+'\n\n'.join(f'## {i}. {title}\n\n{body}' for i,(title,body) in enumerate(sections,1))+'\n'
    (LAB_ROOT/'report/report.md').write_text(text,encoding='utf-8')


if __name__=='__main__':
    build_report()
    print(build_notebook())
