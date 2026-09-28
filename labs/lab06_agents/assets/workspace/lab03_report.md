# Лабораторна робота №3. Реалізація RAG-системи для витягування знань

Варіант 3: Redis Vector Similarity + CSV. Звіт за фактичним локальним запуском 22.09.2026. ПІБ, група: [заповнити]. GUI-скриншоти: [додати за checklist нижче].

## 1. Мета

Реалізувати повний RAG-процес для табличного каталогу, перевірити retrieval та відтворення полів локальною LLM. Відокремити якість пошуку від правильності генерації.

## 2. Теоретичні відомості про RAG

RAG знаходить зовнішні записи й передає їх генератору як контекст. Параметри LLM не змінюються. Наявність джерела дозволяє перевірити відповідь, але не робить її автоматично істинною.

## 3. Семантичні embeddings

Embedding — числове представлення змісту. Одна модель кодує питання й документи; близькість векторів наближено відображає релевантність. Точні числові умови та ID не гарантовано зберігають порядок у цьому просторі.

## 4. Redis Vector Similarity

Запущено один Docker-сервіс redis/redis-stack:7.4.0-v8. Реальний Redis Search індексує HASH-записи та виконує vector KNN, LangChain RedisVectorStore є клієнтською інтеграцією. RedisInsight доступний на localhost:8001; GUI-скриншот ще не підготовлено.

## 5. CSV corpus

Kaggle shopping-dataset: dresses.csv — 100, sports-shoes.csv — 51, earrings.csv — 34. Разом 185 рядків, 24 спільні колонки, 0 точних дублікатів та 0 повторів product_id. Raw файли побайтово збережені, SHA-256 перевірено. Повний набір із 98 CSV лишається поза індексованим каталогом. Докладні типи, пропуски й приклади — у [dataset_audit.md](../dataset_audit.md).

Виявлено 29 розбіжностей формули ціни/знижки понад 1 INR, 22 товари з rating=0 і ratings_count=0. Missing discount означає невідоме значення. При оцінюванні використовуємо зафіксовану final_price, не вигадуємо виправлену ціну.

## 6. CSVLoader

load_csv_documents використовує LangChain CSVLoader для кожного файла. content_columns/metadata_columns зберігають поля, source і row. Pandas застосовано лише для аудиту, таблиць та offline oracle. Двокрапки й багаторядкові клітинки не розбираються через split тексту.

## 7. Row-to-text representation

serialize_record формує підписані title, product_description, category, initial_price, final_price, currency, discount, rating, ratings_count; додає вибрані specification_name/value і product_details. Числа в тексті зберігаються буквально, JSON декодується, HTML-сутності розкриваються. URLs/службові поля не embedded. У metadata збережено source_file, row_id, record_id, product_id, category, похідні числа та fields_json з вихідними клітинками.

## 8. Chunking strategy

Один вихідний товар — один chunk, 185 документів. Row ID нульовий, без заголовка CSV; це не номер фізичного рядка багаторядкового файла. Багаторядкове питання отримує декілька документів через top-k. Batch-of-rows опційний, не виконувався.

## 9. all-MiniLM-L6-v2

Фактично завантажено sentence-transformers/all-MiniLM-L6-v2, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41. CPU, normalize_embeddings=True, 384 dimensions. Sanity check: (3,384), усі значення скінченні, одиничні норми. Індекс містить 185 векторів. Кеш залежить від тексту/моделі/налаштувань.

17 із 185 текстів перевищують 256 word pieces і обрізаються при embedding. Це виміряно embedding_audit, не приховано. Основні поля розміщено на початку, повний документ лишається в Redis та LLM-контексті. Втрата пізніх деталей може знижувати recall.

## 10. Redis index

FT.INFO підтвердив індекс lab03_ec7829ffcaca539f98c4: 185 documents, HASH, FLAT, FLOAT32, dim=384, COSINE. Перевірено TAG поля джерел/категорій та NUMERIC поля row_id, цін, знижки, rating/count. Повторний notebook запуск додав 0 записів; індекс не дублюється. Namespace має fingerprint корпусу й конфігурації, жодного FLUSHDB немає.

## 11. Cosine similarity

cos(a,b)=(a·b)/(||a|| ||b||). Повернений Redis score — distance=1−cos, отже менше краще. Нормалізовані вектори роблять dot product еквівалентним cosine similarity, але цей експеримент явно використовує COSINE. Dot product: NOT EXECUTED (опційний; не твердження про відсутність IP у Redis).

## 12. Retriever

LangChain as_retriever(search_type="similarity") продемонстровано на earrings із TAG filter. Повний експеримент використовує similarity_search_with_score для показу cosine distance і provenance. k=5 — компактний контекст для кількох товарів; tuning k не проводився.

## 13. Evaluation queries

Вісім фіксованих запитів з queries.json, сформованих до першого пошуку. Еталони виведені з джерел. q02/q03/q07 мають прозорі predicate/proxy labels, не повну незалежну людську розмітку. q05 має повнокорпусний numeric oracle, який не надсилається retriever чи LLM.

| ID | Тип | Еталонна умова |
|---|---|---|
| q01 | direct | Exact product ID |
| q02 | semantic | Cotton + Maxi + Casual specification fields; check labels below |
| q03 | filter | Earrings with Plating=Silver-Plated |
| q04 | comparison | Both named products required |
| q05 | multi_condition | Full CSV numeric oracle: category=sports-shoes, rating>=4, discount>0, minimum recorded final_price; oracle never enters retrieval |
| q06 | cross_category | Both named products from different CSVs required |
| q07 | semantic | Walking occurs in source product_description; proxy labels, not exhaustive human relevance |
| q08 | unsupported | No destination-specific date or shipping fee in selected records; must abstain |

## 14. LM Studio integration

На /v1/models перевірено ID mistralai/ministral-3-3b; /api/v0/models показав loaded. ChatOpenAI звертається до http://localhost:1234/v1, temperature=0, max_tokens=800, timeout=120 s, retries=0. Python не завантажував ваги LLM. ID задається LM_STUDIO_MODEL. Модель nomic, присутня в списку сервера, не використовується для embeddings.

## 15. RAG pipeline

Question → MiniLM query embedding → Redis top-5 (+ category filter для q03–q05) → JSON-контекст з provenance → system/user messages → LM Studio answer. Prompt вимагає точних полів, цитат [file.csv:row], відмови за недостатнього контексту й обмеження висновків retrieved records. Усі отримані рядки та промпти доступні в outputs.

## 16. Фактичні результати retrieval

Hit@5 = 0.857143 (6/7); MRR = 0.690476. q08 не входить у знаменник IR-метрик. Середній час одного виклику в notebook 5.578 ms (кеш query embeddings уже прогрітий попереднім smoke run). Перший smoke run мав приблизно 21–28 ms із обчисленням query embeddings. Це не benchmark чистого Redis, масштабування або production latency.

| Запит | Hit@5 | Перший релевантний ранг | Recall@5 | Час, ms |
|---|---:|---:|---:|---:|
| q01 | 1.0000 | 1.0000 | 1.0000 | 5.610 |
| q02 | 1.0000 | 3.0000 | 0.6667 | 8.901 |
| q03 | 1.0000 | 2.0000 | 0.2222 | 4.655 |
| q04 | 1.0000 | 1.0000 | 0.5000 | 4.871 |
| q05 | 0.0000 | n/a | 0.0000 | 4.702 |
| q06 | 1.0000 | 1.0000 | 1.0000 | 5.093 |
| q07 | 1.0000 | 1.0000 | 0.6250 | 5.425 |
| q08 | n/a | n/a | n/a | 5.365 |

q04 має Hit=1, але retrieved лише один із двох потрібних товарів (Recall=0.5). q05 не retrieved product_id 11960808 із final_price=659 INR, що є повнокорпусним мінімумом за заданими умовами. q06 retrieved обидві сутності з різних CSV. Невдачі не відфільтровано.

## 17. Відповіді та ручна перевірка

Отримано 8/8 відповідей без transport errors. Це не означає 8/8 правильних відповідей. Нижче ручний аналіз усіх восьми; повний текст збережено у notebook та outputs/rag/final_answers.csv.

| Запит | Аналіз |
|---|---|
| q01 | Помилка пояснення: ціна 1999 і discount 40% суперечать одне одному; LLM назвала їх узгодженими. Основні числа відтворено. |
| q02 | Підібрано два доречні товари, але порожній discount помилково названо No discount. Третій релевантний товар не retrieved. |
| q03 | Дві доречні позиції; discount 77 записано як суму ₹77, а не відсоток. Цитати мають нестандартний формат і не проходять строгий parser. |
| q04 | Коректна відмова порівняти: COVER STORY відсутній у top-5. Ціна Trendyol правильна. Citation format не відповідає вимозі. |
| q05 | Глобальний oracle 11960808 за 659 INR не retrieved. Названо Puma за 2474 серед кандидатів; твердження No global minimum exists некоректне. Не пояснено конфлікт Duke; citation format невалідний. |
| q06 | 10/10 очікуваних полів присутні й обидві цитати валідні, але арифметика помилкова: 1480*(1-0.76)=355.20, не 352.00. Різниця до raw 355 — лише 0.20 INR, тому висновок про суттєву суперечність хибний. |
| q07 | Названі material/sole material і ціни трьох товарів узгоджені з джерелами. Додано непідтверджене контекстом узагальнення breathable/lightweight; FAUSTO має raw price-discount conflict. |
| q08 | Коректна відмова Insufficient context: відсутні дані про доставку до Kyiv завтра й точну вартість. |

## 18. Field citation verification

Автоматична перевірка розділяє boundary-aware lexical value match, наявність очікуваного citation, наявність потрібного товару у retrieved context і невідомі citations. Вона не перевіряє entailment чи правильність арифметики. Формат citations перевіряється суворо, альтернативний формат може містити людське посилання, але не задовольняти контракт.

| Запит | Знайдених очікуваних полів | Усього перевірюваних полів | Валідних citations |
|---|---:|---:|---:|
| q01 | 4 | 5 | 1 |
| q02 | n/a | 0 | 2 |
| q03 | n/a | 0 | 0 |
| q04 | 5 | 10 | 0 |
| q05 | 1 | 5 | 0 |
| q06 | 10 | 10 | 2 |
| q07 | n/a | 0 | 3 |
| q08 | n/a | 0 | 0 |

Для q02/q03/q07/q08 автоматичних expected-field цілей немає: це n/a, не 100% правильності. q05 має випадковий збіг категорії попри відсутність очікуваного товару. q06 має 10/10 збігів і правильні citations, але хибний арифметичний висновок. Тому загальний lexical match rate не називається accuracy RAG.

## 19. Переваги Redis

Виконано реальний векторний пошук, TAG category filtering та відновлення provenance з metadata. Для малого корпусу FLAT простий і дає точних найближчих сусідів у векторному просторі. Сервер окремий від notebook, а persistent volume зберігає індекс. Висновків про великі корпуси, Redis проти інших БД або production SLA цей запуск не дає.

## 20. Обмеження

Окремий Docker/Search сервіс; залежності інтеграцій; RAM; відсутність гарантії SQL-агрегації; слабкість embeddings для точних ID/чисел; 17 обрізаних текстів; помилки й пропуски джерела; неповні labels; малий query set; похибки LLM у математиці та трактуванні missing. Температура 0 не гарантує побітову відтворюваність LLM.

## 21. Висновки

Повний Варіант 3 реалізовано й виконано. Корпус перетворено через CSVLoader, 185 embeddings внесено в Redis COSINE, контекст використано локальною моделлю. Пошук знайшов хоча б один еталон для 6 із 7 оцінюваних запитів, проте провалив глобальний ціновий oracle та неповністю забезпечив одне порівняння. LLM коректно відмовила за нестачі даних у q04/q08, але також допустила фактичні й арифметичні помилки. Grounding корисний для перевірюваності, але не замінює валідацію.

## Відповідність офіційним крокам Варіанта 3

| Методичка | Реалізація / доказ |
|---|---|
| 2–3 CSV, CSVLoader, конкатенація полів | data.py, documents.py; notebook 3–5; 185 фактичних документів |
| Рядкові chunks та all-MiniLM-L6-v2 | documents.py, embeddings.py; notebook 6–7; 384 dimensions |
| Redis Stack, LangChain RedisVectorStore, cosine | docker-compose.yml, redis_store.py; FT.INFO, 185 vectors |
| Retriever і кілька багаторядкових запитів | retrieval.py, queries.json; q04/q05/q06/q07 |
| Локальна LLM, контекст із Redis, цитування полів | llm.py, rag.py, evaluation.py; 8 відповідей, автоматична й ручна перевірки |
| Висновки про швидкість, filters, точність | розділи 16–21; збережено невдачі |

## Checklist скриншотів (ручний крок)

- [ ] CSV corpus і таблиці schema/missing у notebook, розділ 3.
- [ ] Приклад row-to-text і provenance, розділ 5.
- [ ] Docker Desktop із running/healthy Redis Stack.
- [ ] RedisInsight: індекс і 185 документів (якщо GUI доступний).
- [ ] q01: retrieved rows і cosine distances.
- [ ] q04 або q05: багаторядкова задача та неповне retrieval.
- [ ] LM Studio: loaded model і running server.
- [ ] Повна відповідь q06 разом із retrieved sources; не приховувати арифметичну помилку.
- [ ] Retrieval summary і field/citation verification table.

## NOT EXECUTED / межі перевірки

GUI-скриншоти не створено. Dot product/IP, row batches, no-context baseline, tuning k, інші LLM, масштабний latency benchmark і Python 3.11 не перевірялися. Вони не потрібні для виконаного baseline Варіанта 3. CUDA не переналаштовувалася. Фактичний Python 3.13.5, torch 2.14.0+cu130. Повний notebook виконано (19 первинних code cells); додатковий перегляд збережених відповідей не викликає генерацію повторно.

## Джерела

- Надана методичка «ЛАБ 3.docx», розділи Варіант 3 і Контрольні запитання.
- [Kaggle shopping-dataset](https://www.kaggle.com/datasets/anvitkumar/shopping-dataset).
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): 384 dimensions, 256 word pieces.
- [LangChain RedisVectorStore API](https://reference.langchain.com/python/langchain-redis/vectorstores/RedisVectorStore).
- [Redis Stack image](https://hub.docker.com/r/redis/redis-stack/tags).

## Перевірки реалізації

- `python -m pytest labs/lab03_rag/tests -q`: **22 passed, 1 warning in 34.35s** у середовищі Lab 3.
- Warning — оголошений sunset langchain-community; потрібний CSVLoader фактично працює. Warning не приховувався.
- `compileall` для src/tests, notebook JSON/nbformat/Python syntax і `git diff --check`: успішно.
- Notebook має 37 комірок, із них 20 виконаних code cells без error outputs. Дев’ятнадцять виконано послідовно, двадцята читає збережені відповіді з ручним аналізом без повторної генерації.
- Docker Compose config і PowerShell AST setup.ps1 валідні; setup.ps1 як цілісний сценарій не запускався, його кроки створення середовища й установлення залежностей виконано окремо.
- SHA-256 для 233 попередньо зафіксованих файлів Lab 1, Lab 2 та Lab 3 data: жодної зміни.
- Повторний build_store: inserted=0, document_count=185.
- Git ігнорує CSV, outputs та локальне .venv. Усі зміни обмежені labs/lab03_rag. Комітів і push немає.
