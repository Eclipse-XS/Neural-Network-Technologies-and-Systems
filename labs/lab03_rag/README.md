# Лабораторна робота №3. Реалізація RAG-системи для витягування знань

**Варіант 3: Redis Vector Similarity + CSV.**
CSV → CSVLoader → текст рядка → MiniLM (CPU) → LangChain RedisVectorStore →
Redis retrieval → grounded prompt → локальна LLM у LM Studio → перевірка полів.
Інші лабораторні та CUDA не змінюються.

## Корпус і структура

`data/raw/csv/`: `dresses.csv` (100), `sports-shoes.csv` (51), `earrings.csv` (34).
Разом 185 товарів, 24 спільні колонки. Повний набір із 98 CSV у батьківському data/
не входить до індексу. [Повний аудит](dataset_audit.md).
Дані взято з [Kaggle](https://www.kaggle.com/datasets/anvitkumar/shopping-dataset).
Для іншого checkout потрібні ці три оригінальні CSV; queries.json перевіряє їхні SHA-256.
Дані, кеші та outputs ігноруються Git.

```text
lab03_rag/
  README.md, dataset_audit.md, requirements.txt, setup.ps1
  docker-compose.yml, queries.json
  data/raw/csv/                  # три локальні CSV
  notebooks/01_redis_csv_rag.ipynb
  src/                          # config, data, documents, embeddings,
                                # redis_store, retrieval, llm, rag, evaluation
  tests/                        # офлайн unit/schema tests
  report/report.md
  report/defense_notes.md
  outputs/                      # результати, embeddings, QA
```

## Windows / PyCharm

Команди виконуються з кореня репозиторію. Python ≥3.11; фактично перевірено **3.13.5**.
Python 3.11 окремо не тестувався. Окреме середовище Lab 3 використовує вже встановлений
у репозиторії torch через локальний .pth. setup.ps1 не оновлює CUDA та пакети батьківського
середовища, фіксує його torch як constraint для встановлення.

```powershell
.\labs\lab03_rag\setup.ps1 -Python .\.venv\Scripts\python.exe
```

У PyCharm виберіть `labs/lab03_rag/.venv/Scripts/python.exe` для notebook.
Якщо потрібен іменований Jupyter kernel, зареєструйте вручну:

```powershell
.\labs\lab03_rag\.venv\Scripts\python.exe -m ipykernel install --user --name lab03 --display-name "Python (Lab 3)"
```

Прямі версії інтеграцій зафіксовано в requirements.txt; це не повний transitive lock.
Використано langchain-community 0.4.2 (CSVLoader), langchain-redis 0.2.6,
langchain-huggingface 1.2.2, langchain-openai 1.6.3, sentence-transformers 6.1.0.
Пакет community видає sunset warning, але містить потрібний методичкою CSVLoader.
Redis і HuggingFace використовують окремі інтеграції. Жодної LLM у Python/PyTorch не завантажуємо.

## Redis Stack

Запустіть Docker Desktop (Linux containers):

```powershell
docker compose -f labs/lab03_rag/docker-compose.yml up -d
docker compose -f labs/lab03_rag/docker-compose.yml ps
docker compose -f labs/lab03_rag/docker-compose.yml exec redis redis-cli FT._LIST
```

Один сервіс `redis/redis-stack:7.4.0-v8`; Redis на `127.0.0.1:6379`,
RedisInsight на [localhost:8001](http://localhost:8001). Named volume зберігає дані.
У контейнерному RedisInsight підключення до Redis: host 127.0.0.1, port 6379.
WSL доступний із distro docker-desktop; ще один Redis у WSL не потрібний.
Код перевіряє PING і FT._LIST: звичайного Redis без Search недостатньо.

Індекс `lab03_<fingerprint>`: FLAT, COSINE, FLOAT32, 384 виміри, TAG/NUMERIC metadata.
Назва залежить від корпусу, серіалізації, embedding revision та схеми.
Повторний запуск додає лише відсутні ключі. FLUSHDB/видалення чужих даних не виконується.
Після зміни корпусу старі індекси залишаються для ручного керування.

## LM Studio

Завантажте локальну chat-модель у LM Studio, запустіть Developer → Local Server.
Використовуйте вже наявну модель; велике завантаження не потрібне.

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
$env:LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
$env:LM_STUDIO_MODEL = "mistralai/ministral-3-3b"
```

Наведений ID фактично був завантажений у цьому середовищі. На іншій машині виберіть
власний chat-model ID зі списку. Код не має hardcoded model ID. Задайте ці змінні також
у конфігурації PyCharm/Jupyter та перезапустіть kernel. Опційна LM_STUDIO_API_KEY.
REDIS_URL типово `redis://localhost:6379`.

## Notebook і експеримент

Виконуйте `notebooks/01_redis_csv_rag.ipynb` згори вниз. Комірки короткі; логіка у src/.
Перший запуск MiniLM без кешу потребує `local_files_only=False` у комірці embeddings;
далі використовуйте True. Revision зафіксовано в config.py. Вектори кешуються за текстом
і налаштуваннями. Запуск генерації надсилає реальні HTTP-запити до LM Studio.

Вісім фіксованих queries, k=5: точний/семантичний пошук, category filter, порівняння,
числові умови, різні CSV, insufficient context. Score — **cosine distance**, менше краще.
Hit@5/MRR рахуються лише для семи запитів із мітками. Recall@5 показує coverage,
особливо коли потрібні два товари. Виміряний час включає query embedding, мережу й парсинг;
це не чиста latency Redis. Numeric oracle використовується тільки для оцінювання.

Результати:

- `outputs/retrieval/results.csv`: кожен отриманий рядок, rank, distance, provenance, preview, metadata.
- `outputs/retrieval/summary.csv`, `metrics.json`: усі запити й retrieval metrics.
- `outputs/rag/final_answers.csv`: відповіді або помилки, очікувані поля та перевірки.
- `outputs/rag/q*_prompt.json`: фактичні промпти.
- `outputs/embedding_audit.json`, `index_info.json`, `environment.json`: аудит поточного запуску.

Повторний запуск перезаписує відповідні outputs. Для нового порівняння задайте
окремий `Settings(output_dir=...)` і збережіть той самий queries.json.

## Обмеження

Raw дані мають 29 неузгоджень ціни зі знижкою понад 1 INR та 22 записи без оцінок.
Missing discount не дорівнює нулю. Значення не виправляємо; prompt вимагає називати суперечності.
17 із 185 текстів довші за 256 word pieces MiniLM і обрізаються при embedding;
повний текст лишається в Redis для генерації. Не всі характеристики беруть участь у пошуку.

Vector top-k не гарантує глобальний мінімум. Hit=1 не доводить отримання всіх потрібних рядків.
Перевірка полів — лексичні збіги й валідність посилань; атрибуцію й висновок перевіряємо вручну.
Dot product, row batches і baseline без контексту не входять до обов’язкового baseline.

## Перевірки та звіт

```powershell
.\labs\lab03_rag\.venv\Scripts\python.exe -m pytest labs/lab03_rag/tests -q
.\labs\lab03_rag\.venv\Scripts\python.exe -m compileall -q labs/lab03_rag/src labs/lab03_rag/tests
git diff --check
```

Тести не потребують сервісів, Hugging Face download або CSV-корпусу. Тестові fixtures
явно штучні й не використовуються як експериментальні дані.
[Звіт і checklist скриншотів](report/report.md),
[10 офіційних контрольних запитань і пояснення реалізації](report/defense_notes.md).
GUI-скриншоти залишаються ручним кроком; їх не підмінено згенерованими зображеннями.

Документація: [RedisVectorStore](https://reference.langchain.com/python/langchain-redis/vectorstores/RedisVectorStore),
[Redis Stack image](https://hub.docker.com/r/redis/redis-stack/tags),
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
