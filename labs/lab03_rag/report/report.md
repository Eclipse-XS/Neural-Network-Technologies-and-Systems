# Звіт до лабораторної роботи №3

## Мета

Побудувати відтворювану RAG-систему над трьома CSV-каталогами, розмежувати indexing, retrieval і generation та окремо оцінити якість пошуку й відповідей локальної LLM.

## Реалізація

185 товарів завантажуються через LangChain `CSVLoader`. Один CSV-рядок стає одним `Document`, оскільки ціна, опис, характеристики та provenance мають залишатися однією логічною одиницею. `all-MiniLM-L6-v2` формує нормалізовані 384-вимірні embeddings. Redis Stack зберігає `FLOAT32` vectors у `FLAT`-індексі та виконує пошук за cosine distance. Grounded prompt передає локальній моделі в LM Studio лише retrieved records і вимагає citations формату `[source.csv:row]`.

Semantic baseline оцінено за Hit@5, MRR і Recall@5. Hit@k фіксує наявність хоча б одного релевантного документа, тоді як Recall@k показує повноту множини. Тому успішний Hit@5 не доводить, що контекст достатній для багатотоварної відповіді. Перевірка generation окремо аналізує поля, citations, completeness і відмову за недостатнього контексту.

## Фактичні результати

На семи labeled запитах semantic baseline отримав Hit@5 = 0.8571, MRR = 0.6905 і mean Recall@5 = 0.5734. Leakage-free hybrid router, який обробляє exact ID, атрибути та числові умови без доступу до relevance labels, отримав відповідно 1.0000, 1.0000 і 0.8829. Малий holdout із п'яти запитів дав Hit@5 = 1.0000, MRR = 1.0000 і Recall@5 = 0.9111. Ці значення характеризують лише поточний малий query set і не є загальною accuracy системи.

Зафіксовані failure modes включають суперечливі price fields, помилку одиниць і citation format, неповний multi-record retrieval, неправильне використання similarity для глобальної агрегації, арифметичну помилку та unsupported claim. Поліпшення retrieval не доводить коректність generation.

## Обмеження і подальша робота

Корпус і relevance labels малі, метрики не мають довірчих інтервалів, latency виміряно одиничними запусками. MiniLM обрізає 17 довгих записів до 256 word pieces. Для наступної версії потрібні human-labeled relevance judgments, ширший holdout, deterministic tools для числових операцій, ручний entailment-аудит, повторні latency runs, версіонування індексу й моніторинг якості citations.

Notebook `notebooks/01_redis_csv_rag.ipynb` є основним виконуваним дослідженням; `artifacts/portfolio/` містить таблиці, графіки та перевірені історичні результати.
