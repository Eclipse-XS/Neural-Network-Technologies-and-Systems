# Нотатки до захисту ЛР3

## Що реалізовано

Три CSV-файли містять 185 товарів. `CSVLoader` створює початкові рядкові документи, після чого `documents.py` формує один provenance-preserving `Document` на товар. MiniLM кодує текст у нормалізований 384-вимірний вектор. Redis використовує `FLAT`, `FLOAT32` і cosine distance. Локальна LLM доступна через OpenAI-compatible API LM Studio.

## Що означають метрики

- Hit@5 показує, чи знайдено хоча б один релевантний товар.
- Recall@5 показує частку всіх релевантних товарів у top-5.
- Reciprocal Rank дорівнює оберненому рангу першого релевантного результату; MRR є середнім значенням.

Semantic baseline на семи labeled запитах: Hit@5 `0.8571`, MRR `0.6905`, Recall@5 `0.5734`. Hybrid router: `1.0000`, `1.0000`, `0.8829`. Це retrieval-метрики малого corpus-derived набору, не accuracy відповідей LLM.

## Чому hybrid не є leakage

`answer_query()` і `route_and_retrieve()` отримують текст питання, каталог, optional semantic candidates і `k`. Query ID, expected fields і relevance labels не входять до inference API. Labels застосовуються лише після формування ranked IDs у `evaluate_id_lists()`.

## Ключові обмеження

- Router підтримує вузьку англомовну grammar exact/attribute/numeric запитів.
- Post-rule wording set не є зовнішнім незалежним benchmark.
- MiniLM обрізає 17 довгих записів до 256 word pieces.
- Покращення retrieval не доводить правильність LLM generation.
- Live notebook потребує Redis і LM Studio; offline CI не імітує ці сервіси.

## Типові питання

**Чому один товар є одним chunk?** Щоб не розділяти опис, ціну, характеристики й provenance. Корпус малий, тому character chunking не дає переваги.

**Чому менша cosine distance краща?** Redis повертає distance, а не similarity: для cosine вона монотонно зменшується зі зростанням схожості.

**Чому semantic search не виконує глобальний мінімум?** Найближчі вектори не гарантують охоплення всіх рядків. Агрегації над числами виконує deterministic structured path.

**Що відбувається для невідомого Product ID?** Hybrid path повертає `Insufficient context` і не викликає LLM.
