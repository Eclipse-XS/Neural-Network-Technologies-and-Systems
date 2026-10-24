# Скріншоти для здачі

Виконаний notebook уже містить справжнє зображення, metadata і таблиці відповідей. Звіт містить ті самі результати та побудований за ними графік довжини. Графік є візуалізацією даних, а не скріншотом інтерфейсу.

Live notebook 02 реально виконано на CUDA, але в поточній Codex-сесії немає
доступного Jupyter/PyCharm GUI для створення автентичних знімків інтерфейсу.
Executed notebook і JSON існують локально в ignored `outputs/live/`; це докази
виконання, але не GUI-скріншоти. Якщо потрібні саме GUI-знімки, відкрити виконані
notebooks у PyCharm/Jupyter і зняти такі блоки; пункти нижче ще не означають
створених знімків:

- [ ] Розділ 2: вхідне зображення, RGB metadata та SHA-256.
- [ ] Розділ 3: фактична схема SigLIP → projector → Qwen2 і config-таблиця.
- [ ] Розділ 5: baseline-відповіді, ручні labels і error taxonomy.
- [ ] Розділ 6: temperature 0.2/0.7/1.0, відповіді та графік.
- [ ] Розділ 7: бюджети 24/64/128, EOS/truncation і графік.
- [ ] Розділ 8: left / center / Left з casefold-порівнянням.
- [ ] Фінальна комірка integrity checks без помилок.
- [ ] Notebook 02: CUDA/PyTorch/GPU/VRAM preflight.
- [ ] Notebook 02: processor outputs (`input_ids`, `pixel_values`, image-token positions).
- [ ] Notebook 02: відповідь `left`, 2 continuation tokens, EOS=true, runtime 10.529 с.

Не підміняти ці знімки намальованими вікнами або вигаданим виводом. Усні відповіді викласти за `defense_notes.md`.
