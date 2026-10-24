# Лабораторна робота №4. Stable Diffusion Text-to-Image

Лабораторна досліджує інференс Stable Diffusion 1.5 на RTX 3050 Laptop GPU з 4 ГБ VRAM. Зафіксовано 13 унікальних генерацій разом із параметрами, часом виконання, використанням CUDA-пам'яті та SHA-256. Навчання моделі й LoRA не виконувались.

## Метод і структура

Потік інференсу: `prompt → tokenizer → CLIP text encoder → U-Net + PNDM scheduler → latent → VAE decoder → RGB`.

- `notebooks/01_text_to_image.ipynb` пояснює теорію, виконує або перевіряє експерименти та інтерпретує результати;
- `src/config.py` і `src/experiments.py` задають незмінні конфігурації та матрицю експериментів;
- `src/pipeline.py` завантажує модель і фіксує середовище;
- `src/generation.py` керує генерацією, кешем і provenance;
- `src/evaluation.py` перевіряє артефакти та формує якісний rubric;
- `src/visualization.py` створює comparison grids;
- `artifacts/portfolio/` містить 13 PNG, manifest, візуальні порівняння й історичні метадані.

Використано checkpoint `stable-diffusion-v1-5/stable-diffusion-v1-5` revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, FP16, model CPU offload, PyTorch SDPA, PNDM і роздільність 512×512.

## Контрольовані серії

- seed: 42, 123, 777, 2026;
- inference steps: 20, 30, 50;
- CFG: 5.0, 7.5, 10.0;
- negative prompt: дві пари для seed 42 і 123;
- style prompt: watercolor, isometric, photorealistic.

Baseline із seed 42, 30 steps і CFG 7.5 повторно використовується в кількох групах, тому план містить 13, а не 18 дубльованих генерацій. PNDM виконано; Euler і DDIM лише позначено як невиконані.

## Запуск

```powershell
py -3.11 -m venv labs/lab04_stable_diffusion/.venv
labs/lab04_stable_diffusion/.venv/Scripts/pip install -r labs/lab04_stable_diffusion/requirements.txt
labs/lab04_stable_diffusion/.venv/Scripts/python -m jupyter notebook labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb
```

`PORTFOLIO_MODE=replay` перевіряє committed PNG, метадані й контрольні суми без завантаження ваг. `PORTFOLIO_MODE=full` запускає реальний GPU inference. Replay не є новою генерацією і позначається `HISTORICAL_REPLAY`.

```powershell
$env:PORTFOLIO_MODE='replay'
labs/lab04_stable_diffusion/.venv/Scripts/python.exe -m pytest labs/lab04_stable_diffusion/tests -q
```

## Результати й обмеження

Seed змінював композицію за сталих інших параметрів. Більша кількість steps не дала монотонного покращення на трьох зразках. CFG змінював відповідність prompt і геометрію, але не є метрикою якості. Negative prompt впливав на всю denoising trajectory і не усував усі небажані деталі. Style suffix змінював і фактуру, і композицію.

Час і peak CUDA memory є одиничними спостереженнями; model loading не входить у generation time, а baseline містить warm-up effect. Це не універсальний benchmark. CLIPScore, FID, repeated warmed runs, альтернативні schedulers і LoRA training не виконувались.
