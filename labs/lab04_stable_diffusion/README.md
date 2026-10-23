# Лабораторна робота №4. Stable Diffusion Text-to-Image

**Варіант 1.** Дослідження Stable Diffusion 1.5 на RTX 3050 Laptop GPU з 4 ГБ VRAM. Збережено 13 генерацій із параметрами, runtime metadata і SHA-256. Модель не навчалась і не донавчалась.

## Архітектура і модель

`Prompt → tokenizer → CLIP text encoder → U-Net + scheduler → latent → VAE decoder → RGB`

Використано `stable-diffusion-v1-5/stable-diffusion-v1-5`, revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`, `StableDiffusionPipeline`, FP16, PNDM і model CPU offload. Роздільність — 512×512, batch size — 1.

![Архітектура Stable Diffusion](artifacts/portfolio/stable_diffusion_architecture.png)

## Експеримент

[Notebook](notebooks/01_text_to_image.ipynb) має 21 крок: математичні основи дифузії, архітектура LDM, завантаження та перевірка пайплайну, інспекція токенізації та CLIP-ембедінгів, початковий латентний шум, прямий прохід U-Net, механізм Classifier-Free Guidance (CFG), траєкторія денойзингу шедулера, VAE-декодування у пікселі, baseline генерація, дослідження seed/steps/CFG, negative prompts, 3 стилі на другому сюжеті, статус шедулера, 10-пунктовий аудит цілісності, продуктивність/дашборд, аналіз параметрів LoRA та академічні висновки.

Контрольовані серії:

- seeds: 42, 123, 777, 2026;
- inference steps: 20, 30, 50;
- CFG: 5, 7,5, 10;
- negative prompt: дві пари для seeds 42 і 123;
- styles: watercolor, isometric, photorealistic;
- scheduler: виконано PNDM; Euler і DDIM не запускались.

Baseline повторно використовується в групах seed/steps/CFG, тому матриця містить 13, а не 18 дубльованих генерацій.

![Порівняння seed](artifacts/portfolio/seed_comparison.png)

![Порівняння steps](artifacts/portfolio/steps_comparison.png)

![Порівняння CFG](artifacts/portfolio/cfg_comparison.png)

Seed змінює композицію за сталих prompt, steps і CFG. Збільшення steps не дало монотонного покращення на трьох зразках. CFG = 10 змінив геометрію та стилізацію, тому guidance scale не можна трактувати як accuracy. Negative prompt змінив denoising trajectory, але не прибрав усі псевдонаписи. Style suffix вплинув і на фактуру, і на композицію.

Runtime і peak allocated CUDA memory записані для кожної генерації. Це одиничні observations, не benchmark: baseline містить warm-up effect, повторних warmed runs немає. CLIPScore і FID не обчислювалися; оцінювання є якісним.

## Запуск

```powershell
python -m venv labs/lab04_stable_diffusion/.venv
labs/lab04_stable_diffusion/.venv/Scripts/pip install -r labs/lab04_stable_diffusion/requirements.txt
labs/lab04_stable_diffusion/.venv/Scripts/python -m jupyter notebook labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb
```

Для повного експерименту потрібні CUDA, model weights і достатньо RAM/disk для offload:

```powershell
$env:PORTFOLIO_MODE='full'
python scripts/execute_portfolio_notebooks.py --mode full
```

Не перевстановлюйте PyTorch без перевірки сумісності CUDA. `generate_image()` повторно використовує результат лише після збігу model/config identity, SHA-256, PNG decoding і dimensions.

CPU-only replay перевіряє 13 committed PNG та metadata, після чого заново будує galleries і performance plots. Він не запускає diffusion inference:

```powershell
python scripts/execute_portfolio_notebooks.py --mode replay
```

## Документація верифікації та аудиту

- [Матриця відповідності методичним вимогам (16 пунктів)](LAB4_METHODOLOGY_COMPLIANCE.md)
- [Технічний звіт контролю якості (QA Audit & Verification Report)](LAB4_NOTEBOOK_QA.md)

## Перевірки й обмеження

```powershell
python -m pytest labs/lab04_stable_diffusion/tests -q
```

Тести перевіряють experiment matrix, cache identity, SHA-256, image dimensions, notebook execution та програмну побудову galleries.

Висновки обмежені SD 1.5, однією роздільністю, одним scheduler і одним зображенням на конфігурацію. Visual review не є статистичною оцінкою загальної якості. Euler, DDIM, repeated timing runs, CLIPScore, FID і LoRA training не виконувались.
