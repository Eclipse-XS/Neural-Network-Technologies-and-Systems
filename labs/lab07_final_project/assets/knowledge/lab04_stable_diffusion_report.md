# Лабораторна робота №4

**Тема:** Генерування зображень за допомогою Stable Diffusion.  
**Мета:** дослідити параметри локального text-to-image inference у контрольованих порівняннях.  
**Варіант:** 1. Модель не навчалась і не донавчалась.

## Теорія та архітектура

Під час навчання forward-процес додає шум до латентів зображень. UNet вчиться передбачати шум за латентом, timestep і текстовою умовою. У text-to-image inference починаємо з шуму без вхідного фото. Текст → tokenizer/CLIP text encoder → embeddings; латентний шум → UNet з cross-attention та scheduler → фінальний латент → VAE decoder → RGB. CLIP не створює пікселі. CFG керує силою умови, а не «точністю»; negative prompt змінює опорну гілку CFG.

## Модель і середовище

Методичка називає `runwayml/stable-diffusion-v1-5`. Фактичний API цього ID перенаправляє на `stable-diffusion-v1-5/stable-diffusion-v1-5`; використано цей еквівалент **Stable Diffusion v1.5**, без зміни сімейства. Revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. Є safetensors FP16 усіх компонентів; ваги збережено тільки у стандартному Hugging Face cache.

Фактичний клас: `StableDiffusionPipeline`. Scheduler: `PNDMScheduler`, його повна конфігурація зафіксована у sidecars. Роздільність усіх PNG — 512×512, batch size 1. Safety checker збережено, жоден із 13 результатів ним не заблокований.

Windows, Python 3.13.5; torch 2.14.0+cu130, diffusers 0.40.0, transformers 5.17.0, accelerate 1.15.0, safetensors 0.8.0, Pillow 12.3.0, numpy 2.5.3, pandas 3.0.6, matplotlib 3.11.2, nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.3.0, pytest 9.1.1. CUDA runtime 13.0; GPU NVIDIA GeForce RTX 3050 Laptop GPU, 4.000 GiB VRAM. RAM 15.19 GiB. До завантаження моделі було 8.65 GB вільного диска; доступна RAM у записаному preflight — 236 MiB.

Стратегія: `torch.float16`, `model_cpu_offload`, `AttnProcessor2_0` (SDPA). Attention slicing, xFormers, VAE slicing/tiling не застосовувалися. OOM не було. Наявний CUDA-enabled torch не замінено. Час завантаження моделі, включно з першим отриманням ваг: 42.37 с; він не входить у час inference.

## Організація контрольованих експериментів

Baseline: seed 42, steps 30, CFG 7.5, negative=None. Positive prompt незмінний для baseline, seed, steps, CFG і negative-пар:

> a futuristic city street at sunset, neon signs, wet pavement reflections, detailed architecture, cinematic lighting, pedestrians in the distance

Negative prompt:

> blurry, low quality, watermark, text, distorted architecture, deformed objects, artifacts

Друга семантична сцена: `a small mountain cabin beside a calm lake, surrounded by pine trees, soft morning light`. До неї додається лише стильовий suffix із config.py. Фіксовані seed 42, steps 30, CFG 7.5, negative=None.

| Серія | Змінний фактор | Значення | Сталі фактори |
|---|---|---|---|
| Seed | seed | 42, 123, 777, 2026 | prompt, negative, steps, CFG, model, scheduler, resolution |
| Steps | num_inference_steps | 20, 30, 50 | prompt, negative, seed, CFG, model, scheduler, resolution |
| CFG | guidance_scale | 5, 7.5, 10 | prompt, negative, seed, steps, model, scheduler, resolution |
| Negative | negative_prompt | None / configured, дві пари | усе інше всередині кожної пари |
| Styles | стильовий suffix | watercolor, isometric, photorealistic | друга сцена, seed, steps, CFG, negative, model, scheduler, resolution |

13 унікальних генерацій: 1 baseline + 3 seeds + 2 steps + 2 CFG + 2 negative + 3 styles. Baseline повторно використано для seed=42, steps=30, CFG=7.5 і negative=None при seed=42; seed=123 — для другої negative-пари. SHA-256 PNG і повна конфігурація перевіряються перед повторним використанням. Параметри кожного запуску збережені одразу, тому переривання не втрачає попередніх результатів.

Ключовий фрагмент реалізації (`src/generation.py`):

```python
generator = torch.Generator(device="cpu").manual_seed(spec.seed)
with torch.inference_mode():
    result = pipe(
        prompt=spec.prompt, negative_prompt=spec.negative_prompt,
        num_inference_steps=spec.num_inference_steps,
        guidance_scale=spec.guidance_scale,
        width=spec.width, height=spec.height, generator=generator,
    )
```

## Базова генерація

![Baseline](../outputs/images/baseline/baseline_seed42_steps30_cfg7p5_b0126393ec9c8f32.png)

Це перший реальний inference і офіційний baseline, а не окреме тестове зображення. Фактичні параметри наведено у таблиці нижче.

## Дослідження seed

![Дослідження seed](../outputs/figures/seed_comparison.png)

Seed змінив композицію і палітру: 42 — відкрита вулиця з рожевим небом та пішоходами справа; 123 — вузький простір між яскравими синьо-червоними вивісками; 777 — низький ракурс із широкою калюжею; 2026 — коридор висотних будівель із центральною групою силуетів. Незмінний prompt не задає єдиного розташування об’єктів.

## Дослідження steps

![Дослідження steps](../outputs/figures/steps_comparison.png)

Усі три зображення зберігають центральну висотну будівлю й рожеве небо. На 20 кроках передній план більше схожий на суцільну відбивну поверхню; на 30 і 50 з’являються виразніші лінії мощення/колій. На 50 змінюються фасад зліва та дрібні об’єкти, але однозначної переваги якості над 30 не видно. Час: 20 — 6,76 с; 30 — 18,88 с; 50 — 13,95 с. 30-крокова baseline була першою генерацією, тому ефект прогрівання й стан пам’яті не дозволяють ранжувати швидкість за цими одиничними вимірами.

## Дослідження CFG

![Дослідження CFG](../outputs/figures/cfg_comparison.png)

За CFG 5 фасад зліва має більше дрібних вікон і присутні автомобілі. За 7,5 центральна башта зберігається, змінюються фасад і мощення. За 10 башти стають гладкими округлими формами, сцена — більш стилізованою, а великі кольорові площини виразнішими. Збільшення CFG змінило також геометрію, тому його не можна трактувати як універсальне покращення або «точність».

## Negative prompt, seed 42

![Negative prompt, seed 42](../outputs/figures/negative_prompt_seed42_comparison.png)

Для seed 42 negative prompt змінив фасади, центральну башту та транспорт: автомобілі на лівій частині вулиці стали помітнішими, відбиття — ширшими. Це не локальне видалення недоліків зі сталої картинки: композиція також змінилась. Силуети людей залишаються нечіткими.

## Negative prompt, seed 123

![Negative prompt, seed 123](../outputs/figures/negative_prompt_seed123_comparison.png)

Для seed 123 з negative prompt видно більше людей та автомобілів; великі вертикальні вивіски й характер відбиттів змінилися. Попри слово text у negative prompt, псевдонаписи залишилися. Воно також частково конфліктує з neon signs у positive prompt; гарантованого вилучення тексту чи всіх артефактів немає.

## Порівняння стилів

![Порівняння стилів](../outputs/figures/styles_comparison.png)

У всіх трьох результатах впізнаються хатина, озеро й хвойні дерева. Watercolor має зернисту фактуру паперу й акварельні заливки; isometric — простішу геометрію хатини та ілюстративні дерева, але не демонструє строгої ізометричної проєкції; photorealistic — дрібнішу фактуру лісу й природніше відбиття. Стильовий suffix змінює і композицію: однаковий seed не фіксує геометрію при іншому тексті.

## Таблиця параметрів генерації

| Case | Seed | Steps | CFG | Negative | Style | Time, s |
|---|---:|---:|---:|---|---|---:|
| baseline | 42 | 30 | 7.5 | None | — | 18.876 |
| seed_123 | 123 | 30 | 7.5 | None | — | 10.177 |
| seed_777 | 777 | 30 | 7.5 | None | — | 9.196 |
| seed_2026 | 2026 | 30 | 7.5 | None | — | 9.678 |
| steps_20 | 42 | 20 | 7.5 | None | — | 6.765 |
| steps_50 | 42 | 50 | 7.5 | None | — | 13.950 |
| cfg_5.0 | 42 | 30 | 5.0 | None | — | 9.560 |
| cfg_10.0 | 42 | 30 | 10.0 | None | — | 9.147 |
| negative_42 | 42 | 30 | 7.5 | configured | — | 9.263 |
| negative_123 | 123 | 30 | 7.5 | configured | — | 9.160 |
| style_watercolor | 42 | 30 | 7.5 | None | watercolor | 9.905 |
| style_isometric | 42 | 30 | 7.5 | None | isometric | 9.474 |
| style_photorealistic | 42 | 30 | 7.5 | None | photorealistic | 9.421 |


Для всіх рядків модель, scheduler, resolution і основні controls наведено вище; точні позитивні/негативні рядки та повні метадані — [generations.csv](../outputs/metadata/generations.csv). CSV порожнє поле negative означає None; JSON зберігає явний null.

## Аналіз часу

![Час](../outputs/figures/runtime_by_steps.png)

Сумарний виміряний inference: 134.57 с. По одному запуску на унікальну конфігурацію. CUDA синхронізовано до/після perf_counter; запис файлів не входить у таймер. 20/30/50: 6.765 / 18.876 / 13.950 с. Baseline містить ефект першого запуску; завантаженість RAM та offload також впливають на wall-clock. Ця серія не є вимірюванням чистого масштабування часу за кількістю кроків. Повторні генерації заради timing не виконувалися.

## Обмеження

Лише один primary prompt, друга сцена для стилів, чотири seeds і якісна оцінка без статистики чи метрик якості. Стилі можуть змінювати композицію разом із фактурою. Написи, анатомія й дрібна геометрія SD 1.5 недосконалі. Збережені seed і конфігурація контролюють випадковість у записаному середовищі, але побітове відтворення між різними GPU/версіями не гарантоване. LoRA розглянуто теоретично; донавчання не виконувалось, оскільки Варіант 1 цього не вимагає.

## Висновки

Локальна SD 1.5 успішно виконала 13 унікальних генерацій. Контрольовані порівняння показали зміни композиції від seed, локальної структури від steps і геометрії від CFG. Negative prompt змінив обидві сцени, але не прибрав усі псевдонаписи. Три стильові формулювання дали різну фактуру й рівень реалізму зі збереженням основних об’єктів другої сцени; строга ізометрія не досягнута. FP16 і model CPU offload забезпечили виконання на 4 ГБ VRAM без OOM. Ці якісні спостереження стосуються лише наведеної малої серії.

## Відповідність методичці

| Вимога | Статус | Доказ |
|---|---|---|
| Середовище | PASS | environment.json; CUDA доступна |
| SD 1.5 | PASS | model_id і revision у кожному sidecar |
| StableDiffusionPipeline | PASS | фактичний pipeline_class |
| Baseline 1–2 | PASS | 1 baseline PNG |
| 3–5 seeds | PASS | 42, 123, 777, 2026; seed_comparison.png |
| Steps 20/30/50 | PASS | steps_comparison.png |
| CFG 5/7.5/10 | PASS | cfg_comparison.png |
| Scheduler comparison | OPTIONAL — NOT EXECUTED | PNDM фіксований для всіх випадків |
| Negative prompt 2–3 приклади | PASS | 2 пари, seeds 42 і 123 |
| Друга сцена | PASS | cabin/lake/pines у стильовій серії |
| Три стилі | PASS | watercolor, isometric, photorealistic |
| Параметри кожного зображення | PASS | 13 JSON sidecars і generations.csv |
| Зрозумілі імена | PASS | seed, steps, CFG у назві PNG |
| Код і результати у звіті | PASS | фрагмент коду, PNG, таблиця нижче |
| Контрольні запитання | PASS | defense_notes.md: усі 10 |


Executed notebook і результати автоматичних перевірок зазначено в README. [Відповіді на всі контрольні запитання](defense_notes.md). [Скріншоти GUI, якщо їх окремо вимагають](screenshot_checklist.md).

## Джерела

Надана «ЛАБ 4.docx», НУ «Львівська політехніка», 2025, Варіант 1. [SD 1.5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [Diffusers memory](https://huggingface.co/docs/diffusers/optimization/memory), [Diffusers reproducibility](https://huggingface.co/docs/diffusers/using-diffusers/reusing_seeds).
