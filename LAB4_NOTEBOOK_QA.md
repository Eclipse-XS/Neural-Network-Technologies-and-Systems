# Лабораторна робота №4: Технічний звіт контролю якості (QA Audit & Verification Report)
## Stable Diffusion v1-5 Text-to-Image Inference Pipeline

**Дата аудиту:** 23 жовтня 2026 року  
**Репозиторій:** `Neural-Network-Technologies-and-Systems`  
**Цільовий ноутбук:** [`labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb`](file:///c:/Users/Eclipse/PycharmProjects/neural-network-technologies-and-systems/labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb)  
**Статус верифікації:** **DEFENSE-READY / VERIFIED (21 STEP / 21 CODE CELLS / 0 ERRORS / 24 UNIT TESTS PASSED)**

---

## 1. Середовище виконання та апаратні ресурси (Hardware & Environment Profile)

| Параметр | Значення | Статус перевірки |
| :--- | :--- | :--- |
| **ОС** | Windows 11 Home (Build 26100) | Підтверджено |
| **Python Runtime** | 3.13.5 (CPython, 64-bit) | Підтверджено (`.\.venv-lab04\Scripts\python.exe`) |
| **PyTorch** | `2.14.1+cu130` | Підтверджено (`torch.__version__`) |
| **HuggingFace Diffusers** | `0.40.0` | Підтверджено |
| **HuggingFace Transformers**| `5.17.0` | Підтверджено |
| **Accelerate** | `1.15.0` | Підтверджено |
| **CUDA Driver / Runtime** | CUDA 13.0 (Driver 581.42) | Підтверджено (`torch.cuda.is_available() == True`) |
| **Графічний прискорювач (GPU)**| **NVIDIA GeForce RTX 3050 Laptop GPU** | Підтверджено (`torch.cuda.get_device_name(0)`) |
| **Пам'ять GPU (VRAM)** | 4096 MiB (4.00 GB) | Підтверджено (`get_device_properties(0).total_memory`) |
| **Модель генерації** | `runwayml/stable-diffusion-v1-5` | Підтверджено (Revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`) |
| **Стратегія пам'яті** | `enable_model_cpu_offload()` + FP16 | Підтверджено (пікове VRAM $\approx 2.3-3.2$ GB) |
| **Механізм уваги (Attention)**| PyTorch Scaled Dot Product Attention (SDPA) | Підтверджено (`enable_xformers_memory_efficient_attention()` fallback to SDPA) |

---

## 2. Результати попередньої перевірки (Preflight Verification)

1. **Локальний кеш моделі:** Усі 7 компонентів Stable Diffusion (`tokenizer`, `text_encoder`, `unet`, `vae`, `scheduler`, `feature_extractor`, `safety_checker`) верифіковані в локальному HuggingFace кеші.
2. **Типи даних (dtype):** Весь пайплайн завантажується у `torch.float16`, що знижує споживання VRAM вдвічі порівняно з FP32 без деградації візуальної структури.
3. **Offload-хуки:** Застосування `pipe.enable_model_cpu_offload()` динамічно переміщує активні ваги модулів у VRAM GPU під час відповідної фази інференсу.
4. **Attention Invariants:** PyTorch SDPA оптимізує пам'ять для блоків Cross-Attention до $\mathcal{O}(N)$ замість квадратичної складності $\mathcal{O}(N^2)$.

---

## 3. Архітектурна декомпозиція пайплайну (Architecture Overview)

```
Text Prompt: "a futuristic city street at sunset..."
       │
       ▼
 [ Tokenizer ] ──> Token IDs: [49406, 320, 1374, ..., 49407] (Length: 77)
       │
       ▼
[ CLIP Text Encoder ] ──> Text Embeddings: [1, 77, 768] (Float16)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[ Initial Latent Noise z_T ]               [ Uncond Text Embeddings (Empty / Negative) ]
   Shape: [1, 4, 64, 64]                      Shape: [1, 77, 768]
       │                                         │
       └────────────────────┬────────────────────┘
                            │
               ┌────────────┴─────────────┐
               ▼                          ▼
      [ U-Net Noise Pred ε_cond ]    [ U-Net Noise Pred ε_uncond ]
               │                          │
               └────────────┬─────────────┘
                            ▼
           [ Classifier-Free Guidance (CFG) ]
             ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
                            │
                            ▼
             [ PNDM / Numerical Scheduler Step ]
               z_{t-1} = Update(z_t, ε_guided, t)
                            │  (Повторюється для T кроків)
                            ▼
              [ Final Denoised Latent z_0 ]
                 Shape: [1, 4, 64, 64]
                            │
                            ▼
                    [ VAE Decoder ]
                   x = VAE.decode(z_0 / 0.18215)
                            │
                            ▼
                 [ RGB Image: 512×512×3 ]
```

---

## 4. Структура навчального ноутбука (21-Step Workflow)

Ноутбук `labs/lab04_stable_diffusion/notebooks/01_text_to_image.ipynb` складається з 43 комірок (1 титульна + 21 пара Markdown/Code):

| Крок | Назва кроку | Технічне завдання | Вихідні дані та інваріанти |
| :--- | :--- | :--- | :--- |
| **STEP 01** | Environment and Hardware Preflight | Аудит апаратних ресурсів, версій бібліотек та стану CUDA | Таблиця конфігурації, перевірка RTX 3050 4GB |
| **STEP 02** | Diffusion Fundamentals & Mathematical Formulation | Формулювання прямого ($q$) та зворотного ($p_\theta$) процесів | Математична симуляція forward diffusion (`forward_diffusion_simulation.png`) |
| **STEP 03** | Stable Diffusion Architecture Overview | Декомпозиція компонентів LDM (CLIP, VAE, U-Net, Scheduler) | Академічна діаграма архітектури (`stable_diffusion_architecture.png`) |
| **STEP 04** | Model Loading and Preflight | Завантаження пайплайну SD 1.5 у FP16 з `model_cpu_offload` | Словник `pipeline_context`, перевірка компонентів |
| **STEP 05** | Prompt Tokenization & CLIP Embeddings | BPE-токенізація та проекція в семантичний векторний простір | Токени (довжина 77), shape `[1, 77, 768]`, $\mu \approx 0, \sigma \approx 0.39$ |
| **STEP 06** | Seeded Latent Noise & VAE Downsampling | Вибірка стохастичного шуму $64\times 64\times 4$, 8-кратний просторовий стиск | 4-канальна теплокарта шуму (`latent_channels.png`), $\mu \approx 0, \sigma \approx 1$ |
| **STEP 07** | U-Net Noise Prediction Pass | Аналіз шарів ResNet та Spatial Transformer U-Net | Тензор передбачення шуму shape `[2, 4, 64, 64]` |
| **STEP 08** | Classifier-Free Guidance (CFG) Mechanism | Порівняння $\epsilon_{cond}$ та $\epsilon_{uncond}$, алгебраїчна інваріанта | Перевірка $\epsilon_{cfg} - \epsilon_{uncond} == w \cdot (\epsilon_{cond} - \epsilon_{uncond})$ |
| **STEP 09** | Scheduler and Iterative Denoising | Фіксація латентів на кроках $t \in \{958, 760, 496, 265, 34\}$ | Декодовані зображення траєкторії (`denoising_progression.png`) |
| **STEP 10** | VAE Latent Decoding to RGB Pixel Space | Масштабування $z_0 / 0.18215$ та відновлення у піксельний простір | Тензор $[1, 3, 512, 512]$, квантизація uint8 $[0, 255]$ |
| **STEP 11** | End-to-End Baseline Generation | Еталонна генерація контрольного сюжету (`baseline`, seed=42, cfg=7.5, steps=30) | Зображення $512\times 512$, метадані, SHA-256 |
| **STEP 12** | Controlled Seed Comparison | Оцінка топологічної варіативності (seeds 42, 123, 777, 2026) | Галерея 4 зображень, аналіз розбіжності композиції |
| **STEP 13** | Inference Steps Budget (20 vs 30 vs 50) | Дослідження часового компромісу на обов'язкових кроках методички | Галерея 3 зображень, графік залежності часу від кроків |
| **STEP 14** | Guidance Scale Spectrum (CFG 5.0, 7.5, 10.0) | Дослідження обов'язкового діапазону CFG методички | Галерея 3 зображень, оцінка контрасту та артефактів |
| **STEP 15** | Negative Prompt Dual Guidance | Вплив негативного промпту для двох незалежних seeds (42 та 123) | 2 парні галереї (з/без негативного промпту) |
| **STEP 16** | Style Comparison on Second Scene | Порівняння 3 обов'язкових стилів на гірському будиночку | Галерея 3 зображень: Watercolor, Isometric, Photorealistic |
| **STEP 17** | Scheduler Comparative Analysis | Аналіз чисельних розв'язувачів; статус `NOT_EXECUTED` для Euler/DDIM | Чітке маркування невиконаних планувальників |
| **STEP 18** | 10-Point Experiment Integrity Audit | Повний 10-пунктовий криптографічний аудит маніфесту та sidecar файлів | 10/10 перевірок `PASSED`, статус `PASS` |
| **STEP 19** | Performance & Resource Analytics Dashboard | Трьохпанельний аналітичний дашборд часу та параметрів генерації | Графіки: час vs кроки, CFG спектр, розподіл статусів |
| **STEP 20** | Parameter-Efficient Fine-Tuning (LoRA) Architecture | Математичний розрахунок параметрів адаптації ($W_0 + \frac{\alpha}{r} B A$) | Ранги 4, 8, 16; відсутність тверджень про донавчання моделі |
| **STEP 21** | Defense-Ready Conclusions & Engineering Retrospective | Підсумкові академічні висновки та інженерний ретроспективний аналіз | Підсумок 21 кроку, доказова база виконаної роботи |

---

## 5. Доказова база інспекції внутрішніх компонентів (Internal Inspection Evidence)

Всі параметри зафіксовані в артефакті `labs/lab04_stable_diffusion/artifacts/portfolio/intermediate_inspection.json`:

1. **Токенізація та ембедінги (CLIP):**
   - Промпт: `"a futuristic city street at sunset, neon signs, wet pavement reflections, detailed architecture, cinematic lighting, pedestrians in the distance"`
   - Довжина токенів: 23 інформативні токени + BOS (`49406`) + EOS (`49407`) + 52 токени паддингу = 77 токенів.
   - Розмірність ембедінгів: $[1, 77, 768]$ у форматі FP16.
   - Статистика умовного ембедінгу: $\mu = -0.0075, \sigma = 0.3891$.
   - Статистика безумовного ембедінгу: $\mu = -0.0210, \sigma = 0.3957$.
2. **Початковий латентний шум:**
   - Тензор: $[1, 4, 64, 64]$ ($16\,384$ значень).
   - Статистика: $\mu = -0.0007, \sigma = 0.9995$ ($\mathcal{N}(0, I)$).
   - Артефакт візуалізації: `latent_channels.png`.
3. **Передбачення шуму U-Net та механізм CFG:**
   - Передбачений безумовний шум $\epsilon_\theta(z_t, \varnothing)$: $[1, 4, 64, 64]$, $\mu = 0.0004, \sigma = 0.5401$.
   - Передбачений умовний шум $\epsilon_\theta(z_t, c)$: $[1, 4, 64, 64]$, $\mu = 0.0016, \sigma = 0.5694$.
   - Спрямований шум при $w = 7.5$: $\epsilon_{\text{cfg}} = \epsilon_\varnothing + 7.5 \cdot (\epsilon_c - \epsilon_\varnothing)$, $\mu = 0.0094, \sigma = 0.9912$.
   - Алгебраїчна інваріанта: $\epsilon_{cfg} - \epsilon_{uncond} == w \cdot (\epsilon_{cond} - \epsilon_{uncond})$ програмно підтверджена з точністю $10^{-3}$.
4. **Траєкторія денойзингу:**
   - 5 фіксованих етапів чисельного інтегрування PNDM: $t \in \{958, 760, 496, 265, 34\}$.
   - Збережено у папці `artifacts/portfolio/denoising_steps/` та зведеному зображенні `denoising_progression.png`.
5. **Декодування VAE:**
   - Формула: $x = \mathrm{VAE.decode}(z_0 / 0.18215)$.
   - Розмірність: вхід $[1, 4, 64, 64] \to$ вихід $[1, 3, 512, 512]$, діапазон uint8 $[0, 255]$.
6. **Аналіз параметрів LoRA:**
   - Базовий U-Net SD 1.5: $859\,520\,964$ параметрів.
   - Ранг $r = 4$: 786,432 параметрів (0.0915% від U-Net, $1092.9\times$ скорочення навчальних ваг).
   - Ранг $r = 8$: 1,572,864 параметрів (0.1830% від U-Net, $546.5\times$ скорочення навчальних ваг).
   - Ранг $r = 16$: 3,145,728 параметрів (0.3660% від U-Net, $273.2\times$ скорочення навчальних ваг).

---

## 6. Матриця експериментів (13 Авторитетних Конфігурацій)

| № | Case ID | Експеримент | Seed | Steps | CFG | Сюжет / Промпт | Negative Prompt | Статус виконання | SHA-256 (перші 12 симв.) |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| 1 | `baseline` | Baseline | 42 | 30 | 7.5 | Futuristic city street | Ні | `HISTORICAL_REPLAY` / `LIVE` | `52ab11a9b106...` |
| 2 | `seed_123` | Seed comparison | 123 | 30 | 7.5 | Futuristic city street | Ні | `HISTORICAL_REPLAY` | `ddfd0e7d0259...` |
| 3 | `seed_777` | Seed comparison | 777 | 30 | 7.5 | Futuristic city street | Ні | `HISTORICAL_REPLAY` | `107a3f75ea94...` |
| 4 | `seed_2026`| Seed comparison | 2026 | 30 | 7.5 | Futuristic city street | Ні | `HISTORICAL_REPLAY` | `3451da13499c...` |
| 5 | `steps_20` | Steps comparison | 42 | 20 | 7.5 | Futuristic city street | Ні | `HISTORICAL_REPLAY` / `LIVE` | `f97e01963cff...` |
| 6 | `steps_50` | Steps comparison | 42 | 50 | 7.5 | Futuristic city street | Ні | `HISTORICAL_REPLAY` | `e460e982a21d...` |
| 7 | `cfg_5.0` | CFG comparison | 42 | 30 | 5.0 | Futuristic city street | Ні | `HISTORICAL_REPLAY` | `a4e5cbc393f7...` |
| 8 | `cfg_10.0`| CFG comparison | 42 | 30 | 10.0 | Futuristic city street | Ні | `HISTORICAL_REPLAY` | `fa3fe759fbac...` |
| 9 | `negative_42` | Negative prompt | 42 | 30 | 7.5 | Futuristic city street | Так | `HISTORICAL_REPLAY` | `a32c4c73937f...` |
| 10 | `negative_123`| Negative prompt | 123 | 30 | 7.5 | Futuristic city street | Так | `HISTORICAL_REPLAY` | `5dc98c86d2fc...` |
| 11 | `style_watercolor` | Styles (2nd scene) | 42 | 30 | 7.5 | Mountain cabin, watercolor | Ні | `HISTORICAL_REPLAY` | `a91b4c609eea...` |
| 12 | `style_isometric` | Styles (2nd scene) | 42 | 30 | 7.5 | Mountain cabin, isometric | Ні | `HISTORICAL_REPLAY` | `fff23f8912a3...` |
| 13 | `style_photorealistic` | Styles (2nd scene) | 42 | 30 | 7.5 | Mountain cabin, photorealistic | Ні | `HISTORICAL_REPLAY` | `1fc881e5282f...` |

---

## 7. Результати 10-пунктового аудиту цілісності (10-Point Audit)

```
================================================================================
EXPERIMENT INTEGRITY AUDIT REPORT (10-POINT COMPREHENSIVE VERIFICATION)
================================================================================
[PASS] Point 01: Artifact File Existence: All 13 PNG files exist on disk
[PASS] Point 02: PNG Structure & Decodability: Verified 13 valid PNG decodes via PIL
[PASS] Point 03: Resolution Consistency: Every artifact strictly measures exactly 512x512 pixels
[PASS] Point 04: Metadata Completeness: All 8 critical metadata columns present
[PASS] Point 05: Parameter Boundaries: Parameters strictly conform to controlled experiment matrix
[PASS] Point 06: SHA-256 Cryptographic Integrity: 100% bitwise match across 13 stored digests
[PASS] Point 07: Unique Experiment Identifiers: All 13 case IDs are uniquely identified
[PASS] Point 08: Provenance Tracking: Records carry artifact_id, model revision, and timestamps
[PASS] Point 09: Timing & Numerical Invariants: All wall-clock timings are finite, positive, and non-zero
[PASS] Point 10: One-to-One Artifact Mapping: 13 unique image paths mapped without collisions
--------------------------------------------------------------------------------
OVERALL AUDIT RESULT: PASSED (10 / 10 checks fully verified, 0 anomalies)
================================================================================
```

---

## 8. Результати автоматизованого тестування (Automated Unit Tests)

Запуск команди: `pytest labs/lab04_stable_diffusion/tests -v`

| Назва тесту | Модуль | Тривалість | Результат |
| :--- | :--- | :---: | :---: |
| `test_scheduler_plan_does_not_claim_unrun_results` | `test_evaluation.py` | 0.02s | **PASSED** |
| `test_performance_summary_labels_single_observations` | `test_evaluation.py` | 0.01s | **PASSED** |
| `test_audit_experiment_integrity_passes_on_committed_manifest` | `test_evaluation.py` | 0.04s | **PASSED** |
| `test_one_factor[seed-seed-values0]` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_one_factor[steps-num_inference_steps-values1]` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_one_factor[cfg-guidance_scale-values2]` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_baseline_reuse_and_unique_plan` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_negative_pairs_and_styles` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_invalid_parameters[kwargs0]` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_invalid_parameters[kwargs1]` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_invalid_parameters[kwargs2]` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_invalid_parameters[kwargs3]` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_invalid_parameters[kwargs4]` | `test_experiments.py` | 0.01s | **PASSED** |
| `test_filename_and_complete_identity` | `test_generation_utils.py` | 0.02s | **PASSED** |
| `test_sidecar_roundtrip_and_reuse_requires_integrity` | `test_generation_utils.py` | 0.03s | **PASSED** |
| `test_generate_image_cache_semantics_and_force_regenerate` | `test_generation_utils.py` | 0.02s | **PASSED** |
| `test_executed_notebook` | `test_notebook.py` | 0.12s | **PASSED** |
| `test_experiment_series_are_resolved_and_plotted_programmatically`| `test_notebook.py` | 0.25s | **PASSED** |
| `test_replay_resolver_checks_every_committed_generation` | `test_notebook_helpers.py`| 0.03s | **PASSED** |
| `test_intermediate_inspection_schema_and_contents` | `test_pipeline_inspection.py`| 0.02s | **PASSED** |
| `test_tokenization_and_clip_geometry` | `test_pipeline_inspection.py`| 0.01s | **PASSED** |
| `test_latent_and_unet_cfg_invariants` | `test_pipeline_inspection.py`| 0.01s | **PASSED** |
| `test_denoising_progression_and_vae` | `test_pipeline_inspection.py`| 0.01s | **PASSED** |
| `test_lora_parameter_reduction` | `test_pipeline_inspection.py`| 0.01s | **PASSED** |

**Підсумок тестування:** **24 з 24 тестів пройдено успішно (100% PASS, 5.39s).**

---

## 9. Докази живого виконання інференсу на GPU (Live GPU Inference Proof)

1. **Smoke test на RTX 3050:**
   - Модель: `runwayml/stable-diffusion-v1-5` (revision `451f4fe1...`), FP16, `model_cpu_offload`, SDPA.
   - Час завантаження пайплайну: **7.75 с**.
   - Час генерації 20 кроків: **33.50 с** (включаючи offload-ініціалізацію та синхронізацію CUDA).
   - Згенерований артефакт: `smoke_test_seed42_steps20_cfg7p5_c74f1aa048a725ae.png`.
   - Обчислений SHA-256: `f97e01963cff88727cc527f35f8a5ce180eaffbbc170d6d9616f471bf37f40f1`.
   - **Бітова ідентичність:** Отриманий SHA-256 збігся **байт-у-байт** із зафіксованим у репозиторії хешем артефакту `steps_seed42_steps20_cfg7p5_64dcbe04381d7d04.png`.
2. **Верифікація обходу кешу:**
   - Виклик із `force_regenerate=True` генерує зображення заново (`execution_status=LIVE_GENERATED`, `cache_hit=False`, `generated_in_current_run=True`).
   - Повторний виклик із `force_regenerate=False` миттєво використовує кешований результат (`execution_status=CACHE_REUSED`, `cache_hit=True`, `generated_in_current_run=False`).

---

## 10. Методологічні та наукові обмеження (Limitations)

1. **Одноразові спостереження (Single-run observations):** Кожна точка експерименту представлена однією фіксованою генерацією. Усі таймінги містять варіативність системних процесів та offload-передачі даних і не є усередненим бенчмарком.
2. **Відсутність синтетичних метрик якості (CLIPScore / FID):** Оцінка якості генерації є виключно експертно-візуальною. Жодних штучних чи псевдогенерованих числових оцінок свідомо не фабрикувалося.
3. **Обмеження апаратного бюджету (4 GB VRAM):** Застосування `model_cpu_offload()` забезпечує стабільний інференс без OOM, проте створює накладні витрати CPU-GPU передачі ваг модулів.
4. **LoRA як теоретичний розрахунок:** Вплив LoRA продемонстровано через розрахунок геометрії матриць низького рангу без проведення повного fine-tuning зі зворотним поширенням помилки в середовищі інференсу.
