# Аудит перед реалізацією

Переглянуто кореневі README.md, pyproject.toml, .gitignore; структуру, README, src, tests і notebooks Lab 1, Lab 3, Lab 4 та початковий Lab 5. Проєкт має окремі лабораторні, українські пояснення, pathlib-шляхи, reusable logic у src та CPU-тести без ваг. Кореневий pyproject.toml не задає ML-залежностей. Lab 5 містив README-заготовку та .gitkeep.

До роботи існували зміни README Lab 3/4 і невідстежувані файли цих лабораторних. Початковий git status збережено в outputs/metadata/git_status_before.txt; SHA-256 неігнорованих файлів поза Lab 5 — в outside_lab_before.json. Чинна .gitignore виключає outputs, ваги, virtualenv та кеші; її не редагували. Одна PNG-копія у assets дозволена політикою. Runtime не імпортує інших лабораторних.

Preflight підтвердив Python 3.13.5, PyTorch 2.14.0+cu130, Transformers 5.17.0, Accelerate 1.15.0, Pillow 12.3.0, CUDA 13.0 та RTX 3050 Laptop 4 ГБ. Потрібні пакети вже доступні; встановлень не було. Початково диск мав 11 472 891 904 вільних байти. Кеш LLaVA був відсутній; завантажено одну ревізію, потрібні JSON/TXT та model.safetensors (1 728 158 072 байти), без ONNX. Environment перед завантаженням моделі вже показує weights_cached=true, бо download завершився.

Методичку «ЛАБ 5.docx» прочитано як навчальне джерело; підтверджений варіант і обсяг узято з запиту користувача. Варіант 1 дозволяє HF pipeline без Lab 2. Перевірено всі п’ять контрольних запитань, документацію LLaVA/image-text-to-text, фактичний config checkpoint та встановлений pipeline для chat payload, assistant text і continuation count.

Журнал GPU-запуску: outputs/metadata/execution.log. Transformers вивів рекомендацію batching та deprecation щодо generation_config разом із kwargs; у цій версії виклики успішні, batch_size=1 навмисний. Під час міграції версії слід перевірити API. Перша спроба автоматичного виконання notebook передала зайвий kernel_cmd до nbclient; після вилучення цього аргументу використано наявний project kernel. Inference повторно не запускався.
