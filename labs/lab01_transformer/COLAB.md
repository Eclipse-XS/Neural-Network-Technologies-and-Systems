# Запуск Lab 1 у Google Colab

Інструкція розрахована на GPU runtime і запуск із кореня клонованого репозиторію. Фактичний запуск у Colab у межах цього аудиту не виконувався, тому сумісність позначено як **NOT VERIFIED**. Код не містить Windows-залежних шляхів; notebooks визначають корінь лабораторної від поточної директорії.

## 1. Підготовка runtime

У Colab виберіть `Runtime -> Change runtime type -> T4 GPU`, потім виконайте:

```python
!git clone https://github.com/Eclipse-XS/Neural-Network-Technologies-and-Systems.git
%cd Neural-Network-Technologies-and-Systems
!pip install -r labs/lab01_transformer/requirements.txt
```

Перевірка GPU:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## 2. Дані

Перед запуском перевірте наявність файлів:

```python
from pathlib import Path
root = Path("labs/lab01_transformer")
for name in ("spam.csv", "bbc-text.csv"):
    matches = list((root / "data").rglob(name))
    print(name, matches)
```

Якщо CSV не зберігаються в Git, завантажте їх через панель Files у відповідні каталоги під `labs/lab01_transformer/data/`. Не змінюйте назви колонок: loader виконує strict schema validation.

## 3. Запуск notebooks

Відкрийте потрібний notebook через Colab і дотримуйтеся ручних блоків у документі. Не запускайте фінальну test-комірку до завершення вибору моделі за validation. Для демонстрації вже навченої BBC-моделі checkpoint і vocabulary мають бути сумісними з metadata поточного набору даних.

Для тривалого навчання доцільно підключити Google Drive та задати окремий каталог experiment outputs. Не змішуйте його з baseline і не перезаписуйте наявні checkpoints.

Якщо datasets або checkpoints збережено у Google Drive, підключіть Drive і скопіюйте файли у шляхи, яких очікує проєкт:

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount("/content/drive")
repo = Path("/content/Neural-Network-Technologies-and-Systems")
drive_lab = Path("/content/drive/MyDrive/lab01_transformer")

copies = {
    drive_lab / "data/spam.csv": repo / "labs/lab01_transformer/data/raw/spam.csv",
    drive_lab / "data/bbc-text.csv": repo / "labs/lab01_transformer/data/raw/bbc-text.csv",
    drive_lab / "checkpoints/classification_best.pt": repo / "labs/lab01_transformer/outputs/classification/early_stopping/experiments/d128_h4/checkpoints/best.pt",
    drive_lab / "checkpoints/generation_best.pt": repo / "labs/lab01_transformer/outputs/generation/bbc/early_stopping/experiments/d128_h2/checkpoints/best.pt",
}
for source, destination in copies.items():
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
```

Після роботи скопіюйте лише потрібний новий experiment directory назад у Drive. Не копіюйте його поверх baseline:

```python
source = repo / "labs/lab01_transformer/outputs/generation/bbc/diagnostics_v1"
destination = drive_lab / "outputs/diagnostics_v1"
if source.exists():
    shutil.copytree(source, destination, dirs_exist_ok=True)
```

## 4. Перевірки

```python
!python -B -m pytest labs/lab01_transformer/tests -q -p no:cacheprovider
!python -m compileall -q labs/lab01_transformer/src labs/lab01_transformer/tests
```

Скрипт validation-only діагностики декодування:

```python
!python -B labs/lab01_transformer/scripts/run_generation_diagnostics.py
```

Він використовує наявний checkpoint, однакові validation-контексти та окремий каталог `outputs/generation/bbc/diagnostics_v1`. Скрипт не навчає модель і не використовує test split.

Вибір device у notebooks і скриптах автоматичний: CUDA використовується лише коли `torch.cuda.is_available()` повертає `True`, інакше виконується CPU fallback. **COLAB NOT VERIFIED:** ці команди перевірено статично, але реальний Colab runtime у цьому аудиті не запускався.
