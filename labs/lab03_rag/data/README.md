# Lab 3 dataset acquisition

The experiment uses three unmodified files from Anvit Kumar's Shopping Dataset on Kaggle: `dresses.csv`, `earrings.csv`, and `sports-shoes.csv`.

Source: <https://www.kaggle.com/datasets/anvitkumar/shopping-dataset>

The dataset is not redistributed by this repository. Download it under the terms shown on the Kaggle page, extract the archive, and copy the three files to `data/raw/csv/`. Verify them with:

```powershell
Get-FileHash labs/lab03_rag/data/raw/csv/*.csv -Algorithm SHA256
```

Expected hashes and row counts are stored in `manifest.json`. Evaluation stops if a file differs because the ground truth is tied to this exact corpus. No missing value is replaced by zero. Raw prices and discounts remain unchanged; numeric analysis columns are derived at runtime.
