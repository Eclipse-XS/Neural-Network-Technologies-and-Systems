"""Strict CSV audit and actual LangChain CSVLoader ingestion."""
import csv
import hashlib
from pathlib import Path
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader


def csv_files(directory):
    files = sorted(Path(directory).glob("*.csv"))
    if not 2 <= len(files) <= 3:
        raise ValueError(f"Expected 2–3 CSV files in {directory}; found {len(files)}")
    return files


def inspect_csv(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream, strict=True))
        if not rows or not rows[0] or any(not c.strip() for c in rows[0]):
            raise ValueError("Missing/empty header")
        columns = rows[0]
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate column names")
        if len(rows) < 2 or any(len(row) != len(columns) for row in rows[1:]):
            raise ValueError("Empty corpus or malformed CSV row width")
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if raw.apply(lambda s: s.str.strip().eq("")).all(axis=None):
            raise ValueError("All records are empty")
        inferred = pd.read_csv(path, encoding="utf-8-sig")
    except (ValueError, UnicodeError, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError(f"Unusable CSV {path.name}: {exc}") from exc
    return dict(file=path.name, rows=len(raw), column_count=len(columns), columns=columns,
                missing=raw.apply(lambda s: s.str.strip().eq("").sum()).to_dict(),
                dtypes=inferred.dtypes.astype(str).to_dict(), duplicates=int(raw.duplicated().sum()),
                samples=raw.head(3).to_dict("records"),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def audit_corpus(directory):
    return [inspect_csv(path) for path in csv_files(directory)]


def load_csv_documents(directory):
    """CSVLoader preserves every cell verbatim in metadata, including multiline fields.

    Explicit content_columns lets the raw loader document show the original row;
    downstream serialization does not try to parse that text back into cells.
    """
    documents = []
    for path in csv_files(directory):
        info = inspect_csv(path)
        if {"source", "row"} & set(info["columns"]):
            raise ValueError("CSV columns source/row conflict with CSVLoader provenance")
        documents.extend(CSVLoader(str(path), encoding="utf-8-sig",
                                  metadata_columns=info["columns"],
                                  content_columns=info["columns"]).load())
    return documents
