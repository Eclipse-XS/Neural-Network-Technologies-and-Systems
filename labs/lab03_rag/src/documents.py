"""One complete record per document; raw CSV cells are never rewritten."""
import html
import json
import re
from pathlib import Path
from langchain_core.documents import Document

TEXT_FIELDS = ("title", "product_description", "category", "initial_price", "final_price",
               "currency", "discount", "rating", "ratings_count")
SPEC_FIELDS = ("Sport", "Material", "Fabric", "Occasion", "Type", "Length", "Shape",
               "Base Metal", "Plating", "Cushioning", "Technology", "Sole Material",
               "Neck", "Pattern", "Closure", "Fastening")


def number(value):
    """Derived number only. Missing is None, never a made-up zero."""
    if value is None or not str(value).strip():
        return None
    value = str(value).strip().strip('"').replace("₹", "").replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        raise ValueError(f"Unsupported numeric format: {value!r}")
    return float(value)


def serialize_record(row, fields=TEXT_FIELDS, spec_fields=SPEC_FIELDS):
    fields = tuple(fields)
    if not any(str(row.get(key, "")).strip() for key in fields):
        raise ValueError("No configured textual fields in record")
    lines = ["Product record:"]
    for key in fields:
        if key in row:
            lines.append(f"{key}: {row[key] if str(row[key]).strip() else 'not provided'}")
    if row.get("product_specifications"):
        specs = json.loads(row["product_specifications"])
        for spec in specs:
            key, value = spec.get("specification_name"), spec.get("specification_value")
            if key in spec_fields and value and value != "NA":
                lines.append(f"{key}: {html.unescape(str(value))}")
    if row.get("product_details"):
        details = json.loads(row["product_details"])
        # Full detail is retained for generation; token audit exposes embedding truncation.
        for key in ("description", "material_and_care", "size_and_fit"):
            if details.get(key):
                lines.append(f"{key}: {html.unescape(str(details[key]))}")
    return "\n".join(lines)


def build_documents(loaded, fields=TEXT_FIELDS):
    result = []
    for doc in loaded:
        row = {k: v for k, v in doc.metadata.items() if k not in {"source", "row"}}
        source = Path(doc.metadata["source"]).name
        row_id = int(doc.metadata["row"])
        metadata = dict(source_file=source, row_id=row_id,
                        record_id=f"{source}:{row_id}", product_id=str(row.get("product_id", "")),
                        category=str(row.get("category", "")),
                        fields_json=json.dumps(row, ensure_ascii=False))
        for key in ("rating", "ratings_count", "initial_price", "discount", "final_price"):
            if key in row:
                metadata[key] = number(row[key])
        if all(metadata.get(k) is not None for k in ("initial_price", "final_price", "discount")):
            metadata["price_conflict"] = str(abs(metadata["final_price"] - metadata["initial_price"] *
                                              (1 - metadata["discount"] / 100)) > 1).lower()
        result.append(Document(page_content=serialize_record(row, fields), metadata=metadata))
    return result
