"""Observable execution trace recorder."""
import json
from pathlib import Path
from typing import Any, Dict, List


class ExecutionTracer:
    def __init__(self, trace_path: Path):
        self.trace_path = Path(trace_path).resolve()
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict[str, Any]] = []

    def add_event(self, event_type: str, **data: Any) -> None:
        event = {
            "step": len(self.events) + 1,
            "event_type": event_type,
            **data,
        }
        self.events.append(event)
        self.save()

    def save(self) -> None:
        self.trace_path.write_text(
            json.dumps(self.events, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
