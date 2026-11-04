"""Deterministic recovery and policy enforcement distinguishing autonomous vs recovered tool use."""
import json
import re
import time
from typing import List, Optional, Tuple

from ..domain.enums import ToolCallOrigin
from ..domain.models import EvidenceChunk, ToolExecutionRecord
from .tools import safe_calculate


ARITHMETIC_INTENT_PATTERN = re.compile(
    r"\b(?:calculate|difference|sum|percentage|percent|larger|smaller|by how many|total of)\b|"
    r"різниц|сум|відсот|на скільки|більш[аеи]|менш[аеи]",
    re.IGNORECASE,
)


def has_arithmetic_intent(query: str) -> bool:
    return bool(ARITHMETIC_INTENT_PATTERN.search(query))


def attempt_arithmetic_recovery(
    user_input: str,
    draft_answer: str,
    evidence: List[EvidenceChunk],
) -> Tuple[Optional[str], Optional[ToolExecutionRecord]]:
    """Execute deterministic arithmetic recovery when model failed to call calculate.
    
    Solves DEFECT-03: Never reports this as MODEL_REQUESTED autonomous tool use.
    Tag is strictly RECOVERY_EXECUTED.
    """
    if not evidence or not has_arithmetic_intent(user_input):
        return None, None

    by_source = {}
    answer_numbers = {float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", draft_answer)}
    
    for chunk in evidence:
        source = chunk.document_id
        # Look for explicit unique totals or numbers
        direct = {float(m) for m in re.findall(r"(\d+)\s*(?:унікальн\w*|unique|разом|total)", chunk.text, re.IGNORECASE)}
        if not direct:
            direct = {float(m) for m in re.findall(r"(?:унікальн\w*|unique|разом|total)\s*(?::\s*)?(\d+)", chunk.text, re.IGNORECASE)}
            
        if direct and (direct & answer_numbers):
            by_source.setdefault(source, set()).update(direct & answer_numbers)
        else:
            relevant = " ".join(
                part for part in re.split(r"(?<=[.!?])\s+|\n", chunk.text)
                if re.search(r"unique|унікальн|загальн.*кільк|total", part, re.IGNORECASE)
            )
            values = {float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", relevant)}
            if chunk.lab_number:
                values.discard(float(chunk.lab_number))
            by_source.setdefault(source, set()).update(values & answer_numbers)

    operands = [next(iter(values)) for source, values in sorted(by_source.items()) if len(values) == 1]
    
    if len(operands) == 2:
        if re.search(r"difference|by how many|різниц|на скільки", user_input, re.IGNORECASE):
            expression = f"{max(operands):g} - {min(operands):g}"
        elif re.search(r"\bsum\b|сум", user_input, re.IGNORECASE):
            expression = f"{operands[0]:g} + {operands[1]:g}"
        else:
            expression = None

        if expression:
            start = time.perf_counter()
            calc_result = safe_calculate(expression)
            exec_time_ms = (time.perf_counter() - start) * 1000
            
            tool_record = ToolExecutionRecord(
                tool_name="calculate",
                arguments={"expression": expression},
                result=calc_result,
                origin=ToolCallOrigin.RECOVERY_EXECUTED,  # STRICTLY SEPARATED FROM AUTONOMOUS
                execution_time_ms=exec_time_ms,
            )
            
            # Format clean recovered answer
            result_val = calc_result["result"]
            source_citations = " ".join(sorted({chunk.citation for chunk in evidence}))
            if "різниц" in user_input.lower() or "difference" in user_input.lower():
                recovered_text = (
                    f"Різниця між виміряними значеннями ({max(operands):g} та {min(operands):g}) "
                    f"становить {result_val:g}. {source_citations}"
                )
            else:
                recovered_text = (
                    f"Сума виміряних значень ({operands[0]:g} та {operands[1]:g}) "
                    f"становить {result_val:g}. {source_citations}"
                )
                
            return recovered_text, tool_record

    return None, None
