"""Deterministic recovery and policy enforcement distinguishing autonomous vs recovered tool use."""
import json
import re
import time
from typing import List, Optional, Tuple

from ..domain.enums import ToolCallOrigin
from ..domain.models import EvidenceChunk, ToolExecutionRecord
from .tools import safe_calculate


ARITHMETIC_INTENT_PATTERN = re.compile(
    r"\b(?:calculate|compute|difference|sum|percentage|percent|larger|smaller|by how many|total of|product|multiply|duration)\b|"
    r"обчисл\w*|пораху\w*|різниц\w*|сум\w*|відсот\w*|на скільки|більш[аеи]|менш[аеи]|добут\w*|залиш\w*|триваліст\w*",
    re.IGNORECASE,
)


def has_arithmetic_intent(query: str) -> bool:
    return bool(ARITHMETIC_INTENT_PATTERN.search(query))


def _extract_operands(
    user_input: str,
    draft_answer: str,
    evidence: List[EvidenceChunk],
) -> List[float]:
    """Identify arithmetic operands from user input, draft answer, and evidence text."""
    clean_q = re.sub(r"(\d),(\d)", r"\1\2", user_input)
    q_nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", clean_q) if float(x) not in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}]

    clean_a = re.sub(r"(\d),(\d)", r"\1\2", draft_answer)
    a_nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", clean_a)]

    if len(q_nums) == 2:
        return sorted(q_nums, reverse=True)

    ev_texts = [c.chunk.text for c in evidence]
    ev_all = " ".join(ev_texts)
    clean_ev = re.sub(r"(\d),(\d)", r"\1\2", ev_all)
    ev_nums = [
        float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", clean_ev)
        if float(x) not in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 2024.0, 2025.0, 2026.0}
    ]

    candidates = list(set(q_nums + ev_nums))
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            n1, n2 = max(candidates[i], candidates[j]), min(candidates[i], candidates[j])
            diff = n1 - n2
            prod = n1 * n2
            total = n1 + n2
            if (
                any(abs(an - diff) < 0.01 for an in a_nums)
                or any(abs(an - prod) < 0.01 for an in a_nums)
                or any(abs(an - total) < 0.01 for an in a_nums)
            ):
                return [n1, n2]

    # Fallback to source-separated values if available
    by_source = {}
    answer_numbers = set(a_nums)
    for chunk in evidence:
        source = chunk.document_id
        direct = {float(m) for m in re.findall(r"(\d+)\s*(?:унікальн\w*|unique|разом|total)", chunk.text, re.IGNORECASE)}
        if not direct:
            direct = {float(m) for m in re.findall(r"(?:унікальн\w*|unique|разом|total)\s*(?::\s*)?(\d+)", chunk.text, re.IGNORECASE)}
        if direct and (direct & answer_numbers):
            by_source.setdefault(source, set()).update(direct & answer_numbers)

    operands = [next(iter(values)) for source, values in sorted(by_source.items()) if len(values) == 1]
    if len(operands) == 2:
        return sorted(operands, reverse=True)

    return []


def attempt_arithmetic_recovery(
    user_input: str,
    draft_answer: str,
    evidence: List[EvidenceChunk],
) -> Tuple[Optional[str], Optional[ToolExecutionRecord]]:
    """Execute deterministic arithmetic recovery when model failed to call calculate.

    Solves DEFECT-03: Never reports this as MODEL_REQUESTED autonomous tool use.
    Tag is strictly RECOVERY_EXECUTED.
    """
    if not has_arithmetic_intent(user_input):
        return None, None

    operands = _extract_operands(user_input, draft_answer, evidence)

    if len(operands) == 2:
        n1, n2 = operands[0], operands[1]
        if re.search(r"difference|by how many|різниц|на скільки|залиш|minus|триваліст", user_input, re.IGNORECASE):
            expression = f"{n1:g} - {n2:g}"
        elif re.search(r"product|добут|множен|multipl", user_input, re.IGNORECASE):
            expression = f"{n1:g} * {n2:g}"
        elif re.search(r"\bsum\b|сум|разом|total", user_input, re.IGNORECASE):
            expression = f"{n1:g} + {n2:g}"
        else:
            expression = f"{n1:g} - {n2:g}"

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

        result_val = calc_result["result"]
        source_citations = " ".join(sorted({chunk.citation for chunk in evidence})) if evidence else ""

        if re.search(r"добут|product|multipl", user_input, re.IGNORECASE):
            recovered_text = (
                f"{draft_answer.strip()}\n\n"
                f"[Розраховано: {n1:g} × {n2:g} = {result_val:g}] {source_citations}".strip()
            )
        elif re.search(r"\bsum\b|сум", user_input, re.IGNORECASE):
            recovered_text = (
                f"{draft_answer.strip()}\n\n"
                f"[Розраховано: {n1:g} + {n2:g} = {result_val:g}] {source_citations}".strip()
            )
        else:
            recovered_text = (
                f"{draft_answer.strip()}\n\n"
                f"[Розраховано: {n1:g} - {n2:g} = {result_val:g}] {source_citations}".strip()
            )

        return recovered_text, tool_record

    return None, None
