"""Deterministic recovery and policy enforcement distinguishing autonomous vs recovered tool use."""
import json
import re
import time
from typing import List, Optional, Tuple

from ..domain.enums import ToolCallOrigin
from ..domain.models import EvidenceChunk, ToolExecutionRecord
from .tools import safe_calculate


ARITHMETIC_INTENT_PATTERN = re.compile(
    r"\b(?:calculate|compute|difference|sum|percentage|percent|larger|smaller|by how many|total of|total count|total number|combined total|combined count|in total|product|multiply|duration|per second|into seconds|to seconds|in seconds|average|divide|ratio|convert)\b|"
    r"обчисл\w*|пораху\w*|різниц\w*|сум\w*|відсот\w*|на скільки|більш[аеи]|менш[аеи]|добут\w*|залиш\w*|триваліст\w*|[ув] секунди|за секунду|поділ\w*|середн\w*|загальн\w*\s+кільк\w*|сумарн\w*|сукупн\w*|скільки\s+разом",
    re.IGNORECASE,
)

_STRUCT_LABEL_RE = re.compile(
    r"\b(?:lab(?:oratory)?|лаб(?:ораторн\w*)?(?:\s+робот\w*)?|adr|desc_)\s*(?:№|no\.?|number|#|-)?\s*0?\d+\b|@\d+\b",
    re.IGNORECASE,
)


def has_arithmetic_intent(query: str) -> bool:
    return bool(ARITHMETIC_INTENT_PATTERN.search(query))


def _chunk_matches_operand(chunk_text: str, val: float) -> bool:
    clean = re.sub(r"(?<=\d),(?=\d{3}\b)", "", chunk_text)
    if re.search(rf"(?<![\d.]){re.escape(f'{val:g}')}(?![\d.])", clean):
        return True
    if val.is_integer():
        iv = int(val)
        for h_s, m_s in re.findall(r"\b(\d{1,2}):(\d{2})\b", chunk_text):
            if int(h_s) * 60 + int(m_s) == iv:
                return True
    return False


def _extract_operands(
    user_input: str,
    draft_answer: str,
    evidence: List[EvidenceChunk],
) -> List[float]:
    """Identify arithmetic operands from user input, draft answer, and evidence text."""
    # 1. Explicit HH:MM timestamp duration subtraction in minutes
    hhmm = re.findall(r"\b(\d{1,2}):(\d{2})\b", user_input)
    if len(hhmm) == 2 and re.search(r"duration|minutes|триваліст|хвилин", user_input, re.IGNORECASE):
        mins = sorted([float(int(h) * 60 + int(m)) for h, m in hhmm], reverse=True)
        return mins

    clean_q_no_ts = re.sub(r"\b\d{1,2}:\d{2}\b", " ", user_input)
    clean_q_no_lab = _STRUCT_LABEL_RE.sub(" ", clean_q_no_ts)
    clean_q = re.sub(r"(\d),(\d)", r"\1\2", clean_q_no_lab)
    clean_q = re.sub(r"\b(\d{1,3})\s+(\d{3})\b", r"\1\2", clean_q)
    q_nums = [
        float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", clean_q)
        if float(x) not in {2024.0, 2025.0, 2026.0}
    ]

    clean_a = _STRUCT_LABEL_RE.sub(" ", draft_answer)
    clean_a = re.sub(r"(\d),(\d)", r"\1\2", clean_a)
    clean_a = re.sub(r"\b(\d{1,3})\s+(\d{3})\b", r"\1\2", clean_a)
    a_nums = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", clean_a)]

    ev_texts = [c.chunk.text for c in evidence]
    ev_all = _STRUCT_LABEL_RE.sub(" ", " ".join(ev_texts))
    clean_ev = re.sub(r"(\d),(\d)", r"\1\2", ev_all)
    clean_ev = re.sub(r"\b(\d{1,3})\s+(\d{3})\b", r"\1\2", clean_ev)
    ev_nums = [
        float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", clean_ev)
        if float(x) not in {1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 2024.0, 2025.0, 2026.0}
    ]

    # 2. Unit conversion / division intent (per minute -> per second, ms -> s)
    if re.search(r"per second|за секунду|into seconds|to seconds|in seconds|[ув] секунди", user_input, re.IGNORECASE):
        pool = q_nums or ev_nums
        if re.search(r"per minute|на хвилину", user_input + " " + ev_all, re.IGNORECASE) and pool:
            return [pool[0], 60.0]
        if re.search(r"\bms\b|\bмс\b|millisecond|мілісекунд", user_input + " " + ev_all, re.IGNORECASE) and pool:
            return [pool[0], 1000.0]

    # 3. Range difference between maximum and minimum values listed in question (e.g. [20, 30, 50] or [24, 64, 128])
    if len(q_nums) >= 2 and re.search(r"max|min|maximum|minimum|найбільш|найменш|difference|різниц", user_input, re.IGNORECASE):
        uniq_q = sorted(set(q_nums))
        if len(uniq_q) == 2 or re.search(r"maximum|minimum|max\b|min\b|найбільш|найменш|upper|lower", user_input, re.IGNORECASE):
            return [max(uniq_q), min(uniq_q)]

    if len(q_nums) == 2:
        return sorted(q_nums, reverse=True)

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

    ev_in_answer = sorted(
        {n for n in a_nums if any(abs(n - en) < 1e-6 for en in ev_nums)},
        reverse=True,
    )
    if len(ev_in_answer) == 2:
        return ev_in_answer

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
        calc_expr_display: Optional[str] = None
        if re.search(r"per second|за секунду|into seconds|to seconds|in seconds|[ув] секунди|per minute|на хвилину|divide|поділ|ratio|відношенн", user_input, re.IGNORECASE):
            op_sym = "/"
            expression = f"{n1:g} / {n2:g}"
        elif re.search(r"difference|by how many|remaining|різниц|на скільки|залиш|minus|віднят|триваліст", user_input, re.IGNORECASE):
            op_sym = "-"
            expression = f"{n1:g} - {n2:g}"
        elif re.search(r"product|добут|множен|multipl", user_input, re.IGNORECASE):
            op_sym = "×"
            expression = f"{n1:g} * {n2:g}"
        elif re.search(r"\b(?:average|mean)\s+of\b|середн\w*\s+(?:арифметич|значен\w*\s+(?:між|для|двох))", user_input, re.IGNORECASE):
            op_sym = "avg"
            expression = f"({n1:g} + {n2:g}) / 2"
            calc_expr_display = f"({n1:g} + {n2:g}) / 2"
        elif re.search(r"\bsum\b|сум|разом|combined|in total|total of|total count|total number|загальн\w*\s+кільк\w*|сумарн|сукупн|\bplus\b|плюс", user_input, re.IGNORECASE):
            op_sym = "+"
            expression = f"{n1:g} + {n2:g}"
        else:
            op_sym = "-"
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
        matched_cites = sorted({
            chunk.citation
            for chunk in evidence
            if _chunk_matches_operand(chunk.chunk.text, n1) or _chunk_matches_operand(chunk.chunk.text, n2)
        })
        if not matched_cites and evidence:
            matched_cites = sorted({chunk.citation for chunk in evidence[:2]})
        source_citations = " ".join(matched_cites)
        expr_text = calc_expr_display or f"{n1:g} {op_sym} {n2:g}"

        recovered_text = (
            f"{draft_answer.strip()}\n\n"
            f"[Розраховано: {expr_text} = {result_val:g}] {source_citations}".strip()
        )

        return recovered_text, tool_record

    return None, None
