"""Bounded LLM agent orchestration with observable tool traces and memory synchronization."""
import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from ..config import LAB_ROOT
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunHooks,
    Runner,
    SQLiteSession,
    function_tool,
    set_tracing_disabled,
)

from ..domain.enums import GroundingStatus, ToolCallOrigin
from ..domain.models import EvidenceChunk, ToolExecutionRecord, Turn
from ..observability.tracer import ExecutionTracer
from .recovery import (
    compose_arithmetic_uncertainty_answer,
    determine_arithmetic_operation,
    has_arithmetic_intent,
    run_arithmetic_recovery,
    validate_calculator_execution,
)
from .tools import safe_calculate
from .prompts import SYSTEM_INSTRUCTIONS
from .verifier import (
    KNOWN_TECHNOLOGIES,
    canonicalize_citations_and_claims,
    chunk_searchable_text,
    is_abstention,
    normalize_multi_citations,
    verify_grounding,
)


class CopilotTraceHooks(RunHooks):
    def __init__(self, tracer: ExecutionTracer):
        self.tracer = tracer
        self.tool_records: List[ToolExecutionRecord] = []
        self._start_times: Dict[str, float] = {}
        self._agent_start_times: List[float] = []
        self.llm_calls_count: int = 0
        self.retrieval_calls_count: int = 0
        self.retrieval_ms: float = 0.0
        self.tool_execution_ms: float = 0.0

    async def on_agent_start(self, context, agent):
        self.llm_calls_count += 1
        self._agent_start_times.append(time.perf_counter())

    async def on_tool_start(self, context, agent, tool):
        call_id = context.tool_call_id
        self._start_times[call_id] = time.perf_counter()
        raw_args = context.tool_arguments
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except Exception:
                pass
        self.tracer.add_event(
            "tool_call",
            tool_name=tool.name,
            call_id=call_id,
            arguments=raw_args if isinstance(raw_args, dict) else {},
            origin=ToolCallOrigin.MODEL_REQUESTED.value,
        )

    async def on_tool_end(self, context, agent, tool, result):
        call_id = context.tool_call_id
        start_time = self._start_times.pop(call_id, time.perf_counter())
        exec_ms = (time.perf_counter() - start_time) * 1000.0
        self.tool_execution_ms += exec_ms
        if tool.name == "search_knowledge":
            self.retrieval_calls_count += 1
            self.retrieval_ms += exec_ms

        parsed_result = result
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
            except Exception:
                pass

        parsed_args = context.tool_arguments
        if isinstance(parsed_args, str):
            try:
                parsed_args = json.loads(parsed_args)
            except Exception:
                pass
        if not isinstance(parsed_args, dict):
            parsed_args = {}

        record = ToolExecutionRecord(
            tool_name=tool.name,
            arguments=parsed_args,
            result=parsed_result,
            origin=ToolCallOrigin.MODEL_REQUESTED,
            execution_time_ms=exec_ms,
        )
        self.tool_records.append(record)
        self.tracer.add_event(
            "tool_result",
            tool_name=tool.name,
            call_id=call_id,
            result=parsed_result,
            origin=ToolCallOrigin.MODEL_REQUESTED.value,
            execution_time_ms=exec_ms,
        )


class AgentOrchestrator:
    def __init__(
        self,
        llm_client,
        search_fn: Callable[..., List[EvidenceChunk]],
        list_sources_fn: Callable[[], List[Dict[str, Any]]],
        db_path: Optional[Path] = None,
        max_turns: int = 4,
        timeout_seconds: float = 240.0,
        temperature: float = 0.0,
        max_tokens: int = 900,
    ):
        self.llm_client = llm_client
        self.search_fn = search_fn
        self.list_sources_fn = list_sources_fn
        self.db_path = db_path or LAB_ROOT / "outputs/sessions/assistant_memory.sqlite"
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        set_tracing_disabled(True)

    def _build_tools(
        self,
        current_evidence_accumulator: List[EvidenceChunk],
        retrieval_mode: Optional[Any] = None,
        allow_calculation: bool = True,
    ):
        seen_chunk_keys: Set[Tuple[str, str]] = {
            (c.chunk.document_id, c.chunk.chunk_id) for c in current_evidence_accumulator
        }

        @function_tool
        def search_knowledge(query: str, k: int = 7, document_id: Optional[str] = None) -> Dict[str, Any]:
            """Retrieve evidence chunks from the local document vector index."""
            chunks = self.search_fn(query=query, k=k, document_id=document_id, mode=retrieval_mode)
            new_chunks: List[EvidenceChunk] = []
            for c in chunks:
                key = (c.chunk.document_id, c.chunk.chunk_id)
                if key not in seen_chunk_keys:
                    seen_chunk_keys.add(key)
                    current_evidence_accumulator.append(c)
                    new_chunks.append(c)
                else:
                    new_chunks.append(c)

            note = (
                "Candidates are not proof of relevance. Use only supporting text; otherwise abstain."
                if chunks
                else "No matching evidence found. Do NOT search again; answer directly that the indexed knowledge base does not contain this information."
            )
            # Return compact model-facing chunk payload (omitting 64-char sha256 and nested duplicate metadata)
            compact_chunks = [
                {
                    "chunk_id": c.chunk.chunk_id,
                    "document_id": c.chunk.document_id,
                    "source_id": c.chunk.document_id,
                    "citation": c.citation,
                    "section": c.chunk.section,
                    "text": c.chunk.text,
                    "cosine_distance": round(c.score.score_value, 4),
                    "ranking_position": c.score.ranking_position,
                }
                for c in new_chunks
            ]
            return {
                "chunks": compact_chunks,
                "evidence_note": note,
                "empty": len(chunks) == 0,
            }

        @function_tool
        def list_knowledge_sources() -> Dict[str, Any]:
            """List indexed document IDs, titles, filenames, and status."""
            return {"sources": self.list_sources_fn()}

        @function_tool
        def calculate(expression: str) -> Dict[str, Any]:
            """Compute exact derived arithmetic using retrieved numeric operands.
            Always invoke this tool for ANY mathematical calculation, difference, sum, product, division, unit conversion, or duration.
            Example expressions: '8501 - 8000', '26 * 24', '128 - 24', '5000 / 1000', '(14 * 60 + 47) - (14 * 60 + 20)'."""
            try:
                return safe_calculate(expression)
            except (ZeroDivisionError, ValueError) as exc:
                return {"expression": expression, "error": str(exc), "result": None}

        tools = [search_knowledge, list_knowledge_sources]
        if allow_calculation:
            tools.append(calculate)
        return tools

    async def execute_turn(
        self,
        user_input: str,
        session_id: str,
        history_messages: Optional[List[Dict[str, str]]] = None,
        tracer: Optional[ExecutionTracer] = None,
        retrieval_mode: Optional[Any] = None,
        session_evidence: Optional[List[EvidenceChunk]] = None,
    ) -> Turn:
        start_time = time.perf_counter()
        turn_id = f"turn_{uuid.uuid4().hex[:12]}"

        if tracer is None:
            tracer = ExecutionTracer(LAB_ROOT / f"outputs/traces/{turn_id}.json")

        tracer.add_event("user_input", text=user_input, session_id=session_id)

        sdk_session = SQLiteSession(session_id=session_id, db_path=self.db_path)
        trace_hooks = CopilotTraceHooks(tracer)
        evidence_accum: List[EvidenceChunk] = list(session_evidence or [])
        timings_ms: Dict[str, float] = {
            "prompt_assembly_ms": 0.0,
            "retrieval_ms": 0.0,
            "embedding_ms": 0.0,
            "llm_generation_ms": 0.0,
            "tool_execution_ms": 0.0,
            "verification_ms": 0.0,
            "session_persistence_ms": 0.0,
            "total_ms": 0.0,
        }

        try:
            t_prompt_start = time.perf_counter()
            existing_items = await sdk_session.get_items()
            if not existing_items and history_messages:
                items_to_seed = [{"role": m["role"], "content": m["content"]} for m in history_messages]
                await sdk_session.add_items(items_to_seed)
                existing_items = await sdk_session.get_items()

            # Check pronoun ambiguity on empty history
            has_history = bool(history_messages) or bool(existing_items)
            has_subject = bool(
                re.search(
                    r"\b(?:lab(?:oratory)?\s*\d+|report|model|document|system|paper|file)\b|"
                    r"\b(?:лабораторн\w*\s*(?:робот\w*)?\s*№?\s*\d+|звіт|модел|документ|систем|статт|файл)",
                    user_input,
                    re.IGNORECASE,
                )
            )
            ambiguous = not has_history and not has_subject and bool(
                re.search(
                    r"\b(?:it|its|that model|this model|that file)\b|\b(?:вона|воно|він|цей|ця|те)\b",
                    user_input,
                    re.IGNORECASE,
                )
            )

            if ambiguous:
                clarification = (
                    "Please clarify which document, model, or report you are referring to. "
                    "A new session does not have context for that pronoun."
                )
                tracer.add_event("ambiguity_guard", reason="pronoun_without_session_antecedent")
                total_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
                timings_ms["total_ms"] = total_ms
                return Turn(
                    turn_id=turn_id,
                    session_id=session_id,
                    user_input=user_input,
                    raw_answer=clarification,
                    canonical_answer=clarification,
                    grounding_status=GroundingStatus.INSUFFICIENT_EVIDENCE,
                    tool_executions=[],
                    citations=[],
                    invalid_citations=[],
                    claims=[],
                    evidence=[],
                    timings_ms=timings_ms,
                    telemetry={"llm_calls_count": 0, "retrieval_calls_count": 0, "retries_count": 0, "timeout_type": None},
                    runtime_seconds=round(total_ms / 1000.0, 3),
                )

            # Retrieved numbers and unsolicited tool choices do not establish user intent.
            arith_intent = has_arithmetic_intent(user_input)
            tools = self._build_tools(
                evidence_accum, retrieval_mode=retrieval_mode, allow_calculation=arith_intent
            )
            client = self.llm_client.get_client()
            model_id = self.llm_client.model_id

            agent = Agent(
                name="Knowledge Copilot",
                instructions=SYSTEM_INSTRUCTIONS,
                model=OpenAIChatCompletionsModel(model_id, client),
                tools=tools,
                model_settings=ModelSettings(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                ),
            )
            timings_ms["prompt_assembly_ms"] = round((time.perf_counter() - t_prompt_start) * 1000.0, 2)

            # Execute runner with model-visible multi-turn conversation memory
            t_run_start = time.perf_counter()
            result = await asyncio.wait_for(
                Runner.run(
                    agent,
                    user_input,
                    session=sdk_session,
                    max_turns=self.max_turns,
                    hooks=trace_hooks,
                    run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False),
                ),
                timeout=self.timeout_seconds,
            )
            runner_wall_ms = (time.perf_counter() - t_run_start) * 1000.0
            timings_ms["retrieval_ms"] = round(trace_hooks.retrieval_ms, 2)
            timings_ms["tool_execution_ms"] = round(trace_hooks.tool_execution_ms, 2)
            timings_ms["llm_generation_ms"] = round(max(0.0, runner_wall_ms - trace_hooks.tool_execution_ms), 2)

            raw_answer = str(result.final_output)
            canonical_answer = raw_answer

            t_ver_start = time.perf_counter()
            arithmetic_plan = None
            calc_val = None
            has_calc_calls = any(t.tool_name == "calculate" for t in trace_hooks.tool_records)

            if has_calc_calls:
                calc_val = validate_calculator_execution(user_input, trace_hooks.tool_records, evidence_accum)
                if calc_val.is_valid:
                    arithmetic_plan = calc_val.validated_plan
                    tracer.add_event(
                        "calculator_validation_passed",
                        status=calc_val.status,
                        operands=calc_val.validated_operands,
                    )
                    if (
                        arithmetic_plan
                        and arithmetic_plan.validation_status == "VALIDATED"
                        and arithmetic_plan.source_citations
                        and not is_abstention(canonical_answer)
                    ):
                        missing_cits = [c for c in arithmetic_plan.source_citations if c not in canonical_answer]
                        if missing_cits:
                            canonical_answer = f"{canonical_answer.rstrip()} {' '.join(missing_cits)}".strip()
                else:
                    tracer.add_event(
                        "calculator_validation_failed",
                        status=calc_val.status,
                        reason=calc_val.reason,
                        invalid_operands=calc_val.invalid_operands,
                    )
                    recovered_ans, updated_tools, arithmetic_plan = run_arithmetic_recovery(
                        user_input,
                        raw_answer,
                        evidence_accum,
                        trace_hooks.tool_records,
                        force_recovery=True,
                    )
                    if (
                        arithmetic_plan
                        and arithmetic_plan.validation_status == "VALIDATED"
                        and len(updated_tools) > len(trace_hooks.tool_records)
                    ):
                        recovery_tool = updated_tools[-1]
                        trace_hooks.tool_records.append(recovery_tool)
                        canonical_answer = recovered_ans
                        tracer.add_event(
                            "arithmetic_recovery",
                            recovered_expression=recovery_tool.arguments.get("expression"),
                            origin=ToolCallOrigin.RECOVERY_EXECUTED.value,
                            validation_status=arithmetic_plan.validation_status,
                            decision=arithmetic_plan.canonicalization_decision,
                        )
                    else:
                        canonical_answer = compose_arithmetic_uncertainty_answer(
                            user_input, plan=arithmetic_plan, calc_val=calc_val
                        )
                        tracer.add_event(
                            "arithmetic_uncertainty_abstention",
                            status=arithmetic_plan.validation_status if arithmetic_plan else calc_val.status,
                            reason=calc_val.reason,
                        )
            elif arith_intent:
                recovered_ans, updated_tools, arithmetic_plan = run_arithmetic_recovery(
                    user_input,
                    raw_answer,
                    evidence_accum,
                    trace_hooks.tool_records,
                )
                if len(updated_tools) > len(trace_hooks.tool_records):
                    recovery_tool = updated_tools[-1]
                    rec_val = validate_calculator_execution(
                        user_input, [recovery_tool], evidence_accum, plan=arithmetic_plan
                    )
                    if rec_val.is_valid:
                        trace_hooks.tool_records.append(recovery_tool)
                        if recovered_ans and recovered_ans != raw_answer:
                            canonical_answer = recovered_ans
                        tracer.add_event(
                            "arithmetic_recovery",
                            recovered_expression=recovery_tool.arguments.get("expression"),
                            origin=ToolCallOrigin.RECOVERY_EXECUTED.value,
                            validation_status=arithmetic_plan.validation_status if arithmetic_plan else "UNKNOWN",
                            decision=arithmetic_plan.canonicalization_decision if arithmetic_plan else "UNKNOWN",
                        )
                    else:
                        canonical_answer = compose_arithmetic_uncertainty_answer(
                            user_input, plan=arithmetic_plan, calc_val=rec_val
                        )
                        tracer.add_event(
                            "arithmetic_uncertainty_abstention",
                            status=rec_val.status,
                            reason=rec_val.reason,
                        )

            # Citation Boundary & Safe Content-Based Canonicalization (Phase 3)
            cit_report = canonicalize_citations_and_claims(
                canonical_answer,
                evidence_accum,
                tool_executions=trace_hooks.tool_records,
                user_input=user_input,
            )
            canonical_answer = cit_report.canonical_answer

            # Semantic Grounding Verification (Phase 4)
            grounding_status, cited, invalid_cited, claims = verify_grounding(
                canonical_answer,
                evidence_accum,
                tool_executions=trace_hooks.tool_records,
                user_input=user_input,
            )

            # Guard against ungrounded prose math when source-derived arithmetic lacked valid operands
            if (
                arith_intent
                and arithmetic_plan is not None
                and arithmetic_plan.validation_status in {"MISSING_OPERAND", "AMBIGUOUS_OPERAND", "UNGROUNDED_OPERANDS", "DIVISION_BY_ZERO"}
                and not is_abstention(canonical_answer)
                and (
                    grounding_status != GroundingStatus.GROUNDED
                    or bool(re.search(r"\[(?:Calculated|Розраховано):|\d+\s*[+\-*/=]\s*\d+", canonical_answer))
                    or determine_arithmetic_operation(user_input) != "ambiguous"
                )
            ):
                canonical_answer = compose_arithmetic_uncertainty_answer(
                    user_input, plan=arithmetic_plan, calc_val=calc_val
                )
                tracer.add_event(
                    "arithmetic_uncertainty_abstention",
                    status=arithmetic_plan.validation_status,
                    reason=arithmetic_plan.canonicalization_decision,
                )
                grounding_status, cited, invalid_cited, claims = verify_grounding(
                    canonical_answer,
                    evidence_accum,
                    tool_executions=trace_hooks.tool_records,
                    user_input=user_input,
                )

            # Premise Verification & Repair:
            # Check searchable text (including document_id and section!) so Lab 4 ('lab04_stable_diffusion_report') is recognized for 'stable diffusion'
            all_evidence_lower = " ".join(chunk_searchable_text(c) for c in evidence_accum).lower()
            q_lower = user_input.lower()
            premise_repaired = False

            for tech in KNOWN_TECHNOLOGIES:
                if re.search(r"\b" + re.escape(tech) + r"\b", q_lower):
                    if not re.search(r"\b" + re.escape(tech) + r"\b", all_evidence_lower):
                        has_premise_claim_err = any("Subject '" in (c.unsupported_reason or "") for c in claims)
                        if has_premise_claim_err or not is_abstention(canonical_answer):
                            is_uk = bool(re.search(r"[а-яА-ЯіїєґІЇЄҐ]", user_input))
                            if is_uk:
                                canonical_answer = (
                                    f"У проіндексованих документах не міститься інформація або конфігурація щодо `{tech}`. "
                                    "Надані матеріали не підтверджують використання цього компонента."
                                )
                            else:
                                canonical_answer = (
                                    f"The indexed documents do not contain information or configuration details for `{tech}`. "
                                    "Insufficient evidence in the provided sources to verify this premise."
                                )
                            premise_repaired = True
                            tracer.add_event("premise_repair", absent_technology=tech)
                            break

            # Strip speculative tutorial YAML/code blocks if answer abstained without citations
            if not premise_repaired and is_abstention(canonical_answer) and not cited:
                def _strip_ungrounded_code(m):
                    code_body = m.group(0).strip().lower()
                    key_tokens = [t for t in re.findall(r"[a-zA-Z_]{4,}", code_body)]
                    if key_tokens and not any(t in all_evidence_lower for t in key_tokens[:3]):
                        return ""
                    return m.group(0)

                cleaned_refusal = re.sub(r"```[\s\S]*?```", _strip_ungrounded_code, canonical_answer).strip()
                if (
                    any(
                        re.search(r"\b" + re.escape(tech) + r"\b", q_lower)
                        and not re.search(r"\b" + re.escape(tech) + r"\b", all_evidence_lower)
                        for tech in KNOWN_TECHNOLOGIES
                    )
                    and len(cleaned_refusal) > 350
                ):
                    first_para = cleaned_refusal.split("\n\n")[0].strip()
                    if first_para and is_abstention(first_para):
                        cleaned_refusal = first_para
                if cleaned_refusal and is_abstention(cleaned_refusal) and cleaned_refusal != canonical_answer:
                    canonical_answer = cleaned_refusal
                    premise_repaired = True
                    tracer.add_event("speculative_code_stripped")

            if premise_repaired:
                grounding_status, cited, invalid_cited, claims = verify_grounding(
                    canonical_answer,
                    evidence_accum,
                    tool_executions=trace_hooks.tool_records,
                    user_input=user_input,
                )

            timings_ms["verification_ms"] = round((time.perf_counter() - t_ver_start) * 1000.0, 2)

            # Prevent divergence between OpenAI Agents SDK session and authoritative canonical answer
            t_sess_start = time.perf_counter()
            if canonical_answer != raw_answer:
                try:
                    items = await sdk_session.get_items()
                    if items:
                        last_item = items[-1]
                        if isinstance(last_item, dict) and last_item.get("role") == "assistant":
                            await sdk_session.pop_item()
                            updated_item = dict(last_item)
                            if isinstance(updated_item.get("content"), list):
                                updated_item["content"] = [
                                    {"annotations": [], "text": canonical_answer, "type": "output_text", "logprobs": []}
                                ]
                            else:
                                updated_item["content"] = canonical_answer
                            await sdk_session.add_items([updated_item])
                except Exception as sync_err:
                    tracer.add_event("session_sync_warning", error=str(sync_err))
            timings_ms["session_persistence_ms"] = round((time.perf_counter() - t_sess_start) * 1000.0, 2)
            timings_ms["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)

            telemetry = {
                "model_id": model_id,
                "llm_calls_count": trace_hooks.llm_calls_count,
                "retrieval_calls_count": trace_hooks.retrieval_calls_count,
                "retries_count": 0,
                "timeout_type": None,
                "context_chars": sum(len(c.chunk.text) for c in evidence_accum),
                "citation_repairs_count": cit_report.citation_repairs_count,
                "citation_removals_count": cit_report.citation_removals_count,
                "unsupported_claims_removed_count": cit_report.unsupported_claims_removed_count,
            }

            tracer.add_event(
                "grounding_verification",
                status=grounding_status.value,
                citations=cited,
                invalid_citations=invalid_cited,
                raw_citations=cit_report.raw_citations,
                raw_invalid_citations=cit_report.raw_invalid_citations,
                citation_repairs_count=cit_report.citation_repairs_count,
                citation_removals_count=cit_report.citation_removals_count,
            )

            tracer.add_event(
                "canonical_turn_complete",
                answer=canonical_answer,
                timings_ms=timings_ms,
                telemetry=telemetry,
            )

            return Turn(
                turn_id=turn_id,
                session_id=session_id,
                user_input=user_input,
                raw_answer=raw_answer,
                canonical_answer=canonical_answer,
                grounding_status=grounding_status,
                tool_executions=trace_hooks.tool_records,
                citations=cited,
                invalid_citations=invalid_cited,
                raw_citations=cit_report.raw_citations,
                raw_invalid_citations=cit_report.raw_invalid_citations,
                citation_repairs_count=cit_report.citation_repairs_count,
                citation_removals_count=cit_report.citation_removals_count,
                unsupported_claims_removed_count=cit_report.unsupported_claims_removed_count,
                claims=claims,
                evidence=evidence_accum,
                arithmetic_plan=arithmetic_plan,
                timings_ms=timings_ms,
                telemetry=telemetry,
                runtime_seconds=round(timings_ms["total_ms"] / 1000.0, 3),
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            timeout_type = "ORCHESTRATOR_TIMEOUT" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__
            timings_ms["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            telemetry = {
                "model_id": getattr(self.llm_client, "model_id", "unknown"),
                "llm_calls_count": trace_hooks.llm_calls_count,
                "retrieval_calls_count": trace_hooks.retrieval_calls_count,
                "retries_count": 0,
                "timeout_type": timeout_type,
                "error": error_msg,
            }
            tracer.add_event("execution_error", error=error_msg, timeout_type=timeout_type, timings_ms=timings_ms)
            return Turn(
                turn_id=turn_id,
                session_id=session_id,
                user_input=user_input,
                raw_answer="",
                canonical_answer=f"Error executing request: {error_msg}",
                grounding_status=GroundingStatus.EXECUTION_FAILED,
                tool_executions=trace_hooks.tool_records,
                citations=[],
                invalid_citations=[],
                claims=[],
                evidence=evidence_accum,
                timings_ms=timings_ms,
                telemetry=telemetry,
                runtime_seconds=round(timings_ms["total_ms"] / 1000.0, 3),
            )
        finally:
            try:
                sdk_session.close()
            except Exception:
                pass
