"""Bounded LLM agent orchestration with observable tool traces and memory synchronization."""
import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from .recovery import attempt_arithmetic_recovery, has_arithmetic_intent
from .tools import safe_calculate
from .prompts import SYSTEM_INSTRUCTIONS
from .verifier import (
    KNOWN_TECHNOLOGIES,
    is_abstention,
    normalize_multi_citations,
    verify_grounding,
)


class CopilotTraceHooks(RunHooks):
    def __init__(self, tracer: ExecutionTracer):
        self.tracer = tracer
        self.tool_records: List[ToolExecutionRecord] = []
        self._start_times: Dict[str, float] = {}

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
        exec_ms = (time.perf_counter() - start_time) * 1000
        
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
        self.db_path = db_path or Path("outputs/sessions/assistant_memory.sqlite")
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        set_tracing_disabled(True)

    def _build_tools(
        self,
        current_evidence_accumulator: List[EvidenceChunk],
        retrieval_mode: Optional[Any] = None,
    ):
        @function_tool
        def search_knowledge(query: str, k: int = 7, document_id: Optional[str] = None) -> Dict[str, Any]:
            """Retrieve evidence chunks from the local document vector index."""
            chunks = self.search_fn(query=query, k=k, document_id=document_id, mode=retrieval_mode)
            current_evidence_accumulator.extend(chunks)
            note = (
                "Candidates are not proof of relevance. Use only supporting text; otherwise abstain."
                if chunks
                else "No matching evidence found. Do NOT search again; answer directly that the indexed knowledge base does not contain this information."
            )
            return {
                "chunks": [c.to_dict() for c in chunks],
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
            return safe_calculate(expression)

        return [search_knowledge, list_knowledge_sources, calculate]

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
            tracer = ExecutionTracer(Path(f"outputs/traces/{turn_id}.json"))
            
        tracer.add_event("user_input", text=user_input, session_id=session_id)
        
        sdk_session = SQLiteSession(session_id=session_id, db_path=self.db_path)
        try:
            existing_items = await sdk_session.get_items()
            if not existing_items and history_messages:
                items_to_seed = [{"role": m["role"], "content": m["content"]} for m in history_messages]
                await sdk_session.add_items(items_to_seed)
                existing_items = await sdk_session.get_items()

            # Check pronoun ambiguity on empty history
            has_history = bool(history_messages) or bool(existing_items)
            has_subject = bool(re.search(
                r"\b(?:lab(?:oratory)?\s*\d+|report|model|document|system|paper|file)\b|"
                r"\b(?:лабораторн\w*\s*(?:робот\w*)?\s*№?\s*\d+|звіт|модел|документ|систем|статт|файл)",
                user_input,
                re.IGNORECASE,
            ))
            ambiguous = not has_history and not has_subject and bool(re.search(
                r"\b(?:it|its|that model|this model|that file)\b|\b(?:вона|воно|він|цей|ця|те)\b",
                user_input,
                re.IGNORECASE,
            ))
            
            if ambiguous:
                clarification = (
                    "Please clarify which document, model, or report you are referring to. "
                    "A new session does not have context for that pronoun."
                )
                tracer.add_event("ambiguity_guard", reason="pronoun_without_session_antecedent")
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
                    runtime_seconds=round(time.perf_counter() - start_time, 3),
                )

            evidence_accum: List[EvidenceChunk] = list(session_evidence or [])
            tools = self._build_tools(evidence_accum, retrieval_mode=retrieval_mode)
            trace_hooks = CopilotTraceHooks(tracer)
            
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

            # Execute runner with model-visible multi-turn conversation memory
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
            raw_answer = str(result.final_output)
            canonical_answer = raw_answer
            
            # Check arithmetic recovery if model omitted calculate tool
            used_tool_names = [r.tool_name for r in trace_hooks.tool_records]
            if has_arithmetic_intent(user_input) and "calculate" not in used_tool_names:
                recovered_ans, recovery_tool = attempt_arithmetic_recovery(
                    user_input, raw_answer, evidence_accum
                )
                if recovered_ans and recovery_tool:
                    canonical_answer = recovered_ans
                    trace_hooks.tool_records.append(recovery_tool)
                    tracer.add_event(
                        "arithmetic_recovery",
                        recovered_expression=recovery_tool.arguments.get("expression"),
                        origin=ToolCallOrigin.RECOVERY_EXECUTED.value,
                    )
            
            # Clean formatting: remove literal ungrounded placeholder strings
            canonical_answer = re.sub(r"\s*\[source_id#chunk_id\]", "", canonical_answer).strip()

            # Normalize comma/semicolon-grouped citations [doc#c1, doc#c2] -> [doc#c1] [doc#c2]
            canonical_answer = normalize_multi_citations(canonical_answer)

            # If the model emitted placeholder [source_id#chunk_id] or [document_id#chunk_id] with a real chunk_id:
            chunk_by_id = {c.chunk.chunk_id: c.chunk for c in evidence_accum}
            def _replace_placeholder_doc(m):
                cid = m.group(1)
                if cid in chunk_by_id:
                    return f"[{chunk_by_id[cid].document_id}#{cid}]"
                return m.group(0)
            canonical_answer = re.sub(r"\[(?:source_id|document_id)#([a-zA-Z0-9_-]+)\]", _replace_placeholder_doc, canonical_answer)
            canonical_answer = re.sub(r"\s+([.,;:!?])", r"\1", canonical_answer)

            # Semantic Grounding Verification (DEFECT-02)
            grounding_status, cited, invalid_cited, claims = verify_grounding(
                canonical_answer, evidence_accum, tool_executions=trace_hooks.tool_records
            )

            # Premise Verification & Repair:
            # 1. If user asks about a technology absent from retrieved evidence AND model asserted a premise conflict
            all_evidence_lower = " ".join(c.chunk.text for c in evidence_accum).lower()
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

            # 2. If answer abstained without citations, strip speculative tutorial YAML/code blocks and ungrounded how-to sections
            if not premise_repaired and is_abstention(canonical_answer) and not cited:
                def _strip_ungrounded_code(m):
                    code_body = m.group(0).strip().lower()
                    key_tokens = [t for t in re.findall(r"[a-zA-Z_]{4,}", code_body)]
                    if key_tokens and not any(t in all_evidence_lower for t in key_tokens[:3]):
                        return ""
                    return m.group(0)
                cleaned_refusal = re.sub(r"```[\s\S]*?```", _strip_ungrounded_code, canonical_answer).strip()
                if any(
                    re.search(r"\b" + re.escape(tech) + r"\b", q_lower)
                    and not re.search(r"\b" + re.escape(tech) + r"\b", all_evidence_lower)
                    for tech in KNOWN_TECHNOLOGIES
                ) and len(cleaned_refusal) > 350:
                    first_para = cleaned_refusal.split("\n\n")[0].strip()
                    if first_para and is_abstention(first_para):
                        cleaned_refusal = first_para
                if cleaned_refusal and is_abstention(cleaned_refusal) and cleaned_refusal != canonical_answer:
                    canonical_answer = cleaned_refusal
                    premise_repaired = True
                    tracer.add_event("speculative_code_stripped")

            if premise_repaired:
                grounding_status, cited, invalid_cited, claims = verify_grounding(
                    canonical_answer, evidence_accum, tool_executions=trace_hooks.tool_records
                )

            # Prevent divergence between OpenAI Agents SDK session and authoritative canonical answer
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
            
            tracer.add_event(
                "grounding_verification",
                status=grounding_status.value,
                citations=cited,
                invalid_citations=invalid_cited,
            )
            
            tracer.add_event("canonical_turn_complete", answer=canonical_answer)
            
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
                claims=claims,
                evidence=evidence_accum,
                runtime_seconds=round(time.perf_counter() - start_time, 3),
            )
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            tracer.add_event("execution_error", error=error_msg)
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
                runtime_seconds=round(time.perf_counter() - start_time, 3),
            )
        finally:
            try:
                sdk_session.close()
            except Exception:
                pass
