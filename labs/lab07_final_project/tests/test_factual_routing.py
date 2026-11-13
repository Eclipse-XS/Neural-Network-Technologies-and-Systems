"""Regression for unsolicited calculator calls overriding factual answers."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from knowledge_copilot.agents.orchestrator import AgentOrchestrator
from knowledge_copilot.agents.recovery import has_arithmetic_intent, validate_calculator_execution
from knowledge_copilot.domain.enums import GroundingStatus, ScoreType
from knowledge_copilot.domain.models import Chunk, EvidenceChunk, Score, ToolExecutionRecord


class FactualRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_factual_query_with_numeric_evidence_does_not_trigger_arithmetic_refusal(self):
        query = "What does the smoke relay use Redis and SQLite for?"
        evidence = EvidenceChunk(
            Chunk("chunk-smoke", "smoke-relay", 0,
                  "The smoke relay uses Redis for searchable document vectors and SQLite for durable document metadata and text chunks. Redis listens on port 6380; the relay stores 12 chunks.",
                  "Storage", 40, "test"),
            Score("lexical", ScoreType.RRF_SCORE, 1.0, 1),
        )
        factual_answer = f"The smoke relay uses Redis for searchable document vectors and SQLite for durable document metadata and text chunks {evidence.citation}."
        bad_calculation = ToolExecutionRecord(
            tool_name="calculate", arguments={"expression": "6380 + 12"},
            result={"result": 6392},
        )
        client = Mock(model_id="test-model")
        orchestrator = AgentOrchestrator(client, Mock(return_value=[evidence]), Mock(return_value=[]))
        session = Mock(get_items=AsyncMock(return_value=[]), add_items=AsyncMock())

        async def replay(agent, user_input, **kwargs):
            # Reproduce a model attracted to unrelated numbers if calculate is offered.
            if any(tool.name == "calculate" for tool in agent.tools):
                kwargs["hooks"].tool_records.append(bad_calculation)
            return SimpleNamespace(final_output=factual_answer)

        self.assertFalse(has_arithmetic_intent(query))
        with patch("knowledge_copilot.agents.orchestrator.SQLiteSession", return_value=session), patch(
            "knowledge_copilot.agents.orchestrator.Runner.run", side_effect=replay
        ):
            turn = await orchestrator.execute_turn(query, "regression", tracer=Mock(), session_evidence=[evidence])
        self.assertEqual(turn.grounding_status, GroundingStatus.GROUNDED)
        self.assertEqual(turn.canonical_answer, factual_answer)
        self.assertEqual(turn.citations, [evidence.citation])
        self.assertFalse(turn.invalid_citations)
        self.assertFalse(turn.tool_executions)

        # Genuine calculations still expose the tool and reject unsupported operands/results.
        arithmetic_query = "Calculate the sum of the relay's 12 chunks and 3 additional chunks."
        self.assertTrue(has_arithmetic_intent(arithmetic_query))
        tools = orchestrator._build_tools([evidence], allow_calculation=has_arithmetic_intent(arithmetic_query))
        self.assertIn("calculate", [tool.name for tool in tools])
        valid = {"tool_name": "calculate", "arguments": {"expression": "12 + 3"}, "result": {"result": 15}}
        self.assertTrue(validate_calculator_execution(arithmetic_query, [valid], [evidence]).is_valid)
        wrong_result = dict(valid, result={"result": 16})
        self.assertFalse(validate_calculator_execution(arithmetic_query, [wrong_result], [evidence]).is_valid)
        self.assertFalse(validate_calculator_execution(arithmetic_query, [bad_calculation], [evidence]).is_valid)


if __name__ == "__main__":
    unittest.main()
