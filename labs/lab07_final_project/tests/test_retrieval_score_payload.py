"""Agent-facing retrieval payloads must preserve the underlying score semantics."""
import json
import unittest
from unittest.mock import Mock

from agents.tool_context import ToolContext

from knowledge_copilot.agents.orchestrator import AgentOrchestrator
from knowledge_copilot.domain.enums import ScoreType
from knowledge_copilot.domain.models import Chunk, EvidenceChunk, Score


class RetrievalScorePayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_retrieval_mode_preserves_score_metadata(self):
        for mode, score_type, value in [
            ("dense", ScoreType.COSINE_DISTANCE, .234567),
            ("lexical", ScoreType.BM25_SCORE, 12.34567),
            ("hybrid", ScoreType.RRF_SCORE, .031234),
        ]:
            with self.subTest(mode=mode):
                evidence = EvidenceChunk(
                    Chunk("chunk-test", "test-doc", 0, "Evidence text", "Section", 2, "test"),
                    Score(mode, score_type, value, 1),
                )
                accumulated = []
                orchestrator = AgentOrchestrator(Mock(), Mock(return_value=[evidence]), Mock(return_value=[]))
                tool = next(t for t in orchestrator._build_tools(accumulated) if t.name == "search_knowledge")
                arguments = json.dumps({"query": "test", "k": 5, "document_id": None})
                raw = await tool.on_invoke_tool(ToolContext(
                    context=None, tool_name=tool.name, tool_call_id="test-call",
                    tool_arguments=arguments,
                ), arguments)
                result = json.loads(raw) if isinstance(raw, str) else raw
                chunk = result["chunks"][0]
                self.assertEqual(chunk["score"], evidence.score.to_dict())
                self.assertNotIn("cosine_distance", chunk)
                self.assertEqual(chunk["citation"], evidence.citation)
                self.assertEqual(accumulated, [evidence])
