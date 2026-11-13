"""LM Studio protocol errors, tested without stopping the user's server."""
import unittest
from unittest.mock import Mock, patch

import httpx
from knowledge_copilot.infrastructure.llm_client import LMStudioStatus, probe_lm_studio_readiness


class ModelReadinessTests(unittest.TestCase):
    @patch('knowledge_copilot.infrastructure.llm_client.httpx.get')
    def test_server_unreachable(self, get):
        get.side_effect = httpx.ConnectError('offline')
        self.assertEqual(probe_lm_studio_readiness().status, LMStudioStatus.SERVER_UNREACHABLE)

    @patch('knowledge_copilot.infrastructure.llm_client.httpx.get')
    def test_no_model_loaded(self, get):
        get.return_value = Mock(status_code=200, json=lambda: {'data': []})
        self.assertEqual(probe_lm_studio_readiness().status, LMStudioStatus.NO_MODEL_LOADED)

    @patch('knowledge_copilot.infrastructure.llm_client.httpx.get')
    def test_configured_model_requires_exact_match(self, get):
        get.return_value = Mock(status_code=200, json=lambda: {'data': [{'id': 'different-mistral-model'}]})
        self.assertEqual(probe_lm_studio_readiness(required_model='mistral').status, LMStudioStatus.MODEL_NOT_FOUND)

    @patch('knowledge_copilot.infrastructure.llm_client.httpx.post')
    @patch('knowledge_copilot.infrastructure.llm_client.httpx.get')
    def test_http_200_without_function_call_is_not_ready(self, get, post):
        get.return_value = Mock(status_code=200, json=lambda: {'data': [{'id': 'chat'}]})
        post.return_value = Mock(status_code=200, json=lambda: {'choices': [{'message': {'content': '4'}}]})
        probe = probe_lm_studio_readiness()
        self.assertEqual(probe.status, LMStudioStatus.TOOL_CALLING_UNSUPPORTED)
        self.assertFalse(probe.is_ready)


if __name__ == '__main__':
    unittest.main()
