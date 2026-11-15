"""Synthetic provider responses: no credentials or live network calls in tests."""
import json
import pytest
import requests


@pytest.fixture
def evidence_provider():
    class Provider:
        def respond(self, question, context):
            return {'answer': 'Review the p-value and the computed evidence before interpreting results.',
                    'evidence_ids': [card['id'] for card in context['evidence'][:8]]}
    return Provider()


@pytest.fixture
def mock_gemini(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'TEST-KEY-NOT-REAL')
    monkeypatch.setenv('GEMINI_MODEL', 'test-model')
    calls = []
    class Response:
        status_code = 200
        def json(self):
            return {'candidates': [{'content': {'parts': [{'text': json.dumps({
                'answer': 'Review the computed data quality.', 'evidence_ids': ['E1']})}]}}]}
    def post(*args, **kwargs):
        calls.append(kwargs['json'])
        return Response()
    monkeypatch.setattr(requests, 'post', post)
    return calls
