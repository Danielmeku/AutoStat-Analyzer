import json
import re
import pandas as pd
import pytest
import requests
from ai.context_builder import build_context
from ai.chatbot import ask_assistant, insight_record
from ai.providers.offline_provider import OfflineProvider
from ai.providers.gemini_provider import GeminiProvider
from ai.response_validator import validate_response
from modules.data_loader import demo_data
from modules.hypothesis_testing import correlation_test


def context():
    frame = demo_data()
    return build_context(frame, {'test_result': correlation_test(frame, 'marketing_spend', 'sales')})


def test_context_excludes_raw_values_predictions_and_unapproved_state():
    frame = pd.DataFrame({'private': ['ROW-SECRET-ONE','ROW-SECRET-TWO'], 'x': [1,2]})
    ctx = build_context(frame, {'arbitrary': {'secret': 'STATE-SECRET'}, 'regression_plots': {'row': 'PRIVATE-RESIDUAL'}})
    serialized = json.dumps(ctx)
    assert 'ROW-SECRET' not in serialized
    assert 'STATE-SECRET' not in serialized
    assert 'PRIVATE-RESIDUAL' not in serialized
    assert len(ctx['evidence']) == 2


def test_offline_guide_and_grounded_insights():
    provider, ctx = OfflineProvider(), context()
    answer = ask_assistant(provider, 'Explain the p-value', ctx)
    assert 'p-value' in answer['answer']
    assert not re.search(r'\d', answer['answer'])
    assert all(card_id in {c['id'] for c in ctx['evidence']} for card_id in answer['evidence_ids'])
    saved = insight_record(provider, 'Offline guide', ctx)
    assert saved['provider'] == 'Offline guide'
    assert saved['evidence']
    empty = build_context(demo_data(), {})
    answer = ask_assistant(provider, 'Explain regression', empty)
    assert 'No completed analysis' in answer['answer']
    assert answer['evidence_ids'] == []


@pytest.mark.parametrize('response', [
    {'answer': 'The estimate is 999', 'evidence_ids':['E1']},
    {'answer': 'This proves a cause', 'evidence_ids':['E1']},
    {'answer': 'Look here', 'evidence_ids':['E999']},
    {'answer': '', 'evidence_ids':[]}, {'answer':'Fine','evidence_ids':'E1'}, 'wrong',
])
def test_response_validation_rejects_unsupported_claims_and_references(response):
    with pytest.raises(ValueError):
        validate_response(response, context())


def test_gemini_request_payload_and_secret_handling(monkeypatch):
    captured = {}
    class Response:
        status_code = 200
        def json(self):
            return {'candidates':[{'content':{'parts':[{'text':json.dumps({'answer':'Review the computed evidence.', 'evidence_ids':['E1']})}]}}]}
    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr(requests, 'post', post)
    result = ask_assistant(GeminiProvider('TEST-KEY-NOT-REAL', 'test-model'), 'Explain the data', context())
    assert 'computed' in result['answer']
    assert captured['url'] == 'https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent'
    assert captured['headers']['x-goog-api-key'] == 'TEST-KEY-NOT-REAL'
    assert captured['allow_redirects'] is False
    assert 'verify' not in captured  # Requests retains its TLS verification default.
    assert 'TEST-KEY' not in json.dumps(captured['json'])
    assert captured['timeout'] == (5, 45)


@pytest.mark.parametrize('status', [400,401,403,404,429,500,302])
def test_provider_errors_do_not_echo_secrets_or_remote_body(monkeypatch, status):
    class Response:
        status_code = status
        text = 'SECRET-BODY'
    monkeypatch.setattr(requests, 'post', lambda *a, **k: Response())
    with pytest.raises(ValueError) as error:
        GeminiProvider('SECRET-KEY', 'model').respond('hi', context())
    assert 'SECRET' not in str(error.value)


def test_provider_timeout_blocked_and_invalid_models(monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.Timeout('SECRET-KEY')
    monkeypatch.setattr(requests, 'post', timeout)
    with pytest.raises(ValueError) as error:
        GeminiProvider('SECRET-KEY', 'model').respond('hi', context())
    assert 'SECRET' not in str(error.value)
    with pytest.raises(ValueError):
        GeminiProvider('key', '../invalid-path')
    with pytest.raises(ValueError):
        GeminiProvider('', '')
    class Response:
        status_code = 200
        def json(self): return {'promptFeedback': {'blockReason':'SAFETY'}}
    monkeypatch.setattr(requests, 'post', lambda *a, **k: Response())
    with pytest.raises(ValueError, match='usable structured'):
        GeminiProvider('key', 'model').respond('hi', context())


def test_malformed_provider_parts_are_an_actionable_error(monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {'candidates':[{'content':{'parts':['malformed']}}]}
    monkeypatch.setattr(requests, 'post', lambda *a, **k: Response())
    with pytest.raises(ValueError, match='usable structured'):
        GeminiProvider('key', 'model').respond('Explain', context())
