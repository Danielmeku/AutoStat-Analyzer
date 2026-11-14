import pandas as pd
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_demo_clean_reset_and_categorical_flow():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == '181'
    app.checkbox[0].check()
    app.checkbox[1].check()
    next(b for b in app.button if b.label == 'Apply cleaning').click().run()
    assert not app.exception
    assert app.metric[0].value == '177'
    next(b for b in app.button if b.label == 'Restore original data').click().run()
    assert app.metric[0].value == '181'
    next(r for r in app.radio if r.label == 'Correlation method').set_value('spearman').run()
    assert not app.exception
    app.session_state['active'] = pd.DataFrame({'group': ['a', 'b', 'a']})
    app.run()
    assert not app.exception
    assert any('No numeric columns' in item.value for item in app.info)
    next(r for r in app.radio if r.label == 'Dataset source').set_value('Upload CSV').run()
    assert not app.exception
    assert any('Choose a CSV' in item.value for item in app.info)


def test_run_statistical_analysis_and_invalidate_after_cleaning():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    next(r for r in app.selectbox if r.label == 'Test').set_value('Correlation test').run()
    next(b for b in app.button if b.label == 'Run test').click().run()
    assert not app.exception
    assert app.session_state['test_result']['observations'] > 0
    next(b for b in app.button if b.label == 'Fit regression').click().run()
    assert not app.exception
    assert app.session_state['regression_result']['observations'] > 0
    app.checkbox[0].check()
    next(b for b in app.button if b.label == 'Apply cleaning').click().run()
    assert not app.exception
    assert 'test_result' not in app.session_state
    assert 'regression_result' not in app.session_state


def test_precision_comparison_and_invalidation():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    next(s for s in app.selectbox if s.label == 'Significant figures').set_value(2).run()
    next(b for b in app.button if b.label == 'Compare models').click().run()
    assert not app.exception
    result = app.session_state['comparison_result']
    assert len(result['models']) == 3
    assert result['train_rows'] + result['test_rows'] == 178
    next(s for s in app.selectbox if s.label == 'Significant figures').set_value(3).run()
    assert app.session_state['comparison_result'] == result
    app.checkbox[0].check()
    next(b for b in app.button if b.label == 'Apply cleaning').click().run()
    assert not app.exception
    assert 'comparison_result' not in app.session_state


def test_pdf_generation_and_invalidation_on_precision_and_data():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    next(b for b in app.button if b.label == 'Generate PDF reports').click().run()
    assert not app.exception
    assert app.session_state['pdf_reports']['executive.pdf'].startswith(b'%PDF-')
    next(s for s in app.selectbox if s.label == 'Significant figures').set_value(2).run()
    assert 'pdf_reports' not in app.session_state
    next(b for b in app.button if b.label == 'Generate PDF reports').click().run()
    next(b for b in app.button if b.label == 'Fit regression').click().run()
    assert not app.exception
    assert 'pdf_reports' not in app.session_state
    next(b for b in app.button if b.label == 'Generate PDF reports').click().run()
    app.checkbox[0].check()
    next(b for b in app.button if b.label == 'Apply cleaning').click().run()
    assert not app.exception
    assert 'pdf_reports' not in app.session_state


def test_classification_flow_and_report_invalidation():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=30).run()
    next(b for b in app.button if b.label == 'Generate PDF reports').click().run()
    next(b for b in app.button if b.label == 'Compare classifiers').click().run()
    assert not app.exception
    result = app.session_state['classification_result']
    assert len(result['models']) == 3
    assert result['train_rows'] + result['test_rows'] == 181
    assert 'pdf_reports' not in app.session_state
    next(s for s in app.selectbox if s.label == 'Significant figures').set_value(2).run()
    assert app.session_state['classification_result'] == result
    next(b for b in app.button if b.label == 'Generate PDF reports').click().run()
    assert not app.exception
    app.checkbox[0].check()
    next(b for b in app.button if b.label == 'Apply cleaning').click().run()
    assert 'classification_result' not in app.session_state
    assert 'pdf_reports' not in app.session_state


def test_mixed_predictor_cross_validation_ui():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=60).run()
    next(s for s in app.multiselect if s.label == 'Prediction features').set_value(['marketing_spend', 'region'])
    next(s for s in app.selectbox if s.label == 'Regression cross-validation').set_value(3)
    next(b for b in app.button if b.label == 'Compare models').click().run()
    assert not app.exception
    result = app.session_state['comparison_result']
    assert result['categorical_predictors'] == ['region']
    assert result['cross_validation']['folds'] == 3
    next(s for s in app.selectbox if s.label == 'Classification target').set_value('satisfaction').run()
    next(s for s in app.multiselect if s.label == 'Classification features').set_value(['marketing_spend', 'region'])
    next(s for s in app.selectbox if s.label == 'Classification cross-validation').set_value(3)
    next(b for b in app.button if b.label == 'Compare classifiers').click().run()
    assert not app.exception
    assert app.session_state['classification_result']['cross_validation']['folds'] == 3
    next(b for b in app.button if b.label == 'Generate PDF reports').click().run()
    assert not app.exception
    assert app.session_state['pdf_reports']['detailed.pdf'].startswith(b'%PDF-')


def test_new_tests_structure_offline_insights_and_chat():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=60).run()
    next(s for s in app.selectbox if s.label == 'Test').set_value('One-way ANOVA').run()
    next(s for s in app.selectbox if s.label == 'Group-test grouping column').set_value('region')
    next(b for b in app.button if b.label == 'Run test').click().run()
    assert not app.exception
    assert app.session_state['test_result']['method'] == 'One-way ANOVA'
    next(b for b in app.button if b.label == 'Explore structure').click().run()
    assert not app.exception
    assert app.session_state['exploration_result']['clusters'] == 3
    next(b for b in app.button if b.label == 'Generate insights').click().run()
    assert not app.exception
    assert app.session_state['insight_result']['provider'] == 'Offline guide'
    next(t for t in app.text_input if t.label == 'Your statistical question').set_value('Explain PCA')
    next(b for b in app.button if b.label == 'Ask assistant').click().run()
    assert not app.exception
    assert app.session_state['assistant_messages'][-1]['provider'] == 'Offline guide'
    next(b for b in app.button if b.label == 'Generate PDF reports').click().run()
    assert not app.exception
    assert app.session_state['pdf_reports']['detailed.pdf'].startswith(b'%PDF-')
    app.checkbox[0].check()
    next(b for b in app.button if b.label == 'Apply cleaning').click().run()
    assert 'exploration_result' not in app.session_state
    assert 'insight_result' not in app.session_state
    assert 'assistant_messages' not in app.session_state


def test_gemini_requires_current_summary_consent(monkeypatch):
    import json
    import requests
    monkeypatch.setenv('GEMINI_API_KEY', 'TEST-KEY-NOT-REAL')
    monkeypatch.setenv('GEMINI_MODEL', 'test-model')
    calls = []
    class Response:
        status_code = 200
        def json(self):
            return {'candidates':[{'content':{'parts':[{'text':json.dumps({'answer':'Review the computed data quality.', 'evidence_ids':['E1']})}]}}]}
    def post(*args, **kwargs):
        calls.append(kwargs['json'])
        return Response()
    monkeypatch.setattr(requests, 'post', post)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=60).run()
    next(r for r in app.radio if r.label == 'Assistant mode').set_value('Gemini').run()
    assert not calls
    assert next(b for b in app.button if b.label == 'Generate insights').disabled
    next(c for c in app.checkbox if c.label.startswith('I approve sending')).check().run()
    next(b for b in app.button if b.label == 'Generate insights').click().run()
    assert not app.exception
    assert len(calls) == 1
    assert app.session_state['insight_result']['provider'] == 'Gemini'
    next(s for s in app.selectbox if s.label == 'Significant figures').set_value(2).run()
    assert not next(c for c in app.checkbox if c.label.startswith('I approve sending')).value
    assert 'insight_result' not in app.session_state
    assert next(b for b in app.button if b.label == 'Generate insights').disabled
    assert len(calls) == 1


def test_unconfigured_gemini_has_actionable_message(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_MODEL', raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py', default_timeout=60).run()
    next(r for r in app.radio if r.label == 'Assistant mode').set_value('Gemini').run()
    assert not app.exception
    assert any('GEMINI_API_KEY' in item.value for item in app.info)
    assert next(b for b in app.button if b.label == 'Generate insights').disabled
