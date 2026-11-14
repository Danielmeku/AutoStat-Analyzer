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
