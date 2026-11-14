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
