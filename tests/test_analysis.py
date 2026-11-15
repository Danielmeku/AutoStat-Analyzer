import io
import zipfile
import numpy as np
import pandas as pd
import pytest
from modules.data_loader import load_csv, demo_data
from modules.data_validator import validate_data
from modules.data_cleaner import clean_data
from modules.data_profiler import profile_data
from modules.exploratory_analysis import numeric_summary, correlation_matrix, outlier_summary
from reporting.detailed_report import analysis_bundle


def test_load_csv_and_preserve_values():
    frame = load_csv(b'\xef\xbb\xbfx,y\n1,a\n2,b\n')
    assert frame.shape == (2, 2)
    assert frame.x.tolist() == [1, 2]
    assert load_csv(b'x;y\n1;2', ';').shape == (1, 2)


@pytest.mark.parametrize('content', [b'', b'x,x\n1,2', b'x,\n1,2', b'x,y\n1,2,3', b'x,y\n1', b'x,y\n', b'\xff', b'x\ninf'])
def test_invalid_csv_is_rejected(content):
    with pytest.raises(ValueError):
        load_csv(content)


def test_quality_cleaning_and_original_preserved():
    frame = pd.DataFrame({'x': [1., 1., np.nan], 'y': ['a', 'a', 'b']})
    original = frame.copy(deep=True)
    quality = validate_data(frame)
    assert quality['missing_cells'] == 1
    assert quality['duplicate_rows'] == 1
    assert clean_data(frame, True, True).shape == (1, 2)
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError):
        clean_data(pd.DataFrame({'x': [np.nan]}), drop_missing=True)


def test_profiles_support_mixed_empty_and_date_columns():
    frame = pd.DataFrame({'value': [1, 2, 3], 'date': ['2024-01-01'] * 3, 'empty': [None] * 3})
    profile = profile_data(frame).set_index('Column')
    assert profile.loc['date', 'Suggested role'] == 'Date'
    assert profile.loc['empty', 'Missing'] == 3
    assert profile.loc['value', 'Suggested role'] == 'Numeric'


def test_statistics_match_known_results():
    frame = pd.DataFrame({'x': [1, 2, 3, 4, 100], 'y': [2, 4, 6, 8, 200]})
    assert numeric_summary(frame).loc['x', 'median'] == 3
    assert correlation_matrix(frame).loc['x', 'y'] == pytest.approx(1)
    assert outlier_summary(frame).iloc[0]['IQR flags'] == 1
    assert numeric_summary(pd.DataFrame({'a': ['one']})).empty


def test_bundle_contents_are_real_results():
    frame = demo_data()
    bundle = analysis_bundle(frame, 'demo')
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert len(archive.namelist()) == 6
        assert '181 rows' in archive.read('summary.md').decode()
        profile = pd.read_csv(archive.open('column_profile.csv'))
        assert profile.set_index('Column').loc['sales', 'Missing'] == 3


def test_spearman_and_limits(monkeypatch):
    import modules.data_loader as loader
    frame = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [2, 5, 10, 20]})
    assert correlation_matrix(frame, 'spearman').loc['x', 'y'] == pytest.approx(1)
    monkeypatch.setattr(loader, 'MAX_ROWS', 2)
    with pytest.raises(ValueError):
        loader.load_csv(b'x\n1\n2\n3')
    monkeypatch.setattr(loader, 'MAX_BYTES', 2)
    with pytest.raises(ValueError):
        loader.load_csv(b'x\n1\n')
