import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from pypdf import PdfReader
from modules.machine_learning import compare_regressors
from modules.classification import compare_classifiers
from reporting.pdf_generator import detailed_pdf
from reporting.detailed_report import analysis_bundle


def mixed_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=90)
    group = np.where(x > 0, 'upper', 'lower').astype(object)
    group[0] = None
    return pd.DataFrame({'x': x, 'group': group, 'y': 3*x + rng.normal(0, .1, 90), 'label': np.where(x > 0, 'yes', 'no')})


def test_unknown_categories_and_fold_local_fitting(monkeypatch):
    frame = mixed_data()
    train, holdout = train_test_split(np.arange(len(frame)), test_size=.2, random_state=42)
    frame.loc[holdout, 'group'] = 'holdout-only'
    frame.loc[holdout, 'x'] = 1e6
    original = frame.copy(deep=True)
    encoder_fit, imputer_fit = OneHotEncoder.fit, SimpleImputer.fit
    learned, numeric_sizes = [], []
    def encoder(self, X, y=None):
        fitted = encoder_fit(self, X, y)
        learned.extend(self.categories_[0].tolist())
        return fitted
    def imputer(self, X, y=None):
        if 'x' in X.columns:
            assert X.x.max() < 1e6  # Neither final fit nor any fold sees holdout features.
            numeric_sizes.append(len(X))
        return imputer_fit(self, X, y)
    monkeypatch.setattr(OneHotEncoder, 'fit', encoder)
    monkeypatch.setattr(SimpleImputer, 'fit', imputer)
    result, predictions = compare_regressors(frame, 'y', ['x', 'group'], cv_folds=3)
    pd.testing.assert_frame_equal(frame, original)
    assert 'holdout-only' not in learned
    assert len(numeric_sizes) == 12  # Three models, three fold fits plus final training fit.
    assert set(numeric_sizes) == {48, 72}
    assert result['categorical_predictors'] == ['group']
    assert np.isfinite(predictions['Linear regression']).all()
    for model in result['cross_validation']['models']:
        scores = [row['rmse'] for row in model['fold_results']]
        assert model['metrics']['rmse']['mean'] == pytest.approx(np.mean(scores))
        assert model['metrics']['rmse']['std'] == pytest.approx(np.std(scores, ddof=1))


def test_cv_scores_do_not_use_holdout_targets():
    frame = mixed_data()
    first, _ = compare_regressors(frame, 'y', ['x', 'group'], cv_folds=3)
    _, holdout = train_test_split(np.arange(len(frame)), test_size=.2, random_state=42)
    frame.loc[holdout, 'y'] += 1e6
    second, _ = compare_regressors(frame, 'y', ['x', 'group'], cv_folds=3)
    for a, b in zip(first['cross_validation']['models'], second['cross_validation']['models']):
        assert a['metrics']['rmse']['mean'] == pytest.approx(b['metrics']['rmse']['mean'])
    assert second['models'][0]['rmse'] > first['models'][0]['rmse'] * 1000


def test_mixed_and_categorical_only_classification_and_cv_export():
    frame = mixed_data()
    result = compare_classifiers(frame, 'label', ['x', 'group'], cv_folds=3)
    assert result['cross_validation']['folds'] == 3
    assert result['cross_validation']['training_rows'] == result['train_rows']
    assert len(result['cross_validation']['models']) == 3
    assert result['models'][1]['macro_f1'] > .9
    categorical_only = compare_classifiers(frame, 'label', ['group'])
    assert categorical_only['numeric_predictors'] == []
    with zipfile.ZipFile(io.BytesIO(analysis_bundle(frame, 'mixed', {'classification_result': result}))) as bundle:
        data = json.loads(bundle.read('classification_result.json'))
        assert data['cross_validation']['folds'] == 3
    pdf = PdfReader(io.BytesIO(detailed_pdf(frame, 'mixed', {'classification_result': result})))
    text = ' '.join(' '.join(page.extract_text() for page in pdf.pages).split())
    assert 'fold results' in text
    assert 'macro f1' in text


def test_cv_and_cardinality_guards():
    frame = mixed_data()
    with pytest.raises(ValueError, match='3 folds'):
        compare_regressors(frame, 'y', ['x'], cv_folds=4)
    frame['identifier'] = [str(i) for i in range(len(frame))]
    with pytest.raises(ValueError, match='30 levels'):
        compare_regressors(frame, 'y', ['identifier'])
    rare = pd.DataFrame({'x': range(35), 'target': ['rare']*5+['common']*30})
    with pytest.raises(ValueError, match='training rows in every class'):
        compare_classifiers(rare, 'target', ['x'], cv_folds=5)
