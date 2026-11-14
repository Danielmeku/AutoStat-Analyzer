import io
import zipfile
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from modules.classification import compare_classifiers
from reporting.detailed_report import analysis_bundle


def dataset():
    rng = np.random.default_rng(4)
    x = rng.normal(size=180)
    return pd.DataFrame({'x': x, 'target': np.where(x > 0, 'positive', 'negative')})


def test_classification_metrics_stratification_and_reference_baseline():
    frame = dataset()
    frame.loc[0, 'target'] = None
    frame.loc[1, 'x'] = np.nan
    original = frame.copy(deep=True)
    result = compare_classifiers(frame, 'target', ['x'])
    pd.testing.assert_frame_equal(frame, original)
    data = frame.dropna(subset=['target'])
    train, test = train_test_split(np.arange(len(data)), test_size=.2, stratify=data.target, random_state=42)
    majority = data.target.iloc[train].value_counts().index[0]
    actual = data.target.iloc[test]
    baseline = result['models'][0]
    assert baseline['accuracy'] == pytest.approx((actual == majority).mean())
    assert baseline['balanced_accuracy'] == pytest.approx(.5)
    assert result['models'][1]['macro_f1'] > .9
    assert result['excluded_target_rows'] == 1
    assert result['missing_predictor_cells'] == 1
    for name, details in result['evaluation'].items():
        matrix = np.array(details['confusion_matrix'])
        assert matrix.sum() == len(test)
        assert matrix.trace() / matrix.sum() == pytest.approx(next(m['accuracy'] for m in result['models'] if m['model'] == name))
        assert [m['support'] for m in details['per_class']] == matrix.sum(axis=1).tolist()
    assert all(r['train'] > 0 and r['holdout'] > 0 for r in result['class_distribution'])
    again = compare_classifiers(frame, 'target', ['x'])
    assert result == again


def test_preprocessing_uses_training_only(monkeypatch):
    frame = dataset()
    train, test = train_test_split(np.arange(len(frame)), test_size=.2, stratify=frame.target, random_state=42)
    frame.loc[train[:3], 'x'] = np.nan
    frame.loc[test, 'x'] = 1e6
    median = frame.x.iloc[train].median()
    imputed = frame.x.iloc[train].fillna(median)
    fit_imputer, fit_scaler = SimpleImputer.fit, StandardScaler.fit
    medians, means = [], []
    def imputer(self, X, y=None):
        result = fit_imputer(self, X, y)
        medians.append((len(X), self.statistics_[0]))
        return result
    def scaler(self, X, y=None, sample_weight=None):
        result = fit_scaler(self, X, y, sample_weight=sample_weight)
        means.append((len(X), self.mean_[0]))
        return result
    monkeypatch.setattr(SimpleImputer, 'fit', imputer)
    monkeypatch.setattr(StandardScaler, 'fit', scaler)
    compare_classifiers(frame, 'target', ['x'])
    assert len(medians) == 3
    assert all(n == len(train) and m == median for n, m in medians)
    assert means[0][0] == len(train)
    assert means[0][1] == pytest.approx(imputed.mean())


def test_multiclass_numeric_labels_export_and_missing_predictor_rejection():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({'x': rng.normal(size=90), 'target': np.repeat([1, 2, 10], 30)})
    result = compare_classifiers(frame, 'target', ['x'], .3)
    assert result['classes'] == ['1', '10', '2']
    assert all(np.array(v['confusion_matrix']).shape == (3, 3) for v in result['evaluation'].values())
    with zipfile.ZipFile(io.BytesIO(analysis_bundle(frame, 'classes', {'classification_result': result}))) as bundle:
        assert 'classification_result.json' in bundle.namelist()
    frame['x'] = np.nan
    with pytest.raises(ValueError, match='observed training'):
        compare_classifiers(frame, 'target', ['x'])


@pytest.mark.parametrize('mode', ['small', 'rare', 'one_class', 'too_many', 'target_feature', 'no_features', 'infinite'])
def test_invalid_classification_inputs(mode):
    frame = dataset()
    predictors = ['x']
    if mode == 'small': frame = frame.head(20)
    if mode == 'rare': frame.loc[0, 'target'] = 'rare'
    if mode == 'one_class': frame['target'] = 'only'
    if mode == 'too_many': frame['target'] = np.arange(len(frame))
    if mode == 'target_feature': predictors = ['target']
    if mode == 'no_features': predictors = []
    if mode == 'infinite': frame.loc[0, 'x'] = np.inf
    with pytest.raises(ValueError):
        compare_classifiers(frame, 'target', predictors)


def test_nonconvergence_is_reported_not_silently_accepted(monkeypatch):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    def no_convergence(self, X, y):
        warnings.warn('iteration limit reached', ConvergenceWarning)
        return self
    monkeypatch.setattr(LogisticRegression, 'fit', no_convergence)
    with pytest.raises(ValueError, match='did not converge'):
        compare_classifiers(dataset(), 'target', ['x'])
