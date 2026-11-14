import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from modules.machine_learning import compare_regressors


def test_holdout_is_reproducible_baseline_uses_train_only_and_source_preserved():
    rng = np.random.default_rng(5)
    x = rng.normal(size=150)
    frame = pd.DataFrame({'x': x, 'y': 10 + 4*x + rng.normal(0, .1, 150)})
    frame.loc[0, 'x'] = np.nan
    frame.loc[1, 'y'] = np.nan
    original = frame.copy(deep=True)
    result, predictions = compare_regressors(frame, 'y', ['x'])
    again, again_predictions = compare_regressors(frame, 'y', ['x'])
    assert {k: v for k, v in result.items() if k != 'models'} == {k: v for k, v in again.items() if k != 'models'}
    for first, second in zip(result['models'], again['models']):
        assert first['model'] == second['model']
        for metric in ['rmse', 'mae', 'r_squared']:
            assert first[metric] == pytest.approx(second[metric], abs=1e-12)
    pd.testing.assert_frame_equal(predictions, again_predictions)
    pd.testing.assert_frame_equal(frame, original)
    data = frame.dropna(subset=['y'])
    train, test = train_test_split(np.arange(len(data)), test_size=.2, random_state=42)
    assert len(set(train) & set(test)) == 0
    assert predictions['Mean baseline'].to_numpy() == pytest.approx(np.repeat(data.y.iloc[train].mean(), len(test)))
    assert predictions.Observed.tolist() == data.y.iloc[test].tolist()
    scores = {entry['model']: entry for entry in result['models']}
    assert scores['Linear regression']['rmse'] < scores['Mean baseline']['rmse']
    assert result['excluded_target_rows'] == 1
    assert result['missing_predictor_cells'] == 1
    assert result['train_rows'] + result['test_rows'] == 149


def test_imputation_does_not_learn_from_holdout_features(monkeypatch):
    from sklearn.impute import SimpleImputer
    frame = pd.DataFrame({'x': np.arange(100, dtype=float), 'y': np.arange(100, dtype=float)*2})
    train, test = train_test_split(np.arange(100), test_size=.2, random_state=42)
    # Holdout-only extremes must not shift the median imputation of missing training rows.
    frame.loc[train[:5], 'x'] = np.nan
    frame.loc[test, 'x'] = 1e6
    expected_median = frame.x.iloc[train].median()
    captured = []
    original_fit = SimpleImputer.fit
    def recording_fit(self, X, y=None):
        fitted = original_fit(self, X, y)
        captured.append((len(X), self.statistics_[0]))
        return fitted
    monkeypatch.setattr(SimpleImputer, 'fit', recording_fit)
    result, _ = compare_regressors(frame, 'y', ['x'])
    assert all(np.isfinite(m['rmse']) for m in result['models'])
    assert len(captured) == 3
    assert all(n == len(train) and median == expected_median for n, median in captured)
    frame.loc[train, 'x'] = np.nan
    with pytest.raises(ValueError, match='Training predictors'):
        compare_regressors(frame, 'y', ['x'])


@pytest.mark.parametrize('target,predictors,fraction', [('y', [], .2), ('y', ['y'], .2), ('y', ['x'], .5)])
def test_invalid_comparisons_rejected(target, predictors, fraction):
    with pytest.raises(ValueError):
        compare_regressors(pd.DataFrame({'x': range(50), 'y': range(50)}), target, predictors, fraction)
