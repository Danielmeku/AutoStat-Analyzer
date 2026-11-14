import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from modules.hypothesis_testing import welch_test, correlation_test, chi_square_test
from modules.regression import fit_regression
from reporting.detailed_report import analysis_bundle


def test_welch_matches_reference_and_interval():
    a, b = [1., 2., 3., 5.], [5., 7., 8., 12.]
    frame = pd.DataFrame({'value': a+b+[np.nan], 'group': ['a']*4+['b']*5})
    result = welch_test(frame, 'value', 'group', 'a', 'b')
    expected = stats.ttest_ind(a, b, equal_var=False)
    assert result['pvalue'] == pytest.approx(expected.pvalue)
    assert result['mean_difference'] == pytest.approx(np.mean(a)-np.mean(b))
    assert result['confidence_interval'][1] < 0
    assert result['excluded_rows'] == 1
    with pytest.raises(ValueError):
        welch_test(frame, 'value', 'group', 'a', 'a')


def test_correlations_and_chi_square():
    data = pd.DataFrame({'a': [1., 2., 4., 8., np.nan], 'b': [3., 4., 6., 5., 7.]})
    result = correlation_test(data, 'a', 'b')
    assert result['pvalue'] == pytest.approx(stats.pearsonr([1,2,4,8], [3,4,6,5]).pvalue)
    assert result['observations'] == 4
    result = correlation_test(data, 'a', 'b', 'spearman')
    assert result['statistic'] == pytest.approx(0.8)
    categories = pd.DataFrame({'x': ['a']*30+['b']*30, 'y': ['c']*20+['d']*10+['c']*10+['d']*20})
    result = chi_square_test(categories, 'x', 'y')
    assert result['statistic'] == pytest.approx(20/3)
    assert result['cramers_v'] == pytest.approx(1/3)
    assert result['sparse_cells'] == 0
    with pytest.raises(ValueError):
        correlation_test(pd.DataFrame({'a': [1]*4, 'b': [1,2,3,4]}), 'a', 'b')


def test_ols_recovers_coefficients_and_guards_degenerate_design():
    rng = np.random.default_rng(42)
    x = rng.normal(size=200)
    z = rng.normal(size=200)
    frame = pd.DataFrame({'x': x, 'z': z, 'y': 4 + 3*x - 2*z + rng.normal(0, 0.1, 200)})
    frame.loc[0, 'y'] = np.nan
    result, plots = fit_regression(frame, 'y', ['x', 'z'])
    estimates = {r['term']: r['estimate'] for r in result['coefficients']}
    assert estimates['x'] == pytest.approx(3, abs=.03)
    assert estimates['z'] == pytest.approx(-2, abs=.03)
    assert result['r_squared'] > .99
    assert result['excluded_rows'] == 1
    assert abs(plots.Residual.mean()) < 1e-10
    assert 0 <= result['diagnostics']['breusch_pagan_pvalue'] <= 1
    frame['duplicate_x'] = frame.x * 2
    for predictors in [[], ['y'], ['x', 'duplicate_x']]:
        with pytest.raises(ValueError):
            fit_regression(frame, 'y', predictors)
    with pytest.raises(ValueError):
        fit_regression(frame.head(5), 'y', ['x'])


def test_results_are_in_export_with_exact_selections():
    frame = pd.DataFrame({'x': range(20), 'y': np.arange(20)**2})
    result, _ = fit_regression(frame, 'y', ['x'])
    with zipfile.ZipFile(io.BytesIO(analysis_bundle(frame, 'test', {'regression_result': result}))) as bundle:
        exported = json.loads(bundle.read('regression_result.json'))
        assert exported['target'] == 'y'
        assert exported['observations'] == 20
        assert 'Additional analyses' in bundle.read('summary.md').decode()
