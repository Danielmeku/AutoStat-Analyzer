import numpy as np
import pandas as pd
import pytest
from scipy import stats
from modules.hypothesis_testing import paired_test, group_test, mann_whitney_test, fisher_test


def test_paired_t_and_wilcoxon_match_reference():
    frame = pd.DataFrame({'a': [3., 5, 8, 10, 7, np.nan], 'b': [1., 2, 3, 4, 5, 6]})
    a, b = frame.a.iloc[:5], frame.b.iloc[:5]
    result = paired_test(frame, 'a', 'b')
    reference = stats.ttest_rel(a, b)
    assert result['pvalue'] == pytest.approx(reference.pvalue)
    assert result['confidence_interval'][0] < result['mean_difference'] < result['confidence_interval'][1]
    assert result['excluded_rows'] == 1
    wilcoxon = paired_test(frame, 'a', 'b', 'wilcoxon')
    assert wilcoxon['pvalue'] == pytest.approx(stats.wilcoxon(a-b, method='auto').pvalue)
    assert wilcoxon['rank_biserial'] == 1
    with pytest.raises(ValueError, match='zero'):
        paired_test(pd.DataFrame({'a': [1,2,3], 'b': [1,2,3]}), 'a', 'b')


def test_omnibus_and_independent_rank_tests():
    groups = [[1.,2,4,5], [7.,8,10,12], [8.,10,11,14]]
    frame = pd.DataFrame({'value': np.concatenate(groups), 'group': np.repeat(['a','b','c'], 4)})
    anova = group_test(frame, 'value', 'group')
    assert anova['pvalue'] == pytest.approx(stats.f_oneway(*groups).pvalue)
    assert 0 <= anova['eta_squared'] <= 1
    kruskal = group_test(frame, 'value', 'group', 'kruskal')
    assert kruskal['pvalue'] == pytest.approx(stats.kruskal(*groups).pvalue)
    assert 0 <= kruskal['epsilon_squared'] <= 1
    mw = mann_whitney_test(frame, 'value', 'group', 'a', 'b')
    assert mw['pvalue'] == pytest.approx(stats.mannwhitneyu(groups[0], groups[1]).pvalue)
    assert mw['excluded_rows'] == 4
    assert mw['rank_biserial'] == -1
    with pytest.raises(ValueError):
        group_test(frame.head(1), 'value', 'group')


def test_fisher_sparse_table_and_infinite_odds():
    frame = pd.DataFrame({'a': ['x']*8+['y']*8, 'b': ['x']*8+['y']*8})
    result = fisher_test(frame, 'a', 'b')
    assert result['statistic'] is None  # Infinite odds are labelled rather than silently replaced.
    assert result['pvalue'] == pytest.approx(stats.fisher_exact([[8,0],[0,8]]).pvalue)
    with pytest.raises(ValueError):
        fisher_test(pd.DataFrame({'a': ['x','y','z'], 'b': ['x','x','y']}), 'a', 'b')
