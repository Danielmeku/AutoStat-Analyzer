import numpy as np
import pandas as pd
import pytest
from modules.unsupervised import explore_structure


def test_pca_cluster_aggregates_and_reproducibility_without_mutation():
    rng = np.random.default_rng(12)
    x = np.concatenate([rng.normal(-4,.3,30), rng.normal(4,.3,30)])
    frame = pd.DataFrame({'x': x, 'cluster': x*2+rng.normal(0,.1,60)})
    frame.loc[0, 'x'] = np.nan
    original = frame.copy(deep=True)
    result, plots = explore_structure(frame, ['x','cluster'], clusters=2)
    again, other = explore_structure(frame, ['x','cluster'], clusters=2)
    assert result == again
    pd.testing.assert_frame_equal(plots, other)
    pd.testing.assert_frame_equal(frame, original)
    assert sum(v['ratio'] for v in result['explained_variance']) == pytest.approx(1)
    assert sum(v['rows'] for v in result['cluster_sizes']) == 60
    assert result['silhouette'] > .8
    assert result['imputed_cells'] == 1
    assert len(result['cluster_centers']) == 4
    assert all(isinstance(row['cluster'], str) for row in result['cluster_centers'])
    assert 'PC1' not in result  # Row projections remain separate from aggregate exports.


@pytest.mark.parametrize('features,k', [(['x'],2), (['x','x'],2), (['x','text'],2), (['x','y'],9)])
def test_invalid_structure_requests(features, k):
    frame = pd.DataFrame({'x': range(20), 'y': range(20), 'text': ['a']*20})
    with pytest.raises(ValueError):
        explore_structure(frame, features, k)
