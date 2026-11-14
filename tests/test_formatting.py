import json
import numpy as np
import pandas as pd
import pytest
from reporting.formatting import format_number, display_table, display_result
from reporting.results import json_result


@pytest.mark.parametrize('value,digits,expected', [
    (5e-22, 3, '5e-22'), (0.000123456, 3, '0.000123'),
    (0.012345, 2, '0.012'), (123.456, 3, '123'),
    (123.456, 2, '1.2e+02'), (-0.000001234, 2, '-1.2e-06'),
    (0, 3, '0'), (np.nan, 3, '—'), (np.inf, 3, '—'),
])
def test_significant_figures(value, digits, expected):
    assert format_number(value, digits) == expected


def test_pvalues_and_full_precision_preserved():
    result = {'pvalue': 5e-22, 'jarque_bera_pvalue': 0., 'estimate': 1.23456789, 'observations': 12345}
    shown = display_result(result, 2)
    assert shown['pvalue'] == '5e-22'
    assert shown['jarque_bera_pvalue'] == 'Below numerical precision'
    assert shown['observations'] == 12345
    assert json.loads(json_result(result))['estimate'] == 1.23456789
    table = pd.DataFrame({'pvalue': [5e-22, 0.], 'estimate': [1.23456789, np.nan]})
    shown = display_table(table, 3)
    assert shown.pvalue.tolist() == ['5e-22', 'Below numerical precision']
    assert table.estimate.iloc[0] == 1.23456789
