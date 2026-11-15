import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
from modules.data_loader import demo_data
from modules.data_types import convert_column_types, prepare_working_copy, column_type_schema, dataset_signature
from modules.data_profiler import profile_data
from modules.exploratory_analysis import numeric_summary, correlation_matrix
from modules.regression import fit_regression
from modules.machine_learning import compare_regressors
from modules.classification import compare_classifiers
from reporting.detailed_report import analysis_bundle


def test_numeric_factor_preserves_values_and_missingness_and_changes_analysis_role():
    frame = pd.DataFrame({'rating': [1., 2., np.nan, 3.], 'y': [4, 6, 7, 9]})
    before = frame.copy(deep=True)
    converted = convert_column_types(frame, {'rating': 'Categorical (factor)'})
    assert isinstance(converted.rating.dtype, pd.CategoricalDtype)
    np.testing.assert_allclose(converted.rating.astype(float), frame.rating, equal_nan=True)
    assert converted.rating.isna().equals(frame.rating.isna())
    assert not converted.rating.cat.ordered
    assert profile_data(converted).set_index('Column').loc['rating', 'Suggested role'] == 'Categorical'
    assert 'rating' not in numeric_summary(converted).index
    assert 'rating' not in correlation_matrix(converted).columns
    assert dataset_signature(converted) != dataset_signature(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_typed_copy_survives_cleaning_and_original_values_can_be_restored():
    original = demo_data()
    assignments = {'satisfaction': 'Categorical (factor)'}
    converted = prepare_working_copy(original, assignments, True, True)
    assert len(converted) == 177
    assert isinstance(converted.satisfaction.dtype, pd.CategoricalDtype)
    pd.testing.assert_frame_equal(prepare_working_copy(original, {}), original)
    source = pd.DataFrame({'code': ['001', '002', None]})
    numeric = convert_column_types(source, {'code': 'Numeric'})
    assert numeric.code.iloc[0] == 1
    restored_text = convert_column_types(source, {'code': 'Text'})
    assert restored_text.code.iloc[0] == '001'
    assert restored_text.code.isna().equals(source.code.isna())


@pytest.mark.parametrize('values,treatment', [
    (['1', 'not-a-number'], 'Numeric'), (['1', ''], 'Numeric'),
    (['1', 'inf'], 'Numeric'), (['true', 'perhaps'], 'Boolean'),
    (['2024-01-01', 'not-a-date'], 'Date/time'), ([1, 2], 'Date/time'),
])
def test_invalid_conversions_do_not_mutate_or_silently_coerce(values, treatment):
    original = pd.DataFrame({'x': values, 'y': [3, 4]})
    before = original.copy(deep=True)
    with pytest.raises(ValueError, match='Cannot treat'):
        convert_column_types(original, {'y': 'Categorical (factor)', 'x': treatment})
    pd.testing.assert_frame_equal(original, before)


def test_boolean_date_numeric_and_text_types_preserve_missing_data():
    frame = pd.DataFrame({'flags': ['true', '0', None], 'date': ['2024-01-01', '2024-02-01', None], 'amount': ['2.5', '3', None]})
    converted = convert_column_types(frame, {'flags': 'Boolean', 'date': 'Date/time', 'amount': 'Numeric'})
    assert str(converted['flags'].dtype) == 'boolean'
    assert converted['flags'].iloc[:2].tolist() == [True, False]
    assert str(converted.date.dtype) == 'datetime64[ns, UTC]'
    assert converted.amount.iloc[:2].tolist() == [2.5, 3.0]
    assert converted.isna().equals(frame.isna())
    assert profile_data(converted).set_index('Column').loc['date', 'Suggested role'] == 'Date'
    assert convert_column_types(pd.DataFrame({'flag': [True, False]}), {'flag': 'Numeric'}).flag.tolist() == [1, 0]


def test_factor_flows_through_inference_prediction_and_classification():
    frame = prepare_working_copy(demo_data(), {'satisfaction': 'Categorical (factor)'})
    regression, _ = fit_regression(frame, 'sales', ['marketing_spend', 'satisfaction'])
    assert 'satisfaction' in regression['reference_categories']
    assert len(regression['coefficients']) == 6
    comparison, _ = compare_regressors(frame, 'sales', ['marketing_spend', 'satisfaction'], cv_folds=3)
    assert comparison['categorical_predictors'] == ['satisfaction']
    assert comparison['numeric_predictors'] == ['marketing_spend']
    classification = compare_classifiers(frame, 'region', ['marketing_spend', 'satisfaction'])
    assert classification['categorical_predictors'] == ['satisfaction']
    target_classification = compare_classifiers(frame, 'satisfaction', ['marketing_spend'])
    assert len(target_classification['classes']) == 5


def test_export_schema_records_factor_levels_and_numeric_summaries_exclude_factor():
    frame = prepare_working_copy(demo_data(), {'satisfaction': 'Categorical (factor)'})
    schema = column_type_schema(frame)
    factor = next(column for column in schema['columns'] if column['column'] == 'satisfaction')
    assert factor['levels'] == [1, 2, 3, 4, 5]
    assert factor['dtype'] == 'category'
    assert not factor['ordered']
    with zipfile.ZipFile(io.BytesIO(analysis_bundle(frame, 'typed demo'))) as archive:
        assert json.loads(archive.read('column_types.json')) == schema
        summary = pd.read_csv(archive.open('numeric_summary.csv'), index_col=0)
        assert 'satisfaction' not in summary.index
        profile = pd.read_csv(archive.open('column_profile.csv')).set_index('Column')
        assert profile.loc['satisfaction', 'Type'] == 'category'


def test_schema_changes_invalidate_signatures_even_when_value_hashes_match():
    frame = pd.DataFrame({'x': [1, 2, 1]})
    factor = convert_column_types(frame, {'x': 'Categorical (factor)'})
    np.testing.assert_array_equal(pd.util.hash_pandas_object(frame), pd.util.hash_pandas_object(factor))
    assert dataset_signature(frame) != dataset_signature(factor)
    reordered = factor.copy()
    reordered.x = reordered.x.cat.reorder_categories([2, 1])
    assert dataset_signature(factor) != dataset_signature(reordered)
    with pytest.raises(ValueError, match='Unknown column'):
        convert_column_types(frame, {'absent': 'Numeric'})
