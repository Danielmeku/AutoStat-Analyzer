import io
import zipfile
import numpy as np
import pandas as pd
from pypdf import PdfReader
from modules.data_loader import demo_data
from modules.hypothesis_testing import correlation_test
from modules.regression import fit_regression
from modules.machine_learning import compare_regressors
from reporting.pdf_generator import executive_pdf, detailed_pdf
from reporting.detailed_report import analysis_bundle


def _read(content):
    assert content.startswith(b'%PDF-')
    assert b'/FontFile2' in content  # Fonts are embedded for portable rendering.
    document = PdfReader(io.BytesIO(content))
    return document, '\n'.join(page.extract_text() for page in document.pages)


def test_descriptive_pdfs_are_readable_and_executive_is_one_page():
    frame = demo_data()
    document, text = _read(executive_pdf(frame, '<demo> & sales'))
    assert len(document.pages) == 1
    assert '181 observations' in text
    assert '<demo> & sales' in text
    assert 'No hypothesis test' in text
    _, text = _read(detailed_pdf(frame, 'demo'))
    assert 'Column profile' in text
    assert 'Descriptive statistics' in text
    assert 'Inferential OLS model' not in text


def test_reports_contain_only_completed_analyses_and_chosen_precision():
    frame = demo_data()
    test = correlation_test(frame, 'marketing_spend', 'sales')
    # Ensure tiny p-values exercise the same report formatting used for computed tests.
    test['pvalue'] = 5e-22
    regression, _ = fit_regression(frame, 'sales', ['marketing_spend'])
    comparison, _ = compare_regressors(frame, 'sales', ['marketing_spend'])
    analyses = dict(test_result=test, regression_result=regression, comparison_result=comparison)
    executive = executive_pdf(frame, 'sales', analyses, 2, 'Cleaned working copy')
    detailed = detailed_pdf(frame, 'sales', analyses, 2, 'Cleaned working copy')
    document, text = _read(executive)
    assert len(document.pages) == 1
    assert '5e-22' in text
    assert 'Cleaned working copy' in text
    _, text = _read(detailed)
    assert 'Completed hypothesis test' in text
    assert 'Inferential OLS model (in-sample)' in text
    assert 'Predictive model comparison (holdout)' in text
    assert '5e-22' in text
    assert 'standard error' in text
    with zipfile.ZipFile(io.BytesIO(analysis_bundle(frame, 'sales', analyses, {'executive.pdf': executive, 'detailed.pdf': detailed}))) as bundle:
        assert bundle.read('executive.pdf') == executive
        assert bundle.read('detailed.pdf') == detailed
        assert 'comparison_result.json' in bundle.namelist()


def test_reports_handle_categorical_empty_long_labels_and_pagination():
    for frame in [pd.DataFrame({'category': ['a', 'b', None]}),
                  pd.DataFrame({'empty': [np.nan]*4}),
                  pd.DataFrame({('long<&>label'+str(i))*15: np.arange(50) for i in range(50)})]:
        document, _ = _read(executive_pdf(frame, 'Long <&> name' * 40))
        assert len(document.pages) == 1
        document, text = _read(detailed_pdf(frame, 'Long <&> name' * 40))
        assert 'Data quality' in text
        if len(frame.columns) == 50:
            assert len(document.pages) > 1


def test_classification_in_all_analysis_pdfs_and_ten_class_tables():
    from modules.classification import compare_classifiers
    frame = demo_data()
    regression, _ = fit_regression(frame, 'sales', ['marketing_spend'])
    comparison, _ = compare_regressors(frame, 'sales', ['marketing_spend'])
    classification = compare_classifiers(frame, 'region', ['marketing_spend'])
    analyses = {'test_result': correlation_test(frame, 'marketing_spend', 'sales'),
                'regression_result': regression, 'comparison_result': comparison,
                'classification_result': classification}
    document, text = _read(executive_pdf(frame, 'Synthetic sales', analyses))
    assert len(document.pages) == 1
    assert 'macro-F1' in text
    _, text = _read(detailed_pdf(frame, 'Synthetic sales', analyses))
    assert 'Classification model comparison (stratified holdout)' in ' '.join(text.split())
    assert 'Confusion matrices' in text
    assert 'Majority baseline' in text
    frame = pd.DataFrame({'x': np.arange(100), 'target': np.tile(np.arange(10), 10)})
    result = compare_classifiers(frame, 'target', ['x'])
    _, text = _read(detailed_pdf(frame, 'Ten classes', {'classification_result': result}))
    assert 'C10' in text
