import io
import json
import zipfile
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats, optimize, special
from modules.regression import fit_regression, predict_regression, compare_nested_regressions
from modules.hypothesis_testing import partial_correlation_test
from modules.machine_learning import compare_regressors
from ai.context_builder import build_context
from reporting.detailed_report import analysis_bundle
from reporting.pdf_generator import detailed_pdf
from pypdf import PdfReader


def data(seed=11, n=100):
    rng = np.random.default_rng(seed)
    x, z = rng.normal(size=(2, n))
    return pd.DataFrame({'x': x, 'z': z, 'y': 2+3*x-1.5*z+rng.normal(size=n), 'w': np.linspace(.4, 2, n)})


def test_inference_anova_influence_and_press_match_reference():
    frame = data()
    result, plots = fit_regression(frame, 'y', ['x', 'z'], alpha=.1)
    reference = sm.OLS(frame.y, sm.add_constant(frame[['x', 'z']])).fit()
    np.testing.assert_allclose([c['estimate'] for c in result['coefficients']], reference.params)
    np.testing.assert_allclose([[c['ci_lower'], c['ci_upper']] for c in result['coefficients']], reference.conf_int(alpha=.1))
    assert result['overall_test']['pvalue'] == pytest.approx(reference.f_pvalue)
    assert result['anova'][0]['sum_squares']+result['anova'][1]['sum_squares'] == pytest.approx(result['anova'][2]['sum_squares'])
    influence = reference.get_influence()
    np.testing.assert_allclose(plots['Leverage'], influence.hat_matrix_diag)
    np.testing.assert_allclose(plots['Cook distance'], influence.cooks_distance[0])
    np.testing.assert_allclose(plots['DFFITS'], influence.dffits[0], atol=1e-12)
    np.testing.assert_allclose(plots['Deleted studentized residual'], influence.resid_studentized_external, atol=1e-12)
    np.testing.assert_allclose(plots['Max absolute DFBETAS'], np.max(abs(influence.dfbetas), axis=1), atol=1e-12)
    assert result['leave_one_out']['press'] == pytest.approx(np.sum((reference.resid/(1-influence.hat_matrix_diag))**2))
    assert result['vif'][0]['vif'] >= 1


def test_hc3_and_wls_inference_prediction_match_reference():
    frame = data()
    result, plots = fit_regression(frame, 'y', ['x', 'z'], covariance='HC3', weight_column='w')
    reference = sm.WLS(frame.y, sm.add_constant(frame[['x','z']]), weights=frame.w).fit().get_robustcov_results(cov_type='HC3', use_t=True)
    np.testing.assert_allclose([c['standard_error'] for c in result['coefficients']], reference.bse)
    assert result['overall_test']['pvalue'] == pytest.approx(reference.f_pvalue)
    assert result['influence'] is None
    predicted = predict_regression(plots.attrs['fit'], pd.DataFrame({'x':[0.], 'z':[0.]}))
    expected = reference.get_prediction([[1,0,0]]).summary_frame()
    assert predicted['Fitted mean'][0] == pytest.approx(expected['mean'][0])
    assert 'Prediction lower' not in predicted
    assert not result['diagnostics']['lack_of_fit']['available']
    frame.loc[0,'w'] = 0
    with pytest.raises(ValueError, match='positive'):
        fit_regression(frame, 'y', ['x'], weight_column='w')


def test_prediction_intervals_and_missing_cases():
    frame = data()
    frame.loc[5, 'z'] = np.nan
    result, plots = fit_regression(frame, 'y', ['x', 'z'])
    assert result['excluded_rows'] == 1
    assert 6 not in plots['Row position'].to_list()
    new = pd.DataFrame({'x':[0.,100.], 'z':[0.,0.]})
    predicted = predict_regression(plots.attrs['fit'], new)
    expected = plots.attrs['fit']['model'].get_prediction([[1,0,0],[1,100,0]]).summary_frame()
    np.testing.assert_allclose(predicted['Mean CI lower'], expected.mean_ci_lower)
    np.testing.assert_allclose(predicted['Prediction upper'], expected.obs_ci_upper)
    assert predicted.Extrapolation.to_list() == [False,True]
    assert (predicted['Prediction upper']-predicted['Prediction lower'] > predicted['Mean CI upper']-predicted['Mean CI lower']).all()
    with pytest.raises(ValueError, match='complete'):
        predict_regression(plots.attrs['fit'], pd.DataFrame({'x':[np.nan],'z':[0.]}))


def test_categorical_quadratic_interactions_and_unknown_prediction():
    frame = data()
    frame['group'] = np.where(np.arange(len(frame))%2, 'PRIVATE-CATEGORY', 'REFERENCE-SECRET')
    frame['y'] += 2*frame.x**2+4*(frame.group == 'REFERENCE-SECRET')
    result, plots = fit_regression(frame, 'y', ['x', 'group'], degree=2, interactions=True)
    assert len(result['coefficients']) == 5
    assert result['reference_categories']['group'] == 'PRIVATE-CATEGORY'
    pred = predict_regression(plots.attrs['fit'], pd.DataFrame({'x':[0.,0.], 'group':['REFERENCE-SECRET','PRIVATE-CATEGORY']}))
    assert len(pred) == 2
    with pytest.raises(ValueError, match='Unknown category'):
        predict_regression(plots.attrs['fit'], pd.DataFrame({'x':[0.], 'group':['new']}))
    context = json.dumps(build_context(frame, {'regression_result': result}))
    assert 'PRIVATE-CATEGORY' not in context and 'REFERENCE-SECRET' not in context
    assert 'Row position' not in context


def test_response_transforms_and_boxcox_regression_likelihood():
    frame = data()
    frame['y'] = np.exp(.3*frame.x+.1*frame.z+2+np.random.default_rng(5).normal(0,.1,len(frame)))
    result, plots = fit_regression(frame,'y',['x','z'],response_transform='Log')
    reference = sm.OLS(np.log(frame.y), sm.add_constant(frame[['x','z']])).fit()
    np.testing.assert_allclose([c['estimate'] for c in result['coefficients']], reference.params)
    prediction = predict_regression(plots.attrs['fit'], pd.DataFrame({'x':[0.], 'z':[0.]}))
    assert prediction['Fitted mean (inverse scale)'][0] == pytest.approx(np.exp(reference.params.iloc[0]))
    result, _ = fit_regression(frame,'y',['x','z'],response_transform='Box-Cox')
    design = sm.add_constant(frame[['x','z']]).to_numpy()
    def profile(power):
        y = special.boxcox(frame.y.to_numpy(),power)
        e = y-design@np.linalg.lstsq(design,y,rcond=None)[0]
        return len(y)/2*np.log(np.mean(e**2))-(power-1)*np.log(frame.y).sum()
    optimum=optimize.minimize_scalar(profile,bounds=(-3,3),method='bounded').x
    assert result['response_transformation']['lambda'] == pytest.approx(optimum,abs=1e-5)
    frame.loc[0,'y']=0
    with pytest.raises(ValueError,match='positive'):
        fit_regression(frame,'y',['x'],response_transform='Log')


def test_pure_error_lack_of_fit_and_nested_comparison():
    rng=np.random.default_rng(99)
    x=np.repeat(np.arange(6),6)
    frame=pd.DataFrame({'x':x,'z':rng.normal(size=len(x)),'y':x**2+rng.normal(0,.2,len(x))})
    result,_=fit_regression(frame,'y',['x'])
    assert result['diagnostics']['lack_of_fit']['available']
    assert result['diagnostics']['lack_of_fit']['pvalue'] < .001
    frame.loc[0,'z']=np.nan
    result=compare_nested_regressions(frame,'y',['x'],['x','z'])
    shared=frame.dropna()
    full=sm.OLS(shared.y,sm.add_constant(shared[['x','z']])).fit()
    reduced=sm.OLS(shared.y,sm.add_constant(shared[['x']])).fit()
    assert result['pvalue']==pytest.approx(full.compare_f_test(reduced)[1])
    assert result['observations']==len(shared)


def test_partial_correlation_matches_adjusted_coefficient_test():
    frame=data()
    result=partial_correlation_test(frame,'x','y',['z'])
    reference=sm.OLS(frame.y,sm.add_constant(frame[['x','z']])).fit()
    assert result['pvalue']==pytest.approx(reference.pvalues['x'])
    assert result['statistic']==pytest.approx(reference.tvalues['x'])
    with pytest.raises(ValueError):
        partial_correlation_test(frame,'x','y',['x'])


def test_regularized_candidates_share_holdout_and_fold_local_scaling():
    frame=data()
    result,predictions=compare_regressors(frame,'y',['x','z'],cv_folds=3,regularization=['Ridge','Lasso'],penalty=.2)
    assert len(result['models'])==5
    assert result['cross_validation']['folds']==3
    assert len(predictions)==result['test_rows']
    assert set(predictions.columns)=={'Observed','Mean baseline','Linear regression','Random forest','Ridge','Lasso'}
    original,_=compare_regressors(frame,'y',['x','z'])
    for actual, expected in zip(result['models'][:3], original['models']):
        assert actual['model'] == expected['model']
        for metric in ['rmse', 'mae', 'r_squared']:
            assert actual[metric] == pytest.approx(expected[metric], rel=1e-12)


def test_extended_exports_and_report_include_specification_not_row_diagnostics():
    frame=data(n=30)
    result,_=fit_regression(frame,'y',['x','z'],covariance='HC3')
    encoded=json.loads(json.dumps(result))
    assert 'vif' in encoded and 'anova' in encoded and 'recommendations' in encoded
    with zipfile.ZipFile(io.BytesIO(analysis_bundle(frame,'demo',{'regression_result':result}))) as archive:
        exported=json.loads(archive.read('regression_result.json'))
        assert exported['covariance']=='HC3'
        assert 'regression_plots' not in exported
        assert set(exported['influence']) == {'flagged_rows', 'thresholds', 'note'}
    pdf=detailed_pdf(frame,'demo',{'regression_result':result})
    text='\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages)
    assert 'HC3' in text and 'anova' in text.lower() and 'recommendations' in text.lower()


def test_extended_diagnostics_and_variance_tests_match_references():
    from statsmodels.stats.diagnostic import het_white, het_goldfeldquandt, normal_ad, lilliefors
    from modules.hypothesis_testing import variance_equality_test
    frame=data()
    result,_=fit_regression(frame,'y',['x','z'],extended_diagnostics=True,variance_order='x')
    reference=sm.OLS(frame.y,sm.add_constant(frame[['x','z']])).fit()
    assert result['diagnostics']['white_pvalue']==pytest.approx(het_white(reference.resid,reference.model.exog)[1])
    assert result['diagnostics']['anderson_darling_normal_pvalue']==pytest.approx(normal_ad(reference.resid)[1])
    assert result['diagnostics']['lilliefors_normal_pvalue']==pytest.approx(lilliefors(reference.resid)[1])
    order=np.argsort(frame.x,kind='stable')
    assert result['diagnostics']['goldfeld_quandt_pvalue']==pytest.approx(het_goldfeldquandt(reference.model.endog[order],reference.model.exog[order],split=40,drop=20,alternative='two-sided')[1])
    frame['group']=np.where(frame.x>0,'a','b')
    arrays=[part.y.to_numpy() for _,part in frame.groupby('group')]
    for method, expected in [('levene',stats.levene(*arrays,center='median')),('bartlett',stats.bartlett(*arrays))]:
        result=variance_equality_test(frame,'y','group',method)
        assert result['pvalue']==pytest.approx(expected.pvalue)


def test_rank_guards_predictor_domains_and_no_input_mutation():
    frame=data()
    before=frame.copy(deep=True)
    fit_regression(frame,'y',['x','z'])
    pd.testing.assert_frame_equal(frame,before)
    frame['redundant']=2*frame.x
    with pytest.raises(ValueError,match='dependent'):
        fit_regression(frame,'y',['x','redundant'])
    with pytest.raises(ValueError,match='positive'):
        fit_regression(frame,'y',['x'],predictor_transform='Log')
    with pytest.raises(ValueError,match='numeric predictor'):
        fit_regression(frame,'y',['x'],variance_order='z')
    large = pd.DataFrame(np.random.default_rng(2).normal(size=(100, 12)), columns=[f'x{i}' for i in range(12)])
    large['y'] = np.random.default_rng(4).normal(size=100)
    with pytest.raises(ValueError, match='60 terms'):
        fit_regression(large, 'y', list(large.columns[:-1]), interactions=True)
    frame.loc[0,'x']=np.inf
    with pytest.raises(ValueError,match='finite'):
        fit_regression(frame,'y',['x'])


def test_sensitivity_comparison_is_explicit_and_preserves_primary_data():
    from modules.regression import regression_sensitivity
    frame=data()
    frame.loc[0,'y']+=50
    before=frame.copy(deep=True)
    original,_=fit_regression(frame,'y',['x','z'])
    comparison=regression_sensitivity(frame,original,[1])
    reference,_=fit_regression(frame.iloc[1:],'y',['x','z'])
    assert comparison['sensitivity_observations']==99
    assert comparison['sensitivity_r_squared']==pytest.approx(reference['r_squared'])
    pd.testing.assert_frame_equal(frame,before)
    assert original['observations']==100
    with pytest.raises(ValueError):
        regression_sensitivity(frame,original,[0])
