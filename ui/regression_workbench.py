"""A concise specification → inference → diagnostics → remedies → prediction workflow."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from modules.regression import fit_regression, predict_regression, compare_nested_regressions, regression_sensitivity
from reporting.formatting import format_number, display_table, display_result
from reporting.results import json_result
from ui.presentation import plot_chart, result_details


def _table(rows, digits):
    st.dataframe(display_table(pd.DataFrame(rows), digits), hide_index=True, width='stretch')


def render_regression_workbench(frame):
    digits = st.session_state.get('significant_figures', 3)
    st.subheader('Regression workspace')
    numbers = list(frame.select_dtypes(include='number'))
    if not numbers or len(frame.columns) < 2:
        st.info('Need a numeric target and at least one predictor.')
        return
    controls = st.columns([1, 2])
    target = controls[0].selectbox('Regression target', numbers, index=min(1, len(numbers)-1))
    choices = [c for c in [*numbers, *frame.select_dtypes(exclude='number').columns] if c != target]
    predictors = controls[1].multiselect('Predictors (up to 20)', choices, default=choices[:1])
    with st.expander('Model specification'):
        top = st.columns(3)
        covariance = top[0].selectbox('Coefficient inference', ['Classical', 'HC3'])
        response = top[1].selectbox('Response transformation', ['None', 'Log', 'Square root', 'Box-Cox'])
        predictor_transform = top[2].selectbox('Numeric predictor transformation', ['None', 'Log', 'Square root', 'Reciprocal'])
        middle = st.columns(3)
        degree = middle[0].selectbox('Model terms', [1, 2], format_func=lambda d: 'Linear' if d == 1 else 'Quadratic (numeric terms)')
        weight = middle[1].selectbox('Inverse-variance weight column', ['None', *[c for c in numbers if c not in [target, *predictors]]])
        confidence = middle[2].selectbox('Confidence level', [0.90, 0.95, 0.99], index=1, format_func=lambda x: f'{x:.0%}')
        bottom = st.columns(3, vertical_alignment='bottom')
        variance_order = bottom[0].selectbox('Variance ordering predictor', ['None', *[c for c in predictors if c in numbers]])
        interactions = bottom[1].checkbox('Include pairwise interactions')
        extended = bottom[2].checkbox('Extended diagnostic tests')
    with st.popover('Model help'):
        st.write('Categorical predictors use reference coding; the reference is listed in results. Quadratic terms and pairwise interactions retain main effects. Complete-case rows are used jointly.')
        st.write('HC3 adjusts coefficient uncertainty for heteroscedasticity; it does not fix dependence, bias, the mean model, or individual prediction intervals. WLS requires positive, externally justified inverse-variance weights.')
        st.write('Extended diagnostics add White, Lilliefors and Anderson–Darling tests where the sample/design permits. Selecting a variance ordering predictor requests a two-sided Goldfeld–Quandt test on ordered groups, omitting the middle fifth. Diagnostic tests are unadjusted.')
        st.write('Box-Cox selects a response power using the regression profile likelihood within −3 to 3. This is exploratory selection. Transformed fits report metrics and coefficients on their modeled scale; compare residuals rather than AIC across response scales.')
    if st.button('Fit regression', type='primary'):
        for key in ['regression_result', 'regression_plots', 'regression_fit', 'regression_new_predictions', 'regression_nested_result', 'regression_prediction_inputs']:
            st.session_state.pop(key, None)
        try:
            result, plots = fit_regression(frame, target, predictors, alpha=1-confidence,
                                          covariance=covariance, response_transform=response,
                                          predictor_transform=predictor_transform, degree=degree,
                                          interactions=interactions, weight_column=None if weight == 'None' else weight,
                                          extended_diagnostics=extended, variance_order=None if variance_order == 'None' else variance_order)
            st.session_state['regression_fit'] = plots.attrs.pop('fit')
            st.session_state['regression_result'], st.session_state['regression_plots'] = result, plots
        except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
            st.error(str(exc))
    if not (result := st.session_state.get('regression_result')):
        return
    all_plots = st.session_state['regression_plots']
    fit = st.session_state['regression_fit']
    st.markdown('#### Last fitted model')
    st.write(f"{result['method']} · {result['target']} · {result['covariance']} inference · {1-result['alpha']:.0%} confidence")
    if result['response_transformation']['name'] != 'None':
        st.info(f"Modeled response scale: {result['response_transformation']['name']}" + (f" (λ = {format_number(result['response_transformation']['lambda'], digits)})" if 'lambda' in result['response_transformation'] else '') + '. Metrics below use this scale.')
    if result['reference_categories']:
        with st.expander('Reference categories'):
            _table([{'Predictor': c, 'Reference': v} for c, v in result['reference_categories'].items()], digits)
    columns = st.columns(4)
    for column, (label, value) in zip(columns, [('In-sample R²', result['r_squared']), ('Adjusted R²', result['adjusted_r_squared']),
                                               ('In-sample RMSE', result['metrics']['rmse']), ('Complete observations', result['observations'])]):
        column.metric(label, value if label == 'Complete observations' else format_number(value, digits))
    inference, diagnostics, remedies, prediction = st.tabs(['Inference', 'Diagnostics', 'Remedial review', 'Predict new cases'])
    with inference:
        _table(result['coefficients'], digits)
        left, right = st.columns(2)
        with left:
            st.markdown('#### ANOVA decomposition')
            _table(result['anova'], digits)
        with right:
            st.markdown('#### Overall model test')
            _table([result['overall_test']], digits)
            _table([{'AIC': result['aic'], 'BIC': result['bic'], 'Residual standard error': result['residual_standard_error'],
                     'Excluded rows': result['excluded_rows']}], digits)
            with st.expander('Analytic leave-one-out validation'):
                _table([result['leave_one_out']], digits)
        with st.expander('Compare nested linear models'):
            reduced = st.multiselect('Reduced model predictors', result['predictors'], default=result['predictors'][:1])
            st.write('This comparison refits untransformed classical linear OLS models on identical complete rows, separately from any advanced specification above.')
            if st.button('Compare nested models'):
                try:
                    st.session_state['regression_nested_result'] = compare_nested_regressions(frame, result['target'], reduced, result['predictors'])
                    result['nested_comparison'] = st.session_state['regression_nested_result']
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))
            if nested := st.session_state.get('regression_nested_result'):
                _table([nested], digits)
        result_details(display_result(result, digits))
    with diagnostics:
        plots = all_plots.sample(min(5000, len(all_plots)), random_state=42)
        left, right = st.columns(2)
        with left:
            figure = px.scatter(plots, x='Fitted', y='Residual', title='Residuals vs fitted')
            figure.add_hline(y=0, line_dash='dot', line_color='#F36DAE')
            plot_chart(figure, width='stretch')
        with right:
            plot_chart(px.scatter(plots, x='Observed', y='Fitted', title='Observed vs fitted'), width='stretch')
        residuals = all_plots['Residual'].to_numpy()
        # Quantile thinning preserves tails without plotting every uploaded observation.
        ordered = np.sort(residuals)
        indices = np.unique(np.linspace(0, len(ordered)-1, min(2000, len(ordered))).astype(int))
        qq = pd.DataFrame({'Normal quantile': stats.norm.ppf((indices+.5)/len(ordered)), 'Residual quantile': ordered[indices]})
        left, right = st.columns(2)
        with left:
            figure = px.scatter(qq, x='Normal quantile', y='Residual quantile', title='Normal Q–Q plot')
            slope = (np.quantile(residuals, .75)-np.quantile(residuals, .25))/(stats.norm.ppf(.75)-stats.norm.ppf(.25))
            intercept = np.median(residuals)
            bounds = [qq['Normal quantile'].min(), qq['Normal quantile'].max()]
            figure.add_scatter(x=bounds, y=[intercept+slope*v for v in bounds], mode='lines', name='Quartile reference', line=dict(color='#F36DAE', dash='dot'))
            plot_chart(figure, width='stretch')
        with right:
            plot_chart(px.scatter(plots.sort_values('Row position'), x='Row position', y='Residual', title='Residuals in data order'), width='stretch')
        if 'Studentized residual' in plots:
            scale = plots.assign(**{'√|Studentized residual|': np.sqrt(np.abs(plots['Studentized residual']))})
            left, right = st.columns(2)
            with left:
                plot_chart(px.scatter(scale, x='Fitted', y='√|Studentized residual|', title='Scale–location'), width='stretch')
            with right:
                plot_chart(px.scatter(plots, x='Leverage', y='Cook distance', color='Review flag', title='Leverage & influence'), width='stretch')
        test_rows = [{'Check': key.replace('_', ' ').title(), 'Value': value} for key, value in result['diagnostics'].items() if isinstance(value, (int, float))]
        _table(test_rows, digits)
        with st.expander('Collinearity & lack of fit'):
            _table(result['vif'], digits)
            lof = result['diagnostics']['lack_of_fit']
            if lof['available']:
                _table([lof], digits)
            else:
                st.info(lof['reason'])
        with st.expander('Inspect observation diagnostics'):
            st.write('Row positions refer to the active dataset. Thresholds flag review candidates; observations are never removed automatically.')
            _table(all_plots.loc[all_plots.get('Review flag', pd.Series(True, index=all_plots.index))].head(200), digits)
            st.download_button('Download observation diagnostics', all_plots.to_csv(index=False).encode(), 'autostat-regression-diagnostics.csv', 'text/csv')
        with st.popover('Diagnostic help'):
            st.write(result['diagnostics']['note'])
            st.write('Q–Q departures suggest non-normal errors; a funnel suggests unequal variance; curvature suggests an incomplete mean model. VIF above 5, Cook distance above 4/n and leverage above 2p/n are review heuristics. Deleted studentized residuals, DFFITS and DFBETAS identify unusual and influential observations; none is a deletion rule.')
    with remedies:
        for recommendation in result['recommendations']:
            st.write('• '+recommendation)
        with st.expander('Case-exclusion sensitivity analysis'):
            positions_text = st.text_input('Active row positions to exclude in sensitivity fit', placeholder='e.g. 12, 35')
            st.write('The primary model and active data remain unchanged. Exclude cases only for a justified comparison, not merely because a threshold flagged them.')
            if st.button('Run sensitivity comparison'):
                result.pop('sensitivity', None)
                try:
                    positions = [int(value.strip()) for value in positions_text.split(',') if value.strip()]
                    result['sensitivity'] = regression_sensitivity(frame, result, positions)
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))
            if sensitivity := result.get('sensitivity'):
                _table(sensitivity['coefficients'], digits)
                st.write(sensitivity['note'])
        with st.expander('Interpretation limits'):
            st.write(result['scope'])
            st.write('Prediction intervals assume a correct variance model and independent future errors. WLS requires externally justified weights. Dependence may require GLS or a time/group model; that model is not automatically inferred from row order.')
    with prediction:
        inputs = {c: [fit['levels'][c][0] if c in fit['levels'] else sum(fit['ranges'][c])/2] for c in fit['predictors']}
        edited = st.data_editor(pd.DataFrame(inputs), num_rows='dynamic', hide_index=True, key='regression_prediction_inputs',
                                column_config={c: st.column_config.SelectboxColumn(c, options=levels) for c, levels in fit['levels'].items()})
        if st.button('Predict new observations'):
            st.session_state.pop('regression_new_predictions', None)
            try:
                st.session_state['regression_new_predictions'] = predict_regression(fit, edited)
            except (ValueError, KeyError, TypeError) as exc:
                st.error(str(exc))
        if (predicted := st.session_state.get('regression_new_predictions')) is not None:
            _table(predicted, digits)
            st.download_button('Download predictions', predicted.to_csv(index=False).encode(), 'autostat-new-predictions.csv', 'text/csv')
        with st.popover('Prediction help'):
            st.write('The mean confidence interval describes uncertainty in the conditional mean. An individual prediction interval also includes future observation noise. Extrapolation flags values outside fitted numeric ranges; combinations can still be unsupported inside those ranges.')
            st.write('Inverse-transformed fitted values are not bias-corrected conditional means. Inverse intervals require valid transformation domains. WLS omits individual prediction intervals because a future inverse-variance weight is unknown. HC3 does not make individual prediction intervals robust.')
    st.download_button('Download regression result', json_result(result), 'autostat-regression.json', 'application/json')
