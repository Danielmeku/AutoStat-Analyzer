"""Regression specification, inference, diagnostics and explicit new-case prediction."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats, optimize, special
from modules.diagnostics import regression_diagnostics, collinearity, lack_of_fit, recommendations
from modules.model_evaluation import regression_metrics
from modules.regression_design import make_design, transform_values, inverse_response


def fit_regression(frame, target, predictors, alpha=0.05, *, covariance='Classical',
                   response_transform='None', predictor_transform='None', degree=1,
                   interactions=False, weight_column=None, extended_diagnostics=False, variance_order=None):
    predictors = list(predictors)
    if not 0 < alpha < 1:
        raise ValueError('Significance level must be between zero and one.')
    if covariance not in {'Classical', 'HC3'}:
        raise ValueError('Choose Classical or HC3 covariance.')
    if not predictors or len(predictors) > 20 or target in predictors or len(set(predictors)) != len(predictors):
        raise ValueError('Select 1–20 distinct predictors, excluding the target.')
    columns = [target, *predictors]
    if weight_column:
        if weight_column in columns:
            raise ValueError('Weights must use a separate column, not the target or a predictor.')
        columns.append(weight_column)
    if any(c not in frame for c in columns) or not pd.api.types.is_numeric_dtype(frame[target]):
        raise ValueError('Choose a numeric target and existing predictors.')
    data = frame[columns].dropna().copy()
    if len(data) < 10:
        raise ValueError('Need at least 10 complete observations.')
    if any(data[c].nunique() < 2 for c in [target, *predictors]):
        raise ValueError('All model columns must vary across observations.')
    y_original = data[target].to_numpy(dtype=float)
    if not np.isfinite(y_original).all():
        raise ValueError('The response must be finite.')
    design, terms, levels = make_design(data, predictors, degree=degree, interactions=interactions,
                                       predictor_transform=predictor_transform)
    n, k = design.shape
    if n < k+2:
        raise ValueError('Need at least two residual degrees of freedom after expanding model terms.')
    if np.linalg.matrix_rank(design) < k:
        raise ValueError('Predictors are linearly dependent. Remove redundant columns or terms.')
    weights = None
    if weight_column:
        if not pd.api.types.is_numeric_dtype(data[weight_column]):
            raise ValueError('Weights must be numeric inverse variances.')
        weights = data[weight_column].to_numpy(dtype=float)
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('WLS weights must be finite and strictly positive inverse variances.')
    transformation = {'name': response_transform}
    if response_transform == 'Box-Cox':
        if weights is not None:
            raise ValueError('Box-Cox selection is supported for unweighted regression only.')
        if np.any(y_original <= 0):
            raise ValueError('Box-Cox requires a strictly positive response.')
        log_y = np.log(y_original)
        # Profile the REGRESSION likelihood, not marginal normality of the response.
        def negative_profile(power):
            transformed = special.boxcox(y_original, power)
            errors = transformed-design @ np.linalg.lstsq(design, transformed, rcond=None)[0]
            variance = np.mean(errors**2)
            return float(n/2*np.log(max(variance, np.finfo(float).tiny))-(power-1)*log_y.sum())
        selected = optimize.minimize_scalar(negative_profile, bounds=(-3, 3), method='bounded')
        if not selected.success:
            raise ValueError('Box-Cox likelihood optimization did not converge.')
        transformation.update({'lambda': float(selected.x), 'search_bounds': [-3, 3],
                               'selection': 'In-sample regression profile likelihood; inference conditional on selected lambda.'})
        y = special.boxcox(y_original, selected.x)
    else:
        if response_transform not in {'None', 'Log', 'Square root'}:
            raise ValueError('Unknown response transformation.')
        y = transform_values(y_original, response_transform)
    classical = (sm.WLS(y, design, weights=weights) if weights is not None else sm.OLS(y, design)).fit()
    model = classical if covariance == 'Classical' else classical.get_robustcov_results(cov_type='HC3', use_t=True)
    intervals = model.conf_int(alpha=alpha)
    coefficients = [dict(term=label, estimate=float(model.params[i]), standard_error=float(model.bse[i]),
                         statistic=float(model.tvalues[i]), pvalue=float(model.pvalues[i]),
                         ci_lower=float(intervals[i, 0]), ci_upper=float(intervals[i, 1])) for i, label in enumerate(terms)]
    for i, coefficient in enumerate(coefficients):
        if i and weights is None:
            t_squared = float(classical.tvalues[i]**2)
            coefficient['partial_r_squared'] = t_squared/(t_squared+classical.df_resid)
    if variance_order is not None and variance_order not in predictors:
        raise ValueError('Variance ordering must use a selected numeric predictor.')
    if variance_order is not None and variance_order in levels:
        raise ValueError('Variance ordering requires a numeric predictor.')
    diagnostics = regression_diagnostics(classical, extended=extended_diagnostics, variance_order=None if variance_order is None else data[variance_order].to_numpy(dtype=float))
    if weights is not None:
        diagnostics['note'] += ' WLS checks use whitened residuals; ordinary OLS influence and pure-error tests are not applied.'
    vif = collinearity(design, terms)
    influence_summary = None
    validation = {'press': None, 'predicted_r_squared': None, 'note': 'Analytic leave-one-out diagnostics are available for ordinary least squares; they do not undo model selection.'}
    plots = pd.DataFrame({'Row position': np.flatnonzero(frame[columns].notna().all(axis=1))+1,
                          'Fitted': classical.fittedvalues, 'Residual': classical.resid, 'Observed': y})
    if weights is None and np.max(np.abs(classical.resid)) >= 1e-10:
        influence = classical.get_influence()
        leverage = influence.hat_matrix_diag
        if np.all(leverage < 1-1e-12):
            press = float(np.sum((classical.resid/(1-leverage))**2))
            validation.update(press=press, predicted_r_squared=float(1-press/classical.centered_tss))
        studentized = influence.resid_studentized_internal
        # Compute deleted-residual measures algebraically; avoid n full refits on large uploads.
        denominator = classical.ssr-classical.resid**2/np.maximum(1-leverage, np.finfo(float).eps)
        deleted_variance = denominator/(classical.df_resid-1)
        with np.errstate(invalid='ignore', divide='ignore'):
            external = classical.resid/np.sqrt(deleted_variance*(1-leverage))
            dffits = external*np.sqrt(leverage/(1-leverage))
        cook = influence.cooks_distance[0]
        delta = (design @ classical.normalized_cov_params) * (classical.resid/(1-leverage))[:, None]
        with np.errstate(invalid='ignore', divide='ignore'):
            dfbetas = delta / np.sqrt(deleted_variance[:, None]*np.diag(classical.normalized_cov_params))
        max_dfbetas = np.max(np.abs(dfbetas), axis=1)
        thresholds = {'leverage': 2*k/n, 'cooks_distance': 4/n, 'dffits': 2*np.sqrt(k/n), 'dfbetas': 2/np.sqrt(n), 'studentized_residual': 3.}
        flagged = ((leverage > thresholds['leverage']) | (cook > thresholds['cooks_distance']) |
                   (np.abs(external) > 3) | (np.abs(dffits) > thresholds['dffits']) | (max_dfbetas > thresholds['dfbetas']))
        for name, values in [('Leverage', leverage), ('Studentized residual', studentized), ('Deleted studentized residual', external),
                             ('Cook distance', cook), ('DFFITS', dffits), ('Max absolute DFBETAS', max_dfbetas), ('Review flag', flagged)]:
            plots[name] = values
        influence_summary = {'flagged_rows': int(flagged.sum()), 'thresholds': thresholds,
                             'note': 'Heuristic review thresholds, not deletion rules. Row positions refer to the active dataset; row diagnostics remain local unless explicitly downloaded.'}
    if weights is None:
        diagnostics['lack_of_fit'] = lack_of_fit(data, predictors, y, classical.ssr, k)
    else:
        diagnostics['lack_of_fit'] = {'available': False, 'reason': 'Classical pure-error lack-of-fit test is unavailable for WLS.'}
    anova = [dict(source='Regression', df=int(classical.df_model), sum_squares=float(classical.ess), mean_square=float(classical.ess/classical.df_model)),
             dict(source='Residual', df=int(classical.df_resid), sum_squares=float(classical.ssr), mean_square=float(classical.mse_resid)),
             dict(source='Total', df=n-1, sum_squares=float(classical.centered_tss), mean_square=None)]
    result = dict(method='Weighted least squares' if weights is not None else 'Ordinary least squares', target=str(target), predictors=predictors,
                  observations=n, excluded_rows=len(frame)-n, alpha=alpha, covariance=covariance,
                  response_transformation=transformation, predictor_transformation=predictor_transform,
                  degree=degree, interactions=bool(interactions), weight_column=weight_column,
                  extended_diagnostics=bool(extended_diagnostics), variance_order=variance_order,
                  reference_categories={c: values[0] for c, values in levels.items()},
                  r_squared=float(classical.rsquared), adjusted_r_squared=float(classical.rsquared_adj),
                  coefficients=coefficients, diagnostics=diagnostics, vif=vif, anova=anova,
                  overall_test={'statistic': float(model.fvalue), 'pvalue': float(model.f_pvalue),
                                'kind': 'Classical F test' if covariance == 'Classical' else 'HC3 robust joint Wald F test'},
                  aic=float(classical.aic), bic=float(classical.bic), residual_standard_error=float(np.sqrt(classical.mse_resid)),
                  metrics=regression_metrics(y, classical.fittedvalues), condition_number=float(classical.condition_number),
                  influence=influence_summary, leave_one_out=validation,
                  recommendations=recommendations(diagnostics, vif, influence_summary, alpha),
                  scope='Full-sample fit, not held-out predictive performance. Metrics and coefficients use the modeled response scale. ANOVA is a classical sums-of-squares decomposition (weighted for WLS); HC3 changes coefficient and joint inference, not fit. Independence requires study-design review. Transform selection is exploratory; AIC/BIC across response transformations are not directly comparable. No causal claims.')
    plots.attrs['fit'] = {'model': model, 'classical': classical, 'predictors': predictors, 'levels': levels,
                          'degree': degree, 'interactions': interactions, 'predictor_transform': predictor_transform,
                          'transformation': transformation, 'alpha': alpha, 'weights': weights is not None,
                          'ranges': {c: [float(data[c].min()), float(data[c].max())] for c in predictors if c not in levels}}
    return result, plots


def predict_regression(fit, new_data):
    if not len(new_data) or new_data[fit['predictors']].isna().any().any():
        raise ValueError('Provide complete values for every predictor.')
    new_data = new_data.copy()
    for column in fit['ranges']:
        try:
            new_data[column] = pd.to_numeric(new_data[column], errors='raise')
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Provide numeric prediction values for {column}.') from exc
    design, _, _ = make_design(new_data, fit['predictors'], degree=fit['degree'], interactions=fit['interactions'],
                              predictor_transform=fit['predictor_transform'], levels=fit['levels'])
    predictions = fit['model'].get_prediction(design).summary_frame(alpha=fit['alpha'])
    if fit['weights']:
        # New-observation variance needs a justified future weight, not an assumed value.
        predictions = predictions.drop(columns=['obs_ci_lower', 'obs_ci_upper'])
    result = predictions.rename(columns={'mean': 'Fitted mean', 'mean_se': 'Mean standard error', 'mean_ci_lower': 'Mean CI lower',
                                          'mean_ci_upper': 'Mean CI upper', 'obs_ci_lower': 'Prediction lower', 'obs_ci_upper': 'Prediction upper'})
    if fit['transformation']['name'] != 'None':
        for name in ['Fitted mean', 'Mean CI lower', 'Mean CI upper', 'Prediction lower', 'Prediction upper']:
            if name in result:
                result[name+' (inverse scale)'] = inverse_response(result[name], fit['transformation'])
    result['Extrapolation'] = [any(float(new_data.iloc[i][c]) < bounds[0] or float(new_data.iloc[i][c]) > bounds[1]
                                 for c, bounds in fit['ranges'].items()) for i in range(len(new_data))]
    return result


def compare_nested_regressions(frame, target, reduced, full):
    if not reduced or not set(reduced) < set(full):
        raise ValueError('The reduced predictor set must be a nonempty strict subset of the full model.')
    # Identical rows are essential for a valid extra-sum-of-squares comparison.
    shared = frame[[target, *full]].dropna()
    _, reduced_plots = fit_regression(shared, target, reduced)
    _, full_plots = fit_regression(shared, target, full)
    small, large = reduced_plots.attrs['fit']['classical'], full_plots.attrs['fit']['classical']
    statistic, pvalue, df = large.compare_f_test(small)
    if not np.isfinite([statistic, pvalue]).all():
        raise ValueError('Nested F comparison is undefined for these fits.')
    return {'method': 'Nested-model extra-sum-of-squares F test', 'statistic': float(statistic), 'pvalue': float(pvalue),
            'added_df': int(df), 'observations': len(shared), 'excluded_rows': len(frame)-len(shared),
            'reduced_predictors': list(reduced), 'full_predictors': list(full),
            'assumptions': 'Classical untransformed linear OLS on identical complete cases; independent homoscedastic normal errors. Exploratory model selection is not independent confirmation.'}


def regression_sensitivity(frame, original, excluded_positions):
    """Explicit case-exclusion refit for review; never alters the active dataset."""
    positions = list(excluded_positions)
    if (not positions or len(positions) > 100 or len(set(positions)) != len(positions)
            or any(not isinstance(i, (int, np.integer)) or i < 1 or i > len(frame) for i in positions)):
        raise ValueError('Choose 1–100 distinct active row positions within the dataset.')
    if original['response_transformation']['name'] == 'Box-Cox':
        raise ValueError('Sensitivity refitting with reselected Box-Cox would change the response scale. Use a fixed log/square-root or untransformed specification.')
    excluded = set(positions)
    retained = frame.iloc[[i for i in range(len(frame)) if i+1 not in excluded]]
    alternative, _ = fit_regression(retained, original['target'], original['predictors'], alpha=original['alpha'],
                                    covariance=original['covariance'], response_transform=original['response_transformation']['name'],
                                    predictor_transform=original['predictor_transformation'], degree=original['degree'],
                                    interactions=original['interactions'], weight_column=original['weight_column'])
    baseline = {row['term']: row['estimate'] for row in original['coefficients']}
    if set(baseline) != {row['term'] for row in alternative['coefficients']}:
        raise ValueError('Exclusion changes categorical coding or model terms; a same-specification coefficient comparison is unavailable.')
    return {'method': 'User-requested case-exclusion sensitivity refit', 'excluded_positions': positions,
            'original_observations': original['observations'], 'sensitivity_observations': alternative['observations'],
            'original_r_squared': original['r_squared'], 'sensitivity_r_squared': alternative['r_squared'],
            'coefficients': [{'term': row['term'], 'original_estimate': baseline[row['term']], 'sensitivity_estimate': row['estimate'],
                              'change': row['estimate']-baseline[row['term']]} for row in alternative['coefficients']],
            'note': 'Only this sensitivity fit excludes the explicitly selected active row positions. The active data and primary fit are unchanged. This is exploratory, not a reason to discard valid observations.'}
