"""Diagnostics are evidence to review, never automatic proof of valid assumptions."""
import numpy as np
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_breusch_godfrey, linear_reset, het_white, het_goldfeldquandt, normal_ad, lilliefors
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


def regression_diagnostics(model, extended=False, variance_order=None):
    weighted = not isinstance(model.model, OLS)
    residuals = np.asarray(model.wresid if weighted else model.resid)
    if np.max(np.abs(residuals)) < 1e-10:
        return {'note': 'Near-perfect fit: residual diagnostics are undefined or unreliable.'}
    jb, jb_p, _, _ = jarque_bera(residuals)
    _, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    result = {'jarque_bera_statistic': float(jb), 'jarque_bera_pvalue': float(jb_p),
              'breusch_pagan_pvalue': float(bp_p), 'durbin_watson': float(durbin_watson(residuals)),
              'note': 'Breusch–Pagan and normality checks do not verify independence. Durbin–Watson and serial-correlation checks depend on row order. Diagnostic p-values are unadjusted.'}
    if extended:
        if len(residuals) <= 5000:
            result['anderson_darling_normal_pvalue'] = float(normal_ad(residuals)[1])
            result['lilliefors_normal_pvalue'] = float(lilliefors(residuals, dist='norm', pvalmethod='table')[1])
        else:
            result['extended_normality_note'] = 'Lilliefors and Anderson–Darling not run above 5,000 rows.'
        expanded_terms = model.model.exog.shape[1]*(model.model.exog.shape[1]+1)//2
        if expanded_terms <= 120 and len(residuals) > expanded_terms+2:
            try:
                result['white_pvalue'] = float(het_white(residuals, model.model.exog)[1])
            except (ValueError, AssertionError, np.linalg.LinAlgError):
                result['white_note'] = 'White test unavailable for this auxiliary design.'
        else:
            result['white_note'] = 'White test requires at most 120 auxiliary terms and sufficient residual degrees of freedom.'
    if variance_order is not None and weighted:
        result["goldfeld_quandt_note"] = "The ordered Goldfeld–Quandt test is not applied to WLS."
    if variance_order is not None and not weighted:
        order = np.argsort(variance_order, kind='stable')
        split, drop = int(.4*len(order)), int(.2*len(order))
        groups = [model.model.exog[order][:split], model.model.exog[order][split+drop:]]
        if (min(split, len(order)-split-drop) > model.model.exog.shape[1]+1
                and all(np.linalg.matrix_rank(group) == model.model.exog.shape[1] for group in groups)):
            try:
                statistic, pvalue, _ = het_goldfeldquandt(model.model.endog[order], model.model.exog[order], split=split, drop=drop, alternative='two-sided')
                result['goldfeld_quandt_statistic'] = float(statistic)
                result['goldfeld_quandt_pvalue'] = float(pvalue)
            except (ValueError, np.linalg.LinAlgError):
                result['goldfeld_quandt_note'] = 'Split-sample variance test unavailable for this design.'
        else:
            result['goldfeld_quandt_note'] = 'Ordered split groups need full-rank designs and more observations than model parameters.'
    if len(residuals) <= 5000:
        result['shapiro_wilk_pvalue'] = float(stats.shapiro(residuals).pvalue)
    else:
        result['shapiro_note'] = 'Not run above 5,000 observations; inspect the Q–Q plot and Jarque–Bera result.'
    try:
        if weighted:
            result["note"] += " WLS normality, variance and Durbin–Watson checks use whitened residuals; BG and RESET are not applied."
            return result
        result['breusch_godfrey_lag1_pvalue'] = float(acorr_breusch_godfrey(model, nlags=1)[1])
        if np.linalg.matrix_rank(np.column_stack([model.model.exog, model.fittedvalues**2])) > model.model.exog.shape[1]:
            result['reset_pvalue'] = float(linear_reset(model, power=2, use_f=True).pvalue)
    except (ValueError, np.linalg.LinAlgError):
        result['specification_note'] = 'Serial correlation or RESET check unavailable for this design.'
    return result


def collinearity(design, terms):
    # The diagonal inverse predictor-correlation matrix equals VIF. Compute once
    # rather than fitting two auxiliary regressions per term on large uploads.
    predictors = design[:, 1:]
    centered = predictors-predictors.mean(axis=0)
    scaled = centered/np.sqrt(np.sum(centered**2, axis=0))
    correlation = scaled.T @ scaled
    try:
        values = np.diag(np.linalg.inv(correlation))
    except np.linalg.LinAlgError:
        values = np.array([variance_inflation_factor(design, i) for i in range(1, len(terms))])
    return [{'term': name, 'vif': float(max(1., value)), 'tolerance': float(1/max(1., value))}
            for name, value in zip(terms[1:], values)]


def lack_of_fit(data, predictors, response, residual_ss, parameters):
    """Classical pure-error lack-of-fit F test; requires replicated predictor settings."""
    grouped = data.groupby(predictors, observed=True, dropna=False).indices
    groups, n = len(grouped), len(data)
    pure_df, lack_df = n-groups, groups-parameters
    if pure_df <= 0 or lack_df <= 0:
        return {'available': False, 'reason': 'Requires replicated predictor settings and positive lack-of-fit degrees of freedom.'}
    pure_ss = sum(float(np.sum((response[idx]-np.mean(response[idx]))**2)) for idx in grouped.values())
    if pure_ss <= np.finfo(float).eps:
        return {'available': False, 'reason': 'Pure-error variation is zero; the F test is undefined.'}
    lack_ss = max(0., float(residual_ss-pure_ss))
    statistic = (lack_ss/lack_df)/(pure_ss/pure_df)
    return {'available': True, 'statistic': float(statistic), 'pvalue': float(stats.f.sf(statistic, lack_df, pure_df)),
            'lack_of_fit_df': lack_df, 'pure_error_df': pure_df, 'pure_error_ss': pure_ss, 'lack_of_fit_ss': lack_ss}


def recommendations(diagnostics, vif, influence, alpha):
    items = []
    if any(diagnostics.get(key, 1) < alpha for key in ['breusch_pagan_pvalue', 'white_pvalue', 'goldfeld_quandt_pvalue']):
        items.append('Unequal variance: compare HC3 coefficient inference, a justified response transformation, or WLS with externally justified inverse-variance weights. HC3 does not repair the model or prediction intervals.')
    if diagnostics.get('reset_pvalue', 1) < alpha:
        items.append('Possible mean-model misspecification: inspect residual plots and consider justified quadratic terms, interactions or predictor transformations.')
    if diagnostics.get('jarque_bera_pvalue', 1) < alpha:
        items.append('Non-normal residuals: review the Q–Q plot, data quality and response transformations; small-sample inference and individual prediction intervals need care.')
    if diagnostics.get('breusch_godfrey_lag1_pvalue', 1) < alpha:
        items.append('Possible row-order dependence: establish time/group structure before using a serial model. Random holdouts and HC3 do not address dependence.')
    if any(row['vif'] > 5 for row in vif):
        items.append('Elevated VIF: review redundant predictors, center polynomial inputs where scientifically appropriate, or compare ridge prediction. Do not drop variables solely by a threshold.')
    if influence and influence['flagged_rows']:
        items.append('Influential cases: inspect the diagnostic table and verify source data. Compare justified sensitivity fits; flagged observations are not automatically errors and are never deleted.')
    return items or ['No selected diagnostic crossed its review threshold. This does not verify study design, independence, linearity or causality.']
