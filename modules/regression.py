"""Ordinary least squares inference, explicitly using complete cases."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from modules.diagnostics import regression_diagnostics
from modules.model_evaluation import regression_metrics


def fit_regression(frame, target, predictors, alpha=0.05):
    predictors = list(predictors)
    if not 0 < alpha < 1:
        raise ValueError("Significance level must be between zero and one.")
    if not predictors or len(predictors) > 20 or target in predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Select 1–20 distinct predictors, excluding the target.")
    columns = [target, *predictors]
    if any(c not in frame or not pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
        raise ValueError("This release supports numeric targets and predictors only.")
    data = frame[columns].dropna().astype(float)
    if len(data) < max(10, len(predictors)+3):
        raise ValueError("Not enough complete rows: need at least 10 and at least predictors + 3.")
    if not np.isfinite(data.to_numpy()).all() or any(data[c].nunique() < 2 for c in columns):
        raise ValueError("All model columns must be finite and vary across observations.")
    # Array-based intercept avoids collisions with a user column named 'const'.
    design = np.column_stack([np.ones(len(data)), data[predictors].to_numpy()])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Predictors are linearly dependent. Remove redundant columns.")
    model = sm.OLS(data[target].to_numpy(), design).fit()
    intervals = model.conf_int(alpha=alpha)
    coefficients = [dict(term=label, estimate=float(model.params[i]),
                         standard_error=float(model.bse[i]), pvalue=float(model.pvalues[i]),
                         ci_lower=float(intervals[i, 0]), ci_upper=float(intervals[i, 1]))
                    for i, label in enumerate(["Intercept", *predictors])]
    result = dict(method="Ordinary least squares", target=str(target), predictors=predictors,
                  observations=len(data), excluded_rows=len(frame)-len(data), alpha=alpha,
                  r_squared=float(model.rsquared), adjusted_r_squared=float(model.rsquared_adj),
                  coefficients=coefficients, diagnostics=regression_diagnostics(model),
                  metrics=regression_metrics(data[target], model.fittedvalues),
                  condition_number=float(model.condition_number),
                  scope="Full-sample fit. RMSE, MAE and R-squared are in-sample, not held-out predictive performance. Classical standard errors assume independent, homoscedastic errors; normality supports small-sample inference. No causal claims.")
    plots = pd.DataFrame({"Fitted": model.fittedvalues, "Residual": model.resid, "Observed": data[target].to_numpy()})
    return result, plots
