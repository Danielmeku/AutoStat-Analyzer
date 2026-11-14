"""Diagnostic p-values are evidence to review, not certificates of validity."""
import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson


def regression_diagnostics(model):
    residuals = np.asarray(model.resid)
    # Perfect fits have no meaningful residual-distribution diagnostics.
    if np.max(np.abs(residuals)) < 1e-10:
        return {"note": "Near-perfect fit: residual diagnostics are undefined or unreliable."}
    jb, jb_p, _, _ = jarque_bera(residuals)
    _, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    return {"jarque_bera_statistic": float(jb), "jarque_bera_pvalue": float(jb_p),
            "breusch_pagan_pvalue": float(bp_p), "durbin_watson": float(durbin_watson(residuals)),
            "note": "Jarque–Bera assesses residual normality; Breusch–Pagan assesses heteroscedasticity. Durbin–Watson depends on row order. These do not verify independence or causality."}
