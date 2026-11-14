"""Reproducible holdout comparison. Preprocessing learns exclusively from training rows."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from modules.model_evaluation import regression_metrics


def compare_regressors(frame, target, predictors, test_size=0.2, seed=42):
    predictors = list(predictors)
    if not predictors or len(predictors) > 20 or target in predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Select 1–20 distinct predictors, excluding the target.")
    if test_size not in (0.2, 0.3):
        raise ValueError("Choose a 20% or 30% holdout.")
    columns = [target, *predictors]
    if any(c not in frame or not pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
        raise ValueError("Choose numeric columns for this release.")
    data = frame[columns].dropna(subset=[target]).astype(float)
    if len(data) < 30 or data[target].nunique() < 2:
        raise ValueError("Need at least 30 rows with observed targets and a varying target.")
    if np.isinf(data.to_numpy()).any():
        raise ValueError("Model values must be finite or missing.")
    train, test = train_test_split(np.arange(len(data)), test_size=test_size, random_state=seed)
    xtrain, xtest = data[predictors].iloc[train], data[predictors].iloc[test]
    ytrain, ytest = data[target].iloc[train], data[target].iloc[test]
    if xtrain.isna().all().any() or ytrain.nunique() < 2 or ytest.nunique() < 2:
        raise ValueError("Training predictors need observed values, and both target partitions must vary. Review your data or split.")
    candidates = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=2),
    }
    metrics, prediction_frame = [], pd.DataFrame({"Observed": ytest.to_numpy()})
    for name, estimator in candidates.items():
        model = make_pipeline(SimpleImputer(strategy="median"), estimator)
        model.fit(xtrain, ytrain)
        predicted = model.predict(xtest)
        scores = regression_metrics(ytest, predicted)
        metrics.append({"model": name, **scores, "r_squared": float(r2_score(ytest, predicted))})
        prediction_frame[name] = predicted
    return {
        "method": "Regression model holdout comparison", "target": str(target), "predictors": predictors,
        "train_rows": len(train), "test_rows": len(test), "excluded_target_rows": len(frame)-len(data),
        "missing_predictor_cells": int(data[predictors].isna().sum().sum()),
        "duplicate_rows": int(data.duplicated().sum()), "test_fraction": test_size, "seed": seed,
        "preprocessing": "Predictor medians learned on training rows only. Missing targets excluded; original data unchanged.",
        "models": metrics,
        "scope": "One shared random holdout for all three models. Lower RMSE/MAE is better; higher R-squared is better and can be negative. This is exploratory comparison, not independent confirmation after model selection. Random splitting assumes independent rows; unsuitable for time series or repeated subjects. Duplicate records can leak across partitions. No tuning or cross-validation was performed.",
    }, prediction_frame
