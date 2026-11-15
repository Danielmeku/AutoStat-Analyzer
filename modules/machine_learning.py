"""Reproducible holdout comparison. Preprocessing learns exclusively from training rows."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from modules.model_evaluation import regression_metrics
from modules.preprocessing import prepare_features, feature_transformer, validate_training_features
from modules.cross_validation import training_cross_validation


def compare_regressors(frame, target, predictors, test_size=0.2, seed=42, cv_folds=0, regularization=None, penalty=1.0):
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Regularization strength must be finite and positive.")
    regularization = list(regularization or [])
    if any(name not in {"Ridge", "Lasso"} for name in regularization):
        raise ValueError("Choose Ridge and/or Lasso.")
    predictors = list(predictors)
    if not predictors or len(predictors) > 20 or target in predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Select 1–20 distinct predictors, excluding the target.")
    if test_size not in (0.2, 0.3):
        raise ValueError("Choose a 20% or 30% holdout.")
    columns = [target, *predictors]
    if any(c not in frame for c in columns) or not pd.api.types.is_numeric_dtype(frame[target]):
        raise ValueError("Choose an existing numeric target and valid predictors.")
    data = frame[columns].dropna(subset=[target]).copy()
    data[target] = data[target].astype(float)
    x, numeric, categorical = prepare_features(data, predictors)
    if len(data) < 30 or data[target].nunique() < 2:
        raise ValueError("Need at least 30 rows with observed targets and a varying target.")
    if np.isinf(data[target].to_numpy()).any():
        raise ValueError("Model values must be finite or missing.")
    train, test = train_test_split(np.arange(len(data)), test_size=test_size, random_state=seed)
    xtrain, xtest = x.iloc[train], x.iloc[test]
    ytrain, ytest = data[target].iloc[train], data[target].iloc[test]
    validate_training_features(xtrain)
    if ytrain.nunique() < 2 or ytest.nunique() < 2:
        raise ValueError("Training predictors need observed values, and both target partitions must vary. Review your data or split.")
    candidates = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Random forest": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=seed, n_jobs=2),
    }
    if "Ridge" in regularization:
        candidates["Ridge"] = Ridge(alpha=penalty, solver="lsqr")
    if "Lasso" in regularization:
        candidates["Lasso"] = Lasso(alpha=penalty, max_iter=10000)
    candidates = {name: make_pipeline(feature_transformer(numeric, categorical, scale=name in {"Ridge", "Lasso"}), estimator) for name, estimator in candidates.items()}
    cv = training_cross_validation(candidates, xtrain, ytrain, cv_folds, seed=seed)
    metrics, prediction_frame = [], pd.DataFrame({"Observed": ytest.to_numpy()})
    for name, model in candidates.items():
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
        "preprocessing": "Numeric medians, categorical modes and one-hot categories learned on training rows only. Unseen categories encode as all-zero indicators. Missing targets excluded; original data unchanged.",
        "regularization": {"models": regularization, "penalty": float(penalty), "selection": "User-chosen fixed penalty; no holdout tuning. Numeric scaling learned within training partitions."},
        "models": metrics, "cross_validation": cv, "numeric_predictors": numeric, "categorical_predictors": categorical,
        "scope": "One shared random holdout for all selected models. Lower RMSE/MAE is better; higher R-squared is better and can be negative. This is exploratory comparison, not independent confirmation after model selection. Random splitting assumes independent rows; unsuitable for time series or repeated subjects. Duplicate records can leak across partitions. No hyperparameter tuning. Optional cross-validation is reported separately and uses only training rows.",
    }, prediction_frame
