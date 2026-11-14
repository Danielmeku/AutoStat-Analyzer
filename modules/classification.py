"""Stratified classification evaluation with training-only preprocessing."""
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from modules.preprocessing import prepare_features, feature_transformer, validate_training_features
from modules.cross_validation import training_cross_validation


def compare_classifiers(frame, target, predictors, test_size=0.2, seed=42, cv_folds=0):
    predictors = list(predictors)
    if not predictors or len(predictors) > 20 or target in predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Select 1–20 distinct predictors, excluding the target.")
    if target not in frame or any(c not in frame for c in predictors):
        raise ValueError("Choose an existing target and valid predictors.")
    if test_size not in (0.2, 0.3):
        raise ValueError("Choose a 20% or 30% stratified holdout.")
    data = frame[[target, *predictors]].dropna(subset=[target]).copy()
    if len(data) < 30:
        raise ValueError("Need at least 30 rows with observed class labels.")
    y = data[target].astype(str)
    if y.nunique() != data[target].nunique():
        raise ValueError("Target labels must remain distinct when represented as text.")
    counts = y.value_counts().sort_index()
    if not 2 <= len(counts) <= 10 or counts.min() < 5:
        raise ValueError("Choose a target with 2–10 classes and at least 5 observations in every class.")
    x, numeric, categorical = prepare_features(data, predictors)
    train, test = train_test_split(np.arange(len(data)), test_size=test_size, stratify=y, random_state=seed)
    xtrain, xtest, ytrain, ytest = x.iloc[train], x.iloc[test], y.iloc[train], y.iloc[test]
    validate_training_features(xtrain)
    classes = counts.index.tolist()
    # Estimators receive integer codes; outputs retain exact class labels and order.
    label_codes = {label: i for i, label in enumerate(classes)}
    candidates = {
        "Majority baseline": make_pipeline(feature_transformer(numeric, categorical), DummyClassifier(strategy="most_frequent")),
        "Logistic regression": make_pipeline(feature_transformer(numeric, categorical, scale=True), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
        "Random forest classifier": make_pipeline(feature_transformer(numeric, categorical), RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=2)),
    }
    cv = training_cross_validation(candidates, xtrain, ytrain.map(label_codes), cv_folds, classification=True, seed=seed)
    models, evaluation = [], {}
    for name, model in candidates.items():
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(xtrain, ytrain.map(label_codes).to_numpy(dtype=int))
            except ConvergenceWarning as exc:
                raise ValueError("Logistic regression did not converge. Review feature scale, outliers, or redundant features before comparing again.") from exc
        predicted = np.asarray(classes, dtype=object)[model.predict(xtest).astype(int)]
        precision, recall, f1, support = precision_recall_fscore_support(ytest, predicted, labels=classes, zero_division=0)
        models.append({"model": name, "accuracy": float(accuracy_score(ytest, predicted)),
                       "balanced_accuracy": float(balanced_accuracy_score(ytest, predicted)),
                       "macro_f1": float(np.mean(f1))})
        evaluation[name] = {"confusion_matrix": confusion_matrix(ytest, predicted, labels=classes).tolist(),
                            "per_class": [{"class": label, "precision": float(precision[i]), "recall": float(recall[i]),
                                           "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(classes)]}
    return {
        "method": "Stratified classification holdout comparison", "target": str(target), "predictors": predictors,
        "classes": classes, "train_rows": len(train), "test_rows": len(test),
        "excluded_target_rows": len(frame)-len(data), "test_fraction": test_size, "seed": seed,
        "class_distribution": [{"class": label, "total": int(counts[label]), "train": int((ytrain == label).sum()),
                                "holdout": int((ytest == label).sum())} for label in classes],
        "missing_predictor_cells": int(x.isna().sum().sum()), "duplicate_rows": int(data.duplicated().sum()),
        "models": models, "evaluation": evaluation, "cross_validation": cv, "numeric_predictors": numeric, "categorical_predictors": categorical,
        "preprocessing": "Training-only numeric median and categorical mode imputation and one-hot encoding. Unknown categories encode as all-zero indicators; logistic numeric features also use training-only standard scaling. Missing targets excluded. Original working copy preserved.",
        "settings": "Majority baseline predicts the most frequent training class. Logistic regression uses balanced class weights, L2 regularization and max_iter=1000. Random forest: 100 trees, depth 12, minimum leaf size 2, balanced class weights. No tuning.",
        "scope": "All models share one stratified random holdout. Higher accuracy, balanced accuracy and macro-F1 are better. Balanced accuracy averages class recall; macro-F1 weights classes equally. Undefined class precision/recall/F1 is displayed as zero (for example, a never-predicted class). Confusion rows are actual classes; columns are predicted classes in the listed class order. Numeric target values are treated as discrete labels. Imbalance can hide poor minority-class performance, so inspect class metrics. Selecting models on this holdout is exploratory, not independent confirmation. Requires independent observations; unsuitable for time series or repeated subjects. Duplicate records can leak across partitions. Optional cross-validation is reported separately on training rows only. No probability calibration or inferential coefficient estimates are provided.",
    }
