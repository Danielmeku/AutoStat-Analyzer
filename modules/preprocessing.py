"""Reusable fold-local preprocessing for predictive workflows (not inferential OLS)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_features(frame, predictors):
    x = frame[predictors].copy()
    numeric, categorical = [], []
    for column in predictors:
        if pd.api.types.is_numeric_dtype(x[column]):
            numeric.append(column)
            x[column] = x[column].astype(float)
            if np.isinf(x[column].to_numpy()).any():
                raise ValueError("Predictors must be finite or missing.")
        else:
            categorical.append(column)
            x[column] = x[column].map(lambda v: str(v) if pd.notna(v) else np.nan)
            if x[column].nunique() > 30:
                raise ValueError(f"Categorical predictor {column} has more than 30 levels. Remove identifiers or reduce categories explicitly.")
    return x, numeric, categorical


def validate_training_features(x):
    if x.isna().all().any():
        raise ValueError("Training predictors need at least one observed training value in every training partition/fold.")


def feature_transformer(numeric, categorical, scale=False):
    transformers = []
    if numeric:
        steps = [SimpleImputer(strategy="median")]
        if scale:
            steps.append(StandardScaler())
        transformers.append(("numeric", make_pipeline(*steps), numeric))
    if categorical:
        transformers.append(("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore", sparse_output=True)), categorical))
    return ColumnTransformer(transformers, sparse_threshold=1.0)
