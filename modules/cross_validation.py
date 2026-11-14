"""Cross-validation stays entirely inside the training partition; holdout remains separate."""
import warnings
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, r2_score
from sklearn.exceptions import ConvergenceWarning
from modules.preprocessing import validate_training_features
from modules.model_evaluation import regression_metrics


def training_cross_validation(candidates, x, y, folds=0, classification=False, seed=42):
    if folds not in (0, 3, 5):
        raise ValueError("Cross-validation must be off, 3 folds, or 5 folds.")
    if not folds:
        return {"folds": 0, "scope": "Cross-validation not requested."}
    if classification and y.value_counts().min() < folds:
        raise ValueError("Not enough training rows in every class for the selected fold count. Use fewer folds or more observations.")
    splitter = (StratifiedKFold if classification else KFold)(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(x, y if classification else None))
    details = []
    for name, candidate in candidates.items():
        rows = []
        for fold, (train, valid) in enumerate(splits, 1):
            validate_training_features(x.iloc[train])
            if not classification and y.iloc[valid].nunique() < 2:
                raise ValueError("A regression validation fold has a constant target. Review data or use a different fold count.")
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    model = clone(candidate).fit(x.iloc[train], y.iloc[train])
                except ConvergenceWarning as exc:
                    raise ValueError("Logistic regression did not converge in a cross-validation fold. Review your predictors.") from exc
            predicted = model.predict(x.iloc[valid])
            if classification:
                scores = dict(accuracy=float(accuracy_score(y.iloc[valid], predicted)),
                              balanced_accuracy=float(balanced_accuracy_score(y.iloc[valid], predicted)),
                              macro_f1=float(f1_score(y.iloc[valid], predicted, average="macro", zero_division=0)))
            else:
                scores = {**regression_metrics(y.iloc[valid], predicted), "r_squared": float(r2_score(y.iloc[valid], predicted))}
            rows.append({"fold": fold, "training_rows": len(train), "validation_rows": len(valid), **scores})
        metrics = {key: {"mean": float(np.mean([row[key] for row in rows])),
                         "std": float(np.std([row[key] for row in rows], ddof=1))} for key in scores}
        details.append({"model": name, "metrics": metrics, "fold_results": rows})
    return {"folds": folds, "training_rows": len(x), "models": details,
            "scope": "Shuffled cross-validation uses only the training partition, with a fresh preprocessing/model fit inside every fold. All candidates share folds. Means and sample standard deviations describe fold variation, not confidence intervals. Holdout scores are separate. No tuning or nested cross-validation; independent rows required."}
