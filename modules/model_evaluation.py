import numpy as np


def regression_metrics(actual, predicted):
    errors = np.asarray(actual) - np.asarray(predicted)
    return {"rmse": float(np.sqrt(np.mean(errors**2))), "mae": float(np.mean(np.abs(errors)))}
