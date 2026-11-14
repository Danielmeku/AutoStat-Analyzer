import pandas as pd


def numeric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    summary = numeric.describe().T
    summary["median"] = numeric.median()
    summary["variance"] = numeric.var()
    summary["skewness"] = numeric.skew()
    summary["excess_kurtosis"] = numeric.kurtosis()
    return summary


def outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name in frame.select_dtypes(include="number"):
        values = frame[name].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        count = int(((values < q1 - 1.5 * spread) | (values > q3 + 1.5 * spread)).sum())
        records.append({"Column": name, "IQR flags": count, "Nonmissing rows": len(values)})
    return pd.DataFrame(records)


def correlation_matrix(frame: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    if method not in {"pearson", "spearman"}:
        raise ValueError("Select Pearson or Spearman correlation.")
    return frame.select_dtypes(include="number").corr(method=method, min_periods=3)
