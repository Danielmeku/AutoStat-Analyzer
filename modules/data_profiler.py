"""Column roles are suggestions; original pandas dtypes are preserved."""
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def column_role(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "Empty"
    if is_bool_dtype(series) or series.nunique() == 2:
        return "Binary"
    if is_numeric_dtype(series):
        return "Numeric"
    if len(values) and values.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}").all():
        if pd.to_datetime(values, errors="coerce").notna().all():
            return "Date"
    return "Categorical"


def profile_data(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "Column": str(name), "Type": str(frame[name].dtype),
        "Suggested role": column_role(frame[name]),
        "Unique": int(frame[name].nunique()),
        "Missing": int(frame[name].isna().sum()),
        "Missing %": round(float(frame[name].isna().mean() * 100), 2),
    } for name in frame])
