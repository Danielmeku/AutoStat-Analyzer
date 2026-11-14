"""Explicit, copy-based preparation, with no imputation or encoding by default."""
import pandas as pd


def clean_data(frame: pd.DataFrame, remove_duplicates: bool = False,
               drop_missing: bool = False) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if remove_duplicates:
        result = result.drop_duplicates()
    if drop_missing:
        result = result.dropna()
    if result.empty:
        raise ValueError("These choices would remove every row. Keep some observations.")
    return result.reset_index(drop=True)
