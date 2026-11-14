"""Quality checks are descriptive and never change the input."""
import pandas as pd


def validate_data(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame), "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "missing_percent": float(frame.isna().sum().sum() / frame.size * 100),
        "duplicate_rows": int(frame.duplicated().sum()),
        "constant_columns": [str(c) for c in frame if frame[c].nunique(dropna=True) <= 1],
    }
