"""Presentation precision only: calculations and machine exports retain full precision."""
import math
import numbers
import pandas as pd


def format_number(value, digits=3, pvalue=False):
    if digits not in (2, 3):
        raise ValueError("Choose 2 or 3 significant figures.")
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if not math.isfinite(value):
        return "—"
    if pvalue and value == 0:
        return "Below numerical precision"
    if value == 0:
        return "0"
    # General format chooses scientific notation without rounding small p-values to zero.
    return format(value, f".{digits}g")


def display_result(value, digits=3, key=""):
    if isinstance(value, dict):
        return {k: display_result(v, digits, k) for k, v in value.items()}
    if isinstance(value, list):
        return [display_result(v, digits, key) for v in value]
    if isinstance(value, numbers.Real) and not isinstance(value, (numbers.Integral, bool)):
        return format_number(value, digits, pvalue="pvalue" in key)
    return value


def display_table(frame, digits=3):
    result = frame.copy()
    for col in result:
        if pd.api.types.is_float_dtype(result[col]):
            result[col] = result[col].map(lambda v: format_number(v, digits, pvalue="pvalue" in str(col)))
    return result
