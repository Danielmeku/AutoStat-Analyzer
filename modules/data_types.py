"""Explicit, loss-checked column typing and reproducible working-copy preparation."""
import hashlib
import json
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from modules.data_cleaner import clean_data

TYPE_OPTIONS = ['Original type', 'Categorical (factor)', 'Numeric', 'Text', 'Date/time', 'Boolean']


def _boolean(value):
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {'true', 'false', '0', '1'}:
        return value.strip().lower() in {'true', '1'}
    raise ValueError('Boolean values must be true/false or 0/1.')


def convert_column_types(original, assignments):
    """Always convert from original values; failures never mutate input or drop rows."""
    result = original.copy(deep=True)
    for column, treatment in assignments.items():
        if column not in original:
            raise ValueError(f'Unknown column: {column}.')
        if treatment not in TYPE_OPTIONS:
            raise ValueError(f'Unsupported data type for {column}.')
        source = original[column]
        try:
            if treatment == 'Original type':
                continue
            if treatment == 'Categorical (factor)':
                converted = source.astype('category')
            elif treatment == 'Numeric':
                if is_datetime64_any_dtype(source):
                    raise ValueError('Dates cannot be implicitly converted to numeric timestamps.')
                converted = pd.to_numeric(source, errors='raise')
                if is_bool_dtype(converted):
                    converted = converted.astype('Int64')
                values = converted.to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(values).any():
                    raise ValueError('Infinite numeric values are unsupported.')
            elif treatment == 'Text':
                converted = source.astype('string')
            elif treatment == 'Date/time':
                source_dtype = source.cat.categories.dtype if isinstance(source.dtype, pd.CategoricalDtype) else source.dtype
                if is_numeric_dtype(source_dtype) or is_bool_dtype(source_dtype):
                    raise ValueError('Dates require ISO date/time values, not numeric epoch guesses.')
                converted = pd.to_datetime(source, format='ISO8601', errors='raise', utc=True)
            else:
                converted = source.map(_boolean).astype('boolean')
            if not source.isna().equals(converted.isna()):
                raise ValueError('Conversion would introduce missing values. Correct those values first.')
            result[column] = converted
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f'Cannot treat {column} as {treatment}: {exc}') from None
    return result


def prepare_working_copy(original, assignments, remove_duplicates=False, drop_missing=False):
    return clean_data(convert_column_types(original, assignments), remove_duplicates, drop_missing)


def preparation_description(assignments, remove_duplicates=False, drop_missing=False):
    parts = []
    if remove_duplicates:
        parts.append('duplicate rows removed')
    if drop_missing:
        parts.append('incomplete rows dropped')
    parts.extend(f'{column}: {treatment}' for column, treatment in assignments.items() if treatment != 'Original type')
    return 'Working copy · '+' · '.join(parts) if parts else 'Original data — no preparation applied'


def _json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return value if isinstance(value, (str, bool, int, float)) or value is None else str(value)


def column_type_schema(frame):
    columns = []
    for name in frame:
        series = frame[name]
        column = {'column': str(name), 'dtype': str(series.dtype)}
        if isinstance(series.dtype, pd.CategoricalDtype):
            column.update(treatment='Categorical (factor)', ordered=series.cat.ordered,
                          levels=[_json_scalar(value) for value in series.cat.categories],
                          level_dtype=str(series.cat.categories.dtype))
        elif is_bool_dtype(series):
            column['treatment'] = 'Boolean'
        elif is_datetime64_any_dtype(series):
            column['treatment'] = 'Date/time'
        elif is_numeric_dtype(series):
            column['treatment'] = 'Numeric'
        else:
            column['treatment'] = 'Text'
        columns.append(column)
    return {'columns': columns, 'note': 'CSV alone does not preserve data types. Factor levels are labels, not numeric quantities; date/time values use the recorded pandas dtype.'}


def dataset_signature(frame):
    """Include schema: pandas value hashes alone may match numeric and factor data."""
    schema = json.dumps(column_type_schema(frame), sort_keys=True, ensure_ascii=False)
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(values+schema.encode()).hexdigest()
