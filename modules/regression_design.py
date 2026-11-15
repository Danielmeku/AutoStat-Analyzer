"""Explicit, reproducible regression terms without evaluating dataset column names."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import special


def transform_values(values, kind):
    values = np.asarray(values, dtype=float)
    if kind in {'Log', 'Box-Cox'} and np.any(values <= 0):
        raise ValueError(f'{kind} requires strictly positive values.')
    if kind == 'Square root' and np.any(values < 0):
        raise ValueError('Square root requires nonnegative values.')
    if kind == 'Reciprocal' and np.any(values == 0):
        raise ValueError('Reciprocal requires nonzero values.')
    operations = {'None': lambda x: x, 'Log': np.log, 'Square root': np.sqrt, 'Reciprocal': lambda x: 1/x}
    if kind not in operations:
        raise ValueError('Unknown predictor transformation.')
    result = operations[kind](values)
    if not np.isfinite(result).all():
        raise ValueError('Transformation produced nonfinite values.')
    return result


def make_design(data, predictors, *, degree=1, interactions=False, predictor_transform='None', levels=None):
    if degree not in (1, 2):
        raise ValueError('Choose linear or quadratic terms.')
    fitted_levels = {} if levels is None else levels.copy()
    arrays, names, numeric, origins = [], [], [], []
    for column in predictors:
        if column in fitted_levels or not pd.api.types.is_numeric_dtype(data[column]):
            if levels is None:
                categories = sorted(data[column].astype(str).unique())
                if not 2 <= len(categories) <= 20:
                    raise ValueError('Categorical predictors require 2–20 observed levels.')
                fitted_levels[column] = categories
            categories = fitted_levels[column]
            if len(names)+len(categories)-1 > 60:
                raise ValueError('Expanded model exceeds 60 terms. Reduce features or interactions.')
            values = data[column].astype(str)
            if not values.isin(categories).all():
                raise ValueError(f'Unknown category for {column}; use a fitted category.')
            for category in categories[1:]:
                arrays.append((values == category).to_numpy(dtype=float))
                names.append(f'{column} [{category} vs {categories[0]}]')
                origins.append(column)
        else:
            values = transform_values(data[column], predictor_transform)
            arrays.append(values)
            names.append(str(column) if predictor_transform == 'None' else f'{predictor_transform}({column})')
            numeric.append((names[-1], values))
            origins.append(column)
    if degree == 2:
        if len(names)+len(numeric) > 60:
            raise ValueError('Expanded model exceeds 60 terms. Reduce features or interactions.')
        for name, values in numeric:
            arrays.append(values**2)
            names.append(f'{name}²')
    if interactions:
        # Pairwise interactions between original encoded main effects; retain hierarchy.
        main_count = len(names) - (len(numeric) if degree == 2 else 0)
        extra_count = sum(origins[i] != origins[j] for i, j in combinations(range(main_count), 2))
        if len(names)+extra_count > 60:
            raise ValueError('Expanded model exceeds 60 terms. Reduce features or interactions.')
        for i, j in combinations(range(main_count), 2):
            # Dummy terms of the same categorical variable cannot coexist.
            if origins[i] == origins[j]:
                continue
            arrays.append(arrays[i] * arrays[j])
            names.append(f'{names[i]} × {names[j]}')
    if len(names) > 60:
        raise ValueError('Expanded model exceeds 60 terms. Reduce features or interactions.')
    design = np.column_stack([np.ones(len(data)), *arrays])
    if not np.isfinite(design).all():
        raise ValueError('Model terms must be finite.')
    return design, ['Intercept', *names], fitted_levels


def inverse_response(values, transformation):
    kind, power = transformation['name'], transformation.get('lambda')
    values = np.asarray(values, dtype=float)
    if kind == 'None':
        result = values
    elif kind == 'Log':
        result = np.exp(values)
    elif kind == 'Square root':
        if np.any(values < 0):
            raise ValueError('A fitted interval crosses the square-root inverse domain.')
        result = values**2
    else:
        result = special.inv_boxcox(values, power)
    if not np.isfinite(result).all():
        raise ValueError('Prediction cannot be inverted on the original response scale.')
    return result
