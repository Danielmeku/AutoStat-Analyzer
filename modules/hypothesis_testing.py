"""Explicit statistical tests on complete observations; no automatic test selection."""
import numpy as np
import pandas as pd
from scipy import stats


def _numeric(frame, columns):
    if any(c not in frame or not pd.api.types.is_numeric_dtype(frame[c]) for c in columns):
        raise ValueError("Select numeric columns for this test.")
    data = frame[columns].dropna().astype(float)
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Test values must be finite.")
    return data


def _result(method, statistic, pvalue, n, excluded, alpha, **extra):
    if not 0 < alpha < 1:
        raise ValueError("Significance level must be between zero and one.")
    if not np.isfinite([statistic, pvalue]).all():
        raise ValueError("This test is undefined for these data.")
    return dict(method=method, statistic=float(statistic), pvalue=float(pvalue),
                observations=int(n), excluded_rows=int(excluded), alpha=float(alpha),
                conclusion="Reject the null hypothesis" if pvalue < alpha else "Do not reject the null hypothesis",
                **extra)


def welch_test(frame, value, group, first, second, alpha=0.05):
    if value == group or first == second:
        raise ValueError("Choose a numeric outcome and two distinct groups.")
    _numeric(frame, [value])
    selected = frame.loc[frame[group].isin([first, second]), [value, group]].dropna()
    a = selected.loc[selected[group] == first, value].to_numpy(dtype=float)
    b = selected.loc[selected[group] == second, value].to_numpy(dtype=float)
    if min(len(a), len(b)) < 2:
        raise ValueError("Each selected group needs at least two nonmissing observations.")
    if not np.isfinite(np.concatenate([a, b])).all() or (a.var(ddof=1) == 0 and b.var(ddof=1) == 0):
        raise ValueError("Groups need finite values and nonzero sampling variation.")
    test = stats.ttest_ind(a, b, equal_var=False)
    variance_a, variance_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(variance_a + variance_b)
    df = (variance_a + variance_b)**2 / (variance_a**2/(len(a)-1) + variance_b**2/(len(b)-1))
    difference = float(a.mean() - b.mean())
    margin = float(stats.t.ppf(1-alpha/2, df) * se)
    return _result("Welch independent two-sample t-test", test.statistic, test.pvalue,
                   len(selected), len(frame)-len(selected), alpha,
                   outcome=str(value), group=str(group), first_group=str(first), second_group=str(second),
                   first_n=len(a), second_n=len(b), first_mean=float(a.mean()), second_mean=float(b.mean()),
                   mean_difference=difference, confidence_interval=[difference-margin, difference+margin],
                   degrees_of_freedom=float(df),
                   null_hypothesis="The two population means are equal.",
                   assumptions="Independent observations and groups; approximately normal group means. Small or strongly skewed samples need care. Other groups and incomplete rows are excluded.")


def correlation_test(frame, x, y, method="pearson", alpha=0.05):
    if x == y or method not in {"pearson", "spearman"}:
        raise ValueError("Choose two different columns and Pearson or Spearman.")
    data = _numeric(frame, [x, y])
    if len(data) < 3 or any(data[c].nunique() < 2 for c in [x, y]):
        raise ValueError("Correlation needs at least three complete pairs and two varying columns.")
    test = (stats.pearsonr if method == "pearson" else stats.spearmanr)(data[x], data[y])
    return _result(f"{method.title()} correlation test", test.statistic, test.pvalue,
                   len(data), len(frame)-len(data), alpha, x=str(x), y=str(y),
                   null_hypothesis=f"Population {method} correlation is zero.",
                   assumptions="Independent paired observations. Pearson inference assumes bivariate normality; Spearman uses an asymptotic p-value that can be inaccurate for small samples. Association is not causation.")


def chi_square_test(frame, x, y, alpha=0.05):
    if x == y:
        raise ValueError("Choose two distinct categorical columns.")
    data = frame[[x, y]].dropna()
    if data[x].nunique() > 30 or data[y].nunique() > 30:
        raise ValueError("Use categorical variables with at most 30 levels each.")
    table = pd.crosstab(data[x], data[y])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Each variable must have at least two observed categories.")
    statistic, pvalue, df, expected = stats.chi2_contingency(table, correction=False)
    return _result("Pearson chi-square independence test (no continuity correction)", statistic, pvalue,
                   len(data), len(frame)-len(data), alpha, x=str(x), y=str(y),
                   degrees_of_freedom=int(df), minimum_expected_count=float(expected.min()),
                   sparse_cells=int((expected < 5).sum()),
                   cramers_v=float(np.sqrt(statistic / (len(data) * min(table.shape[0]-1, table.shape[1]-1)))),
                   null_hypothesis="The two categorical variables are independent.",
                   assumptions="Independent observations and mutually exclusive categories. Expected counts below 5 make the chi-square approximation unreliable; consider an exact method.")
