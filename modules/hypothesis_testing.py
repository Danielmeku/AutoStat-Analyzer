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


def paired_test(frame, first, second, method="t", alpha=0.05):
    if first == second or method not in {"t", "wilcoxon"}:
        raise ValueError("Select two different paired measurements and a supported test.")
    data = _numeric(frame, [first, second])
    if len(data) < 3:
        raise ValueError("Need at least three complete paired observations.")
    differences = data[first].to_numpy() - data[second].to_numpy()
    if np.all(differences == 0):
        raise ValueError("All paired differences are zero; this test is undefined.")
    extras = dict(first=str(first), second=str(second), mean_difference=float(differences.mean()))
    if method == "t":
        if differences.std(ddof=1) == 0:
            raise ValueError("Paired t-test requires variation in the paired differences.")
        test = stats.ttest_rel(data[first], data[second])
        margin = stats.t.ppf(1-alpha/2, len(data)-1) * stats.sem(differences)
        extras.update(confidence_interval=[float(differences.mean()-margin), float(differences.mean()+margin)],
                      cohen_dz=float(differences.mean()/differences.std(ddof=1)), degrees_of_freedom=len(data)-1)
        assumptions = "Rows must be genuinely paired; pairs are independent. Differences should be approximately normal for small-sample inference. Difference is first minus second."
        null = "The population mean paired difference is zero."
    else:
        test = stats.wilcoxon(differences, alternative="two-sided", zero_method="wilcox", method="auto")
        nonzero = differences[differences != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        extras["rank_biserial"] = float((ranks[nonzero > 0].sum()-ranks[nonzero < 0].sum())/ranks.sum())
        extras["zero_differences"] = int((differences == 0).sum())
        assumptions = "Genuinely paired measurements, independent pairs, and a symmetric distribution of paired differences. Zero differences are discarded for ranking; ties affect inference. SciPy auto-selects the p-value method."
        null = "The paired-difference distribution is symmetric about zero."
    return _result("Paired t-test" if method == "t" else "Wilcoxon signed-rank test", test.statistic, test.pvalue,
                   len(data), len(frame)-len(data), alpha, null_hypothesis=null, assumptions=assumptions, **extras)


def group_test(frame, outcome, group, method="anova", alpha=0.05):
    if outcome == group or method not in {"anova", "kruskal"}:
        raise ValueError("Select an outcome, grouping column and supported group test.")
    _numeric(frame, [outcome])
    data = frame[[outcome, group]].dropna()
    levels = list(data[group].unique())
    if not 2 <= len(levels) <= 10:
        raise ValueError("Choose a grouping column with 2–10 observed groups.")
    arrays = [data.loc[data[group] == level, outcome].to_numpy(dtype=float) for level in levels]
    if min(map(len, arrays)) < 2 or data[outcome].nunique() < 2:
        raise ValueError("Every group needs two observations, and the outcome must vary.")
    if method == "anova":
        if all(np.var(a) == 0 for a in arrays):
            raise ValueError("ANOVA requires within-group variation.")
        test = stats.f_oneway(*arrays)
        overall = data[outcome].mean()
        between = sum(len(a)*(a.mean()-overall)**2 for a in arrays)
        effect = {"eta_squared": float(between/((data[outcome]-overall)**2).sum())}
        null = "All population group means are equal."
        assumptions = "Independent groups/observations, approximately normal within-group errors, and equal variances. Significant omnibus results do not identify which groups differ; adjusted post-hoc tests are not performed."
    else:
        test = stats.kruskal(*arrays)
        effect = {"epsilon_squared": float(max(0, min(1, (test.statistic-len(arrays)+1)/(len(data)-len(arrays)))))}
        null = "The population group distributions are identical."
        assumptions = "Independent observations and rankable outcomes. Interpreting this as a median comparison requires similarly shaped distributions. Significant omnibus results do not identify specific differing groups. No post-hoc tests are performed."
    return _result("One-way ANOVA" if method == "anova" else "Kruskal–Wallis test", test.statistic, test.pvalue,
                   len(data), len(frame)-len(data), alpha, outcome=str(outcome), group=str(group),
                   groups=[{"group": str(level), "n": len(a), "mean": float(a.mean())} for level, a in zip(levels, arrays)],
                   null_hypothesis=null, assumptions=assumptions, **effect)


def mann_whitney_test(frame, outcome, group, first, second, alpha=0.05):
    if outcome == group or first == second:
        raise ValueError("Choose an outcome and two distinct groups.")
    _numeric(frame, [outcome])
    data = frame.loc[frame[group].isin([first, second]), [outcome, group]].dropna()
    a = data.loc[data[group] == first, outcome].to_numpy(dtype=float)
    b = data.loc[data[group] == second, outcome].to_numpy(dtype=float)
    if min(len(a), len(b)) < 2 or data[outcome].nunique() < 2:
        raise ValueError("Each group needs two nonmissing values and the outcome must vary.")
    test = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    return _result("Mann–Whitney U test", test.statistic, test.pvalue, len(data), len(frame)-len(data), alpha,
                   outcome=str(outcome), group=str(group), first_group=str(first), second_group=str(second),
                   first_n=len(a), second_n=len(b), rank_biserial=float(2*test.statistic/(len(a)*len(b))-1),
                   null_hypothesis="The two population distributions are identical.",
                   assumptions="Independent groups and observations, rankable outcomes. Median interpretations need similar distribution shapes. SciPy auto-selects exact/asymptotic inference; ties affect inference. Other groups are excluded.")


def fisher_test(frame, x, y, alpha=0.05):
    if x == y or not 0 < alpha < 1:
        raise ValueError("Choose two distinct categorical variables and a valid significance level.")
    data = frame[[x, y]].dropna()
    table = pd.crosstab(data[x], data[y])
    if table.shape != (2, 2):
        raise ValueError("Fisher's exact test requires exactly two observed levels in each variable.")
    test = stats.fisher_exact(table.to_numpy(), alternative="two-sided")
    return dict(method="Fisher's exact test", statistic=float(test.statistic) if np.isfinite(test.statistic) else None,
                pvalue=float(test.pvalue), observations=len(data), excluded_rows=len(frame)-len(data), alpha=alpha,
                conclusion="Reject the null hypothesis" if test.pvalue < alpha else "Do not reject the null hypothesis",
                x=str(x), y=str(y), row_levels=[str(v) for v in table.index], column_levels=[str(v) for v in table.columns],
                contingency_counts=table.to_numpy().tolist(),
                statistic_note="Statistic is the sample odds ratio; a missing value indicates infinite/undefined odds due to empty cells.",
                null_hypothesis="The odds ratio is one (independence in the two-by-two table).",
                assumptions="Independent observations with mutually exclusive categories; conditional exact inference given margins. The odds ratio orientation follows the displayed row/column level order.")


def partial_correlation_test(frame, x, y, controls, alpha=0.05):
    """Pearson partial correlation after linear adjustment on the same complete cases."""
    controls = list(controls)
    if x == y or not controls or len(controls) > 20 or len(set([x, y, *controls])) != len(controls)+2:
        raise ValueError('Choose distinct numeric variables and 1–20 distinct control variables.')
    data = _numeric(frame, [x, y, *controls])
    design = np.column_stack([np.ones(len(data)), data[controls].to_numpy()])
    df = len(data)-len(controls)-2
    if df <= 0 or np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError('Need independent control columns and positive residual test degrees of freedom.')
    residuals = [data[c].to_numpy()-design @ np.linalg.lstsq(design, data[c], rcond=None)[0] for c in [x, y]]
    if any(np.std(r) <= np.finfo(float).eps*max(1., np.std(data[c])) for r, c in zip(residuals, [x,y])):
        raise ValueError('A selected variable has no variation after adjustment.')
    coefficient = float(np.clip(np.corrcoef(*residuals)[0, 1], -1, 1))
    if abs(coefficient) >= 1-1e-14:
        raise ValueError('Perfect adjusted correlation makes finite t inference undefined.')
    statistic = coefficient*np.sqrt(df/(1-coefficient**2))
    return _result('Pearson partial correlation', statistic, 2*stats.t.sf(abs(statistic), df), len(data), len(frame)-len(data), alpha,
                   x=str(x), y=str(y), controls=controls, partial_correlation=coefficient,
                   partial_r_squared=coefficient**2, degrees_of_freedom=df,
                   null_hypothesis='Population linear partial correlation is zero.',
                   assumptions='Independent observations, linear adjustment and normal-model t inference. Adjustment is conditional association, not causal control. Missing rows excluded jointly.')


def variance_equality_test(frame, outcome, group, method='levene', alpha=0.05):
    if outcome == group or method not in {'levene', 'bartlett'}:
        raise ValueError('Choose distinct outcome/group columns and Levene or Bartlett.')
    _numeric(frame, [outcome])
    data = frame[[outcome, group]].dropna()
    levels = list(data[group].unique())
    if not 2 <= len(levels) <= 10:
        raise ValueError('Variance comparison requires 2–10 groups.')
    arrays = [data.loc[data[group] == level, outcome].to_numpy(dtype=float) for level in levels]
    if min(map(len, arrays)) < 3 or any(np.var(values) <= 0 for values in arrays):
        raise ValueError('Each variance-test group needs at least three varying observations.')
    test = stats.levene(*arrays, center='median') if method == 'levene' else stats.bartlett(*arrays)
    return _result('Levene median-centered variance test' if method == 'levene' else 'Bartlett variance test',
                   test.statistic, test.pvalue, len(data), len(frame)-len(data), alpha,
                   outcome=str(outcome), group=str(group),
                   groups=[{'group': str(level), 'n': len(values), 'sample_variance': float(np.var(values, ddof=1))} for level, values in zip(levels, arrays)],
                   null_hypothesis='Population variances are equal across groups.',
                   assumptions='Independent observations and groups. Bartlett is sensitive to non-normality; median-centered Levene is more robust. Non-rejection does not establish equal variances.')
