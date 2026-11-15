"""UI controllers keep statistical calculations independent of Streamlit."""
import plotly.express as px
import pandas as pd
import streamlit as st
from ui.presentation import result_details, plot_chart
from modules.hypothesis_testing import welch_test, correlation_test, chi_square_test, partial_correlation_test, variance_equality_test, paired_test, group_test, mann_whitney_test, fisher_test
from reporting.results import json_result
from reporting.formatting import format_number, display_result, display_table
from modules.machine_learning import compare_regressors
from ui.evaluation import render_cross_validation


def _scatter(*args, **kwargs):
    figure = px.scatter(*args, **kwargs)
    digits = st.session_state.get("significant_figures", 3)
    figure.update_traces(hovertemplate=f"x: %{{x:.{digits}g}}<br>y: %{{y:.{digits}g}}<extra></extra>")
    return figure


def render_tests(frame):
    digits = st.session_state.get("significant_figures", 3)
    st.subheader("Statistical tests")
    numbers = list(frame.select_dtypes(include="number"))
    columns = list(frame.columns)
    method = st.selectbox("Test", ["Welch two-sample t-test", "Correlation test", "Chi-square independence", "Paired t-test", "Wilcoxon signed-rank", "One-way ANOVA", "Kruskal–Wallis", "Mann–Whitney U", "Fisher exact (2×2)", "Partial correlation", "Variance equality"])
    with st.popover("About this test"):
        descriptions = {
            "Welch two-sample t-test": "Compare the means of two independent groups. Allows unequal variances. Check independence and the shape of small samples.",
            "Variance equality": "Compare population variances across independent groups. Median-centered Levene is more robust to non-normality; Bartlett assumes normal group distributions.",
            "Partial correlation": "Measure linear association between two numeric variables after adjusting for selected numeric controls. Adjustment does not establish causation.",
            "Correlation test": "Measure association between two numeric variables. Pearson describes linear association; Spearman describes monotonic rank association. Neither establishes causation.",
            "Chi-square independence": "Test whether two categorical variables are associated. Requires independent observations and adequate expected cell counts.",
            "Paired t-test": "Compare the mean difference between paired measurements, such as before and after. Small samples require approximately normal differences.",
            "Wilcoxon signed-rank": "Compare paired measurements using ranks. Assumes a symmetric distribution of differences; zero differences are omitted for ranking.",
            "One-way ANOVA": "Compare means across independent groups. Assumes approximately normal errors and equal variances. A significant result does not identify which groups differ.",
            "Kruskal–Wallis": "Compare distributions across independent groups using ranks. Interpreting this as a median comparison requires similarly shaped distributions.",
            "Mann–Whitney U": "Compare distributions of two independent groups using ranks. It is not automatically a test of medians.",
            "Fisher exact (2×2)": "Test association in a two-by-two categorical table. Useful for small counts; assumes independent observations.",
        }
        st.write(descriptions[method])
        st.markdown("**Reading the numbers**")
        st.write("The statistic measures departure from the null hypothesis on the test’s own scale. The p-value measures how incompatible the data are with that null model; it is not the probability the hypothesis is true. A p-value below your chosen significance level is evidence against the null, subject to the assumptions.")
        st.write("Effect sizes describe magnitude; confidence intervals describe uncertainty under the model. Review excluded rows and assumptions. A nonsignificant result does not prove no difference. Multiple unadjusted tests increase false positives.")
    alpha = st.select_slider("Significance level", options=[0.01, 0.05, 0.10], value=0.05)
    operation = None
    if method in {"Welch two-sample t-test", "Mann–Whitney U"}:
        groups = [c for c in columns if 2 <= frame[c].nunique() <= 30]
        if numbers and groups:
            value = st.selectbox("Outcome", numbers)
            group = st.selectbox("Grouping column", groups)
            levels = list(frame[group].dropna().unique())
            first = st.selectbox("First group", levels)
            second = st.selectbox("Second group", levels, index=1)
            operation = lambda: (welch_test if method == "Welch two-sample t-test" else mann_whitney_test)(frame, value, group, first, second, alpha)
        else:
            st.info("Need a numeric outcome and a grouping column with 2–30 levels.")
    elif method == "Variance equality":
        groups = [c for c in columns if 2 <= frame[c].nunique() <= 10]
        if numbers and groups:
            value = st.selectbox("Variance outcome", numbers)
            group = st.selectbox("Variance grouping column", groups)
            variance_method = st.selectbox("Variance test method", ["levene", "bartlett"], format_func=lambda v: "Levene (median-centered)" if v == "levene" else "Bartlett")
            operation = lambda: variance_equality_test(frame, value, group, variance_method, alpha)
        else:
            st.info("Need a numeric outcome and 2–10 groups.")
    elif method in {"One-way ANOVA", "Kruskal–Wallis"}:
        groups = [c for c in columns if 2 <= frame[c].nunique() <= 10]
        if numbers and groups:
            value = st.selectbox("Group-test outcome", numbers)
            group = st.selectbox("Group-test grouping column", groups)
            operation = lambda: group_test(frame, value, group, "anova" if method == "One-way ANOVA" else "kruskal", alpha)
        else:
            st.info("Need a numeric outcome and a grouping column with 2–10 levels.")
    elif method in {"Paired t-test", "Wilcoxon signed-rank"}:
        if len(numbers) >= 2:
            first = st.selectbox("First paired measurement", numbers)
            second = st.selectbox("Second paired measurement", numbers, index=1)
            operation = lambda: paired_test(frame, first, second, "t" if method == "Paired t-test" else "wilcoxon", alpha)
        else:
            st.info("Need two numeric paired measurement columns.")
    elif method == "Partial correlation":
        if len(numbers) >= 3:
            x = st.selectbox("First adjusted variable", numbers)
            y = st.selectbox("Second adjusted variable", numbers, index=1)
            controls = st.multiselect("Control variables", [c for c in numbers if c not in {x, y}])
            operation = lambda: partial_correlation_test(frame, x, y, controls, alpha)
        else:
            st.info("Need at least three numeric columns for partial correlation.")
    elif method == "Correlation test":
        if len(numbers) >= 2:
            x = st.selectbox("First numeric variable", numbers)
            y = st.selectbox("Second numeric variable", numbers, index=1)
            correlation = st.selectbox("Test correlation method", ["pearson", "spearman"])
            operation = lambda: correlation_test(frame, x, y, correlation, alpha)
        else:
            st.info("Need at least two numeric variables.")
    else:
        groups = [c for c in columns if 2 <= frame[c].nunique() <= 30]
        if len(groups) >= 2:
            x = st.selectbox("First categorical variable", groups)
            y = st.selectbox("Second categorical variable", groups, index=1)
            operation = lambda: (fisher_test if method == "Fisher exact (2×2)" else chi_square_test)(frame, x, y, alpha)
        else:
            st.info("Need at least two categorical variables with 2–30 levels.")
    if operation and st.button("Run test", type="primary"):
        st.session_state.pop("test_result", None)
        try:
            st.session_state["test_result"] = operation()
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if result := st.session_state.get("test_result"):
        st.markdown("#### Last completed test")
        st.write(result["method"])
        st.write(f"{result['conclusion']} at α = {format_number(result['alpha'], digits)}. This does not prove either hypothesis.")
        st.write(f"Statistic: {format_number(result['statistic'], digits)} · p-value: {format_number(result['pvalue'], digits, pvalue=True)} · observations: {result['observations']} · excluded rows: {result['excluded_rows']}")
        effects = {key.replace('_', ' ').title(): display_result(result[key], digits)
                   for key in ['partial_correlation', 'partial_r_squared', 'mean_difference', 'confidence_interval',
                               'cohen_dz', 'rank_biserial', 'eta_squared', 'epsilon_squared', 'cramers_v'] if key in result}
        if effects:
            st.dataframe(pd.DataFrame([effects]), hide_index=True, width="stretch")
        st.info(result["assumptions"])
        if result.get("sparse_cells", 0):
            st.warning("Some expected cell counts are below 5. The chi-square p-value may be unreliable.")
        result_details(display_result(result, digits))
        st.download_button("Download test result", json_result(result), "autostat-test.json", "application/json")


def render_regression(frame):
    from ui.regression_workbench import render_regression_workbench
    render_regression_workbench(frame)


def render_comparison(frame):
    digits = st.session_state.get("significant_figures", 3)
    st.subheader("Predictive model comparison")
    numbers = list(frame.select_dtypes(include="number"))
    if not numbers or len(frame.columns) < 2:
        st.info("Need a numeric target and at least one other predictor.")
        return
    target = st.selectbox("Prediction target", numbers, index=min(1, len(numbers)-1))
    choices = [c for c in [*numbers, *frame.select_dtypes(exclude="number").columns] if c != target]
    predictors = st.multiselect("Prediction features", choices, default=choices[:1])
    fraction = st.selectbox("Holdout fraction", [0.2, 0.3], format_func=lambda v: f"{v:.0%}")
    cv_folds = st.selectbox("Regression cross-validation", [0, 3, 5], format_func=lambda v: "Off" if v == 0 else f"{v} folds (training only)")
    with st.expander("Regularized models"):
        regularization = st.multiselect("Additional models", ["Ridge", "Lasso"])
        penalty = st.number_input("Regularization strength", min_value=0.0001, value=1.0, format="%.4f")
    if st.button("Compare models", type="primary"):
        for key in ["comparison_result", "comparison_predictions"]:
            st.session_state.pop(key, None)
        try:
            with st.spinner("Training and evaluating models…"):
                result, predictions = compare_regressors(frame, target, predictors, fraction, cv_folds=cv_folds, regularization=regularization, penalty=penalty)
            st.session_state["comparison_result"] = result
            st.session_state["comparison_predictions"] = predictions
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if result := st.session_state.get("comparison_result"):
        st.markdown("#### Last completed comparison")
        st.write(f"Target: {result['target']} · Features: {', '.join(result['predictors'])}")
        st.write(f"Training rows: {result['train_rows']} · Holdout rows: {result['test_rows']} · Excluded missing targets: {result['excluded_target_rows']}")
        st.dataframe(display_table(pd.DataFrame(result["models"]), digits), hide_index=True, width="stretch")
        st.info(result["scope"])
        if result["duplicate_rows"]:
            st.warning(f"Found {result['duplicate_rows']} duplicate model rows. Repeated records may leak information across the random split; review whether deduplication is appropriate.")
        render_cross_validation(result, digits)
        predictions = st.session_state["comparison_predictions"]
        model = st.selectbox("Inspect holdout predictions", [entry["model"] for entry in result["models"]])
        chart = predictions.sample(min(5000, len(predictions)), random_state=42)
        plot_chart(_scatter(chart, x="Observed", y=model, title=f"Holdout observations vs predictions · {model}"), width="stretch")
        result_details(display_result(result, digits))
        st.download_button("Download comparison result", json_result(result), "autostat-comparison.json", "application/json")
