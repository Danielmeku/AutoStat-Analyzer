"""UI controllers keep statistical calculations independent of Streamlit."""
import plotly.express as px
import pandas as pd
import streamlit as st
from modules.hypothesis_testing import welch_test, correlation_test, chi_square_test
from modules.regression import fit_regression
from reporting.results import json_result


def render_tests(frame):
    st.subheader("Statistical tests")
    st.caption("Choose a test based on your study design. These are two-sided, unadjusted tests; repeated testing increases false positives. Missing observations are excluded only for selected variables.")
    numbers = list(frame.select_dtypes(include="number"))
    columns = list(frame.columns)
    method = st.selectbox("Test", ["Welch two-sample t-test", "Correlation test", "Chi-square independence"])
    alpha = st.select_slider("Significance level", options=[0.01, 0.05, 0.10], value=0.05)
    operation = None
    if method == "Welch two-sample t-test":
        groups = [c for c in columns if 2 <= frame[c].nunique() <= 30]
        if numbers and groups:
            value = st.selectbox("Outcome", numbers)
            group = st.selectbox("Grouping column", groups)
            levels = list(frame[group].dropna().unique())
            first = st.selectbox("First group", levels)
            second = st.selectbox("Second group", levels, index=1)
            operation = lambda: welch_test(frame, value, group, first, second, alpha)
        else:
            st.info("Need a numeric outcome and a grouping column with 2–30 levels.")
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
            operation = lambda: chi_square_test(frame, x, y, alpha)
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
        st.caption("Results reflect the last Run test action; changing selections requires running again.")
        st.write(result["method"])
        st.write(f"{result['conclusion']} at α = {result['alpha']}. This does not prove either hypothesis.")
        st.write(f"Statistic: {result['statistic']:.5g} · p-value: {result['pvalue']:.5g} · observations: {result['observations']} · excluded rows: {result['excluded_rows']}")
        st.info(result["assumptions"])
        if result.get("sparse_cells", 0):
            st.warning("Some expected cell counts are below 5. The chi-square p-value may be unreliable.")
        st.json(result)
        st.download_button("Download test result", json_result(result), "autostat-test.json", "application/json")


def render_regression(frame):
    st.subheader("Linear regression laboratory")
    st.caption("Fit numeric outcomes using ordinary least squares. Coefficients describe associations holding the other selected predictors fixed.")
    numbers = list(frame.select_dtypes(include="number"))
    if len(numbers) < 2:
        st.info("Need at least two numeric columns to fit a model.")
        return
    target = st.selectbox("Regression target", numbers, index=1)
    choices = [c for c in numbers if c != target]
    predictors = st.multiselect("Predictors (up to 20)", choices, default=choices[:1])
    if st.button("Fit regression", type="primary"):
        st.session_state.pop("regression_result", None)
        st.session_state.pop("regression_plots", None)
        try:
            result, plots = fit_regression(frame, target, predictors)
            st.session_state["regression_result"] = result
            st.session_state["regression_plots"] = plots
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if result := st.session_state.get("regression_result"):
        st.markdown("#### Last fitted model")
        st.caption("Changing selections requires fitting again. Intervals below have 95% confidence.")
        st.write(f"Target: {result['target']} · Predictors: {', '.join(result['predictors'])}")
        left, middle, right = st.columns(3)
        left.metric("In-sample R²", f"{result['r_squared']:.3f}")
        middle.metric("In-sample RMSE", f"{result['metrics']['rmse']:.3f}")
        right.metric("Complete observations", result["observations"])
        st.write(f"Excluded incomplete rows: {result['excluded_rows']}")
        st.dataframe(pd.DataFrame(result["coefficients"]), hide_index=True, width="stretch")
        st.info(result["scope"])
        st.json(result["diagnostics"])
        if result["condition_number"] > 1000:
            st.warning("The design matrix has a high condition number. Review scaling and possible multicollinearity.")
        plots = st.session_state["regression_plots"].sample(min(5000, result["observations"]), random_state=42)
        st.plotly_chart(px.scatter(plots, x="Fitted", y="Residual", title="Residuals vs fitted (up to 5,000 rows)"), width="stretch")
        st.plotly_chart(px.scatter(plots, x="Observed", y="Fitted", title="Observed vs fitted (in-sample)"), width="stretch")
        st.download_button("Download regression result", json_result(result), "autostat-regression.json", "application/json")
