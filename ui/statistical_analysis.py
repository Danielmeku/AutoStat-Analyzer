"""UI controllers keep statistical calculations independent of Streamlit."""
import plotly.express as px
import pandas as pd
import streamlit as st
from modules.hypothesis_testing import welch_test, correlation_test, chi_square_test
from modules.regression import fit_regression
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
        st.write(f"{result['conclusion']} at α = {format_number(result['alpha'], digits)}. This does not prove either hypothesis.")
        st.write(f"Statistic: {format_number(result['statistic'], digits)} · p-value: {format_number(result['pvalue'], digits, pvalue=True)} · observations: {result['observations']} · excluded rows: {result['excluded_rows']}")
        st.info(result["assumptions"])
        if result.get("sparse_cells", 0):
            st.warning("Some expected cell counts are below 5. The chi-square p-value may be unreliable.")
        st.json(display_result(result, digits))
        st.download_button("Download test result", json_result(result), "autostat-test.json", "application/json")


def render_regression(frame):
    digits = st.session_state.get("significant_figures", 3)
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
        left.metric("In-sample R²", format_number(result['r_squared'], digits))
        middle.metric("In-sample RMSE", format_number(result['metrics']['rmse'], digits))
        right.metric("Complete observations", result["observations"])
        st.write(f"Excluded incomplete rows: {result['excluded_rows']}")
        st.dataframe(display_table(pd.DataFrame(result["coefficients"]), digits), hide_index=True, width="stretch")
        st.info(result["scope"])
        st.json(display_result(result["diagnostics"], digits))
        if result["condition_number"] > 1000:
            st.warning("The design matrix has a high condition number. Review scaling and possible multicollinearity.")
        plots = st.session_state["regression_plots"].sample(min(5000, result["observations"]), random_state=42)
        st.plotly_chart(_scatter(plots, x="Fitted", y="Residual", title="Residuals vs fitted (up to 5,000 rows)"), width="stretch")
        st.plotly_chart(_scatter(plots, x="Observed", y="Fitted", title="Observed vs fitted (in-sample)"), width="stretch")
        st.download_button("Download regression result", json_result(result), "autostat-regression.json", "application/json")


def render_comparison(frame):
    digits = st.session_state.get("significant_figures", 3)
    st.subheader("Predictive model comparison")
    st.write("Compare a training-mean baseline, linear regression, and a random forest on the same unseen rows.")
    st.caption("Use independent observations. For time series or repeated subjects, a different splitting strategy is needed.")
    numbers = list(frame.select_dtypes(include="number"))
    if not numbers or len(frame.columns) < 2:
        st.info("Need a numeric target and at least one other predictor.")
        return
    target = st.selectbox("Prediction target", numbers, index=min(1, len(numbers)-1))
    choices = [c for c in [*numbers, *frame.select_dtypes(exclude="number").columns] if c != target]
    predictors = st.multiselect("Prediction features", choices, default=choices[:1])
    fraction = st.selectbox("Holdout fraction", [0.2, 0.3], format_func=lambda v: f"{v:.0%}")
    cv_folds = st.selectbox("Regression cross-validation", [0, 3, 5], format_func=lambda v: "Off" if v == 0 else f"{v} folds (training only)")
    st.caption("Numeric and categorical features supported (max 30 levels per categorical feature). Training-only imputation and one-hot encoding; unknown categories encode as all zeros. Cross-validation adds training time. Original data is preserved.")
    if st.button("Compare models", type="primary"):
        for key in ["comparison_result", "comparison_predictions"]:
            st.session_state.pop(key, None)
        try:
            with st.spinner("Training and evaluating models…"):
                result, predictions = compare_regressors(frame, target, predictors, fraction, cv_folds=cv_folds)
            st.session_state["comparison_result"] = result
            st.session_state["comparison_predictions"] = predictions
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if result := st.session_state.get("comparison_result"):
        st.markdown("#### Last completed comparison")
        st.caption("Changing selections requires comparing again. These models are separate from the full-sample inferential OLS fit.")
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
        st.plotly_chart(_scatter(chart, x="Observed", y=model, title=f"Holdout observations vs predictions · {model}"), width="stretch")
        st.json(display_result(result, digits))
        st.download_button("Download comparison result", json_result(result), "autostat-comparison.json", "application/json")
