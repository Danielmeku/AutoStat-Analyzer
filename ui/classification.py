import pandas as pd
import plotly.express as px
import streamlit as st
from ui.presentation import result_details, plot_chart
from modules.classification import compare_classifiers
from ui.evaluation import render_cross_validation
from reporting.formatting import display_table, display_result
from reporting.results import json_result


def render_classification(frame):
    digits = st.session_state.get("significant_figures", 3)
    st.subheader("Classification laboratory")
    targets = [c for c in frame if 2 <= frame[c].nunique() <= 10]
    if not targets:
        st.info("No classification targets with 2–10 classes were detected.")
        return
    controls = st.columns([1.2, 1.8, 1, 1.3])
    target = controls[0].selectbox("Classification target", targets)
    choices = [c for c in [*frame.select_dtypes(include="number").columns, *frame.select_dtypes(exclude="number").columns] if c != target]
    if not choices:
        st.info("Need at least one predictor distinct from the target.")
        return
    predictors = controls[1].multiselect("Classification features", choices, default=choices[:1])
    fraction = controls[2].selectbox("Classification holdout", [0.2, 0.3], format_func=lambda v: f"{v:.0%}")
    cv_folds = controls[3].selectbox("Classification cross-validation", [0, 3, 5], format_func=lambda v: "Off" if v == 0 else f"{v} folds (training only)")
    if st.button("Compare classifiers", type="primary"):
        st.session_state.pop("classification_result", None)
        try:
            with st.spinner("Evaluating classifiers…"):
                st.session_state["classification_result"] = compare_classifiers(frame, target, predictors, fraction, cv_folds=cv_folds)
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if result := st.session_state.get("classification_result"):
        st.markdown("#### Last completed classification")
        st.write(f"Target: {result['target']} · Features: {', '.join(result['predictors'])}")
        st.write(f"Training rows: {result['train_rows']} · Holdout rows: {result['test_rows']} · Excluded missing targets: {result['excluded_target_rows']}")
        st.dataframe(display_table(pd.DataFrame(result["models"]), digits), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(result["class_distribution"]), hide_index=True, width="stretch")
        st.info(result["scope"])
        if result["duplicate_rows"]:
            st.warning(f"Found {result['duplicate_rows']} duplicate model rows. Review possible leakage across the random split.")
        render_cross_validation(result, digits)
        selected = st.selectbox("Inspect classifier", list(result["evaluation"]))
        evaluation = result["evaluation"][selected]
        # Plotly otherwise treats numeric-looking class names as a numeric axis.
        labels = [f"Class: {label}" for label in result["classes"]]
        figure = px.imshow(evaluation["confusion_matrix"], x=labels, y=labels, text_auto=True,
                            labels={"x": "Predicted class", "y": "Actual class", "color": "Rows"},
                            color_continuous_scale="Teal", title=f"Holdout confusion matrix · {selected}")
        figure.update_traces(hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Rows: %{z:d}<extra></extra>")
        left, right = st.columns(2)
        with left:
            plot_chart(figure, width="stretch")
        with right:
            st.markdown("#### Per-class performance")
            st.dataframe(display_table(pd.DataFrame(evaluation["per_class"]), digits), hide_index=True, width="stretch")
        result_details(display_result({k: v for k, v in result.items() if k != "evaluation"}, digits))
        st.download_button("Download classification result", json_result(result), "autostat-classification.json", "application/json")
