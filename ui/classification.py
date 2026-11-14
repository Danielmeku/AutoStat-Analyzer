import pandas as pd
import plotly.express as px
import streamlit as st
from modules.classification import compare_classifiers
from reporting.formatting import display_table, display_result
from reporting.results import json_result


def render_classification(frame):
    digits = st.session_state.get("significant_figures", 3)
    st.subheader("Classification laboratory")
    st.write("Compare a majority-class baseline, logistic regression and a random forest on the same stratified holdout.")
    st.caption("Choose a categorical or discrete numeric target (2–10 classes). Features are numeric in this release. Need 30 observed targets and at least 5 rows in every class.")
    targets = [c for c in frame if 2 <= frame[c].nunique() <= 10]
    if not targets:
        st.info("No classification targets with 2–10 classes were detected.")
        return
    target = st.selectbox("Classification target", targets)
    choices = [c for c in frame.select_dtypes(include="number") if c != target]
    if not choices:
        st.info("Need at least one numeric predictor distinct from the target.")
        return
    predictors = st.multiselect("Classification features", choices, default=choices[:1])
    fraction = st.selectbox("Classification holdout", [0.2, 0.3], format_func=lambda v: f"{v:.0%}")
    st.caption("Fixed seed: 42. Stratification preserves approximate class proportions; imputation and logistic scaling are learned only from training rows.")
    if st.button("Compare classifiers", type="primary"):
        st.session_state.pop("classification_result", None)
        try:
            with st.spinner("Evaluating classifiers…"):
                st.session_state["classification_result"] = compare_classifiers(frame, target, predictors, fraction)
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if result := st.session_state.get("classification_result"):
        st.markdown("#### Last completed classification")
        st.caption("Selections alone do not recompute results; compare again after changing them.")
        st.write(f"Target: {result['target']} · Features: {', '.join(result['predictors'])}")
        st.write(f"Training rows: {result['train_rows']} · Holdout rows: {result['test_rows']} · Excluded missing targets: {result['excluded_target_rows']}")
        st.dataframe(display_table(pd.DataFrame(result["models"]), digits), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(result["class_distribution"]), hide_index=True, width="stretch")
        st.info(result["scope"])
        if result["duplicate_rows"]:
            st.warning(f"Found {result['duplicate_rows']} duplicate model rows. Review possible leakage across the random split.")
        selected = st.selectbox("Inspect classifier", list(result["evaluation"]))
        evaluation = result["evaluation"][selected]
        # Plotly otherwise treats numeric-looking class names as a numeric axis.
        labels = [f"Class: {label}" for label in result["classes"]]
        figure = px.imshow(evaluation["confusion_matrix"], x=labels, y=labels, text_auto=True,
                            labels={"x": "Predicted class", "y": "Actual class", "color": "Rows"},
                            color_continuous_scale="Teal", title=f"Holdout confusion matrix · {selected}")
        figure.update_traces(hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Rows: %{z:d}<extra></extra>")
        st.plotly_chart(figure, width="stretch")
        st.dataframe(display_table(pd.DataFrame(evaluation["per_class"]), digits), hide_index=True, width="stretch")
        st.json(display_result({k: v for k, v in result.items() if k != "evaluation"}, digits))
        st.download_button("Download classification result", json_result(result), "autostat-classification.json", "application/json")
