import pandas as pd
import streamlit as st
from reporting.formatting import display_table


def render_cross_validation(result, digits):
    cv = result.get("cross_validation", {})
    if not cv.get("folds"):
        return
    st.markdown("#### Training-only cross-validation")
    summary, folds = [], []
    for model in cv["models"]:
        for metric, values in model["metrics"].items():
            summary.append({"model": model["model"], "metric": metric, "mean": values["mean"], "sample_std": values["std"]})
        folds.extend({"model": model["model"], **row} for row in model["fold_results"])
    st.dataframe(display_table(pd.DataFrame(summary), digits), hide_index=True, width="stretch")
    with st.expander("Inspect fold scores"):
        st.dataframe(display_table(pd.DataFrame(folds), digits), hide_index=True, width="stretch")
