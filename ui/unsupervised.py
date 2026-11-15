import pandas as pd
import plotly.express as px
import streamlit as st
from ui.presentation import result_details, plot_chart
from modules.unsupervised import explore_structure
from reporting.formatting import display_table, display_result
from reporting.results import json_result


def render_structure(frame):
    digits = st.session_state.get("significant_figures", 3)
    st.subheader("PCA and clustering")
    numbers = list(frame.select_dtypes(include="number"))
    if len(numbers) < 2:
        st.info("Need at least two numeric features for PCA and K-means.")
        return
    controls = st.columns([2, 1])
    features = controls[0].multiselect("Structure features", numbers, default=numbers[:2])
    clusters = controls[1].selectbox("Number of clusters", list(range(2, 9)), index=1)
    if st.button("Explore structure", type="primary"):
        for key in ["exploration_result", "exploration_plots"]:
            st.session_state.pop(key, None)
        try:
            with st.spinner("Calculating components and clusters…"):
                result, plots = explore_structure(frame, features, clusters)
            st.session_state["exploration_result"] = result
            st.session_state["exploration_plots"] = plots
        except (ValueError, KeyError) as exc:
            st.error(str(exc))
    if result := st.session_state.get("exploration_result"):
        st.markdown("#### Last completed structure exploration")
        plots = st.session_state["exploration_plots"]
        chart = plots.sample(min(5000, len(plots)), random_state=42)
        figure = px.scatter(chart, x="PC1", y="PC2", color="Cluster", title="PCA projection coloured by K-means cluster")
        figure.update_traces(hovertemplate=f"PC1: %{{x:.{digits}g}}<br>PC2: %{{y:.{digits}g}}<extra></extra>")
        left, right = st.columns(2)
        with left:
            plot_chart(figure, width="stretch")
        with right:
            plot_chart(px.bar(pd.DataFrame(result["cluster_sizes"]), x="cluster", y="rows", title="Cluster membership", color_discrete_sequence=["#22D3EE"]), width="stretch")
        for key in ["explained_variance", "component_weights", "cluster_sizes", "cluster_centers"]:
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            st.dataframe(display_table(pd.DataFrame(result[key]), digits), hide_index=True, width="stretch")
        st.info(result["scope"])
        result_details(display_result({k: v for k, v in result.items() if not isinstance(v, list)}, digits))
        st.download_button("Download structure result", json_result(result), "autostat-structure.json", "application/json")
