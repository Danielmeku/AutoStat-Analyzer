"""AutoStat Analyzer: an in-memory exploratory analysis workspace."""
import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.data_loader import load_csv, demo_data
from modules.data_validator import validate_data
from modules.data_cleaner import clean_data
from modules.data_profiler import profile_data
from modules.exploratory_analysis import numeric_summary, correlation_matrix, outlier_summary
from ai.insight_teller import quality_insights
from reporting.executive_report import executive_report
from reporting.detailed_report import analysis_bundle
from ui.statistical_analysis import render_tests, render_regression

st.set_page_config(page_title="AutoStat Analyzer", page_icon="📊", layout="wide")
st.title("AutoStat Analyzer")
st.caption("STATISTICAL WORKSPACE · v0.2")
st.write("Understand your data before you model it. Upload, review, explore, and export your findings.")
with st.sidebar:
    st.header("Your workspace")
    source = st.radio("Dataset source", ["Demo dataset", "Upload CSV"])
    uploaded = st.file_uploader("Choose a UTF-8 CSV", type=["csv"]) if source == "Upload CSV" else None
    delimiter = st.selectbox("CSV separator", [",", ";", "\t"], format_func=lambda x: {",": "Comma", ";": "Semicolon", "\t": "Tab"}[x]) if source == "Upload CSV" else ","
    st.caption("Limits: 10 MB · 100,000 rows · 200 columns")
    st.divider()
    st.caption("Uploads stay in this server session's memory. No data is saved to disk or sent to an AI provider. Downloads are saved only by you.")
    if st.button("Reset working copy"):
        st.session_state.pop("fingerprint", None)
        st.session_state.pop("active", None)
        st.rerun()
    st.caption("To remove an upload completely, also clear the file picker or close the session.")

if source == "Upload CSV" and uploaded is None:
    st.info("Choose a CSV in the sidebar, or switch to the demo dataset to explore the workspace.")
    st.stop()
try:
    raw = uploaded.getvalue() if uploaded is not None else b"demo-v1"
    original = load_csv(raw, delimiter) if uploaded is not None else demo_data()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
name = uploaded.name if uploaded is not None else "Demo · regional sales (synthetic)"
fingerprint = hashlib.sha256(raw + delimiter.encode()).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:
    st.session_state["fingerprint"] = fingerprint
    st.session_state["active"] = original.copy()
    st.session_state["preparation"] = "Original data — no cleaning applied"
frame = st.session_state["active"]
# Invalidate all computed analyses whenever active data change (including cleaning/reset).
analysis_fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes() + str(list(frame.columns)).encode()).hexdigest()
if st.session_state.get("analysis_fingerprint") != analysis_fingerprint:
    for result_key in ["test_result", "regression_result", "regression_plots"]:
        st.session_state.pop(result_key, None)
    st.session_state["analysis_fingerprint"] = analysis_fingerprint
quality = validate_data(frame)
st.subheader(name)
st.caption(st.session_state["preparation"])
a, b, c, d = st.columns(4)
a.metric("Observations", f"{quality['rows']:,}")
b.metric("Variables", quality["columns"])
c.metric("Missing cells", f"{quality['missing_percent']:.1f}%")
d.metric("Duplicate rows", quality["duplicate_rows"])

review, explore, tests, regression, insights, export = st.tabs(["01 · Data review", "02 · Explore", "03 · Statistical tests", "04 · Regression", "05 · Insights", "06 · Export"])
with review:
    st.subheader("Review before you transform")
    st.dataframe(frame.head(500), hide_index=True, width="stretch")
    st.caption("Preview shows up to 500 rows. Analyses use the complete active dataset.")
    st.dataframe(profile_data(frame), hide_index=True, width="stretch")
    if quality["constant_columns"]:
        st.warning("Constant or empty columns: " + ", ".join(quality["constant_columns"]))
    st.markdown("#### Prepare a working copy")
    st.caption("Choices always start from the original upload. Your original data is preserved.")
    duplicates = st.checkbox("Remove duplicate rows")
    missing = st.checkbox("Drop rows containing any missing value")
    try:
        proposed = clean_data(original, duplicates, missing)
        st.write(f"Proposed result: {len(proposed):,} rows ({len(original) - len(proposed):,} removed).")
        if st.button("Apply cleaning", type="primary"):
            st.session_state["active"] = proposed
            st.session_state["preparation"] = f"Working copy · remove duplicates: {duplicates} · drop incomplete rows: {missing}"
            st.rerun()
    except ValueError as exc:
        st.warning(str(exc))
    if st.button("Restore original data"):
        st.session_state["active"] = original.copy()
        st.session_state["preparation"] = "Original data — no cleaning applied"
        st.rerun()

with explore:
    numbers = list(frame.select_dtypes(include="number").columns)
    categories = list(frame.select_dtypes(exclude="number").columns)
    st.subheader("Distributions & relationships")
    st.caption("Missing values are omitted per chart or pair. Charts use up to 5,000 reproducibly sampled rows; tables use all rows.")
    chart_data = frame.sample(min(5000, len(frame)), random_state=42)
    if numbers:
        variable = st.selectbox("Numeric variable", numbers)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.histogram(chart_data, x=variable, nbins=30, color_discrete_sequence=["#0D9488"]), width="stretch")
        with right:
            st.plotly_chart(px.box(chart_data, y=variable, color_discrete_sequence=["#0D9488"]), width="stretch")
        st.markdown("#### Descriptive statistics")
        st.dataframe(numeric_summary(frame), width="stretch")
        if len(numbers) >= 2:
            x = st.selectbox("Horizontal axis", numbers)
            y = st.selectbox("Vertical axis", numbers, index=1)
            st.plotly_chart(px.scatter(chart_data, x=x, y=y, opacity=0.65), width="stretch")
            method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)
            matrix = correlation_matrix(frame, method)
            st.plotly_chart(px.imshow(matrix, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title=f"{method.title()} correlation · pairwise complete rows"), width="stretch")
            st.caption("At least 3 paired observations are required. Constant columns have undefined correlations. Correlation does not imply causation.")
        st.markdown("#### Potential outliers")
        st.dataframe(outlier_summary(frame), hide_index=True, width="stretch")
        st.caption("Flags fall outside Q1 − 1.5 × IQR or Q3 + 1.5 × IQR. They are not automatically removed.")
    else:
        st.info("No numeric columns detected. Categorical exploration is available below.")
    if categories:
        category = st.selectbox("Categorical / date variable", categories)
        counts = frame[category].astype("string").fillna("(missing)").value_counts().head(20).rename_axis("Value").reset_index(name="Count")
        st.plotly_chart(px.bar(counts, x="Value", y="Count", title="Top 20 values"), width="stretch")
        st.dataframe(counts, hide_index=True, width="stretch")

with tests:
    render_tests(frame)

with regression:
    render_regression(frame)

with insights:
    st.subheader("Evidence-based observations")
    st.caption("Generated by transparent rules from the active dataset. This is not an AI chatbot.")
    for finding in quality_insights(quality):
        st.info(finding)
    st.markdown("#### Next development milestones")
    st.write("Model comparison with held-out evaluation, PDF reports, and an optional AI assistant are planned. This release includes hypothesis testing and linear regression with diagnostics.")

with export:
    st.subheader("Take your analysis with you")
    st.write("Export the active working copy and reproducible descriptive results. The ZIP contains a Markdown summary, profiles, descriptive statistics, IQR flags, Pearson correlations, and any completed statistical test or regression result as JSON.")
    st.download_button("Download analysis bundle", analysis_bundle(frame, name, {key: st.session_state[key] for key in ["test_result", "regression_result"] if key in st.session_state}), "autostat-analysis.zip", "application/zip")
    st.download_button("Download summary", executive_report(frame, name), "autostat-summary.md", "text/markdown")
    st.download_button("Download active dataset", frame.to_csv(index=False).encode("utf-8"), "autostat-working-copy.csv", "text/csv")
    st.caption("Exports contain dataset information. Store and share them according to your data's sensitivity. CSV text is preserved; use care when opening untrusted data in spreadsheet software.")
