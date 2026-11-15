"""AutoStat Analyzer: an in-memory exploratory analysis workspace."""
import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.data_loader import load_csv, demo_data
from modules.data_validator import validate_data
from modules.data_types import dataset_signature, prepare_working_copy, preparation_description
from modules.data_profiler import profile_data
from modules.exploratory_analysis import numeric_summary, correlation_matrix, outlier_summary
from reporting.executive_report import executive_report
from reporting.detailed_report import analysis_bundle
from reporting.pdf_generator import executive_pdf, detailed_pdf
from reporting.results import json_result
from ui.presentation import apply_style, workspace_header, plot_chart
from ui.data_preparation import render_column_types
from ui.classification import render_classification
from ui.unsupervised import render_structure
from ui.assistant import render_assistant
from ui.statistical_analysis import render_tests, render_regression, render_comparison
from reporting.formatting import display_table, format_number

st.set_page_config(page_title="AutoStat Analyzer", page_icon="📊", layout="wide")
apply_style()
workspace_header()
with st.sidebar:
    st.header("Your workspace")
    source = st.radio("Dataset source", ["Demo dataset", "Upload CSV"])
    uploaded = st.file_uploader("Choose a UTF-8 CSV", type=["csv"]) if source == "Upload CSV" else None
    delimiter = st.selectbox("CSV separator", [",", ";", "\t"], format_func=lambda x: {",": "Comma", ";": "Semicolon", "\t": "Tab"}[x]) if source == "Upload CSV" else ","
    digits = st.selectbox("Significant figures", [2, 3], index=1, key="significant_figures")

    st.divider()
    step = st.radio("Analysis workflow", ["Data review", "Explore", "Statistical tests", "Regression", "Model comparison", "Classification", "PCA & clusters", "Export"])
    with st.popover("Analysis guide"):
        st.write("Review data quality and study design, then explore distributions and relationships. Choose a test for your design or specify a regression model. Inspect residuals, collinearity and influence before considering remedies. Use held-out model comparisons for prediction; export the analyses you have completed.")
    st.divider()
    if st.button("Reset working copy"):
        st.session_state.pop("fingerprint", None)
        st.session_state.pop("active", None)
        st.rerun()
    with st.expander("Data privacy"):
        st.caption("Uploads stay in session memory. Gemini actions send an aggregate summary and your question to Google. Clear the upload or close the session to release your data.")

if source == "Upload CSV" and uploaded is None:
    st.info("Choose a CSV in the sidebar, or switch to the demo dataset to explore the workspace.")
    render_assistant(None, {}, "no-data", digits)
    st.stop()
try:
    raw = uploaded.getvalue() if uploaded is not None else b"demo-v1"
    original = load_csv(raw, delimiter) if uploaded is not None else demo_data()
except ValueError as exc:
    st.error(str(exc))
    render_assistant(None, {}, "no-data", digits)
    st.stop()
name = uploaded.name if uploaded is not None else "Demo · regional sales (synthetic)"
fingerprint = hashlib.sha256(raw + delimiter.encode()).hexdigest()
if st.session_state.get("fingerprint") != fingerprint:
    st.session_state["fingerprint"] = fingerprint
    st.session_state["active"] = original.copy()
    st.session_state["column_type_overrides"] = {}
    st.session_state["applied_cleaning"] = {"remove_duplicates": False, "drop_missing": False}
    for key in list(st.session_state):
        if key.startswith("column_type_choice_"):
            st.session_state.pop(key, None)
    st.session_state["preparation"] = "Original data — no cleaning applied"
frame = st.session_state["active"]
# Invalidate all computed analyses whenever active data change (including cleaning/reset).
analysis_fingerprint = dataset_signature(frame)
if st.session_state.get("analysis_fingerprint") != analysis_fingerprint:
    for result_key in ["test_result", "regression_result", "regression_plots", "regression_fit", "regression_new_predictions", "regression_nested_result", "comparison_result", "comparison_predictions", "classification_result", "exploration_result", "exploration_plots", "insight_result", "assistant_messages"]:
        st.session_state.pop(result_key, None)
    st.session_state["analysis_fingerprint"] = analysis_fingerprint
if st.session_state.get("last_digits", digits) != digits:
    st.session_state.pop("insight_result", None)
    st.session_state.pop("assistant_messages", None)
st.session_state["last_digits"] = digits
quality = validate_data(frame)
st.subheader(name)
with st.container(key="overview"):
    a, b, c, d = st.columns(4)
    a.metric("Observations", f"{quality['rows']:,}")
    b.metric("Variables", quality["columns"])
    c.metric("Missing cells", f"{format_number(quality['missing_percent'], digits)}%")
    d.metric("Duplicate rows", quality["duplicate_rows"])


def invalidate_commentary():
    evidence = {key: st.session_state[key] for key in ["test_result", "regression_result", "comparison_result", "classification_result", "exploration_result"] if key in st.session_state}
    signature = hashlib.sha256((analysis_fingerprint + str(digits) + json_result(evidence)).encode()).hexdigest()
    if st.session_state.get("completed_evidence_signature") != signature:
        for key in ["insight_result", "assistant_messages", "assistant_signature"]:
            st.session_state.pop(key, None)
        st.session_state["completed_evidence_signature"] = signature

invalidate_commentary()

analyses = {key: st.session_state[key] for key in ["test_result", "regression_result", "comparison_result", "classification_result", "exploration_result", "insight_result"] if key in st.session_state}
report_signature = hashlib.sha256((analysis_fingerprint + name + st.session_state["preparation"] + str(digits) + json_result(analyses)).encode()).hexdigest()
if st.session_state.get("pdf_signature") != report_signature:
    st.session_state.pop("pdf_reports", None)
    st.session_state.pop("pdf_signature", None)

if step == "Data review":
    st.subheader("Review before you transform")
    st.dataframe(frame.head(500), hide_index=True, width="stretch")
    st.dataframe(display_table(profile_data(frame), digits), hide_index=True, width="stretch")
    if quality["constant_columns"]:
        st.warning("Constant or empty columns: " + ", ".join(quality["constant_columns"]))
    render_column_types(original)
    st.markdown("#### Prepare a working copy")
    options = st.columns(2)
    duplicates = options[0].checkbox("Remove duplicate rows")
    missing = options[1].checkbox("Drop rows containing any missing value")
    proposed = None
    try:
        proposed = prepare_working_copy(original, st.session_state.get("column_type_overrides", {}), duplicates, missing)
        st.write(f"Proposed result: {len(proposed):,} rows ({len(original) - len(proposed):,} removed).")
    except ValueError as exc:
        st.warning(str(exc))
    actions = st.columns(2)
    if actions[0].button("Apply cleaning", type="primary", disabled=proposed is None, width="stretch"):
        st.session_state["active"] = proposed
        st.session_state["applied_cleaning"] = {"remove_duplicates": duplicates, "drop_missing": missing}
        st.session_state["preparation"] = preparation_description(st.session_state.get("column_type_overrides", {}), duplicates, missing)
        st.rerun()
    if actions[1].button("Restore original data", width="stretch"):
        st.session_state.pop("fingerprint", None)
        st.rerun()

if step == "Explore":
    numbers = list(frame.select_dtypes(include="number").columns)
    categories = list(frame.select_dtypes(exclude="number").columns)
    st.subheader("Distributions & relationships")
    chart_data = frame.sample(min(5000, len(frame)), random_state=42)
    if numbers:
        variable = st.selectbox("Numeric variable", numbers)
        left, right = st.columns(2)
        with left:
            plot_chart(px.histogram(chart_data, x=variable, nbins=30, title="Distribution", color_discrete_sequence=["#B06CFF"]), width="stretch")
        with right:
            plot_chart(px.box(chart_data, y=variable, title="Spread & outliers", color_discrete_sequence=["#24D9ED"]), width="stretch")
        st.markdown("#### Descriptive statistics")
        st.dataframe(display_table(numeric_summary(frame), digits), width="stretch")
        if len(numbers) >= 2:
            controls = st.columns(3)
            x = controls[0].selectbox("Horizontal axis", numbers)
            y = controls[1].selectbox("Vertical axis", numbers, index=1)
            method = controls[2].radio("Correlation method", ["pearson", "spearman"], horizontal=True)
            relationship, correlation = st.columns(2)
            with relationship:
                plot_chart(px.scatter(chart_data, x=x, y=y, opacity=0.75, title="Variable relationship", color_discrete_sequence=["#24D9ED"]), width="stretch")
            matrix = correlation_matrix(frame, method)
            with correlation:
                plot_chart(px.imshow(matrix, zmin=-1, zmax=1, color_continuous_scale=["#22D3EE", "#171B35", "#A78BFA"], title=f"{method.title()} correlation"), width="stretch")
        st.markdown("#### Potential outliers")
        st.dataframe(outlier_summary(frame), hide_index=True, width="stretch")
    else:
        st.info("No numeric columns detected. Categorical exploration is available below.")
    if categories:
        category = st.selectbox("Categorical / date variable", categories)
        counts = frame[category].astype("string").fillna("(missing)").value_counts().head(20).rename_axis("Value").reset_index(name="Count")
        category_chart, category_table = st.columns([2, 1])
        with category_chart:
            plot_chart(px.bar(counts, x="Value", y="Count", title="Top 20 values", color_discrete_sequence=["#F36DAE"]), width="stretch")
        with category_table:
            st.dataframe(counts, hide_index=True, width="stretch")

if step == "Statistical tests":
    render_tests(frame)

if step == "Regression":
    render_regression(frame)

if step == "Model comparison":
    render_comparison(frame)

if step == "Classification":
    render_classification(frame)

if step == "PCA & clusters":
    render_structure(frame)

if step == "Export":
    st.subheader("Take your analysis with you")
    if st.button("Generate PDF reports", type="primary"):
        with st.spinner("Preparing reports from calculated results…"):
            pdfs = {
                "executive.pdf": executive_pdf(frame, name, analyses, digits, st.session_state["preparation"]),
                "detailed.pdf": detailed_pdf(frame, name, analyses, digits, st.session_state["preparation"]),
            }
        st.session_state["pdf_reports"] = pdfs
        st.session_state["pdf_signature"] = report_signature
    pdfs = st.session_state.get("pdf_reports", {})
    if pdfs:
        pdf_actions = st.columns(2)
        pdf_actions[0].download_button("Download executive PDF", pdfs["executive.pdf"], "autostat-executive.pdf", "application/pdf")
        pdf_actions[1].download_button("Download detailed PDF", pdfs["detailed.pdf"], "autostat-detailed.pdf", "application/pdf")
    downloads = st.columns(3)
    downloads[0].download_button("Download analysis bundle", analysis_bundle(frame, name, analyses, pdfs), "autostat-analysis.zip", "application/zip")
    downloads[1].download_button("Download summary", executive_report(frame, name), "autostat-summary.md", "text/markdown")
    downloads[2].download_button("Download active dataset", frame.to_csv(index=False).encode("utf-8"), "autostat-working-copy.csv", "text/csv")

# Chat is available on every page and uses the latest completed analysis.
invalidate_commentary()
calculated = {key: st.session_state[key] for key in ["test_result", "regression_result", "comparison_result", "classification_result", "exploration_result"] if key in st.session_state}
render_assistant(frame, calculated, analysis_fingerprint, digits)

# A calculation on another page can change report contents during this run.
invalidate_commentary()
analyses = {key: st.session_state[key] for key in ["test_result", "regression_result", "comparison_result", "classification_result", "exploration_result", "insight_result"] if key in st.session_state}
report_signature = hashlib.sha256((analysis_fingerprint + name + st.session_state["preparation"] + str(digits) + json_result(analyses)).encode()).hexdigest()
if st.session_state.get("pdf_signature") != report_signature:
    st.session_state.pop("pdf_reports", None)
    st.session_state.pop("pdf_signature", None)
