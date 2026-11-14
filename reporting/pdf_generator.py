"""In-memory, evidence-based PDF reports; no network or filesystem output."""
import io
from pathlib import Path
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from html import escape
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from modules.data_validator import validate_data
from modules.data_profiler import profile_data
from modules.exploratory_analysis import numeric_summary, outlier_summary
from reporting.formatting import format_number

# Bundle fonts with each PDF so it renders consistently across browsers/machines.
FONT_DIR = Path(reportlab.__file__).parent / "fonts"
pdfmetrics.registerFont(TTFont("AutoStat", str(FONT_DIR / "Vera.ttf")))
pdfmetrics.registerFont(TTFont("AutoStat-Bold", str(FONT_DIR / "VeraBd.ttf")))
STYLES = getSampleStyleSheet()
for style in STYLES.byName.values():
    if hasattr(style, "fontName"):
        style.fontName = "AutoStat-Bold" if "Bold" in style.fontName else "AutoStat"
STYLES.add(ParagraphStyle(name="Compact", fontName="AutoStat", fontSize=9, leading=12, spaceAfter=5))
STYLES.add(ParagraphStyle(name="Cell", fontName="AutoStat", fontSize=7, leading=9, wordWrap="CJK"))
STYLES.add(ParagraphStyle(name="ReportTitle", fontName="AutoStat", fontSize=22, leading=25, textColor=colors.HexColor("#0D9488"), spaceAfter=12))


def _paragraph(text, style="Compact", limit=1200):
    # Dataset-provided text is never treated as ReportLab markup.
    text = str(text)
    if len(text) > limit:
        text = text[:limit] + " [shortened; see machine exports for full text]"
    return Paragraph(escape(text), STYLES[style])


def _number(value, digits, key=""):
    if value is None:
        return "Unavailable"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return format_number(value, digits, pvalue="pvalue" in key)
    return str(value)


def _table(frame, digits, width=499):
    headings = [str(c) for c in frame.columns]
    data = [[_paragraph(c, "Cell") for c in headings]]
    for row in frame.itertuples(index=False, name=None):
        data.append([_paragraph(_number(v, digits, headings[i]), "Cell", limit=180) for i, v in enumerate(row)])
    table = Table(data, colWidths=[width / len(headings)] * len(headings), repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF1F5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _charts(frame, digits):
    missing = frame.isna().sum().sort_values(ascending=False).head(5)
    sets = [("Missing cells by column (top 5)", list(missing.index), missing.to_list())]
    numeric = frame.select_dtypes(include="number")
    varying = [c for c in numeric if numeric[c].notna().any()]
    if varying:
        column = varying[0]
        values = frame[column].dropna().to_numpy(dtype=float)
        try:
            counts, edges = np.histogram(values, bins=6)
            sets.append((f"Distribution: {column}", [format_number(v, digits) for v in edges[:-1]], counts.tolist()))
        except (ValueError, OverflowError):
            pass
    elif len(frame.columns):
        column = frame.columns[0]
        counts = frame[column].astype("string").fillna("(missing)").value_counts().head(5)
        sets.append((f"Top values: {column}", list(counts.index), counts.to_list()))
    cells = []
    for title, labels, values in sets:
        drawing = Drawing(230, 135)
        chart = VerticalBarChart()
        chart.x, chart.y, chart.width, chart.height = 28, 32, 190, 92
        chart.data = [[int(v) for v in values]]
        chart.categoryAxis.categoryNames = [str(v)[:10] for v in labels]
        chart.categoryAxis.labels.fontSize = 6
        chart.categoryAxis.labels.fontName = "AutoStat"
        chart.valueAxis.labels.fontSize = 6
        chart.valueAxis.labels.fontName = "AutoStat"
        chart.valueAxis.valueMin = 0
        chart.valueAxis.valueMax = max(1, max(values, default=0)) * 1.15
        chart.bars[0].fillColor = colors.HexColor("#0D9488")
        drawing.add(chart)
        cells.append([_paragraph(title, "Compact", limit=65), drawing])
    return Table([cells], colWidths=[499 / len(cells)] * len(cells))


def _analysis_summary(analyses, digits):
    lines = []
    if test := analyses.get("test_result"):
        selection = (f"{test.get('outcome')} by {test.get('group')} ({test.get('first_group')} vs {test.get('second_group')})"
                     if "outcome" in test else f"{test.get('x')} vs {test.get('y')}")
        lines.append(f"{test['method']}: {selection}. p = {format_number(test['pvalue'], digits, pvalue=True)}; n = {test['observations']}. {test['conclusion']} at alpha = {test['alpha']}.")
    if model := analyses.get("regression_result"):
        lines.append(f"OLS target {model['target']}: in-sample R-squared = {format_number(model['r_squared'], digits)}; RMSE = {format_number(model['metrics']['rmse'], digits)}; n = {model['observations']}. This is not holdout performance.")
    if comparison := analyses.get("comparison_result"):
        best = min(comparison["models"], key=lambda m: m["rmse"])
        lines.append(f"Holdout comparison target {comparison['target']}: lowest observed RMSE = {format_number(best['rmse'], digits)} ({best['model']}); holdout n = {comparison['test_rows']}. One random split, not independent confirmation after selection.")
    if classification := analyses.get("classification_result"):
        best = max(classification["models"], key=lambda m: m["macro_f1"])
        lines.append(f"Classification target {classification['target']}: highest observed macro-F1 = {format_number(best['macro_f1'], digits)} ({best['model']}); stratified holdout n = {classification['test_rows']}. Review minority-class recall; one split only.")
    return lines or ["No hypothesis test, OLS model, or predictive comparison was completed. This report describes exploratory results only."]


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("AutoStat", 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(48, 28, "AutoStat Analyzer | Calculated results, no external AI")
    canvas.drawRightString(A4[0]-48, 28, f"Page {doc.page}")
    canvas.restoreState()


def _build(story, title):
    output = io.BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4, rightMargin=48, leftMargin=48,
                                 topMargin=40, bottomMargin=45, title=title, author="AutoStat Analyzer")
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return output.getvalue()


def executive_pdf(frame, dataset_name, analyses=None, digits=3, preparation="Original data"):
    """A deliberately bounded one-page overview with at most four analysis summaries."""
    quality = validate_data(frame)
    story = [_paragraph("AutoStat Analyzer", "ReportTitle"), _paragraph("Executive statistical summary", "Heading2"),
             _paragraph(f"Dataset: {dataset_name}", limit=90), _paragraph(f"Preparation: {preparation}", limit=120),
             _paragraph(f"{quality['rows']:,} observations | {quality['columns']} variables | Missing cells: {format_number(quality['missing_percent'], digits)}% | Duplicate rows: {quality['duplicate_rows']:,}"),
             Spacer(1, 8), _charts(frame, digits), Spacer(1, 10), _paragraph("Key calculated findings", "Heading2")]
    story.extend(_paragraph(line, limit=230) for line in _analysis_summary(analyses or {}, digits))
    story.extend([_paragraph("Limitations and next steps", "Heading2"),
                  _paragraph("Check data quality and study design before interpreting significance. Missing values are excluded per analysis; prediction features use training-only median imputation. Associations do not establish causation. Repeated testing increases false positives. Review the detailed report for exact variables, exclusions, assumptions and diagnostics."),
                  _paragraph(f"Display: up to {digits} significant figures; small values use scientific notation (e.g. 5e-22). Machine-readable exports retain full precision. Chart labels may be shortened; histogram labels show bin starts. No raw observation rows are included.")])
    return _build(story, "AutoStat executive summary")


def _result_details(result, digits, prefix=""):
    story = []
    for key, value in result.items():
        label = (prefix + key).replace("_", " ")
        if isinstance(value, dict):
            story.extend(_result_details(value, digits, prefix=label + " / "))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            story.extend([_paragraph(label, "Heading3"), _table(pd.DataFrame(value), digits), Spacer(1, 8)])
        elif isinstance(value, list):
            story.append(_paragraph(label + ": " + ", ".join(_number(v, digits, key) for v in value)))
        else:
            story.append(_paragraph(label + ": " + _number(value, digits, key)))
    return story


def detailed_pdf(frame, dataset_name, analyses=None, digits=3, preparation="Original data"):
    quality = validate_data(frame)
    story = [_paragraph("AutoStat Analyzer", "ReportTitle"), _paragraph("Detailed statistical report", "Heading1"),
             _paragraph(f"Dataset: {dataset_name}"), _paragraph(f"Preparation: {preparation}"),
             _paragraph("Data quality", "Heading2")]
    story.extend(_result_details(quality, digits))
    story.extend([_charts(frame, digits), _paragraph("Column profile", "Heading2"), _table(profile_data(frame), digits)])
    summary = numeric_summary(frame)
    if not summary.empty:
        story.extend([_paragraph("Descriptive statistics", "Heading2"),
                      _paragraph("All active rows; missing values excluded separately per column. Sample standard deviations and variances use n-1. IQR flags are review candidates, not automatic errors."),
                      _table(summary[["count", "mean", "std", "min", "median", "max"]].reset_index(names="Column"), digits),
                      _paragraph("Potential outliers", "Heading2"), _table(outlier_summary(frame), digits)])
    sections = {"test_result": "Completed hypothesis test", "regression_result": "Inferential OLS model (in-sample)",
                "comparison_result": "Predictive model comparison (holdout)"}
    for key, title in sections.items():
        if result := (analyses or {}).get(key):
            story.append(_paragraph(title, "Heading1"))
            story.extend(_result_details(result, digits))
    if classification := (analyses or {}).get("classification_result"):
        story.append(_paragraph("Classification model comparison (stratified holdout)", "Heading1"))
        story.extend(_result_details({k: v for k, v in classification.items() if k != "evaluation"}, digits))
        story.append(_paragraph("Confusion matrices: rows are actual classes; columns are predictions. C1, C2, etc. refer to the class order below."))
        for i, label in enumerate(classification["classes"]):
            story.append(_paragraph(f"C{i+1}: {label}"))
        for model_name, details in classification["evaluation"].items():
            story.append(_paragraph(model_name, "Heading2"))
            labels = [f"C{i+1}" for i in range(len(classification["classes"]))]
            matrix = pd.DataFrame(details["confusion_matrix"], columns=labels)
            matrix.insert(0, "Actual", labels)
            story.extend([_table(matrix, digits), Spacer(1, 8),
                          _paragraph("Per-class holdout metrics", "Heading3"),
                          _table(pd.DataFrame(details["per_class"]), digits)])
    if not analyses:
        story.append(_paragraph("No statistical test, OLS model or predictive comparison was completed."))
    story.extend([_paragraph("Interpretation and limitations", "Heading2"),
                  _paragraph("These observations are derived from calculated results using rules, not an external language model. Test p-values are unadjusted; non-rejection does not prove the null. OLS uses classical standard errors. Diagnostic p-values do not verify all assumptions. Holdout comparisons use one random split, not cross-validation; time series and repeated subjects require other split strategies. Consult the analysis-specific assumptions above. No analysis establishes causation."),
                  _paragraph(f"Numbers use up to {digits} significant figures; very small values use scientific notation. A computed zero p-value is labelled below numerical precision. Machine exports retain full precision. Long text and chart labels may be shortened; consult JSON/CSV exports for original labels. This report includes aggregate results, not raw rows.")])
    return _build(story, "AutoStat detailed report")
