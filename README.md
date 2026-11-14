# AutoStat Analyzer

A working **v0.1 exploratory analysis MVP** based on the development blueprint. Built with Python, Streamlit, pandas, and Plotly. No API key is needed.

## Run

Use Python 3.12. In a terminal at the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```

Open the Streamlit address printed by the terminal. In GitHub Codespaces, use the forwarded port 8501. The included devcontainer installs dependencies and forwards that port.

## Try the first milestone

1. Start with the built-in synthetic regional-sales dataset (181 rows).
2. Review column types, missing cells, duplicate rows, and data previews.
3. Select both cleaning options and press **Apply cleaning**: 177 rows remain. Restore the original whenever needed.
4. Explore histograms, boxplots, scatterplots, Pearson/Spearman correlations, categorical frequencies, descriptive statistics, and IQR flags.
5. Download a Markdown summary, ZIP analysis bundle, or the active CSV.
6. Upload your own UTF-8 CSV using comma, semicolon, or tab separation.

Uploads are limited to 10 MB, 100,000 observations and 200 columns. Headers must be unique and nonempty; malformed rows and numeric infinity are rejected. No implicit numeric conversion, imputation, encoding, or row deletion is performed. Suggested column roles do not alter data types. Numeric binary columns remain available for descriptive numeric exploration.

## Statistical scope

Summaries use all available rows; charts reproducibly sample at most 5,000 rows. Standard deviation and variance use sample definitions. Correlations use pairwise complete rows with at least three pairs; constant columns return undefined correlations. Different pairs can use different observations. IQR flags identify candidates for review, not errors. Associations are descriptive and do not establish causation.

This release implements foundation, data review/preparation, EDA, deterministic quality insights, and descriptive exports. **Not yet implemented:** hypothesis testing, regression/diagnostics, machine learning, conversational AI, and PDF reporting. Original empty module placeholders for future phases are retained. The ZIP is a descriptive report bundle, not a fitted-model report.

## Privacy

Uploaded data stays in the Streamlit server session's memory; the application does not persist it to disk or send it to AI services. Downloads may contain sensitive information. Resetting the working copy restores the original data; remove the uploaded file from its picker or close the session to release it. Infrastructure may have its own retention behavior. For sensitive datasets, use trusted/private hosting. Never commit datasets or API keys. CSV exports preserve user text; treat untrusted CSVs carefully in spreadsheet applications.

## Tests

```bash
python -m pytest -q
```

Tests cover CSV ingestion/rejection, mixed-type profiling, known statistical results, non-destructive cleaning, export contents, and the Streamlit demo/clean/reset journey. GitHub Actions runs the suite on Python 3.12.

## Architecture and next steps

- `modules/`: ingestion, validation, preparation, profiling, and descriptive calculations.
- `ai/insight_teller.py`: transparent rules using computed results. No language model is connected.
- `reporting/`: Markdown summary and ZIP exports.
- `app.py`: UI orchestration and per-session working copy.

Next: suitable hypothesis tests with assumptions and effect sizes; regression with held-out evaluation and diagnostics; PDF reports; then a provider-independent AI assistant receiving only explicitly approved aggregate results. Gemini credentials are unnecessary for this milestone and should later be supplied through environment variables or Streamlit secrets, never source code.
