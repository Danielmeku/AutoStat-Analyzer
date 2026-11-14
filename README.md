# AutoStat Analyzer

A working **v0.2 statistical analysis MVP** based on the development blueprint. Built with Python, Streamlit, pandas, Plotly, SciPy, and Statsmodels. No API key is needed.

## Run

Use Python 3.12. In a terminal at the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0
```

Open the Streamlit address printed by the terminal. In GitHub Codespaces, use the forwarded port 8501. The included devcontainer installs dependencies and forwards that port.

## Try the application

1. Start with the built-in synthetic regional-sales dataset (181 rows).
2. Review column types, missing cells, duplicate rows, and data previews.
3. Select both cleaning options and press **Apply cleaning**: 177 rows remain. Restore the original whenever needed.
4. Explore histograms, boxplots, scatterplots, Pearson/Spearman correlations, categorical frequencies, descriptive statistics, and IQR flags.
5. Download a Markdown summary, ZIP analysis bundle, or the active CSV.
6. In **Statistical tests**, select a two-sided Welch t-test, Pearson/Spearman correlation test, or chi-square independence test. Review assumptions, choose variables, and press **Run test**.
7. In **Regression**, select a numeric target and predictors, then press **Fit regression**. Review coefficient confidence intervals, in-sample metrics, and residual diagnostics.
8. Download the result JSON, or the analysis bundle containing your last completed test and model. Results are cleared when the active data changes; changing variable selections alone requires running/fitting again.
9. Upload your own UTF-8 CSV using comma, semicolon, or tab separation.

Uploads are limited to 10 MB, 100,000 observations and 200 columns. Headers must be unique and nonempty; malformed rows and numeric infinity are rejected. No implicit numeric conversion, imputation, encoding, or row deletion is performed. Suggested column roles do not alter data types. Numeric binary columns remain available for descriptive numeric exploration.

## Statistical scope

Summaries use all available rows; charts reproducibly sample at most 5,000 rows. Standard deviation and variance use sample definitions. Correlations use pairwise complete rows with at least three pairs; constant columns return undefined correlations. Different pairs can use different observations. IQR flags identify candidates for review, not errors. Associations are descriptive and do not establish causation.

This release implements foundation, data review/preparation, EDA, statistical tests, OLS regression with diagnostics, deterministic quality insights, and analysis exports. **Not yet implemented:** ANOVA, paired/nonparametric tests, logistic regression, held-out prediction/model comparison, machine learning, conversational AI, and PDF reporting.

### Tests and models

- **Welch t-test:** independent groups, unequal variances allowed, at least two complete observations per group; reports group means, their difference (first minus second), degrees of freedom and a confidence interval at the selected significance level. Other groups are excluded and counted.
- **Correlation tests:** at least three complete pairs and varying numeric columns. Pearson tests zero linear correlation under its distributional assumptions. Spearman tests zero rank correlation using an asymptotic p-value; small samples need care.
- **Chi-square independence:** categorical variables with 2–30 levels, no continuity correction; reports Cramér’s V and flags expected counts below 5. Sparse-table results need an appropriate exact method, which is not implemented here.
- **OLS:** 1–20 numeric predictors, at least 10 complete observations and at least predictors + 3 rows. Constant columns and rank-deficient designs are rejected. Coefficients use classical standard errors and 95% confidence intervals. Diagnostics include Jarque–Bera, Breusch–Pagan, Durbin–Watson and a condition-number warning. Near-perfect fits skip residual diagnostics. Durbin–Watson depends on row ordering.

Tests assume independent observations; the software cannot verify study design. P-values are two-sided and unadjusted for multiple testing; failure to reject is not proof of the null. **Model metrics are in-sample only** and do not estimate performance on new data. No analysis establishes causation. Each result records its exact variables, excluded rows, sample size and limitations; unavailable numerical diagnostic values export as JSON null.

## Privacy

Uploaded data stays in the Streamlit server session's memory; the application does not persist it to disk or send it to AI services. Downloads may contain sensitive information. Resetting the working copy restores the original data; remove the uploaded file from its picker or close the session to release it. Infrastructure may have its own retention behavior. For sensitive datasets, use trusted/private hosting. Never commit datasets or API keys. CSV exports preserve user text; treat untrusted CSVs carefully in spreadsheet applications.

## Tests

```bash
python -m pytest -q
```

Tests cover CSV ingestion/rejection, mixed-type profiling, known statistical results, non-destructive cleaning, export contents, statistical reference values, regression recovery and degeneracy checks, analysis export contents, and Streamlit test/model/clean/reset journeys. GitHub Actions runs the suite on Python 3.12.

## Architecture and next steps

- `modules/`: ingestion, validation, preparation, profiling, and descriptive calculations.
- `ai/insight_teller.py`: transparent rules using computed results. No language model is connected.
- `reporting/`: Markdown summary and ZIP exports.
- `ui/`: statistical analysis controls and result displays.
- `app.py`: UI orchestration and per-session working copy.

Next: broader tests and models; held-out evaluation and model comparison; PDF reports; then a provider-independent AI assistant receiving only explicitly approved aggregate results. Gemini credentials are unnecessary for this milestone and should later be supplied through environment variables or Streamlit secrets, never source code.
