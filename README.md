# AutoStat Analyzer

A working **v0.5 statistical analysis MVP** based on the development blueprint. Built with Python, Streamlit, pandas, Plotly, SciPy, Statsmodels, scikit-learn, and ReportLab. No API key is needed.

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
9. In **Model comparison**, choose numeric predictors and a target, then press **Compare models** to compare a mean baseline, linear regression and random forest on the same 20% or 30% holdout.
10. Select **2 or 3 significant figures** in the sidebar. Small values use scientific notation (for example `5e-22` = 5 × 10⁻²²). This changes presentation only; integer counts and full-precision machine exports are preserved. P-values that underflow to zero are labelled “Below numerical precision”, rather than claiming an exact zero.
11. In **Export**, press **Generate PDF reports** to download a one-page executive PDF and a detailed PDF. Generate them after running the analyses you want included. The ZIP then includes both PDFs alongside the full-precision results.
12. In **Classification**, select a categorical or discrete numeric target and numeric features, then press **Compare classifiers**. Inspect accuracy, balanced accuracy, macro-F1, the confusion matrix and each class’s precision/recall/F1. Results are included in JSON, ZIP and generated PDFs.
13. Upload your own UTF-8 CSV using comma, semicolon, or tab separation.

Uploads are limited to 10 MB, 100,000 observations and 200 columns. Headers must be unique and nonempty; malformed rows and numeric infinity are rejected. No implicit numeric conversion, imputation, encoding, or row deletion is performed. Suggested column roles do not alter data types. Numeric binary columns remain available for descriptive numeric exploration.

## Statistical scope

Summaries use all available rows; charts reproducibly sample at most 5,000 rows. Standard deviation and variance use sample definitions. Correlations use pairwise complete rows with at least three pairs; constant columns return undefined correlations. Different pairs can use different observations. IQR flags identify candidates for review, not errors. Associations are descriptive and do not establish causation.

This release implements foundation, data review/preparation, EDA, statistical tests, OLS regression with diagnostics, held-out regression and classification model comparison, deterministic quality insights, executive/detailed PDF reports, and analysis exports. **Not yet implemented:** ANOVA, paired/nonparametric tests, inferential logistic regression, categorical feature encoding, clustering, cross-validation/tuning, and conversational AI.

### Tests and models

- **Welch t-test:** independent groups, unequal variances allowed, at least two complete observations per group; reports group means, their difference (first minus second), degrees of freedom and a confidence interval at the selected significance level. Other groups are excluded and counted.
- **Correlation tests:** at least three complete pairs and varying numeric columns. Pearson tests zero linear correlation under its distributional assumptions. Spearman tests zero rank correlation using an asymptotic p-value; small samples need care.
- **Chi-square independence:** categorical variables with 2–30 levels, no continuity correction; reports Cramér’s V and flags expected counts below 5. Sparse-table results need an appropriate exact method, which is not implemented here.
- **OLS:** 1–20 numeric predictors, at least 10 complete observations and at least predictors + 3 rows. Constant columns and rank-deficient designs are rejected. Coefficients use classical standard errors and 95% confidence intervals. Diagnostics include Jarque–Bera, Breusch–Pagan, Durbin–Watson and a condition-number warning. Near-perfect fits skip residual diagnostics. Durbin–Watson depends on row ordering.

Tests assume independent observations; the software cannot verify study design. P-values are two-sided and unadjusted for multiple testing; failure to reject is not proof of the null. **Inferential OLS metrics are in-sample only** and do not estimate performance on new data. The separate Model comparison tab reports held-out metrics. No analysis establishes causation. Each result records its exact variables, excluded rows, sample size and limitations; unavailable numerical diagnostic values export as JSON null.

### Predictive evaluation

The model comparison uses one shared random train/test split (seed 42), with a choice of 20% or 30% holdout. It requires 30 observed target rows, varying targets in each partition, and 1–20 numeric predictors. Rows with missing targets are excluded; missing predictors use medians learned exclusively from training rows. Columns entirely missing in training are rejected. The baseline predicts the training target mean. Random forest uses 100 trees, maximum depth 12, minimum leaf size 2, and two workers. No hyperparameters are tuned. Metric tables show holdout RMSE, MAE and R²; R² may be negative. Fixed seeds reproduce splits and predictions within floating-point tolerance.

A single holdout supports exploratory comparison; choosing models from it makes it a validation set, not independent final confirmation. No confidence interval or cross-validation score is claimed. Independent observations are required; time series and repeated subjects need other split strategies. Duplicates are counted and flagged because they may leak across partitions; they are not silently removed. The original working copy is preserved. Results retain exact target/features, split size, seed and missing-value handling in JSON and the ZIP bundle. Changing active data clears all analyses; changing only display precision keeps them.

### Classification

Classification supports categorical or explicitly selected discrete numeric targets with 2–10 classes, at least 30 observed targets overall, and at least 5 observations per class. Numeric values are treated as class labels, not continuous outcomes. Features must be 1–20 numeric columns, excluding the target; categorical feature encoding is future work. Missing targets are excluded and counted. A 20% or 30% stratified random holdout (seed 42) is shared by all classifiers. Predictor medians and logistic regression standardization are fitted only on training rows; entirely missing training features are rejected. Original data is preserved.

The baseline predicts the most frequent training class. Logistic regression uses L2 regularization, balanced class weights and a 1,000-iteration limit; nonconvergence produces an actionable error rather than a completed comparison. Random forest uses 100 trees, maximum depth 12, minimum leaf size 2, balanced class weights and two workers. Labels use internal integer codes and are displayed/exported in the explicit lexicographically sorted text class order, preserving original label text. No tuning, threshold optimization, probability calibration or coefficient inference is performed.

All scores are holdout scores. Accuracy is the fraction correct; balanced accuracy is average recall over classes; macro-F1 weights class F1 scores equally. Precision, recall or F1 that would be undefined are displayed as zero. Confusion matrix rows are actual classes and columns are predictions. Per-class support counts refer to holdout rows. Class distributions show both training and holdout support. Imbalance can conceal poor minority-class recall, so inspect all metrics rather than accuracy alone. A single holdout is exploratory after model selection; independent confirmation requires another test set or an appropriate validation design. Independent observations are required; time series and repeated subjects need different split strategies. Duplicate model rows are counted and flagged as a potential leakage source.

Classification results retain their target/features, split, class order/distributions, preprocessing, settings and limitations. JSON/ZIP and generated PDFs include the last completed classification. Data changes invalidate results and PDFs; display precision changes preserve computed classification results but require regenerating PDFs. Changing selections alone does not recompute results.

### PDF reports

PDFs are generated on demand in server-session memory, without network calls or disk writes. The executive report is a bounded one-page overview of dataset quality, two compact charts where possible, completed analysis highlights, and interpretation limits. The detailed PDF includes column profiles, descriptive statistics, IQR flags, and only the tests, inferential OLS, and predictive regression/classification comparisons actually run. Classification reports include class distributions, metric tables, labelled confusion counts and per-class scores. Result fields record variables, exclusions, confidence intervals, diagnostics and model scope; no unperformed analyses or AI-generated findings are inserted.

Both use your selected 2 or 3 significant figures, including scientific p-values. Fonts are embedded for consistent rendering. Long labels may be shortened and chart labels show histogram bin starts; full labels/results remain in JSON and CSV. PDFs contain aggregate results, not raw observation rows. Generated reports are cleared when active data, completed analyses, dataset identity, preparation description, or display precision changes. Changing analysis selections without running them leaves reports describing the last completed analysis, consistent with the displayed results.

## Privacy

Uploaded data stays in the Streamlit server session's memory; the application does not persist it to disk or send it to AI services. Downloads may contain sensitive information. Resetting the working copy restores the original data; remove the uploaded file from its picker or close the session to release it. Infrastructure may have its own retention behavior. For sensitive datasets, use trusted/private hosting. Never commit datasets or API keys. CSV exports preserve user text; treat untrusted CSVs carefully in spreadsheet applications.

## Tests

```bash
python -m pytest -q
```

Tests cover CSV ingestion/rejection, mixed-type profiling, known statistical results, non-destructive cleaning, export contents, statistical reference values, regression recovery and degeneracy checks, analysis export contents, train-only imputation, reproducible holdout predictions, significant-figure formatting, PDF parsing/content, embedded fonts, one-page executive layout and detailed pagination, report invalidation, stratified classification/reference metrics and train-only scaling, numeric/multiclass labels, and Streamlit test/model/comparison/clean/reset journeys. GitHub Actions runs the suite on Python 3.12.

## Architecture and next steps

- `modules/`: ingestion, validation, preparation, profiling, and descriptive calculations.
- `ai/insight_teller.py`: transparent rules using computed results. No language model is connected.
- `reporting/`: number formatting, Markdown/JSON/CSV bundles, and in-memory executive/detailed PDFs.
- `ui/`: statistical analysis controls and result displays.
- `app.py`: UI orchestration and per-session working copy.

Next: broader tests; cross-validation and categorical predictors; then a provider-independent AI assistant receiving only explicitly approved aggregate results. Gemini credentials are unnecessary for this milestone and should later be supplied through environment variables or Streamlit secrets, never source code.
