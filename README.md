# AutoStat Analyzer

An interactive statistical analysis workspace with a regression specification, diagnostics, remedies and prediction workflow. Built with Python, Streamlit, pandas, Plotly, SciPy, Statsmodels, scikit-learn, and ReportLab. All analyses and the offline guide work without an API key; live Gemini is optional.

## Expanded analysis workflow

The supplied regression lecture notes now guide the analysis process. See [Analysis methods](docs/analysis-methods.md) for the source-to-feature mapping, statistical assumptions and implementation limits.

- **Regression → Model specification:** numeric/categorical predictors, reference coding, quadratic terms, pairwise interactions, predictor/response transformations, regression-profile Box–Cox, confidence levels, classical/HC3 inference and inverse-variance WLS.
- **Inference:** coefficient tests/intervals, partial R², ANOVA, overall model inference, adjusted R², AIC/BIC, residual error, analytic OLS PRESS and nested linear-model comparisons on identical complete cases.
- **Diagnostics:** residual, Q–Q, ordered-residual, scale–location and influence plots; normality/variance/specification checks (including optional White, Goldfeld–Quandt, Lilliefors and Anderson–Darling); VIF/tolerance; pure-error lack-of-fit when applicable; leverage, Cook distance, studentized residuals, DFFITS and DFBETAS review measures.
- **Remedial review:** evidence-based suggestions and explicit case-exclusion sensitivity comparisons without changing the active dataset or primary model. **Predict new cases** distinguishes conditional-mean and individual intervals, reports numeric extrapolation and preserves the fitted encoding and transformation.
- **Statistical tests:** partial correlation with numeric controls and Levene/Bartlett variance comparisons, with effect sizes shown directly in the results. **Model comparison:** optional ridge/lasso with user-selected fixed penalties and training-only scaling/CV.
- Reports and assistant evidence identify the fitted method, covariance and modeled response scale. Row-level diagnostics and new-case predictions have separate CSV downloads and do not enter AI summaries.

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
9. In **Model comparison**, choose numeric and/or categorical predictors and a numeric target, then press **Compare models** to compare a mean baseline, linear regression and random forest on the same 20% or 30% holdout.
10. Select **2 or 3 significant figures** in the sidebar. Small values use scientific notation (for example `5e-22` = 5 × 10⁻²²). This changes presentation only; integer counts and full-precision machine exports are preserved. P-values that underflow to zero are labelled “Below numerical precision”, rather than claiming an exact zero.
11. In **Export**, press **Generate PDF reports** to download a one-page executive PDF and a detailed PDF. Generate them after running the analyses you want included. The ZIP then includes both PDFs alongside the full-precision results.
12. In **Classification**, select a categorical or discrete numeric target and numeric/categorical features, then press **Compare classifiers**. Inspect accuracy, balanced accuracy, macro-F1, the confusion matrix and each class’s precision/recall/F1. Results are included in JSON, ZIP and generated PDFs.
13. Optionally choose **3 or 5 folds** in either comparison tab before comparing. Cross-validation uses the training partition only. Inspect mean scores, sample standard deviations and per-fold scores; final holdout scores remain separate. Cross-validation is off by default and adds training time.
14. Try **Paired t-test**, **Wilcoxon signed-rank**, **Mann–Whitney U**, **One-way ANOVA**, **Kruskal–Wallis**, or **Fisher exact** in Statistical tests; review the design assumptions before running them.
15. In **PCA & clusters**, select 2–20 numeric features and a cluster count, then press **Explore structure**. Inspect standardized PCA components, variance explained, cluster sizes/centers and silhouette.
16. In **Assistant & insights**, use **Offline guide → Generate insights**, or ask a question such as “Explain the p-value” or “What does PCA mean?” This guide uses templates and calculated evidence, not a language model. Saved insights join JSON/ZIP and detailed PDFs; chat history stays in the session.
17. To use a language model, configure Gemini as described below, review the summary payload, then press Generate insights or Ask assistant to send it and your current question. No raw rows or previous chat are forwarded.
18. Upload your own UTF-8 CSV using comma, semicolon, or tab separation.

Uploads are limited to 10 MB, 100,000 observations and 200 columns. Headers must be unique and nonempty; malformed rows and numeric infinity are rejected. No implicit numeric conversion, imputation, encoding, or row deletion is performed. Suggested column roles do not alter data types. Numeric binary columns remain available for descriptive numeric exploration.

## Statistical scope

Summaries use all available rows; charts reproducibly sample at most 5,000 rows. Standard deviation and variance use sample definitions. Correlations use pairwise complete rows with at least three pairs; constant columns return undefined correlations. Different pairs can use different observations. IQR flags identify candidates for review, not errors. Associations are descriptive and do not establish causation.

This release implements foundation, data review/preparation, EDA, statistical tests, OLS regression with diagnostics, held-out regression and classification model comparison, PCA/K-means exploration, grounded offline insights/chat, optional Gemini, executive/detailed PDF reports, and analysis exports. **Still outside this release:** inferential logistic coefficients, automated post-hoc comparisons/multiple-testing correction, time-series/group-aware validation, hyperparameter tuning, hierarchical clustering, production accounts/storage and deployment. Gemini sends evidence only on explicit action buttons; live credentialed validation depends on your key/model access.

### Tests and models

- **Welch t-test:** independent groups, unequal variances allowed, at least two complete observations per group; reports group means, their difference (first minus second), degrees of freedom and a confidence interval at the selected significance level. Other groups are excluded and counted.
- **Paired t-test / Wilcoxon signed-rank:** genuinely paired measurements in each row, independent pairs, at least three complete pairs. Paired t assumes approximately normal differences for small samples and reports a difference interval and Cohen’s dz. Wilcoxon assumes symmetric differences, discards zero differences for ranking, and reports signed rank-biserial effect. Identical differences/zero-only differences are handled explicitly.
- **ANOVA / Kruskal–Wallis:** 2–10 groups with at least two observations per group. Classical ANOVA assumes equal variances and approximately normal errors; eta-squared is descriptive. Kruskal–Wallis compares distributions; a median interpretation needs similar shapes. Epsilon-squared is an omnibus effect estimate. No post-hoc comparisons are performed.
- **Mann–Whitney U:** two independent groups and at least two observations each; compares distributions rather than automatically comparing medians. Reports signed rank-biserial effect. Other groups are excluded and counted.
- **Fisher exact:** an observed 2×2 categorical table, independent observations, two-sided conditional inference. Reports table orientation and sample odds ratio. Infinite/undefined odds from empty cells are labelled unavailable; the exact p-value remains available.
- **Correlation tests:** at least three complete pairs and varying numeric columns. Pearson tests zero linear correlation under its distributional assumptions. Spearman tests zero rank correlation using an asymptotic p-value; small samples need care.
- **Chi-square independence:** categorical variables with 2–30 levels, no continuity correction; reports Cramér’s V and flags expected counts below 5. Sparse 2×2 tables can use the Fisher exact option; larger sparse tables need an appropriate method outside this release.
- **Regression:** 1–20 numeric/categorical predictors, up to 60 expanded terms and at least 10 complete observations with two residual degrees of freedom. Classical OLS, HC3 inference and inverse-variance WLS are available. Rank-deficient designs are rejected. Choose 90%, 95% or 99% intervals; numeric and response transformations are explicit. See the method guide for the diagnostic and interval assumptions.

Tests assume independent observations; the software cannot verify study design. Pairwise tests use two-sided alternatives; omnibus tests use standard upper-tail reference distributions. All p-values are unadjusted for multiple testing; failure to reject is not proof of the null. **Inferential OLS metrics are in-sample only** and do not estimate performance on new data. The separate Model comparison tab reports held-out metrics. No analysis establishes causation. Each result records its exact variables, excluded rows, sample size and limitations; unavailable numerical diagnostic values export as JSON null.

### Predictive evaluation

The model comparison uses one shared random train/test split (seed 42), with a choice of 20% or 30% holdout. It requires 30 observed target rows, varying targets in each partition, and 1–20 numeric/categorical predictors. Rows with missing targets are excluded. Numeric predictors use training medians; categorical predictors use training modes and one-hot encoding. Unknown categories encode as all-zero indicators. Categorical predictors are limited to 30 levels each to avoid exploding identifier features. Columns entirely missing in any training partition/fold are rejected. The baseline predicts the training target mean. Random forest uses 100 trees, maximum depth 12, minimum leaf size 2, and two workers. No hyperparameters are tuned. Metric tables show holdout RMSE, MAE and R²; R² may be negative. Fixed seeds reproduce splits and predictions within floating-point tolerance.

A single holdout supports exploratory comparison; choosing models from it makes it a validation set, not independent final confirmation. No confidence interval is claimed; optional training-only cross-validation scores are reported separately. Independent observations are required; time series and repeated subjects need other split strategies. Duplicates are counted and flagged because they may leak across partitions; they are not silently removed. The original working copy is preserved. Results retain exact target/features, split size, seed and missing-value handling in JSON and the ZIP bundle. Changing active data clears all analyses; changing only display precision keeps them.

### Classification

Classification supports categorical or explicitly selected discrete numeric targets with 2–10 classes, at least 30 observed targets overall, and at least 5 observations per class. Numeric values are treated as class labels, not continuous outcomes. Features may be 1–20 numeric or categorical columns, excluding the target (up to 30 levels per categorical feature). Missing targets are excluded and counted. A 20% or 30% stratified random holdout (seed 42) is shared by all classifiers. Numeric medians, categorical modes/one-hot categories, and logistic numeric standardization are fitted only on training rows; entirely missing training features are rejected. Original data is preserved.

The baseline predicts the most frequent training class. Logistic regression uses L2 regularization, balanced class weights and a 1,000-iteration limit; nonconvergence produces an actionable error rather than a completed comparison. Random forest uses 100 trees, maximum depth 12, minimum leaf size 2, balanced class weights and two workers. Labels use internal integer codes and are displayed/exported in the explicit lexicographically sorted text class order, preserving original label text. No tuning, threshold optimization, probability calibration or coefficient inference is performed.

All scores are holdout scores. Accuracy is the fraction correct; balanced accuracy is average recall over classes; macro-F1 weights class F1 scores equally. Precision, recall or F1 that would be undefined are displayed as zero. Confusion matrix rows are actual classes and columns are predictions. Per-class support counts refer to holdout rows. Class distributions show both training and holdout support. Imbalance can conceal poor minority-class recall, so inspect all metrics rather than accuracy alone. A single holdout is exploratory after model selection; independent confirmation requires another test set or an appropriate validation design. Independent observations are required; time series and repeated subjects need different split strategies. Duplicate model rows are counted and flagged as a potential leakage source.

Classification results retain their target/features, split, class order/distributions, preprocessing, settings and limitations. JSON/ZIP and generated PDFs include the last completed classification. Data changes invalidate results and PDFs; display precision changes preserve computed classification results but require regenerating PDFs. Changing selections alone does not recompute results.

### Fold-local preprocessing and cross-validation

Predictive pipelines support mixed or categorical-only features. Mode imputation and one-hot category discovery occur on training rows only; unseen categories are ignored and yield all-zero indicators for that feature. This avoids leakage but does not guarantee good extrapolation to new categories. High-cardinality fields (more than 30 observed levels) are rejected; remove identifiers or explicitly regroup levels before upload. Numeric imputation uses training medians. Logistic regression scales numeric columns only. The inferential **Regression** workspace supports reference-coded categorical predictors with explicitly reported reference levels; it uses complete cases and remains separate from predictive training-only encoding/imputation.

Cross-validation is optional (off, 3 folds, or 5 folds) and runs **inside the training partition**, excluding the separate holdout. Regression uses shuffled K-fold; classification uses shuffled stratified K-fold. Fixed seed 42 and the same fold assignments are used for all candidates. Every fold fits a fresh preprocessing/model pipeline; no globally imputed or encoded feature matrix is reused. Classification needs at least the selected number of folds in every training class; choose fewer folds or more data otherwise. Missing-all training-fold features and constant regression validation targets produce explicit errors rather than skipped or undefined scores.

Results include all fold scores, fold sizes, metric means and sample standard deviations (ddof=1). Standard deviations describe fold variation, not confidence intervals. Scores are not nested-CV estimates and no hyperparameters are tuned. Model selection using holdout results still makes those results exploratory; CV does not undo repeated holdout inspection. Independent observations remain necessary. Fold results are included in JSON, ZIP, detailed PDF, and UI tables; integer counts and full-precision machine outputs remain unchanged.

### PCA and K-means exploration

Select 2–20 numeric features and 2–8 clusters with at least 10 active rows. Empty/constant features, infinite values and insufficient distinct feature vectors are rejected. Median imputation and z-score standardization use **all active rows** for descriptive exploration. PCA uses full SVD and reports variance ratios, cumulative ratios and the first two component weights. Component signs are arbitrary. K-means uses the full standardized feature space (seed 42, ten initializations), not only the plotted projection. Cluster centers are transformed back into original feature units.

Silhouette uses at most a reproducible 2,000-row sample and is unavailable if that sample does not support it; inertia depends on scaling and the cluster count. No automatic choice of cluster count, natural-population claim or predictive validation is made. Plots show up to 5,000 sampled rows. Row projections and assignments stay in session memory; exports and assistant summaries include aggregate results only.

### Assistant architecture and optional Gemini

`ai/context_builder.py` explicitly whitelists computed aggregate evidence. It never serializes arbitrary session state, raw rows, category values, row predictions, residual arrays, PCA coordinates or past chat. Evidence cards include dataset-quality counts, completed test results/assumptions, OLS coefficients/diagnostics, model/validation scores and PCA/cluster summaries. Names are shortened in the assistant context; computed numbers use selected display precision. Aggregates and variable names can still be sensitive, so review the payload.

The provider-independent assistant supports a transparent **Offline guide** and optional **Gemini**. Offline replies use topic templates plus evidence cards; it is not a language model and does not calculate arbitrary new analyses. Gemini receives the reviewed context and current question only, when you press Generate insights or Ask assistant. Consent and assistant state reset when active data, evidence or display precision changes. No request is made just by selecting Gemini. Chat retains up to ten exchanges in session memory and is excluded from reports. Saved insight commentary includes its provider and cited evidence in JSON/ZIP and detailed PDFs.

Provider prose must be qualitative: numeric facts are rendered from computed cards. Validation rejects unknown evidence references, inserted digits, malformed responses and selected unsupported certainty/causal phrases. This is a bounded reference/format check, **not semantic fact-checking**: AI interpretations can still be wrong and require review. Gemini output is never executed and cannot run tools or calculations. Dataset labels/questions are treated as untrusted prompt data. Network and authentication/quota/blocked-response errors are shown without exposing keys or remote error bodies. TLS verification stays enabled, requests use timeouts and redirects are disabled. There is no automatic billable retry loop.

#### Enable live Gemini

1. Add `GEMINI_API_KEY` securely using GitHub Codespaces secrets (grant this repository access) or your hosting platform’s secret manager. Never paste a real key into source, issues or chat.
2. Set `GEMINI_MODEL` to a current model identifier your account can use with the `generateContent` API and JSON output. This app intentionally does not guess an available model. Use an identifier, not a URL/path.
3. Restart/reconnect the Codespace so variables are injected, activate the virtual environment and restart Streamlit. Alternatively, Streamlit deployment secrets can supply the same two names; `.streamlit/secrets.toml` is ignored by Git.
4. If outbound networking is restricted, allow `generativelanguage.googleapis.com` over HTTPS. No data is sent until you select Gemini, approve the displayed summary and press Generate insights or Ask assistant.
5. Test a nonsensitive demo summary first. Availability, quotas and cost depend on your account/model; free access is not assumed.

The integration uses Gemini’s REST `v1beta/models/{model}:generateContent` endpoint. Automated tests mock provider responses to cover request shape, consent, validation, timeouts and common HTTP errors. **No live Gemini call has been validated in this development environment:** no key/model is configured, and the Google API documentation URL returned HTTP 403 from this environment. The complete offline workflow is tested.

### PDF reports

PDFs are generated on demand in server-session memory, without network calls or disk writes. The executive report is a bounded one-page overview of dataset quality, two compact charts where possible, up to four analysis highlights, and interpretation limits. Additional exploration and assistant commentary are explicitly referred to the detailed report. The detailed PDF includes column profiles, descriptive statistics, IQR flags, and only the tests, fitted regression specifications, predictive comparisons and PCA/clustering actually run. Saved assistant commentary is separately labelled by provider and contains its computed evidence references. Classification reports include class distributions, metric tables, labelled confusion counts and per-class scores. Result fields record variables, exclusions, confidence intervals, diagnostics and model scope; no unperformed analyses or provider-generated numeric calculations are inserted.

Both use your selected 2 or 3 significant figures, including scientific p-values. Fonts are embedded for consistent rendering. Long labels may be shortened and chart labels show histogram bin starts; full labels/results remain in JSON and CSV. PDFs contain aggregate results, not raw observation rows. Generated reports are cleared when active data, completed analyses, dataset identity, preparation description, or display precision changes. Changing analysis selections without running them leaves reports describing the last completed analysis, consistent with the displayed results.

## Privacy

Uploaded data stays in the Streamlit server session's memory; the application does not persist it to disk. The offline guide makes no external requests. Optional Gemini sends the explicitly approved aggregate summary and current question to Google; it does not receive raw rows or previous chat. Your external provider may have its own retention policies. Downloads may contain sensitive information. Resetting the working copy restores the original data; remove the uploaded file from its picker or close the session to release it. Infrastructure may have its own retention behavior. For sensitive datasets, use trusted/private hosting. Never commit datasets or API keys. CSV exports preserve user text; treat untrusted CSVs carefully in spreadsheet applications.

## Tests

```bash
python -m pytest -q
```

Tests cover CSV ingestion/rejection, mixed-type profiling, known statistical results, non-destructive cleaning, export contents, statistical reference values, regression recovery and degeneracy checks, analysis export contents, train-only imputation, reproducible holdout predictions, significant-figure formatting, PDF parsing/content, embedded fonts, one-page executive layout and detailed pagination, report invalidation, stratified classification/reference metrics, train-only scaling, mixed-feature encoding, unseen categories, train-only cross-validation fits and fold-score summaries, numeric/multiclass labels, extended statistical reference tests, PCA/clustering reproducibility, aggregate-only AI context, offline chat, Gemini request/error mocks, evidence invalidation, and Streamlit end-to-end user journeys. GitHub Actions runs the suite on Python 3.12.

## Architecture and next steps

- `modules/`: ingestion, validation, preparation, profiling, and descriptive calculations.
- `ai/`: aggregate context, rule-based guide, provider protocol, optional Gemini, response/reference validation and grounded insight records.
- `reporting/`: number formatting, Markdown/JSON/CSV bundles, and in-memory executive/detailed PDFs.
- `ui/`: statistical analysis controls and result displays.
- `app.py`: UI orchestration and per-session working copy.

Next: live Gemini validation with your configured credentials; broader inference and validation designs; deployment hardening. No key is needed for the offline assistant or statistical workflows.
