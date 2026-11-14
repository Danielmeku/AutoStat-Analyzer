"""An explicit aggregate-only allowlist: never serialize dataframes or arbitrary session data."""
from modules.data_validator import validate_data
from reporting.formatting import format_number


def build_context(frame, analyses, digits=3):
    quality = validate_data(frame)
    cards = []
    def add(source, title, statement):
        cards.append({"id": f"E{len(cards)+1}", "source": source, "title": title, "statement": statement})
    def label(value):
        return str(value)[:80]
    def fmt(value, p=False):
        return format_number(value, digits, pvalue=p)
    add("data_quality", "Dataset overview", f"Active dataset: {quality['rows']} observations and {quality['columns']} variables.")
    add("data_quality", "Missing and repeated data", f"Missing cells: {quality['missing_cells']} ({fmt(quality['missing_percent'])}%). Duplicate rows after first occurrences: {quality['duplicate_rows']}.")
    if test := analyses.get("test_result"):
        selection = f"{label(test.get('outcome', test.get('first', test.get('x', ''))))} / {label(test.get('group', test.get('second', test.get('y', ''))))}"
        add("test_result", "Hypothesis test", f"{test['method']} ({selection}): statistic {fmt(test['statistic'])}; p-value {fmt(test['pvalue'], True)}; alpha {fmt(test['alpha'])}; complete observations {test['observations']}; excluded rows {test['excluded_rows']}. {test['conclusion']}. Null: {test['null_hypothesis']} Assumptions: {test['assumptions']}")
    if model := analyses.get("regression_result"):
        add("regression_result", "In-sample OLS", f"Target {label(model['target'])}: full-sample R-squared {fmt(model['r_squared'])}, RMSE {fmt(model['metrics']['rmse'])}; observations {model['observations']}; excluded {model['excluded_rows']}. These are in-sample fit metrics, not predictive validation. Classical standard errors assume independent homoscedastic errors.")
        for coefficient in model['coefficients'][:5]:
            add("regression_result", "OLS coefficient", f"Term {label(coefficient['term'])}: estimate {fmt(coefficient['estimate'])}, p-value {fmt(coefficient['pvalue'], True)}, confidence interval [{fmt(coefficient['ci_lower'])}, {fmt(coefficient['ci_upper'])}]. Association conditional on selected predictors, not causation.")
        diagnostics = model['diagnostics']
        add("regression_result", "Residual diagnostics", f"Jarque–Bera p-value {fmt(diagnostics.get('jarque_bera_pvalue'), True)}; Breusch–Pagan p-value {fmt(diagnostics.get('breusch_pagan_pvalue'), True)}; Durbin–Watson {fmt(diagnostics.get('durbin_watson'))}. {diagnostics.get('note', '')}")
    for key in ["comparison_result", "classification_result"]:
        if result := analyses.get(key):
            ordered_models = sorted(result['models'], key=lambda m: m['rmse'] if key == 'comparison_result' else -m['macro_f1'])
            for model in ordered_models:
                scores = ", ".join(f"{metric.replace('_', ' ')} {fmt(score)}" for metric, score in model.items() if metric != 'model')
                add(key, "Holdout model", f"{model['model']} for target {label(result['target'])}: {scores}. Training rows {result['train_rows']}, holdout rows {result['test_rows']}, excluded missing targets {result['excluded_target_rows']}. One holdout; exploratory after model selection. Independent rows required; duplicates can leak across splits.")
            cv = result.get('cross_validation', {})
            if cv.get('folds'):
                add(key, "Cross-validation", f"{cv['folds']}-fold validation used only the {cv['training_rows']} training rows, with preprocessing learned inside each fold. Reported fold standard deviations are not confidence intervals.")
                metric = 'rmse' if key == 'comparison_result' else 'macro_f1'
                for model in cv['models']:
                    values = model['metrics'][metric]
                    add(key, 'Fold-score summary', f"{model['model']}: training-only fold mean {metric} {fmt(values['mean'])}; sample standard deviation {fmt(values['std'])}. Fold variation is not a confidence interval or held-out test score.")
    if result := analyses.get("exploration_result"):
        add("exploration_result", "PCA", f"First two standardized components explain {fmt(sum(r['ratio'] for r in result['explained_variance'][:2])*100)}% of variance. All {result['observations']} active rows used after median imputation; descriptive only.")
        add("exploration_result", "K-means", f"Requested clusters {result['clusters']}; standardized inertia {fmt(result['inertia_standardized'])}; silhouette {fmt(result['silhouette'])}, evaluated on at most {result['silhouette_sample_limit']} rows. Cluster labels are arbitrary and do not establish natural populations.")
    return {"purpose": "Explain computed statistical evidence; no raw rows, category values, row-level predictions, or chat history are included.",
            "limitations": "Associations do not establish causation. Test p-values are unadjusted. Model comparisons are exploratory, not independent confirmation after selection. Names/aggregates can still be sensitive; review before sharing.",
            "evidence": cards}
