"""A clearly labelled deterministic guide, not a language model."""


class OfflineProvider:
    def respond(self, question, context):
        query = question.lower()
        sources = []
        if any(term in query for term in ['p-value', 'p value', 'significan', 'hypothesis', 'anova', 'wilcoxon', 'fisher', 'kruskal', 'mann', 'paired']):
            answer = "A p-value describes how incompatible the data are with the null under the test assumptions. Compare it with the preselected significance level; non-rejection is not proof of the null. Check independence, distributions and multiple testing before interpretation."
            sources = ['test_result']
        elif any(term in query for term in ['leverage', 'influenc', 'cook', 'dffits', 'dfbetas']):
            answer = 'Influence measures identify observations whose removal could change a fitted regression. High leverage concerns predictor location; large residuals concern outcome disagreement. Review source data and justified sensitivity analyses rather than automatically deleting flagged cases.'
            sources = ['regression_result']
        elif any(term in query for term in ['vif', 'multicollinear', 'ridge', 'lasso']):
            answer = 'Collinearity can inflate coefficient uncertainty even when overall fit is strong. Review redundant features and the scientific model. Regularized predictive models shrink coefficients and should be compared using training-only preprocessing and validation, not treated as classical coefficient inference.'
            sources = ['regression_result', 'comparison_result']
        elif any(term in query for term in ['box-cox', 'box cox', 'transform', 'weighted least', 'wls', 'hc3']):
            answer = 'Transformations change the modeled response or predictor scale. Weighted least squares needs justified inverse-variance weights. Heteroscedasticity-consistent covariance adjusts coefficient uncertainty but does not fix dependence, bias, or individual prediction intervals. Review diagnostics after a justified refit.'
            sources = ['regression_result']
        elif any(term in query for term in ['prediction interval', 'predict new', 'mean interval']):
            answer = 'A mean confidence interval describes uncertainty in the conditional mean. A prediction interval also includes individual future error. Extrapolation and model assumptions matter. Inverse-transformed fitted values are not automatically bias-corrected means.'
            sources = ['regression_result']
        elif any(term in query for term in ['breusch', 'heteroscedastic']):
            answer = 'The Breusch–Pagan test looks for evidence that residual variance changes with predictors. A small p-value challenges the constant-variance assumption behind classical coefficient uncertainty. Review residual plots and consider an appropriate robust method; this app does not automatically change the fitted model.'
            sources = ['regression_result']
        elif any(term in query for term in ['jarque', 'normality']):
            answer = 'The Jarque–Bera test assesses whether residual skewness and kurtosis are compatible with normality. It does not verify independence, linearity or causality. Normality matters particularly for small-sample classical inference, and large samples can reveal minor departures.'
            sources = ['regression_result']
        elif any(term in query for term in ['r-squared', 'r squared', 'coefficient', 'residual', 'regression', 'diagnostic', 'breusch', 'jarque', 'normality', 'heteroscedastic', 'confidence']):
            answer = "Inferential regression describes conditional associations and in-sample fit. Coefficient uncertainty depends on assumptions; residual checks are evidence to review, not certificates of validity. Use the separate holdout comparison for predictive performance."
            sources = ['regression_result', 'comparison_result']
        elif any(term in query for term in ['classif', 'accuracy', 'recall', 'precision', 'f1', 'confusion']):
            answer = "Inspect balanced accuracy and macro-averaged F score alongside accuracy, especially for imbalanced targets. Confusion rows are actual classes and columns are predictions. Per-class recall helps reveal missed minority-class observations. A never-predicted class has precision reported as zero."
            sources = ['classification_result']
        elif any(term in query for term in ['cross', 'holdout', 'best', 'model', 'predict']):
            answer = "Compare models on the same holdout and inspect training-only cross-validation for stability. Lower prediction error is preferable; higher classification scores are preferable. Choosing models after inspecting a holdout makes it exploratory rather than independent confirmation. Fold variation is not a confidence interval."
            sources = ['comparison_result', 'classification_result']
        elif any(term in query for term in ['pca', 'cluster', 'component', 'silhouette']):
            answer = "PCA compresses standardized features along directions of variation. K-means groups similar standardized observations; the projection can hide structure in other dimensions. Cluster labels are arbitrary and silhouette is descriptive, not evidence of causal or natural populations."
            sources = ['exploration_result']
        elif any(term in query for term in ['missing', 'duplicate', 'clean']):
            answer = "Review missingness and repeated records before changing the working copy. Repeats may be meaningful. Predictive imputation and encoding are learned within training partitions; descriptive structure exploration uses all active rows."
            sources = ['data_quality']
        else:
            answer = "This offline guide explains the current evidence and common statistical concepts; it does not perform new analyses or answer arbitrary dataset questions. Run the relevant analysis first, then ask about significance, regression, model performance, missingness, PCA or clustering."
        cards = [c for c in context['evidence'] if not sources or c['source'] in sources]
        if not sources:
            # Cover each completed workflow before adding additional coefficient/model cards.
            representative = {}
            for card in cards:
                representative.setdefault(card['source'], card)
            selected = list(representative.values())
            cards = selected + [c for c in cards if c not in selected]
        if question.startswith('Explain the important findings'):
            answer = 'The evidence cards below summarize the completed analyses. Review missingness and study design first, distinguish descriptive fit from predictive validation, and treat model selection and clustering as exploratory. Numerical facts come directly from the statistical engine; no causal conclusion follows.'
        if sources and not cards:
            answer += " No completed analysis for that topic is available in the current summary."
        return {"answer": answer, "evidence_ids": [c['id'] for c in cards[:8]]}
