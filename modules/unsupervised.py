"""Exploratory PCA and K-means; aggregate exports exclude row projections/assignments."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def explore_structure(frame, features, clusters=3, seed=42):
    features = list(features)
    if not 2 <= len(features) <= 20 or len(set(features)) != len(features):
        raise ValueError("Choose 2–20 distinct numeric features.")
    if any(c not in frame or not pd.api.types.is_numeric_dtype(frame[c]) for c in features):
        raise ValueError("PCA and K-means currently support numeric features only.")
    data = frame[features].astype(float)
    if len(data) < 10 or not 2 <= clusters <= 8 or len(data) <= clusters:
        raise ValueError("Need at least 10 rows and 2–8 clusters, fewer than the row count.")
    if np.isinf(data.to_numpy()).any() or any(data[c].nunique() < 2 for c in features):
        raise ValueError("Every feature needs finite, varying observed values; empty/constant features are unsupported.")
    imputed = SimpleImputer(strategy="median").fit_transform(data)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(imputed)
    if len(np.unique(standardized, axis=0)) < clusters:
        raise ValueError("Not enough distinct feature vectors for this cluster count.")
    with threadpool_limits(limits=2):
        pca = PCA(n_components=min(len(features), len(data)), svd_solver="full")
        coordinates = pca.fit_transform(standardized)
        kmeans = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(standardized)
        labels = kmeans.labels_
        if len(np.unique(labels)) != clusters:
            raise ValueError("K-means did not produce the requested number of clusters. Review repeated rows or choose fewer clusters.")
        try:
            silhouette = float(silhouette_score(standardized, labels, sample_size=min(2000, len(data)), random_state=seed))
        except ValueError:
            silhouette = None
    ratios = pca.explained_variance_ratio_
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    result = {
        "method": "Standardized PCA and K-means exploration", "features": features, "observations": len(data),
        "imputed_cells": int(data.isna().sum().sum()), "clusters": clusters, "seed": seed,
        "explained_variance": [{"component": f"PC{i+1}", "ratio": float(r), "cumulative_ratio": float(ratios[:i+1].sum())} for i, r in enumerate(ratios)],
        "component_weights": [{"feature": name, "PC1": float(pca.components_[0, i]), "PC2": float(pca.components_[1, i])} for i, name in enumerate(features)],
        "cluster_sizes": [{"cluster": str(i+1), "rows": int((labels == i).sum())} for i in range(clusters)],
        "cluster_centers": [{"cluster": str(i+1), "feature": name, "value": float(centers[i, j])} for i in range(clusters) for j, name in enumerate(features)],
        "inertia_standardized": float(kmeans.inertia_), "silhouette": silhouette,
        "silhouette_sample_limit": 2000,
        "preprocessing": "Median imputation and z-score standardization fitted on all active rows for descriptive exploration; no holdout prediction is claimed.",
        "scope": "PCA weights refer to standardized features; component signs are arbitrary. K-means operates in the full standardized feature space, not just the two plotted components. Centers are back-transformed to original feature units. Cluster labels are arbitrary and do not prove natural groups or causation. Inertia depends on feature scaling and cluster count. Silhouette ranges from minus one to one and uses at most a reproducible 2000-row sample; it is unavailable if that sample cannot support the metric. No cluster-count selection was performed. Row coordinates/assignments stay in session memory and are not included in aggregate report exports.",
    }
    plots = pd.DataFrame({"PC1": coordinates[:, 0], "PC2": coordinates[:, 1], "Cluster": (labels+1).astype(str)})
    return result, plots
