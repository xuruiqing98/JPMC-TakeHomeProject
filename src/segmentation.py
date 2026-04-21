# src/segmentation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ======================
# 1. Feature Selection
# ======================
def prepare_segmentation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select interpretable features for segmentation.
    
    Focus on demographic + labor-force variables (business meaningful),
    and avoid high-cardinality explosion.
    """

    selected_cols = [
        "age",
        "education",
        "marital stat",
        "class of worker",
        "full or part time employment stat",
        "weeks worked in year",
        "weight"
    ]

    df_seg = df[selected_cols].copy()

    return df_seg


# ======================
# 2. Encoding
# ======================
def encode_features(df_seg: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical variables.
    """

    categorical_cols = df_seg.select_dtypes(include=["object", "string"]).columns.tolist()

    # 不编码 weight
    if "weight" in categorical_cols:
        categorical_cols.remove("weight")

    df_encoded = pd.get_dummies(df_seg, columns=categorical_cols, drop_first=True)

    return df_encoded


# ======================
# 3. Find Best K
# ======================
def find_best_k(X_scaled, k_range=range(3, 7)):
    """
    Use silhouette score to select optimal K.
    """

    best_k = None
    best_score = -1

    print("\nSelecting best K using silhouette score:")

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)
        print(f"K = {k}, Silhouette Score = {score:.4f}")

        if score > best_score:
            best_k = k
            best_score = score

    print(f"\nBest K selected: {best_k}")

    return best_k


# ======================
# 4. Run Clustering
# ======================
def run_kmeans(df_encoded: pd.DataFrame, n_clusters: int):
    scaler = StandardScaler()

    X = df_encoded.drop(columns=["weight"])
    weights = df_encoded["weight"]

    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = model.fit_predict(X_scaled)

    return clusters, model, X_scaled


# ======================
# 5. Weighted Summary
# ======================
def summarize_clusters(df_original: pd.DataFrame, clusters: np.ndarray):
    """
    Weighted cluster profiling (CRITICAL for census data).
    """

    df = df_original.copy()
    df["cluster"] = clusters

    print("\n=== Cluster Population Share (Weighted) ===")
    cluster_weight = df.groupby("cluster")["weight"].sum()
    cluster_pct = cluster_weight / cluster_weight.sum()
    print(cluster_pct)

    print("\n=== Weighted Numeric Summary ===")

    def weighted_mean(x, w):
        return np.sum(x * w) / np.sum(w)

    numeric_cols = ["age", "weeks worked in year"]

    for col in numeric_cols:
        values = df.groupby("cluster").apply(
            lambda g: weighted_mean(g[col], g["weight"])
        )
        print(f"\n{col}:")
        print(values)

    print("\n=== Top Categories per Cluster ===")

    categorical_cols = df.select_dtypes(include=["object","string"]).columns

    for col in categorical_cols:
        print(f"\n{col}:")
        top_values = df.groupby("cluster").apply(
            lambda g: g[col].value_counts().index[0]
        )
        print(top_values)


# ======================
# 6. Main Pipeline
# ======================
def main():
    print("Running segmentation pipeline...")

    from load_data import load_project_data
    df = load_project_data(add_target=False)

    df_seg = prepare_segmentation_features(df)
    df_encoded = encode_features(df_seg)

    # Scale first for K selection
    scaler = StandardScaler()
    X_temp = scaler.fit_transform(df_encoded.drop(columns=["weight"]))

    best_k = find_best_k(X_temp)

    clusters, model, _ = run_kmeans(df_encoded, best_k)

    summarize_clusters(df_seg, clusters)

    print("\n[Insight] Clusters reflect lifecycle and labor-force participation patterns rather than purely income segmentation.")
    print("\n[Business Application]")
    print("Segmentation can be combined with model predictions to guide targeting decisions.")
    print("For example, high-probability individuals within high-value clusters can be prioritized,")
    print("while low-priority segments can be deprioritized to optimize marketing ROI.")

if __name__ == "__main__":
    main()