from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from load_data import load_project_data
from preprocess import infer_feature_schema, prepare_feature_frame


RANDOM_STATE = 42
TARGET_COL = "label_binary"
WEIGHT_COL = "weight"


def make_one_hot_encoder(drop_first: bool = False) -> OneHotEncoder:
    """
    Create a OneHotEncoder that is compatible across sklearn versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first" if drop_first else None,
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first" if drop_first else None,
            sparse=False,
        )


def fit_preprocessor(
    df_train: pd.DataFrame,
    target_col: str = TARGET_COL,
    weight_col: str = WEIGHT_COL,
    drop_first: bool = False,
) -> Dict[str, object]:
    """
    Fit preprocessing objects using training data only.

    Returns
    -------
    Dict[str, object]
        Contains schema, fitted encoder, fill values, and encoded training matrix.
    """
    schema = infer_feature_schema(df_train, weight_col=weight_col)
    X_train_raw, schema = prepare_feature_frame(df_train, schema=schema)

    y_train = df_train[target_col].copy()
    sample_weight_train = df_train[weight_col].copy() if weight_col in df_train.columns else None

    # Fit fill values on training data only
    numeric_fill_values = {
        col: X_train_raw[col].median()
        for col in schema.numeric_cols
        if col in X_train_raw.columns
    }
    categorical_fill_value = "Missing"

    # Apply missing value handling
    X_train_filled = X_train_raw.copy()
    for col, median_value in numeric_fill_values.items():
        X_train_filled[col] = X_train_filled[col].fillna(median_value)

    for col in schema.categorical_cols:
        if col in X_train_filled.columns:
            X_train_filled[col] = X_train_filled[col].fillna(categorical_fill_value)

    # Fit encoder on training categorical data only
    encoder = None
    if schema.categorical_cols:
        encoder = make_one_hot_encoder(drop_first=drop_first)
        encoder.fit(X_train_filled[schema.categorical_cols])

    X_train_encoded = transform_features(
        df=df_train,
        schema=schema,
        encoder=encoder,
        numeric_fill_values=numeric_fill_values,
        categorical_fill_value=categorical_fill_value,
    )

    return {
        "schema": schema,
        "encoder": encoder,
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_value": categorical_fill_value,
        "X_train_encoded": X_train_encoded,
        "y_train": y_train,
        "sample_weight_train": sample_weight_train,
    }


def transform_features(
    df: pd.DataFrame,
    schema,
    encoder: Optional[OneHotEncoder],
    numeric_fill_values: Dict[str, float],
    categorical_fill_value: str = "Missing",
) -> pd.DataFrame:
    """
    Transform a dataframe using a fitted preprocessing setup.
    """
    X_raw, _ = prepare_feature_frame(df, schema=schema)
    X_filled = X_raw.copy()

    for col, median_value in numeric_fill_values.items():
        if col in X_filled.columns:
            X_filled[col] = X_filled[col].fillna(median_value)

    for col in schema.categorical_cols:
        if col in X_filled.columns:
            X_filled[col] = X_filled[col].fillna(categorical_fill_value)

    numeric_part = (
        X_filled[schema.numeric_cols].copy()
        if schema.numeric_cols
        else pd.DataFrame(index=X_filled.index)
    )

    if encoder is None or not schema.categorical_cols:
        return numeric_part

    encoded_array = encoder.transform(X_filled[schema.categorical_cols])
    encoded_cols = encoder.get_feature_names_out(schema.categorical_cols)

    categorical_encoded = pd.DataFrame(
        encoded_array,
        columns=encoded_cols,
        index=X_filled.index,
    )

    X_encoded = pd.concat([numeric_part, categorical_encoded], axis=1)
    return X_encoded


def get_candidate_models() -> Dict[str, object]:
    """
    Define candidate models for comparison.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # Optional: include XGBoost if installed
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def find_best_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    sample_weight: Optional[pd.Series] = None,
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes F1 score on validation data.
    """
    thresholds = np.linspace(0.10, 0.90, 81)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_predictions(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    sample_weight: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics at a given decision threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "precision": float(precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob, sample_weight=sample_weight)),
    }

    return metrics


def get_feature_importance(
    model,
    feature_names: pd.Index,
) -> pd.DataFrame:
    """
    Extract feature importance from the trained model when available.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return importance_df


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    models_dir = project_root / "models"

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_project_data(add_target=True)

    # Split raw data first to avoid preprocessing leakage
    df_train_valid, df_test = train_test_split(
        df,
        test_size=0.20,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    df_train, df_valid = train_test_split(
        df_train_valid,
        test_size=0.25,  # 0.25 of 0.80 = 0.20
        stratify=df_train_valid[TARGET_COL],
        random_state=RANDOM_STATE,
    )

    print("Data split completed.")
    print(f"Train shape: {df_train.shape}")
    print(f"Validation shape: {df_valid.shape}")
    print(f"Test shape: {df_test.shape}")

    # Fit preprocessing on training data only
    prep = fit_preprocessor(df_train, target_col=TARGET_COL, weight_col=WEIGHT_COL, drop_first=False)

    schema = prep["schema"]
    encoder = prep["encoder"]
    numeric_fill_values = prep["numeric_fill_values"]
    categorical_fill_value = prep["categorical_fill_value"]

    X_train = prep["X_train_encoded"]
    y_train = prep["y_train"]
    w_train = prep["sample_weight_train"]

    X_valid = transform_features(
        df=df_valid,
        schema=schema,
        encoder=encoder,
        numeric_fill_values=numeric_fill_values,
        categorical_fill_value=categorical_fill_value,
    )
    y_valid = df_valid[TARGET_COL].copy()
    w_valid = df_valid[WEIGHT_COL].copy()

    X_test = transform_features(
        df=df_test,
        schema=schema,
        encoder=encoder,
        numeric_fill_values=numeric_fill_values,
        categorical_fill_value=categorical_fill_value,
    )
    y_test = df_test[TARGET_COL].copy()
    w_test = df_test[WEIGHT_COL].copy()

    assert X_train.isnull().sum().sum() == 0, "Training features contain NaN values."
    assert X_valid.isnull().sum().sum() == 0, "Validation features contain NaN values."
    assert X_test.isnull().sum().sum() == 0, "Test features contain NaN values."

    models = get_candidate_models()

    model_selection_results = {}
    best_model_name = None
    best_model = None
    best_threshold = 0.50
    best_valid_f1 = -1.0

    # Train and validate each candidate
    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")

        model.fit(X_train, y_train, sample_weight=w_train)

        valid_prob = model.predict_proba(X_valid)[:, 1]
        threshold, valid_best_f1 = find_best_threshold(
            y_valid,
            valid_prob,
            sample_weight=w_valid
        )
        valid_metrics = evaluate_predictions(
            y_valid,
            valid_prob,
            threshold,
            sample_weight=w_valid
        )

        print(f"Validation metrics for {model_name}:")
        print(json.dumps(valid_metrics, indent=2))

        thresholds = [0.3, 0.5, 0.7, 0.85]
        threshold_sensitivity = {}

        print("\nThreshold sensitivity analysis:")

        for t in thresholds:
            metrics = evaluate_predictions(y_valid, valid_prob, t, sample_weight=w_valid)
            threshold_sensitivity[str(t)] = metrics
            print(f"\nThreshold = {t}")
            print(metrics)

        print("\n[Business Interpretation]")
        print(
            f"For {model_name}, the selected threshold ({threshold:.2f}) is treated as a "
            "business decision, not only a statistical setting."
        )
        print(
            "A higher threshold prioritizes precision and reduces unnecessary marketing cost, "
            "while a lower threshold improves recall but may increase wasted outreach."
        )

        model_selection_results[model_name] = {
            "validation_metrics": valid_metrics,
            "threshold_sensitivity": threshold_sensitivity,
        }

        if valid_best_f1 > best_valid_f1:
            best_valid_f1 = valid_best_f1
            best_model_name = model_name
            best_model = model
            best_threshold = threshold

    print("\nBest model selected:")
    print(f"Model: {best_model_name}")
    print(f"Validation best threshold: {best_threshold:.3f}")
    print(f"Validation best F1: {best_valid_f1:.4f}")
    print("Selection is based on both predictive performance and business usability.")

    print("\n[Business Decision]")
    print(
        f"The final threshold ({best_threshold:.2f}) is selected to balance precision and recall "
        "from a marketing resource allocation perspective."
    )
    print(
        "This means the model is used to prioritize likely high-value individuals while limiting "
        "unnecessary spending on lower-probability cases."
    )

    if best_model_name == "logistic_regression":
        print("\n[Model Selection Rationale]")
        print(
            "Logistic Regression is selected not only for competitive performance, but also for "
            "interpretability, stability, and ease of deployment."
        )
        print(
            "In decision-critical business settings, transparent models are often preferred over "
            "slightly more complex alternatives with limited practical gain."
        )

    # Refit best model on train + validation data
    df_train_full = pd.concat([df_train, df_valid], axis=0)

    prep_full = fit_preprocessor(
        df_train_full,
        target_col=TARGET_COL,
        weight_col=WEIGHT_COL,
        drop_first=False,
    )

    schema_full = prep_full["schema"]
    encoder_full = prep_full["encoder"]
    numeric_fill_values_full = prep_full["numeric_fill_values"]
    categorical_fill_value_full = prep_full["categorical_fill_value"]

    X_train_full = prep_full["X_train_encoded"]
    y_train_full = prep_full["y_train"]
    w_train_full = prep_full["sample_weight_train"]

    X_test_final = transform_features(
        df=df_test,
        schema=schema_full,
        encoder=encoder_full,
        numeric_fill_values=numeric_fill_values_full,
        categorical_fill_value=categorical_fill_value_full,
    )

    best_model_final = get_candidate_models()[best_model_name]
    best_model_final.fit(X_train_full, y_train_full, sample_weight=w_train_full)

    test_prob = best_model_final.predict_proba(X_test_final)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_prob, best_threshold, sample_weight=w_test)
    
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))

    # Save metrics
    metrics_output = {
        "best_model": best_model_name,
        "selected_threshold": best_threshold,
        "validation_results": model_selection_results,
        "test_metrics": test_metrics,
    }

    with open(results_dir / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    # Save feature importance
    feature_importance_df = get_feature_importance(best_model_final, X_train_full.columns)
    if not feature_importance_df.empty:
        feature_importance_df.to_csv(results_dir / "feature_importance.csv", index=False)

    # Save model artifact
    artifact = {
        "model": best_model_final,
        "schema": schema_full,
        "encoder": encoder_full,
        "numeric_fill_values": numeric_fill_values_full,
        "categorical_fill_value": categorical_fill_value_full,
        "threshold": best_threshold,
        "feature_names": X_train_full.columns.tolist(),
        "target_col": TARGET_COL,
        "weight_col": WEIGHT_COL,
    }

    joblib.dump(artifact, models_dir / "best_model.joblib")

    print("\nArtifacts saved:")
    print(f"- Metrics: {results_dir / 'model_metrics.json'}")
    if not feature_importance_df.empty:
        print(f"- Feature importance: {results_dir / 'feature_importance.csv'}")
    print(f"- Model artifact: {models_dir / 'best_model.joblib'}")


if __name__ == "__main__":
    main()