from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np


# Integer-coded variables that should be treated as categorical rather than numeric.
CODED_CATEGORICAL_COLS = [
    "detailed industry recode",
    "detailed occupation recode",
    "major industry code",
    "major occupation code",
    "own business or self employed",
    "veterans benefits",
    "year",
]

# True numeric variables identified during EDA.
TRUE_NUMERIC_COLS = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "dividends from stocks",
    "weight",
    "num persons worked for employer",
    "weeks worked in year",
]

# Columns that must not appear in model features.
LEAKAGE_COLS = [
    "label",
    "label_binary",
]

# Default weight column name.
DEFAULT_WEIGHT_COL = "weight"


@dataclass
class FeatureSchema:
    """
    Container for feature typing metadata used during preprocessing.
    """
    numeric_cols: List[str]
    categorical_cols: List[str]
    coded_categorical_cols: List[str]
    dropped_cols: List[str]
    weight_col: Optional[str]


def infer_feature_schema(
    df: pd.DataFrame,
    weight_col: str = DEFAULT_WEIGHT_COL,
) -> FeatureSchema:
    """
    Infer the modeling schema for the census dataset.

    This function enforces the feature semantics established during EDA:
    - integer-coded categorical variables are treated as categorical
    - 'year' is treated as categorical
    - 'label' and 'label_binary' are excluded from features
    - 'weight' is separated for sample weighting and can optionally be excluded
      from modeling features downstream

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    weight_col : str, default='weight'
        Name of the sampling weight column.

    Returns
    -------
    FeatureSchema
        Structured schema describing numeric, categorical, and dropped columns.
    """
    df_cols = set(df.columns)

    numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_candidates = df.select_dtypes(include=["object", "string"]).columns.tolist()

    coded_categorical_cols = [col for col in CODED_CATEGORICAL_COLS if col in df_cols]

    # Remove coded categorical variables from numeric candidates.
    numeric_cols = [col for col in numeric_candidates if col not in coded_categorical_cols]

    # Add coded categorical variables to the categorical side.
    categorical_cols = categorical_candidates + [
        col for col in coded_categorical_cols if col not in categorical_candidates
    ]

    # Remove leakage columns from both lists.
    numeric_cols = [col for col in numeric_cols if col not in LEAKAGE_COLS]
    categorical_cols = [col for col in categorical_cols if col not in LEAKAGE_COLS]

    # Keep only the numeric columns confirmed during EDA when available.
    true_numeric_cols = [col for col in TRUE_NUMERIC_COLS if col in df_cols]
    numeric_cols = [col for col in numeric_cols if col in true_numeric_cols]

    # Remove weight from feature columns by default; keep it separately for sample weighting.
    effective_weight_col = weight_col if weight_col in df_cols else None
    if effective_weight_col in numeric_cols:
        numeric_cols.remove(effective_weight_col)

    dropped_cols = [col for col in LEAKAGE_COLS if col in df_cols]
    if effective_weight_col is not None:
        dropped_cols.append(effective_weight_col)

    return FeatureSchema(
        numeric_cols=sorted(numeric_cols),
        categorical_cols=sorted(categorical_cols),
        coded_categorical_cols=sorted(coded_categorical_cols),
        dropped_cols=sorted(dropped_cols),
        weight_col=effective_weight_col,
    )


def prepare_feature_frame(
    df: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
) -> Tuple[pd.DataFrame, FeatureSchema]:
    """
    Create a clean feature dataframe using the inferred schema.

    The output dataframe contains:
    - true numeric variables
    - categorical variables (including coded categorical variables)

    It excludes:
    - label columns
    - sample weight column

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    schema : FeatureSchema or None, default=None
        Optional precomputed schema. If None, the schema is inferred.

    Returns
    -------
    Tuple[pd.DataFrame, FeatureSchema]
        Feature dataframe and the schema used to construct it.
    """
    if schema is None:
        schema = infer_feature_schema(df)

    selected_cols = schema.numeric_cols + schema.categorical_cols
    X = df[selected_cols].copy()

    return X, schema


def extract_target(
    df: pd.DataFrame,
    target_col: str = "label_binary",
) -> pd.Series:
    """
    Extract the target vector.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, default='label_binary'
        Name of the target column.

    Returns
    -------
    pd.Series
        Target series.

    Raises
    ------
    KeyError
        If the target column does not exist.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    return df[target_col].copy()


def extract_sample_weight(
    df: pd.DataFrame,
    weight_col: str = DEFAULT_WEIGHT_COL,
) -> Optional[pd.Series]:
    """
    Extract the sample weight column for weighted modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    weight_col : str, default='weight'
        Name of the weight column.

    Returns
    -------
    pd.Series or None
        Sample weight series if present; otherwise None.
    """
    if weight_col not in df.columns:
        return None

    return df[weight_col].copy()




def encode_categorical_features(
    X: pd.DataFrame,
    schema: FeatureSchema,
    drop_first: bool = False,
) -> Tuple[pd.DataFrame, Optional[OneHotEncoder]]:
    """
    One-hot encode categorical variables using sklearn encoder.

    Returns both encoded dataframe and fitted encoder.
    """

    numeric_part = (
        X[schema.numeric_cols].copy()
        if schema.numeric_cols
        else pd.DataFrame(index=X.index)
    )
    categorical_part = (
        X[schema.categorical_cols].copy()
        if schema.categorical_cols
        else pd.DataFrame(index=X.index)
    )

    if categorical_part.empty:
        return numeric_part, None

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first" if drop_first else None,
        sparse_output=False
    )

    encoded_array = encoder.fit_transform(categorical_part)
    encoded_cols = encoder.get_feature_names_out(schema.categorical_cols)

    categorical_encoded = pd.DataFrame(
        encoded_array,
        columns=encoded_cols,
        index=X.index
    )

    X_encoded = pd.concat([numeric_part, categorical_encoded], axis=1)

    return X_encoded, encoder


def preprocess_for_modeling(
    df: pd.DataFrame,
    target_col: str = "label_binary",
    weight_col: str = DEFAULT_WEIGHT_COL,
    drop_first: bool = False,
) -> Dict[str, object]:
    """
    Full preprocessing pipeline for classification modeling.

    Steps
    -----
    1. Infer feature schema
    2. Build feature frame
    3. Extract target
    4. Extract sample weights
    5. One-hot encode categorical variables

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, default='label_binary'
        Target column name.
    weight_col : str, default='weight'
        Sample weight column name.
    drop_first : bool, default=False
        Whether to drop first dummy level during one-hot encoding.

    Returns
    -------
    Dict[str, object]
        Dictionary with:
        - X_raw: pre-encoded feature dataframe
        - X_encoded: encoded feature matrix
        - y: target series
        - sample_weight: weight series or None
        - schema: FeatureSchema object
        - encoder: fitted OneHotEncoder or None
    """
    schema = infer_feature_schema(df, weight_col=weight_col)
    X_raw, schema = prepare_feature_frame(df, schema=schema)
    # Simple missing value handling
    for col in schema.numeric_cols:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col].fillna(X_raw[col].median())

    for col in schema.categorical_cols:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col].fillna("Missing")
    y = extract_target(df, target_col=target_col)
    sample_weight = extract_sample_weight(df, weight_col=weight_col)
    X_encoded, encoder = encode_categorical_features(X_raw, schema, drop_first=drop_first)

    return {
        "X_raw": X_raw,
        "X_encoded": X_encoded,
        "y": y,
        "sample_weight": sample_weight,
        "schema": schema,
        "encoder": encoder,
    }


def prepare_segmentation_features(
    df: pd.DataFrame,
    include_weight: bool = False,
) -> pd.DataFrame:
    """
    Prepare interpretable features for segmentation.

    Selected variables are based on marketing interpretability and behavioral relevance
    (e.g., demographics and labor market participation), rather than predictive power alone.
    """

    segmentation_cols = [
        "age",
        "education",
        "marital stat",
        "class of worker",
        "major occupation code",
        "major industry code",
        "full or part time employment stat",
        "weeks worked in year",
        "sex",
        "citizenship",
        "capital gains",
        "dividends from stocks",
    ]

    if include_weight and DEFAULT_WEIGHT_COL in df.columns:
        segmentation_cols.append(DEFAULT_WEIGHT_COL)

    segmentation_cols = [col for col in segmentation_cols if col in df.columns]

    return df[segmentation_cols].copy()


if __name__ == "__main__":
    from load_data import load_project_data

    df = load_project_data(add_target=True)

    processed = preprocess_for_modeling(df)

    X_raw = processed["X_raw"]
    X_encoded = processed["X_encoded"]
    y = processed["y"]
    sample_weight = processed["sample_weight"]
    schema = processed["schema"]

    print("Preprocessing completed successfully.")
    print(f"Raw feature shape: {X_raw.shape}")
    print(f"Encoded feature shape: {X_encoded.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sample weight available: {sample_weight is not None}")
    print(f"Encoder available: {processed['encoder'] is not None}")

    print("\nNumeric columns:")
    print(schema.numeric_cols)

    print("\nCategorical columns:")
    print(schema.categorical_cols[:10], "..." if len(schema.categorical_cols) > 10 else "")

    print("\nDropped columns:")
    print(schema.dropped_cols)