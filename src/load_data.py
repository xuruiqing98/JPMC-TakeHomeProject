from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_column_names(columns_path: str | Path) -> List[str]:
    """
    Load column names from the census column file.

    Parameters
    ----------
    columns_path : str or Path
        Path to the file containing one column name per line.

    Returns
    -------
    List[str]
        Cleaned list of column names.

    Raises
    ------
    FileNotFoundError
        If the columns file does not exist.
    ValueError
        If no valid column names are found.
    """
    columns_path = Path(columns_path)

    if not columns_path.exists():
        raise FileNotFoundError(f"Columns file not found: {columns_path}")

    with columns_path.open("r", encoding="utf-8") as f:
        columns = [line.strip() for line in f if line.strip()]

    if not columns:
        raise ValueError(f"No column names found in: {columns_path}")

    return columns


def load_raw_data(data_path: str | Path, columns_path: str | Path) -> pd.DataFrame:
    """
    Load the raw census dataset without modifying the source data.

    Parameters
    ----------
    data_path : str or Path
        Path to the raw data file.
    columns_path : str or Path
        Path to the columns file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with assigned column names.

    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If the number of loaded columns does not match the dataframe width.
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    columns = load_column_names(columns_path)

    df = pd.read_csv(
        data_path,
        header=None,
        names=columns,
        sep=",",
        skipinitialspace=True,
        encoding="utf-8",
    )

    if df.shape[1] != len(columns):
        raise ValueError(
            f"Column mismatch: dataframe has {df.shape[1]} columns, "
            f"but {len(columns)} column names were loaded."
        )

    return df


def create_eda_copy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a safe copy of the raw dataframe for exploratory analysis.

    This ensures that the original loaded dataframe remains unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Independent copy of the dataframe.
    """
    return df.copy()


def add_binary_label(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Add a binary income label for analysis or modeling.

    Mapping
    -------
    0 -> income <= 50K
    1 -> income > 50K

    This function returns a new dataframe and does not modify the input in place.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_col : str, default="label"
        Name of the original label column.

    Returns
    -------
    pd.DataFrame
        Dataframe with an added 'label_binary' column.

    Raises
    ------
    KeyError
        If the label column does not exist.
    """
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataframe.")

    df_out = df.copy()

    df_out["label_binary"] = df_out[label_col].apply(
        lambda x: 1 if "50000+" in str(x) else 0
    )

    return df_out


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "label_binary",
    drop_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    target_col : str, default="label_binary"
        Name of the target column.
    drop_cols : list[str] or None, default=None
        Additional columns to remove from X.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X, y

    Raises
    ------
    KeyError
        If the target column does not exist.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    drop_cols = drop_cols or []

    cols_to_drop = [target_col] + drop_cols
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    return X, y


def get_default_paths(project_root: str | Path | None = None) -> Tuple[Path, Path]:
    """
    Return default paths for the raw data and column files.

    Parameters
    ----------
    project_root : str or Path or None
        Root folder of the project. If None, infer it from this file location.

    Returns
    -------
    Tuple[Path, Path]
        data_path, columns_path
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    else:
        project_root = Path(project_root)

    data_path = project_root / "data" / "census-bureau.data"
    columns_path = project_root / "data" / "census-bureau.columns"

    return data_path, columns_path


def load_project_data(
    project_root: str | Path | None = None,
    add_target: bool = False,
) -> pd.DataFrame:
    """
    Convenience function to load project data using the standard folder structure.

    Parameters
    ----------
    project_root : str or Path or None
        Root folder of the project. If None, infer it automatically.
    add_target : bool, default=False
        Whether to add the binary target column.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    data_path, columns_path = get_default_paths(project_root)
    df = load_raw_data(data_path, columns_path)

    if add_target:
        df = add_binary_label(df)

    return df


if __name__ == "__main__":
    df_raw = load_project_data(add_target=False)
    print("Raw dataset loaded successfully.")
    print(f"Shape: {df_raw.shape}")
    print("\nFirst 5 rows:")
    print(df_raw.head())

    df_with_target = add_binary_label(df_raw)
    print("\n'label_binary' column added successfully.")
    print(df_with_target["label_binary"].value_counts())