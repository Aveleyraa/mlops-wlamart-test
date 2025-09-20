from __future__ import annotations
import json, os
from typing import Dict, List, Optional
import pandas as pd
from pipelines.predictive_maintenance.utils.utils_preprocess import (
    build_telemetry_features,
    build_error_counts,
    build_component_age_days,
    assemble_features,
)

def load_feature_order(path_or_json: str) -> List[str]:
    """
    Load the feature list used during training, preserving order.

    Accepts either:
      - A path to a JSON file, or
      - A JSON string (array or pandas Series).

    Parameters
    ----------
    path_or_json : str
        Path to JSON file or JSON string.

    Returns
    -------
    List[str]
        Ordered list of feature names.

    Raises
    ------
    ValueError
        If input cannot be parsed into a list.
    """
    if os.path.exists(path_or_json):
        return pd.read_json(path_or_json, typ="series").tolist()
    try:
        return pd.read_json(path_or_json, typ="series").tolist()
    except Exception:
        data = json.loads(path_or_json)
        if isinstance(data, list):
            return data
        raise ValueError("feature_order must be a JSON array (list of strings).")

def align_columns(df: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    """Asegura columnas/orden de entrenamiento; rellena faltantes con 0."""
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    # elimina columnas extra si el modelo no las espera
    return df[feature_order]



def align_columns(df: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    """
    Align a DataFrame to the training feature order.

    - Adds any missing columns (filled with 0).
    - Reorders columns to match `feature_order` and drops extras.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    feature_order : List[str]
        Expected training column order.

    Returns
    -------
    pd.DataFrame
        DataFrame with the exact columns and order required by the model.
    """
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    return df[feature_order]

def features_from_raw_window(
    telemetry_window: pd.DataFrame,
    errors_window: pd.DataFrame,
    maint_window: pd.DataFrame,
    machines_dim: pd.DataFrame,
    feature_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Build model-ready features from recent raw windows (telemetry, errors, maintenance).

    - Ensures datetime dtypes.
    - Computes telemetry features (e.g., 3h/24h), error counts, and component age.
    - Joins with machine dimensions.
    - Optionally aligns columns to `feature_order`.

    Parameters
    ----------
    telemetry_window : pd.DataFrame
        Recent telemetry rows (must include 'datetime' and machine identifier).
    errors_window : pd.DataFrame
        Recent error logs (must include 'datetime' and machine identifier).
    maint_window : pd.DataFrame
        Recent maintenance records (must include 'datetime' and machine identifier).
    machines_dim : pd.DataFrame
        Machine dimension table for static attributes/keys.
    feature_order : Optional[List[str]]
        Exact training column order to enforce (adds missing with 0, drops extras).

    Returns
    -------
    pd.DataFrame
        Feature matrix matching training logic (and order if provided).
    """
    # Ensure datetime dtype
    for df, col in [(telemetry_window, "datetime"), (errors_window, "datetime"), (maint_window, "datetime")]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    telemetry_feat = build_telemetry_features(telemetry_window)
    error_counts   = build_error_counts(errors_window, telemetry_window)
    comp_age       = build_component_age_days(maint_window, telemetry_window)

    features = assemble_features(telemetry_feat, error_counts, comp_age, machines_dim)
    if feature_order:
        features = align_columns(features, feature_order)

    return features
