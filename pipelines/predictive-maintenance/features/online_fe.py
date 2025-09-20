# pipelines/predictive_maintenance/features/online_fe.py
from __future__ import annotations
import json, os
from typing import Dict, List, Optional
import pandas as pd

# Reusa tus funciones de FE (NO dupliques lógica)
from pipelines.predictive_maintenance.utils.utils_preprocess import (
    build_telemetry_features,
    build_error_counts,
    build_component_age_days,
    assemble_features,
)

def load_feature_order(path_or_json: str) -> List[str]:
    """
    Carga la lista de columnas usadas en entrenamiento (en el MISMO orden).
    Acepta ruta local o string JSON.
    """
    if os.path.exists(path_or_json):
        return pd.read_json(path_or_json, typ="series").tolist()
    try:
        return pd.read_json(path_or_json, typ="series").tolist()
    except Exception:
        data = json.loads(path_or_json)
        if isinstance(data, list):
            return data
        raise ValueError("feature_order no es una lista válida")

def align_columns(df: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    """Asegura columnas/orden de entrenamiento; rellena faltantes con 0."""
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    # elimina columnas extra si el modelo no las espera
    return df[feature_order]

# ---------- Dos modos de entrada ----------

def features_from_already_aggregated(df_features: pd.DataFrame,
                                     feature_order: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Caso simple: el cliente te manda ya las columnas agregadas tal cual las usaste en train.
    """
    X = df_features.copy()
    if feature_order:
        X = align_columns(X, feature_order)
    return X

def features_from_raw_window(
    telemetry_window: pd.DataFrame,
    errors_window: pd.DataFrame,
    maint_window: pd.DataFrame,
    machines_dim: pd.DataFrame,
    feature_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Caso 'crudo': recibes últimas 24–48h de telemetría/errores/mantenimientos para uno o varios machineID.
    Construye EXACTAMENTE las features como en training (3H/24H).
    """
    # Asegura dtypes
    for df, col in [(telemetry_window, "datetime"), (errors_window, "datetime"), (maint_window, "datetime")]:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])

    telemetry_feat = build_telemetry_features(telemetry_window)
    error_counts   = build_error_counts(errors_window, telemetry_window)
    comp_age       = build_component_age_days(maint_window, telemetry_window)

    features = assemble_features(telemetry_feat, error_counts, comp_age, machines_dim)

    # Elimina columnas que no son de entrada al modelo (ej. machineID, datetime, categóricas crudas)
    # y/o mapea categóricas igual que en training si aplica
    # Aquí asumo que en training ya hiciste one-hot o encoding y guardaste el resultado numérico.

    # Si guardaste feature_order.json en training, úsalo:
    if feature_order:
        features = align_columns(features, feature_order)

    return features
