from __future__ import annotations
import json
import os
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# --------- Rutas estándar SageMaker ----------
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
METRICS_DIR = "/opt/ml/output/metrics"


# =========================
# Features/Target helpers
# =========================
def split_X_y(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Tuple[str, ...] = ("datetime", "machineID"),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa X e y eliminando columnas de ID/tiempo si aparecen.
    """
    feats = [c for c in df.columns if c not in set(drop_cols + (target_col,))]
    X = df[feats].copy()
    y = df[target_col].copy()
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    One-hot para categóricas; passthrough para numéricas.
    """
    cat_cols = [
        c
        for c in X.columns
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c])
    ]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer(
        transformers=[("cat", ohe, cat_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return pre


# =========================
# Métricas / evaluación
# =========================
def evaluate_multiclass(
    y_true: pd.Series, y_pred: np.ndarray, labels: List[str]
) -> dict:
    """
    Métricas multiclass + matriz de confusión, listo para volcar a JSON.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "multiclass_classification_metrics": {
            "accuracy": {"value": float(acc)},
            "macro_precision": {"value": float(macro_prec)},
            "macro_recall": {"value": float(macro_rec)},
            "macro_f1": {"value": float(macro_f1)},
            "per_class": {
                "labels": labels,
                "precision": [float(x) for x in prec],
                "recall": [float(x) for x in rec],
                "f1": [float(x) for x in f1],
            },
            "confusion_matrix": {"labels": labels, "matrix": cm},
        }
    }


# =========================
# Persistencia (modelo/metrics)
# =========================
def save_model(pipeline, filename: str = "model.joblib", model_dir: str | None = None) -> str:
    """
    Guarda el modelo (pipeline) en SM_MODEL_DIR (o ruta dada) y devuelve path absoluto.
    """
    model_dir = model_dir or SM_MODEL_DIR
    ensure_dir(model_dir)
    path = os.path.join(model_dir, filename)
    joblib.dump(pipeline, path)
    return path


def save_metrics(metrics: dict, filename: str = "evaluation.json", out_dir: str = METRICS_DIR) -> str:
    """
    Serializa métricas en JSON para SageMaker ModelMetrics.
    """
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return out_path

# ---------- Helpers ----------
def _resolve_csv_path(path_or_prefix: str) -> str:
    """
    Si recibimos un prefijo (termina con '/'), anexar 'data.csv'.
    Si ya viene un .csv, lo devolvemos tal cual.
    """
    if path_or_prefix.endswith(".csv"):
        return path_or_prefix
    if not path_or_prefix.endswith("/"):
        path_or_prefix = path_or_prefix + "/"
    return path_or_prefix + "data.csv"


def load_split(csv_path: str) -> pd.DataFrame:
    """Cargar split (S3 o local). Requiere s3fs/pyarrow si es S3."""
    path = _resolve_csv_path(csv_path)
    df = pd.read_csv(path)
    # tipificar columnas clave si existen
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    # asegurar tipo categórico (evita OneHot de valores num convertidos a str)
    if "model" in df.columns and not pd.api.types.is_categorical_dtype(df["model"]):
        df["model"] = df["model"].astype("category")
    if "failure" in df.columns and not pd.api.types.is_categorical_dtype(df["failure"]):
        df["failure"] = df["failure"].astype("category")
    return df


def train_lgbm_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """Entrenar Pipeline(preprocesamiento + LightGBM)."""
    pre = build_preprocessor(X_train)
    lgbm = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,  # ayuda con desbalance
    )
    pipe = Pipeline(steps=[("preprocess", pre), ("model", lgbm)])
    pipe.fit(X_train, y_train)
    return pipe