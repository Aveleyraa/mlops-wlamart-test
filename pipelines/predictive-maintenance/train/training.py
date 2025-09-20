#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py
--------
Consume splits de preprocess.py (CSV en S3), entrena LightGBM multiclass
y emite:
  - Modelo: /opt/ml/model/model.joblib  (SageMaker lo empacará como model.tar.gz)
  - Métricas: /opt/ml/output/metrics/evaluation.json

Ejemplo de invocación (SageMaker SKLearn Estimator hyperparameters):
  --train-data s3://.../features/train/
  --validate-data s3://.../features/validate/
  --test-data s3://.../features/test/
  --target-col failure
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from utils.utils_training import (
    load_split,
    split_X_y,
    train_lgbm_pipeline,
    evaluate_multiclass,
    save_model,
    save_metrics,
)

# ---------- Paths SageMaker ----------
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
METRICS_DIR = "/opt/ml/output/metrics"
EVAL_JSON = os.path.join(METRICS_DIR, "evaluation.json")


# ---------- Main ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM multiclass from preprocessed CSVs.")
    p.add_argument("--train-data", required=True, help="Ruta a train/ o train/data.csv")
    p.add_argument("--validate-data", required=True, help="Ruta a validate/ o validate/data.csv")
    p.add_argument("--test-data", required=True, help="Ruta a test/ o test/data.csv")
    p.add_argument("--target-col", default="failure", help="Nombre de la columna objetivo")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Cargar splits
    print("[data] loading splits ...")
    df_train = load_split(args.train_data)
    df_val = load_split(args.validate_data)
    df_test = load_split(args.test_data)

    # 2) Preparar X/y
    drop_cols = ("datetime", "machineID")
    X_train, y_train = split_X_y(df_train, args.target_col, drop_cols)
    X_val, y_val = split_X_y(df_val, args.target_col, drop_cols)
    X_test, y_test = split_X_y(df_test, args.target_col, drop_cols)

    # etiquetas en orden consistente
    labels = sorted(list(y_train.astype("category").cat.categories))
    print(f"[data] train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    print(f"[data] target classes: {labels}")

    # 3) Entrenamiento
    print("[train] training LightGBM ...")
    pipe = train_lgbm_pipeline(X_train, y_train, X_val, y_val, class_weight="balanced")

    # 4) Evaluación (val + test)
    print("[eval] evaluating ...")
    y_val_pred = pipe.predict(X_val)
    y_test_pred = pipe.predict(X_test)

    metrics_val = evaluate_multiclass(y_val, y_val_pred, labels)
    metrics_test = evaluate_multiclass(y_test, y_test_pred, labels)

    # Combinar en un único JSON
    combined = {
        "dataset": {
            "train_rows": int(len(df_train)),
            "validate_rows": int(len(df_val)),
            "test_rows": int(len(df_test)),
        },
        "validation": metrics_val,
        "test": metrics_test,
        "context": {
            "company_name": args.company_name,
            "project_name": args.project_name,
            "environment": args.environment,
        },
    }

    # 5) Guardar modelo y métricas
    print("[save] saving model & metrics ...")
    save_model(pipe, os.path.join(MODEL_DIR, "model.joblib"))
    save_metrics(combined, EVAL_JSON)

    # Logging adicional
    print(json.dumps(combined["validation"], indent=2))
    print("[done] training completed.")


if __name__ == "__main__":
    main()
