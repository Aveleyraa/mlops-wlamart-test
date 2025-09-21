#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocess.py
-------------
Modular feature engineering for Predictive Maintenance (Azure PdM dataset style).

- Loads raw tables (telemetry, errors, maintenance, failures, machines)
- Builds telemetry features (3H mean/std + 24H rolling mean/std resampled to 3H)
- Builds 24H rolling error counts resampled to 3H
- Builds component "age" in days since last replacement
- Joins with machines table
- Labels with failures (forward-fill next few hours)
- Chronological split into train/validate/test
- Writes CSVs to output directories

Usage in SageMaker SKLearnProcessor:
    python preprocess.py \
      --input-telemetry-s3 s3://.../PdM_telemetry.csv \
      --input-errors-s3    s3://.../PdM_errors.csv \
      --input-maint-s3     s3://.../PdM_maint.csv \
      --input-failures-s3  s3://.../PdM_failures.csv \
      --input-machines-s3  s3://.../PdM_machines.csv \
      --train-cutoff "2015-08-01 00:00:00" \
      --validate-cutoff "2015-09-01 00:00:00"

Default outputs (SageMaker-friendly):
  /opt/ml/processing/train/data.csv
  /opt/ml/processing/validate/data.csv
  /opt/ml/processing/test/data.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, List
from fastapi import FastAPI, Request
import uuid
import numpy as np
import pandas as pd
from utils.utils_preprocess import (
    save_split_csv,
    assemble_features,
    load_raw_tables,
    build_telemetry_features,
    build_error_counts,
    build_component_age_days,
    label_with_failures,
    chronological_split
)




def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PdM data and produce train/val/test splits.")
    parser.add_argument("--input-telemetry-s3", required=True, help="Path to PdM_telemetry.csv (S3 or local)")
    parser.add_argument("--input-errors-s3",    required=True, help="Path to PdM_errors.csv (S3 or local)")
    parser.add_argument("--input-maint-s3",     required=True, help="Path to PdM_maint.csv  (S3 or local)")
    parser.add_argument("--input-failures-s3",  required=True, help="Path to PdM_failures.csv (S3 or local)")
    parser.add_argument("--input-machines-s3",  required=True, help="Path to PdM_machines.csv (S3 or local)")

    parser.add_argument("--train-cutoff", default="2015-08-01 00:00:00", help="Train end datetime (exclusive)")
    parser.add_argument("--validate-cutoff", default="2015-09-01 00:00:00", help="Validation end datetime (exclusive)")

    # SageMaker default output dirs
    parser.add_argument("--train-out", default="/opt/ml/processing/train", help="Output dir for train split")
    parser.add_argument("--validate-out", default="/opt/ml/processing/validate", help="Output dir for validate split")
    parser.add_argument("--test-out", default="/opt/ml/processing/test", help="Output dir for test split")

    args = parser.parse_args()

    # Load
    telemetry, errors, maint, failures, machines = load_raw_tables(
        args.input_telemetry_s3, args.input_errors_s3, args.input_maint_s3,
        args.input_failures_s3, args.input_machines_s3
    )

    # Build features
    telemetry_feat = build_telemetry_features(telemetry)
    error_counts = build_error_counts(errors, telemetry)
    comp_age = build_component_age_days(maint, telemetry)

    # Assemble & label
    final_feat = assemble_features(telemetry_feat, error_counts, comp_age, machines)
    labeled_features = label_with_failures(final_feat, failures, backfill_hours=7)

    # Split & save
    train, validate, test = chronological_split(
        labeled_features, args.train_cutoff, args.validate_cutoff
    )
    save_split_csv(
        train=train, validate=validate, test=test,
        train_dir=args.train_out, validate_dir=args.validate_out, test_dir=args.test_out
    )

    # Minimal log
    print("✅ Preprocessing complete.")
    print(f"Train rows: {len(train):,} | Validate rows: {len(validate):,} | Test rows: {len(test):,}")
    print(f"Saved:\n  - {os.path.join(args.train_out, 'data.csv')}\n  - {os.path.join(args.validate_out, 'data.csv')}\n  - {os.path.join(args.test_out, 'data.csv')}")


if __name__ == "__main__":
    main()
