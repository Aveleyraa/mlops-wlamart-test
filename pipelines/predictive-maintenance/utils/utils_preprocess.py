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


SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
METRICS_DIR = "/opt/ml/output/metrics"


def load_raw_tables(
    telemetry_path: str,
    errors_path: str,
    maint_path: str,
    failures_path: str,
    machines_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSVs (local paths or S3 URLs) and standardize dtypes.

    - Parses 'datetime' columns to pandas datetime.
    - Casts categorical fields: errors.errorID, maint.comp, failures.failure, machines.model.

    Parameters
    ----------
    telemetry_path, errors_path, maint_path, failures_path, machines_path : str
        CSV locations (e.g., '/path/file.csv' or 's3://bucket/key.csv').

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (telemetry, errors, maint, failures, machines)
    """
    telemetry = pd.read_csv(telemetry_path)
    errors = pd.read_csv(errors_path)
    maint = pd.read_csv(maint_path)
    failures = pd.read_csv(failures_path)
    machines = pd.read_csv(machines_path)

    # datetimes
    telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
    errors["datetime"] = pd.to_datetime(errors["datetime"])
    maint["datetime"] = pd.to_datetime(maint["datetime"])
    failures["datetime"] = pd.to_datetime(failures["datetime"])

    # categories
    if "errorID" in errors.columns:
        errors["errorID"] = errors["errorID"].astype("category")
    if "comp" in maint.columns:
        maint["comp"] = maint["comp"].astype("category")
    if "failure" in failures.columns:
        failures["failure"] = failures["failure"].astype("category")
    if "model" in machines.columns:
        machines["model"] = machines["model"].astype("category")

    return telemetry, errors, maint, failures, machines


def build_telemetry_features(
    telemetry: pd.DataFrame,
    resample_rule: str = "3H",
    rolling_hours: int = 24,
    signals: List[str] = ("volt", "rotate", "pressure", "vibration"),
) -> pd.DataFrame:
    """
    Compute resampled and rolling telemetry features.

    - Resamples each signal by `resample_rule` and computes 3H mean/std.
    - Computes `rolling_hours` rolling mean/std, then resamples and takes the
      first value per window.
    - Returns a wide table (flattened columns) indexed by ['datetime','machineID'].

    Parameters
    ----------
    telemetry : pd.DataFrame
        Raw telemetry with columns: ['datetime', 'machineID'] + signals.
    resample_rule : str, optional
        Pandas offset alias for resampling (e.g., '3H').
    rolling_hours : int, optional
        Rolling window size (in original sampling steps) for 24H-like stats.
    signals : List[str], optional
        Signal column names to aggregate.

    Returns
    -------
    pd.DataFrame
        Concatenated features: mean_3h, sd_3h, mean_24h, sd_24h per signal.
    """
    # pivot by datetime x machineID for each signal, then resample
    def _agg_3h(func_name: str) -> pd.DataFrame:
        frames = []
        for col in signals:
            pv = pd.pivot_table(
                telemetry, index="datetime", columns="machineID", values=col
            )
            agg = (
                pv.resample(resample_rule, closed="left", label="right")
                .agg(func_name)
                .unstack()
            )
            frames.append(agg)
        out = pd.concat(frames, axis=1)
        suffix = "mean_3h" if func_name == "mean" else "sd_3h"
        out.columns = [f"{s}{suffix}" for s in signals]
        out = out.reset_index()
        return out

    telemetry_mean_3h = _agg_3h("mean")

    frames = []
    for col in signals:
        pv = pd.pivot_table(
            telemetry, index="datetime", columns="machineID", values=col
        )
        roll_mean = (
            pv.rolling(rolling_hours).mean().resample(resample_rule, closed="left", label="right").first().unstack()
        )
        frames.append(roll_mean)
    telemetry_mean_24h = pd.concat(frames, axis=1)
    telemetry_mean_24h.columns = [f"{s}mean_24h" for s in signals]
    telemetry_mean_24h = telemetry_mean_24h.reset_index()
    telemetry_mean_24h = telemetry_mean_24h.loc[
        ~telemetry_mean_24h["voltmean_24h"].isnull()
    ]

    frames = []
    for col in signals:
        pv = pd.pivot_table(
            telemetry, index="datetime", columns="machineID", values=col
        )
        roll_std = (
            pv.rolling(rolling_hours).std().resample(resample_rule, closed="left", label="right").first().unstack()
        )
        frames.append(roll_std)
    telemetry_sd_24h = pd.concat(frames, axis=1)
    telemetry_sd_24h.columns = [f"{s}sd_24h" for s in signals]
    telemetry_sd_24h = telemetry_sd_24h.loc[~telemetry_sd_24h["voltsd_24h"].isnull()]
    telemetry_sd_24h = telemetry_sd_24h.reset_index()

    # 3H std (not rolling)
    frames = []
    for col in signals:
        pv = pd.pivot_table(
            telemetry, index="datetime", columns="machineID", values=col
        )
        agg = (
            pv.resample(resample_rule, closed="left", label="right")
            .std()
            .unstack()
        )
        frames.append(agg)
    telemetry_sd_3h = pd.concat(frames, axis=1)
    telemetry_sd_3h.columns = [f"{s}sd_3h" for s in signals]
    telemetry_sd_3h = telemetry_sd_3h.reset_index()

    # merge feature blocks
    telemetry_feat = pd.concat(
        [
            telemetry_mean_3h,
            telemetry_sd_3h.iloc[:, 2:6],
            telemetry_mean_24h.iloc[:, 2:6],
            telemetry_sd_24h.iloc[:, 2:6],
        ],
        axis=1,
    ).dropna()

    return telemetry_feat


def build_error_counts(
    errors: pd.DataFrame,
    telemetry: pd.DataFrame,
    resample_rule: str = "3H",
    rolling_hours: int = 24,
) -> pd.DataFrame:
    """
    Compute 24h rolling error counts per machine, resampled to `resample_rule`.

    - One-hot encodes error types (1..5), aligns to the telemetry time grid,
      fills gaps with 0, applies a `rolling_hours` sum, then resamples and takes
      the first value per window.
    - Returns a wide DataFrame with columns:
      ['datetime','machineID','error1count',...,'error5count'].
    """
    error_count = pd.get_dummies(errors.set_index("datetime")).reset_index()
    error_count.columns = ["datetime", "machineID", "error1", "error2", "error3", "error4", "error5"]
    error_count = error_count.groupby(["machineID", "datetime"]).sum().reset_index()

    # align with telemetry grid, fill missing hours with zeros
    error_count = (
        telemetry[["datetime", "machineID"]]
        .merge(error_count, on=["machineID", "datetime"], how="left")
        .fillna(0.0)
    )

    fields = [f"error{i}" for i in range(1, 6)]
    frames = []
    for col in fields:
        pv = pd.pivot_table(
            error_count, index="datetime", columns="machineID", values=col
        )
        roll = (
            pv.rolling(rolling_hours)
            .sum()
            .resample(resample_rule, closed="left", label="right")
            .first()
            .unstack()
        )
        frames.append(roll)

    out = pd.concat(frames, axis=1)
    out.columns = [f"{c}count" for c in fields]
    out = out.reset_index().dropna()
    return out



def build_component_age_days(
    maint: pd.DataFrame,
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-machine days since last replacement for components (comp1..comp4).

    - One-hot encodes maintenance events, aligns to telemetry timestamps, and
      forward-fills the last replacement datetime per machine.
    - Drops early records before 2015-01-01 (dataset-specific cleanup).
    - Returns a DataFrame with ['datetime','machineID','comp1','comp2','comp3','comp4']
      where each comp* is the age in days (float). Values are NaN if no prior replacement.

    Parameters
    ----------
    maint : pd.DataFrame
        Maintenance logs with ['datetime','machineID','comp'] (categorical comp1..comp4).
    telemetry : pd.DataFrame
        Telemetry grid used to define the time axis (must include ['datetime','machineID']).

    Returns
    -------
    pd.DataFrame
        Per-timestamp component ages in days.
    """
    comp_rep = pd.get_dummies(maint.set_index("datetime")).reset_index()
    comp_rep.columns = ["datetime", "machineID", "comp1", "comp2", "comp3", "comp4"]
    comp_rep = comp_rep.groupby(["machineID", "datetime"]).sum().reset_index()

    # add timepoints where no components were replaced
    comp_rep = (
        telemetry[["datetime", "machineID"]]
        .merge(comp_rep, on=["datetime", "machineID"], how="outer")
        .fillna(0)
        .sort_values(by=["machineID", "datetime"])
    )

    components = ["comp1", "comp2", "comp3", "comp4"]
    for comp in components:
        # set timestamp at replacement, else None
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[~comp_rep[comp].isnull(), comp] = comp_rep.loc[~comp_rep[comp].isnull(), "datetime"]
        # forward fill last replacement date per machine
        comp_rep[comp] = comp_rep.groupby("machineID")[comp].ffill()

    # remove 2014 (dataset-specific cleanup)
    comp_rep = comp_rep.loc[comp_rep["datetime"] > pd.to_datetime("2015-01-01")]

    # days since last replacement
    for comp in components:
        comp_rep[comp] = (comp_rep["datetime"] - comp_rep[comp]) / np.timedelta64(1, "D")

    return comp_rep

def assemble_features(
    telemetry_feat: pd.DataFrame,
    error_counts: pd.DataFrame,
    comp_age: pd.DataFrame,
    machines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all feature blocks with machine metadata into one matrix.

    - Joins telemetry features, rolling error counts, and component ages on
      ['datetime','machineID'].
    - Joins machine attributes on ['machineID'].
    - Left joins preserve the telemetry feature rows.

    Parameters
    ----------
    telemetry_feat : pd.DataFrame
        Base features indexed by ['datetime','machineID'].
    error_counts : pd.DataFrame
        Rolling error counts per ['datetime','machineID'].
    comp_age : pd.DataFrame
        Component ages per ['datetime','machineID'].
    machines : pd.DataFrame
        Machine dimension table keyed by ['machineID'].

    Returns
    -------
    pd.DataFrame
        Unified feature frame ready for modeling.
    """
    final_feat = telemetry_feat.merge(error_counts, on=["datetime", "machineID"], how="left")
    final_feat = final_feat.merge(comp_age, on=["datetime", "machineID"], how="left")
    final_feat = final_feat.merge(machines, on=["machineID"], how="left")
    return final_feat



def label_with_failures(
    final_feat: pd.DataFrame,
    failures: pd.DataFrame,
    backfill_hours: int = 7,
) -> pd.DataFrame:
    """
    Attach failure labels to features and backfill imminent failures.

    - Left-joins `failures` on ['datetime','machineID'].
    - Backfills the 'failure' label per machine up to `backfill_hours` steps
      (assumes an hourly grid).
    - Casts to categorical and fills missing as 'none'.

    Parameters
    ----------
    final_feat : pd.DataFrame
        Feature matrix with ['datetime','machineID'].
    failures : pd.DataFrame
        Failure events with ['datetime','machineID','failure'].
    backfill_hours : int, optional
        Max backfill window (steps/hours) for imminent failures, by machine.

    Returns
    -------
    pd.DataFrame
        Labeled features with a categorical 'failure' column.
    """
    labeled = final_feat.merge(failures, on=["datetime", "machineID"], how="left")

    # Backward fill per machine to cover the next N hours window
    labeled = labeled.sort_values(["machineID", "datetime"])
    labeled["failure"] = labeled.groupby("machineID")["failure"].apply(
        lambda s: s.fillna(method="bfill", limit=backfill_hours)
    )

    # cast to categorical with 'none'
    labeled["failure"] = labeled["failure"].astype("category")
    labeled["failure"] = labeled["failure"].cat.add_categories("none")
    labeled["failure"] = labeled["failure"].fillna("none")
    return labeled


def chronological_split(
    labeled_features: pd.DataFrame,
    train_cutoff: str | None,
    validate_cutoff: str | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split into train/validate/test.

    - If both cutoffs are provided:
        train:   datetime < train_cutoff
        validate: train_cutoff ≤ datetime < validate_cutoff
        test:    datetime ≥ validate_cutoff
    - If any cutoff is None: split by time at 70%/15%/15%.

    Parameters
    ----------
    labeled_features : pd.DataFrame
        Input with a 'datetime' column.
    train_cutoff : str | None
        ISO-like timestamp for the train/validate boundary (e.g., '2015-06-01').
    validate_cutoff : str | None
        ISO-like timestamp for the validate/test boundary.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, validate, test) in chronological order.
    """
    df = labeled_features.sort_values("datetime")
    if train_cutoff and validate_cutoff:
        train = df[df["datetime"] < pd.to_datetime(train_cutoff)]
        validate = df[(df["datetime"] >= pd.to_datetime(train_cutoff)) & (df["datetime"] < pd.to_datetime(validate_cutoff))]
        test = df[df["datetime"] >= pd.to_datetime(validate_cutoff)]
        return train, validate, test

    # fallback: 70/15/15 by time
    times = df["datetime"].sort_values().values
    t1 = times[int(0.7 * len(times))]
    t2 = times[int(0.85 * len(times))]
    train = df[df["datetime"] < t1]
    validate = df[(df["datetime"] >= t1) & (df["datetime"] < t2)]
    test = df[df["datetime"] >= t2]
    return train, validate, test

# ---------------------------
# I/O helpers
# ---------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)



def resolve_csv_path(path_or_prefix: str) -> str:
    """
    Resolve a path or prefix to a CSV file path.

    - If the input ends with '.csv', return it as is.
    - If the input ends with '/', append 'data.csv'.
    - Otherwise, add a trailing '/' and then append 'data.csv'.

    Parameters
    ----------
    path_or_prefix : str
        File path or directory prefix.

    Returns
    -------
    str
        Full CSV file path.
    """
    if path_or_prefix.endswith(".csv"):
        return path_or_prefix
    if not path_or_prefix.endswith("/"):
        path_or_prefix += "/"
    return path_or_prefix + "data.csv"


def load_csv(path_or_prefix: str) -> pd.DataFrame:
    """
    Load a CSV from local or S3 (requires s3fs/pyarrow for S3) and lightly normalize dtypes.

    - Resolves directory prefixes to '<prefix>/data.csv'.
    - Parses 'datetime' to pandas datetime if present.
    - Casts 'model' and 'failure' to categorical if present.

    Parameters
    ----------
    path_or_prefix : str
        File path or directory/prefix (local or s3://).

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with normalized dtypes.
    """
    path = resolve_csv_path(path_or_prefix)
    df = pd.read_csv(path)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    for cat_col in ("model", "failure"):
        if cat_col in df.columns and not pd.api.types.is_categorical_dtype(df[cat_col]):
            df[cat_col] = df[cat_col].astype("category")

    return df


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_split_csv(
    train: pd.DataFrame,
    validate: pd.DataFrame,
    test: pd.DataFrame,
    train_dir: str,
    validate_dir: str,
    test_dir: str,
    filename: str = "data.csv",
) -> None:
    """
    Save train/validate/test DataFrames to CSV files.

    - Ensures output directories exist (via `ensure_dir`).
    - Writes each split to `<dir>/<filename>` without index.

    Parameters
    ----------
    train, validate, test : pd.DataFrame
        Data splits to persist.
    train_dir, validate_dir, test_dir : str
        Target directories for each split.
    filename : str, optional
        CSV filename to use in each directory (default: 'data.csv').
    """
    ensure_dir(train_dir)
    ensure_dir(validate_dir)
    ensure_dir(test_dir)
    train.to_csv(os.path.join(train_dir, filename), index=False)
    validate.to_csv(os.path.join(validate_dir, filename), index=False)
    test.to_csv(os.path.join(test_dir, filename), index=False)


def split_X_y(
    df: pd.DataFrame,
    target_col: str,
    drop_cols: Tuple[str, ...] = ("datetime", "machineID"),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split features (X) and target (y), dropping common ID/time columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_col : str
        Name of the target column.
    drop_cols : Tuple[str, ...], optional
        Columns to exclude from features (default: ('datetime','machineID')).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (features) and y (target).
    """
    feats = [c for c in df.columns if c not in set(drop_cols + (target_col,))]
    X = df[feats].copy()
    y = df[target_col].copy()
    return X, y

