# services/online/lambda_app.py
import os, json, boto3, pandas as pd
from pipelines.predictive_maintenance.features.online_fe import (
    features_from_already_aggregated,
    features_from_raw_window,
    load_feature_order,
)

REGION = os.environ.get("AWS_REGION", "us-east-2")
ENDPOINT_NAME = os.environ["SM_ENDPOINT_NAME"]
FEATURE_ORDER_S3 = os.environ.get("FEATURE_ORDER_S3")  # s3://bucket/path/feature_order.json

s3 = boto3.client("s3", region_name=REGION)
smrt = boto3.client("sagemaker-runtime", region_name=REGION)

_feature_order = None

def _load_feature_order_once():
    global _feature_order
    if _feature_order is not None or not FEATURE_ORDER_S3:
        return
    # s3://bucket/key
    assert FEATURE_ORDER_S3.startswith("s3://")
    bkt, key = FEATURE_ORDER_S3[5:].split("/", 1)
    body = s3.get_object(Bucket=bkt, Key=key)["Body"].read().decode("utf-8")
    _feature_order = load_feature_order(body)

def handler(event, context):
    """
    Espera un JSON con:
    {
      "mode": "aggregated" | "raw",
      "records": [ {...}, {...} ],            # aggregated
      "raw": {
        "telemetry": [ {...}, ... ],
        "errors":    [ {...}, ... ],
        "maint":     [ {...}, ... ],
        "machines":  [ {...}, ... ]
      }
    }
    Devuelve predicciones del endpoint.
    """
    _load_feature_order_once()

    if isinstance(event, str):
        event = json.loads(event)

    mode = event.get("mode", "aggregated")

    if mode == "aggregated":
        df = pd.DataFrame(event["records"])
        X = features_from_already_aggregated(df, _feature_order)
    elif mode == "raw":
        raw = event["raw"]
        telemetry = pd.DataFrame(raw["telemetry"])
        errors    = pd.DataFrame(raw["errors"])
        maint     = pd.DataFrame(raw["maint"])
        machines  = pd.DataFrame(raw["machines"])
        X = features_from_raw_window(telemetry, errors, maint, machines, _feature_order)
    else:
        return {"statusCode": 400, "body": "Unsupported mode"}

    # Invoca el endpoint (CSV sin header)
    body = X.to_csv(index=False, header=False)
    resp = smrt.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Accept="application/json",
        Body=body,
    )
    preds = json.loads(resp["Body"].read().decode("utf-8"))
    return {"statusCode": 200, "predictions": preds}
