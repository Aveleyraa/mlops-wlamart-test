from fastapi import FastAPI
from pydantic import BaseModel
import os, json, boto3, pandas as pd
import uuid
from fastapi import FastAPI, Request
from pipelines.predictive_maintenance.features.online_fe import (
    features_from_already_aggregated, features_from_raw_window, load_feature_order
)
from pipelines.predictive_maintenance.utils.utils_logging import get_logger
logger = get_logger("fastapi")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = str(uuid.uuid4())
    logger.info("Request received", extra={"path": str(request.url.path), "rid": rid})
    response = await call_next(request)
    logger.info("Request done", extra={"status": response.status_code, "rid": rid})
    return response

app = FastAPI()
REGION = os.environ.get("AWS_REGION", "us-east-2")
ENDPOINT = os.environ["SM_ENDPOINT_NAME"]
FEATURE_ORDER_S3 = os.environ.get("FEATURE_ORDER_S3")

s3 = boto3.client("s3", region_name=REGION)
smrt = boto3.client("sagemaker-runtime", region_name=REGION)
_feature_order = None

def _load_feature_order_once():
    global _feature_order
    if _feature_order is not None or not FEATURE_ORDER_S3: return
    bkt, key = FEATURE_ORDER_S3[5:].split("/", 1)
    body = s3.get_object(Bucket=bkt, Key=key)["Body"].read().decode("utf-8")
    _feature_order = load_feature_order(body)

class PredictAggregated(BaseModel):
    records: list[dict]

class PredictRaw(BaseModel):
    telemetry: list[dict]
    errors: list[dict]
    maint: list[dict]
    machines: list[dict]

@app.post("/predict/aggregated")
def predict_agg(payload: PredictAggregated):
    _load_feature_order_once()
    X = features_from_already_aggregated(pd.DataFrame(payload.records), _feature_order)
    body = X.to_csv(index=False, header=False)
    r = smrt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv", Accept="application/json", Body=body)
    return json.loads(r["Body"].read().decode())

@app.post("/predict/raw")
def predict_raw(payload: PredictRaw):
    _load_feature_order_once()
    X = features_from_raw_window(pd.DataFrame(payload.telemetry), pd.DataFrame(payload.errors),
                                 pd.DataFrame(payload.maint), pd.DataFrame(payload.machines),
                                 _feature_order)
    body = X.to_csv(index=False, header=False)
    r = smrt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv", Accept="application/json", Body=body)
    return json.loads(r["Body"].read().decode())
