# Predictive Maintenance (AWS + SageMaker)

End-to-end pipeline for **training, registering, and deploying** a Predictive Maintenance model on **Amazon SageMaker**, managed with  **GitHub Actions**, and MLOps practices (pre-commit, tests, IaC).

## 🗂 Repository Structure

```
.
├── ci/
│   ├── ci.yml
│   ├── deploy-infra.yml
│   └── run-sagemaker-pipeline.yml
├── data/                            
├── infra/
│   ├── cloudformation.yml          
│   ├── parameters_dev.json
│   └── parameters_prod.json
├── notebooks/
├── pipelines/
│   ├── get_pipeline_definition.py
│   ├── run_pipeline.py
│   ├── predictive-maintenance/
│   │   ├── pipeline.py             # SageMaker Pipeline definition
│   │   ├── requirements.txt        # deps for Processing/Training steps
│   │   ├── preprocess/
│   │   │   └── preprocess.py       # batch feature engineering
│   │   ├── train/
│   │   │   └── training.py         # training + eval + export
│   │   └── utils/
│   │       ├── utils_preprocess.py # reusable FE helpers
│   │       └── utils_training.py   # training utilities
├── tests/
├── pyproject.toml                  # Poetry (deps + toolchain)
├── .pre-commit-config.yml          # quality gates
└── README.md
```

##  Requirements

- **Python 3.10** (or the version pinned in `pyproject.toml`)
- **AWS CLI** configured with permissions for SageMaker, S3, IAM, CloudFormation
- **jq** (for JSON parameter handling in infra deploys)

##  Connect GitHub Actions to AWS using OIDC

1. Create an IAM Identity Provider in your AWS account for GitHub OIDC. [AWS link example](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html)

2. Create an IAM Role in your AWS account with a trust policy that allows GitHub Actions to assume it:
```bash
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<AWS_ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:<GITHUB_ORG>/<GITHUB_REPOSITORY>:ref:refs/heads/<GITHUB_BRANCH>"
        }
      }
    }
  ]
}
```
3. Attach permissions to the IAM Role that allow it to access the AWS resources you need.

4. Create GitHub Actions workflow (in this repo there are 4 differents fo different purposes):

##  Add GitHub Secrets & Variables

In your repository:

**Settings → Secrets and variables → Actions**

### Secrets (sensitive)
- `AWS_ROLE_TO_ASSUME` → your OIDC role ARN  
  _Example:_ `arn:aws:iam::123456789012:role/github-oidc-actions`
- `SAGEMAKER_PIPELINE_ROLE_ARN` → SageMaker execution role used by your pipeline  
  _Example:_ `arn:aws:iam::123456789012:role/AmazonSageMakerServiceCatalogProductsUseRole`


### Variables (non-sensitive defaults; you can override per workflow)
- `AWS_ACCOUNT_ID` → `123456789012`
- `AWS_DEFAULT_REGION` → `us-east-2`
- `COMPANY_NAME` → `walmart`
- `PROJECT_NAME` → `predictive-maintenance`
- `ENV` → `dev` _(or `prod`)_

---


##  Infrastructure Deployment


1. Go to **GitHub → Actions → Deploy Infra**.
2. Click **Run workflow** (top-right).
3. Choose the input **`env`** (e.g., `dev` or `prod`).
4. Click **Run workflow** and wait for the job to finish.

**What this workflow does**
1. Assumes your AWS role via OIDC (using `AWS_ROLE_TO_ASSUME`).
2. Validates the CloudFormation template (`infra/cloudformation.yml`).
3. Loads parameters from `infra/parameters_<env>.json`.
4. Creates/updates the CloudFormation stack `walmart-pdm-infra-<env>`.
5. Prints the stack **Outputs** (e.g., bucket names, ARNs) at the end of the job.

```bash
aws cloudformation deploy \
  --region $AWS_DEFAULT_REGION \
  --template-file infra/cloudformation.yml \
  --stack-name walmart-pdm-infra-$ENV \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides $(jq -r '.[] | "\(.ParameterKey)=\(.ParameterValue)"' infra/parameters_${ENV}.json | paste -sd " ")
```

> This creates buckets, roles, and (optionally) a Lambda for orchestration.
You can verify on AWS Console cloudformation the stack Status.

---

### (Training Pipeline via `run-sagemaker-pipeline.yml`)

1. Go to **GitHub → Actions → Run SageMaker Pipeline**.
2. Click **Run workflow** (top-right).
3. Choose inputs:
   - **`env`**: environment to target (e.g., `dev` or `prod`).
   - **`execute`**: `"true"` to **start** the pipeline execution (use `"false"` if you only want to *create/update* the pipeline definition without running it, depending on your script’s behavior).
4. Click **Run workflow** and wait for the job to finish.

**What this workflow does**
1. Assumes your AWS role via OIDC using the secret **`AWS_ROLE_TO_ASSUME`**.
2. Sets region from **`AWS_DEFAULT_REGION`** (repo variable).
3. Checks out the repo and installs your project with **Poetry**.
4. Invokes:
   ```bash
   pipelines/run_pipeline.py \
     --module-name pipelines.predictive-maintenance.pipeline \
     --execute-pipeline "<execute>" \
     --role-arn "<SAGEMAKER_PIPELINE_ROLE_ARN>" \
     --kwargs '{
       "region": "<AWS_DEFAULT_REGION>",
       "company_name": "<COMPANY_NAME>",
       "project_name": "<PROJECT_NAME>",
       "environment": "<env>",
       "model_package_group_name": "walmart-pdm-<env>"
     }'

Artifacts you should expect

Feature outputs and intermediate files in:

s3://walmart-predictive-maintenance-<env>-staging-data/...

s3://walmart-predictive-maintenance-<env>-model-data/...

Trained model tarball in the Training job ModelArtifacts S3 path.

An evaluation.json under the model’s output/metrics/ prefix.

A new Model Package in the Model Registry for model_package_group_name = walmart-pdm-<env>.

## 🧭 Training Pipeline (SageMaker)

`pipelines/predictive-maintenance/pipeline.py` defines:

1. **Preprocessing Step** (Processing / ScriptProcessor):
   - Runs `preprocess/preprocess.py` on source data (S3).
   - Writes **train/validate/test** processed splits to S3 (`staging-data`).

2. **Training Step** (Training / Script Mode):
   - Runs `train/training.py` (e.g., LightGBM).
   - Produces **model.tar.gz** and **evaluation.json** (metrics) in S3.

3. **Registration Step** (Model Registry):
   - Registers the model and attaches metrics from `evaluation.json`.



## 🚢 Model Deployment (Online Endpoint)

Once a model is approved/registered:

1) **Create/Update Endpoint** (via your `deploy-model` script or console).
   - Define `Model` → `EndpointConfig` → `Endpoint`.
   - Keep the endpoint name, e.g., `walmart-pdm-dev`.

### 2) Invoke the endpoint

You can invoke the endpoint from **your local terminal** (with AWS CLI configured), **AWS CloudShell**, or a **CI job**. The endpoint only does **prediction** — any **feature engineering must be done client-side** first, so the payload matches the model’s expected feature order.

#### A) Prepare the payload
1. Compute features **in the same order** used at training time (e.g., the columns in `train/data.csv` after preprocessing).
2. Save them as **CSV without header** (default for SageMaker SKLearn serving). Example with two rows:
   ```csv
   voltmean_3h,rotatemean_3h,pressuremean_3h,vibrationmean_3h,voltsd_3h,rotatesd_3h,pressuresd_3h,vibrationsd_3h,error1count,error2count,error3count,error4count,error5count,comp1,comp2,comp3,comp4,model,age
   166.28,453.78,106.18,51.99,24.27,23.62,11.17,3.39,1,0,1,0,0,22.125,217.125,157.125,172.125,model3,18
   175.41,445.45,100.88,54.25,34.91,11.00,10.58,2.92,1,0,1,0,0,22.250,217.250,157.250,172.250,model3,18

#### B) Invoke with AWS CLI (CSV)
```bash
aws sagemaker-runtime invoke-endpoint \
  --region us-east-2 \
  --endpoint-name walmart-pdm-dev \
  --content-type text/csv \
  --body fileb://payload.csv \
  /tmp/response.json
cat /tmp/response.json
```

> If requests arrive **raw**, implement a microservice/Lambda that runs the **same feature engineering** as `preprocess.py` (use `utils_preprocess.py`) and then calls the endpoint. Persist a `feature_order.json` from training and **align** columns at inference-time.

---

## Services Overview

This repo includes two app-level services that sit **on top of** the SageMaker training & model hosting flow:

- **Online inference (FastAPI)** → `services/fast_app.py`  
  A small web service that receives raw inputs, runs the **same feature engineering** used at training time, and then **calls the SageMaker endpoint** (or performs local inference, if configured).

- **Batch inference (Lambda)** → `services/lambda_app.py`  
  An AWS Lambda that’s triggered by S3 (or a schedule), runs the **same feature engineering** on incoming files, and writes scored outputs back to S3. It can call a **SageMaker Batch Transform job** or invoke the online endpoint in chunks.

Both services import the shared preprocessing utilities from:
`pipelines/predictive-maintenance/utils/utils_preprocess.py`  
This guarantees the **exact same transforms & column order** as training.

---

## Online Inference Service — FastAPI (`services/fast_app.py`)

### What it does
1. Accepts raw JSON payloads (or CSV) via `POST /predict`.
2. Converts input to a DataFrame and applies **feature engineering**  
   (3H means/std, 24H rolling stats, error counts, component age in days, merges with machines metadata, encoding, column ordering).
3. Sends the **prepared features** to your SageMaker **endpoint** (`InvokeEndpoint`)  
   **or** performs local inference if the container includes a model artifact.
4. Returns predictions (e.g., predicted label and/or probabilities).

### Typical request/response (JSON)
**Request** (example; raw fields _before_ feature engineering):
```json
{
  "machineID": 1,
  "datetime": "2015-01-04T09:00:00",
  "volt": 166.28,
  "rotate": 453.78,
  "pressure": 106.18,
  "vibration": 51.99,
  "errorIDs": [1, 5],
  "model": "model3",
  "age": 18
}
```
## Batch Inference Lambda — (`services/lambda_app.py`)

### What it does
- **Trigger**: S3 `PUT` to a specific prefix (e.g., `raw/batch-inference/`), or a scheduled **EventBridge** rule.
- Reads the incoming file(s) from **S3** (CSV/Parquet).
- Applies the **same feature engineering** from `utils_preprocess.py`.
- **Two modes**:
  1. Submit a **SageMaker Batch Transform** job using the latest approved model/package; **or**
  2. Make **chunked** calls to the online endpoint (`InvokeEndpoint`) and aggregate results.
- Writes scored outputs to **S3** (e.g., `model-data/batch-scoring/<date>/predictions.csv`).

---

### Environment variables (Lambda)
- `AWS_DEFAULT_REGION` — e.g., `us-east-2`
- `INPUT_BUCKET` / `INPUT_PREFIX` — where raw batch files arrive
- `OUTPUT_BUCKET` / `OUTPUT_PREFIX` — where to write predictions

**If using endpoint:**
- `PDM_ENDPOINT_NAME` — e.g., `walmart-pdm-dev`

**If using Batch Transform:**
- `MODEL_PACKAGE_ARN` **or** `MODEL_ARTIFACT_S3_URI`
- `BT_INSTANCE_TYPE` / `BT_INSTANCE_COUNT` (optional overrides)

---

### Required IAM permissions (Lambda role)
- `s3:GetObject`, `s3:PutObject` for input/output buckets
- `sagemaker:CreateTransformJob` (if using Batch Transform)
- `sagemaker:InvokeEndpoint` (if invoking endpoint)
- `logs:*` for CloudWatch logging

---

### Trigger (S3 event example)
- **S3 bucket** → **Properties** → **Event notifications** → **PUT** → prefix `raw/batch-inference/` → **target Lambda**



---

## 🧪 Tests & Quality

- **Unit tests** under `tests/` (preprocess, training, utils).
- **pre-commit** runs:
  - ruff, black, mypy
  - YAML/JSON/TOML validations
  - notebook output cleanup (nbstripout)
  - secret scanning (detect-secrets)


---

## 🛠 Troubleshooting

- **Pipeline failures**:
  - Check *CloudWatch Logs* for Processing/Training steps.
  - Verify roles/permissions (S3 read/write, SageMaker).
  - Inspect S3 input/output paths in `pipeline.py`.

- **Endpoint 4xx/5xx**:
  - Confirm payload format (`text/csv` w/o header vs JSON).
  - Ensure **feature alignment** exactly as in training.
  - Check container logs for the endpoint (if using a custom handler).

- **Missing metrics in registry**:
  - Confirm `evaluation.json` path produced by training and mapped in `ModelMetrics`.

---


# Next Steps

## implement MLflow 


## 1) Pick a Tracking Backend

### Production 
Run an MLflow Tracking Server with:
- **Backend store**: Amazon **RDS** (PostgreSQL)
- **Artifact store**: Amazon **S3** (e.g., `s3://walmart-pdm-mlflow-artifacts-<env>/`)
- **Compute**: ECS/Fargate service in a private subnet (behind an internal ALB)

Example server command (container CMD):
```bash
mlflow server \
  --backend-store-uri postgresql+psycopg2://<user>:<pwd>@<rds-host>:5432/mlflow \
  --default-artifact-root s3://walmart-pdm-mlflow-artifacts-<env>/ \
  --host 0.0.0.0 --port 5000
```
**Secure it** with VPC security groups, an internal ALB, and auth (e.g., Cognito/SSO or basic auth).

### Dev (quick start)
Use a **local file backend** with **local artifacts**:
```bash
# 1) Point MLflow to a local folder (created if it doesn't exist)
export MLFLOW_TRACKING_URI="file:./mlruns"

# 2) Launch the MLflow UI
mlflow ui \
  --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --host 0.0.0.0 \
  --port 5001
```
Open the UI at: **http://localhost:5001**  
Your runs will be stored under the `./mlruns/` directory.


---

## 2) Add Dependencies

In `pyproject.toml` add:
```toml
[tool.poetry.dependencies]
mlflow = "^2.14.0"
boto3 = "^1.34.0"        # usually present in AWS projects
lightgbm = "^4.3.0"      # if you log LGBM params/metrics
```
Then install:
```bash
poetry install
```


---

## 3) Configure Secrets & Environment Variables

### GitHub Actions
Add to **Settings → Secrets and variables → Actions**:
- **Secret**: `MLFLOW_TRACKING_URI` → e.g., `https://mlflow.<your-domain>` (prod) or leave unset (dev/local)

In your training workflow/job:
```yaml
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

### SageMaker (Processing/Training)
Ensure the job container sees `MLFLOW_TRACKING_URI`. Either:
- Pass it via `environment={...}` on your `Estimator`/`Processor`, or
- Read from code when present.

Example:
```python
estimator = SKLearn(
    ...,
    environment={"MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "")},
)
```


---

## 4) Instrument `training.py`

Add MLflow calls around training. Here you can create a .py file for:

1. Points MLflow to the tracking server via MLFLOW_TRACKING_URI and uses the experiment walmart-pdm-<env>.

2. Enables LightGBM autologging (to capture standard params/metrics automatically).

3. Starts an MLflow run and:

Logs selected hyperparameters (objective, classes, learning rate, leaves, etc.).

Fits the model using a validation set (eval_set) and metric (multi_logloss).

(You must replace the ... with real computations) computes validation metrics (accuracy, macro-F1, logloss) and logs them.

Writes an evaluation.json and logs it as an artifact.

Saves the trained LightGBM booster to model.lgb and logs it as an artifact.

Sets tags (env, Git SHA, SageMaker pipeline exec ARN) for traceability.

Prints the MLflow run_id for reference.


The use of MLflow purely for **experiment tracking** (params, metrics, artifacts).
---

## 5) View & Compare Runs

- Open the MLflow UI (prod URL or local UI).
- Choose the experiment `walmart-pdm-<env>`.
- Compare runs by `val_accuracy`, `val_f1_macro`, `val_logloss`, review artifacts (plots, `evaluation.json`).


---



