# Predictive Maintenance (AWS + SageMaker + Poetry)

End-to-end pipeline for **training, registering, and deploying** a Predictive Maintenance model on **Amazon SageMaker**, managed with **Poetry**, **GitHub Actions**, and MLOps best practices (pre-commit, tests, IaC).

## 🗂 Repository Structure

```
.
├── ci/
│   ├── ci.yml
│   ├── deploy-infra.yml
│   └── run-sagemaker-pipeline.yml
├── data/                           # optional local CSVs for quick tests
├── infra/
│   ├── cloudformation.yml          # Infra: buckets, roles, Lambda, etc.
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
├── Makefile                        # optional local shortcuts
├── pyproject.toml                  # Poetry (deps + toolchain)
├── .pre-commit-config.yml          # quality gates
└── README.md
```

## ✅ Requirements

- **Python 3.10** (or the version pinned in `pyproject.toml`)
- **Poetry** for dependency management
- **AWS CLI** configured with permissions for SageMaker, S3, IAM, CloudFormation
- **jq** (for JSON parameter handling in infra deploys)
- **Docker** (optional: if you build API/Lambda containers)

## 🔐 Conventions & Variables

- Region: `AWS_DEFAULT_REGION` (e.g., `us-east-2`)
- AWS Account ID: `AWS_ACCOUNT_ID`
- Bucket scheme (convention):
  - `walmart-predictive-maintenance-<ENV>-raw-data`
  - `walmart-predictive-maintenance-<ENV>-staging-data`
  - `walmart-predictive-maintenance-<ENV>-model-data`
  - `walmart-predictive-maintenance-<ENV>-artifacts`
- SageMaker role: `arn:aws:iam::<ACCOUNT_ID>:role/AmazonSageMakerServiceCatalogProductsUseRole`

> Adjust names in `infra/*.json`, `pipeline.py`, and/or `Makefile` as needed.

---

## 🚀 Getting Started (Local)

1) **Clone & install deps**
```bash
pip install -U pip
pip install poetry==1.8.3
poetry install
```

2) **Install pre-commit (optional but recommended)**
```bash
poetry run pre-commit install
```

3) **Local preprocessing (optional)**
```bash
poetry run python pipelines/predictive-maintenance/preprocess/preprocess.py \
  --input-telemetry-s3 data/PdM_telemetry.csv \
  --input-errors-s3    data/PdM_errors.csv \
  --input-maint-s3     data/PdM_maint.csv \
  --input-failures-s3  data/PdM_failures.csv \
  --input-machines-s3  data/PdM_machines.csv \
  --train-out   /tmp/pdm/train \
  --validate-out /tmp/pdm/validate \
  --test-out    /tmp/pdm/test
```

4) **Run tests**
```bash
poetry run pytest -q
```

---

## 🏗 Infrastructure Deployment

1) Review/edit params in `infra/parameters_<ENV>.json`.
2) Deploy the CloudFormation stack:
```bash
aws cloudformation deploy \
  --region $AWS_DEFAULT_REGION \
  --template-file infra/cloudformation.yml \
  --stack-name walmart-pdm-infra-$ENV \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides $(jq -r '.[] | "\(.ParameterKey)=\(.ParameterValue)"' infra/parameters_${ENV}.json | paste -sd " ")
```

> This creates buckets, roles, and (optionally) a Lambda for orchestration.

---

## 🧪 CI with GitHub Actions

- **`ci/ci.yml`**: dependency install, lint (ruff/black), type-check (mypy), tests, coverage.
- **`ci/run-sagemaker-pipeline.yml`**: runs the SageMaker pipeline from `pipelines/predictive-maintenance/pipeline.py`.
- **`ci/deploy-infra.yml`**: deploys/updates CloudFormation in `dev/prod`.

> Workflows consume GitHub **secrets** (AWS creds, roles, etc.). Adapt variable/secret names to your org.

---

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

### Run the pipeline from your machine

```bash
poetry run python pipelines/run_pipeline.py \
  --module-name pipelines.predictive-maintenance.pipeline \
  --execute-pipeline true \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/AmazonSageMakerServiceCatalogProductsUseRole \
  --kwargs '{
      "region": "us-east-2",
      "company_name": "walmart",
      "project_name": "predictive-maintenance",
      "environment": "dev",
      "model_package_group_name": "walmart-pdm-dev"
  }'
```

> Alternative: call the **Makefile** (`make train ENV=dev`) or let `ci/run-sagemaker-pipeline.yml` do it.

---

## 🚢 Model Deployment (Online Endpoint)

Once a model is approved/registered:

1) **Create/Update Endpoint** (via your `deploy-model` script or console).
   - Define `Model` → `EndpointConfig` → `Endpoint`.
   - Keep the endpoint name, e.g., `walmart-pdm-dev`.

2) **Invoke the endpoint** (CSV example with pre-aligned features):
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

## 🌐 Online Inference (optional microservice/Lambda)

- Suggested module: `pipelines/predictive-maintenance/features/online_fe.py`
  - Reuse `utils_preprocess.py`
  - Align columns to `feature_order.json` exported during training
- Invoker service:
  - `services/online/lambda_app.py` (Lambda + API Gateway/Function URL), or
  - `services/online/fastapi_app.py` (Docker/ECS/Fargate)

> Typical env vars: `SM_ENDPOINT_NAME`, `FEATURE_ORDER_S3`, `AWS_REGION`.

---

## 📦 Batch Transform (optional)

For **batch scoring** on S3:
- Define `ci/batch-transform.yml` to submit a **TransformJob** with:
  - `ModelName` (from registry)
  - `TransformInput` (S3 CSV/Parquet)
  - `TransformOutput` (destination S3 prefix)
  - `TransformResources` (instance type/count)
- If you need **feature engineering** first, run a **Processing Job** before the Transform (using the same `preprocess.py` or a “score-only” variant).

---

## 🧪 Tests & Quality

- **Unit tests** under `tests/` (preprocess, training, utils).
- **pre-commit** runs:
  - ruff, black, mypy
  - YAML/JSON/TOML validations
  - notebook output cleanup (nbstripout)
  - secret scanning (detect-secrets)

Local:
```bash
poetry run pytest -q
poetry run pre-commit run --all-files
```

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

## 🤝 Contributing

1) Create feature branch: `feat/<name>`
2) `poetry install`, `pre-commit install`
3) `pytest`, `ruff`, `black`, `mypy`
4) Open a Pull Request → CI will run

---

## 📄 License

MIT (or your choice)

---

### Final Notes

- Keep a **single source of truth** for S3 paths, roles, buckets, and model/package names (variables/JSON params).
- Avoid duplicating FE logic: **reuse** `utils_preprocess.py` for both batch (preprocess) and **online** (`online_fe.py`).
- Persist training artifacts (e.g., `feature_order.json`, `preprocessor.joblib`) to ensure **inference alignment**.
