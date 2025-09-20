import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import FrameworkProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
print("ESTE ES EL BASE DIR: ", BASE_DIR)


def get_sagemaker_client(region):
    """
    Get the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns
    -------
        sagemaker.session.Session instance

    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """
    Get the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns
    -------
        sagemaker.session.Session instance

    """
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """
    Get the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns
    -------
        PipelineSession instance

    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    """
    Get the pipeline custom tags.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns
    -------
        tags

    """
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name=None,
    pipeline_name=None,
    base_job_prefix=None,
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
    company_name=None,
    project_name=None,
    environment=None
):
    """
    Build a SageMaker Pipeline for LightGBM training (Walmart use case).
    - Step 1: preprocess (feature engineering) -> writes features to S3
    - Step 2: training (Script Mode, LightGBM via requirements.txt)
    - Step 3: register model in Model Registry with metrics
    """
    # Sessions / roles
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    pipeline_session = get_pipeline_session(region, default_bucket)

    env = environment
    base_prefix = base_job_prefix

    # -------------------------
    # Pipeline parameters
    # -------------------------
    processing_instance_count = ParameterInteger("ProcessingInstanceCount", default_value=1)
    train_instance_count = ParameterInteger("TrainInstanceCount", default_value=1)

    processing_instance_type_param = ParameterString(
        "ProcessingInstanceType", default_value=processing_instance_type
    )
    training_instance_type_param = ParameterString(
        "TrainInstanceType", default_value=training_instance_type
    )

    model_package_group_name_param = ParameterString(
        "ModelPackageGroupName",
        default_value=model_package_group_name or f"{project_name}-registry",
    )

    # Raw-data inputs
    input_data_telemetry = ParameterString(
        "InputDataUrlTelemetry",
        default_value=f"s3://{company_name}-{project_name}-{env}-raw-data/telemetry/",
    )
    input_data_errors = ParameterString(
        "InputDataUrlErrors",
        default_value=f"s3://{company_name}-{project_name}-{env}-raw-data/errors/",
    )
    input_data_maint = ParameterString(
        "InputDataUrlMaint",
        default_value=f"s3://{company_name}-{project_name}-{env}-raw-data/maint/",
    )
    input_data_failures = ParameterString(
        "InputDataUrlFailures",
        default_value=f"s3://{company_name}-{project_name}-{env}-raw-data/failures/",
    )
    input_data_machines = ParameterString(
        "InputDataUrlMachines",
        default_value=f"s3://{company_name}-{project_name}-{env}-raw-data/machines/",
    )

    # Where preprocess will write features (aligns with training hyperparameters below)
    features_prefix = f"s3://{default_bucket}/{base_prefix}/features"

    # -------------------------
    # Step 1: Processing (feature engineering)
    # -------------------------
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type_param,
        instance_count=processing_instance_count,
        base_job_name=f"{base_prefix}/sklearn-preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step1_args = sklearn_processor.run(
        source_dir=BASE_DIR,
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"{features_prefix}/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"{features_prefix}/test"),
            ProcessingOutput(output_name="validate", source="/opt/ml/processing/validate", destination=f"{features_prefix}/validate"),
        ],
        code="preprocess/preprocess.py",
        arguments=[
            "--input-telemetry-s3", input_data_telemetry,
            "--input-errors-s3",    input_data_errors,
            "--input-maint-s3",     input_data_maint,
            "--input-failures-s3",  input_data_failures,
            "--input-machines-s3",  input_data_machines,
            "--company-name",       company_name,
            "--project-name",       project_name,
            "--environment",        env,
        ],
    )

    step_process = ProcessingStep(
        name="PreprocessWalmartData",
        step_args=step1_args,
    )

    # -------------------------
    # 2) Training step (Script Mode, includes LightGBM via requirements.txt)
    # -------------------------
    estimator = SKLearn(
        entry_point="training.py",
        source_dir=os.path.join(os.path.dirname(__file__), "train"),
        framework_version="1.2-1",
        role=role,
        instance_type=training_instance_type_param,
        instance_count=train_instance_count,
        sagemaker_session=pipeline_session,
        # contains lightgbm, pyarrow, s3fs...
        dependencies=[os.path.join(os.path.dirname(__file__), "requirements.txt")],
        hyperparameters={
            "train-data": f"{features_prefix}/train/",      # written by preprocess.py
            "test-data": f"{features_prefix}/test/",
            "validate-data": f"{features_prefix}/validate/",
            "target-col": "failure",
            "company_name": company_name,
            "project_name": project_name,
            "environment": env,
        },
    )

    step_train = TrainingStep(
        name="TrainLightGBM",
        estimator=estimator,
        inputs={},  # URIs passed via hyperparameters
    )

    # -------------------------
    # 3) Load metrics from evaluation.json and attach to the model package
    #    training script should write to /opt/ml/output/metrics/evaluation.json
    #    which lands at .../output/metrics/evaluation.json in S3
    # -------------------------
    eval_report = PropertyFile(
        name="EvaluationReport",
        output_name="metrics",
        path="evaluation.json",
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=step_train.properties.ModelArtifacts.S3ModelArtifacts.replace(
                "model.tar.gz", "output/metrics/evaluation.json"
            ),
            content_type="application/json",
        )
    )

    # -------------------------
    # 4) Register in the Model Registry
    # -------------------------
    model = step_train.get_expected_model()
    step_register = ModelStep(
        name="RegisterModel",
        model=model,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_metrics=model_metrics,
        model_package_group_name=model_package_group_name_param,
        approve_model=False,  # can be auto-approved later with a ConditionStep
    )

    # -------------------------
    # Pipeline assembly
    # -------------------------
    pipeline = Pipeline(
        name=pipeline_name or f"{project_name}-pipeline",
        parameters=[
            processing_instance_type_param,
            processing_instance_count,
            training_instance_type_param,
            train_instance_count,
            model_package_group_name_param,
            input_data_telemetry,
            input_data_errors,
            input_data_maint,
            input_data_failures,
            input_data_machines,
        ],
        steps=[step_process, step_train, step_register],
        sagemaker_session=pipeline_session,
    )
    return pipeline
