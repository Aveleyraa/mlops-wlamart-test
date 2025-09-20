# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A CLI to create or update and run pipelines."""
from __future__ import absolute_import

import argparse
import json
import sys
import traceback

from pipelines._utils import get_pipeline_driver, convert_struct, get_pipeline_custom_tags


def main():  # pragma: no cover
    """
    Create or updates and runs the pipeline.

    Create or update the pipeline and runs it.
    """
    parser = argparse.ArgumentParser(
        "Create or update and runs the pipeline for the pipeline script."
    )

    parser.add_argument(
        "-n",
        "--module-name",
        dest="module_name",
        type=str,
        help="Module name of the pipeline to import.",
    )
    parser.add_argument(
        "-kwargs",
        "--kwargs",
        dest="kwargs",
        default=None,
        help="Dict string of keyword arguments for the pipeline generation (if supported)",
    )
    parser.add_argument(
        "-role-arn",
        "--role-arn",
        dest="role_arn",
        type=str,
        help="The role arn for the pipeline service execution role.",
    )
    parser.add_argument(
        "-description",
        "--description",
        dest="description",
        type=str,
        default=None,
        help="The description of the pipeline.",
    )
    parser.add_argument(
        "-tags",
        "--tags",
        dest="tags",
        default=None,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )
    parser.add_argument(
        "-execute-pipeline",
        "--execute-pipeline",
        dest="execute_pipeline",
        type=str,
        required=True,
        help="""List of dict strings of '[{"Key": "string", "Value": "string"}, ..]'""",
    )   
    args = parser.parse_args()

    execute_pipeline = args.execute_pipeline.lower() == 'true'
    print(f"Valor para ejecutar el pipeline de sagemaker: {execute_pipeline}")

    if args.module_name is None or args.role_arn is None:
        parser.print_help()
        sys.exit(2)
    tags = convert_struct(args.tags)

    try:
        pipeline = get_pipeline_driver(args.module_name, args.kwargs)
        print("###### Creating/updating a SageMaker Pipeline with the following definition:")
        parsed = json.loads(pipeline.definition())
        print(json.dumps(parsed, indent=2, sort_keys=True))

        # all_tags = get_pipeline_custom_tags(args.module_name, args.kwargs, tags)
        print("🧪 Debug de valores antes de upsert:")
        print("Pipeline name:", pipeline.name)
        print("Base job prefix:", pipeline.base_job_prefix if hasattr(pipeline, 'base_job_prefix') else "N/A")
        print("Role ARN:", args.role_arn)
        print("Definition length:", len(pipeline.definition()))


        upsert_response = pipeline.upsert(
            role_arn=args.role_arn, description=args.description
        )
        print("\n###### Created/Updated SageMaker Pipeline: Response received:")
        print(upsert_response)

        if execute_pipeline:
            print(f"El valor para ejecutar el pipeline es TRUE. Valor: {execute_pipeline}")
            pipeline_execution = pipeline.start()
            print(f"\n###### Execution started with PipelineExecutionArn: {pipeline_execution.arn}")
            print("Waiting for the execution to finish...")
            # Setting the attempts and delay (in seconds) will modify the overall time the pipeline waits.
            # If the execution is taking a longer time, update these parameters to a larger value.
            # Eg: The total wait time is calculated as 60 * 120 = 7200 seconds (2 hours)
            pipeline_execution.wait(max_attempts=120, delay=60) # max wait: 24 hours
            print("\n#####Execution completed. Execution step details:")
            print(pipeline_execution.list_steps())
        else:
            print(f"El valor para ejecutar el pipeline es FALSE. Valor: {execute_pipeline}")

    # Todo print the status?
    except Exception as e:  # pylint: disable=W0703
        print(f"Exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
