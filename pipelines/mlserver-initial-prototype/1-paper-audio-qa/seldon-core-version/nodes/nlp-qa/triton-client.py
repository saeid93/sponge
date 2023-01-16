import argparse
from functools import partial
import os
import sys

import numpy as np

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype
from datasets import load_dataset

gateway_endpoint = "localhost:32000"
deployment_name = 'audio-qa'
namespace = "default"
endpoint = f"http://{gateway_endpoint}/seldon/{namespace}/{deployment_name}/v2/models/infer"
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")
input_data = ds[0]["audio"]["array"]

try:
    triton_client = httpclient.InferenceServerClient(
                url=endpoint, verbose=False, concurrency=1)
except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

try:
    triton_client.infer(
                        inputs=input_data
                        )
except InferenceServerException as e:
            print("inference failed: " + str(e))
            if FLAGS.streaming:
                triton_client.stop_stream()
            sys.exit(1)