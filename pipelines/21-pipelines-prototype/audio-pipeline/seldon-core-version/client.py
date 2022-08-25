import os
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
from transformers import pipeline
from datasets import load_dataset


# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'audio'
namespace = "default"
sc = SeldonClient(
    gateway_endpoint=gateway_endpoint,
    gateway="istio",
    transport="rest",
    deployment_name=deployment_name,
    namespace=namespace)

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")


translator  = pipeline(task="automatic-speech-recognition", model="facebook/s2t-small-librispeech-asr")


# image = np.array(image)
response = sc.predict(
    data=ds[0]["audio"]["array"]
)

print(response.response['jsonData']['text'])
