import os
from PIL import Image
import numpy as np
from seldon_core.seldon_client import SeldonClient
from transformers import pipeline
from datasets import load_dataset


# single node inferline
gateway_endpoint="localhost:32000"
deployment_name = 'nlp'
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
# path = "/home/cc/infernece-pipeline-joint-optimization/pipelines/21-pipelines-prototype/audio-pipeline/seldon-core-version/sample-dataset.mp3"
# translator  = pipeline(task="automatic-speech-recognition", model="facebook/s2t-small-librispeech-asr")
# audio_dataset = Dataset.from_dict({"audio": [path]}).cast_column("audio", Audio())
# audio_dataset[0]


translator  = pipeline(task="automatic-speech-recognition", model="facebook/s2t-small-librispeech-asr")
print(translator(ds[0]["audio"]["array"]))


# image = np.array(image)
# response = sc.predict(
#     data=ds[0]["audio"]["array"]
# )


# print(response.response['data']['ndarray'][0])
