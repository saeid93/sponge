import requests
from transformers import pipeline
from datasets import load_dataset
from pprint import PrettyPrinter
import numpy as np
pp = PrettyPrinter(indent=4)


ds = load_dataset(
    "hf-internal-testing/librispeech_asr_demo",
    "clean",
    split="validation")

input_data = ds[0]["audio"]["array"]

task = "automatic-speech-recognition"
model_name = 'facebook/s2t-small-librispeech-asr'
data_batch_size = 256
model_batch_size = 1
# batch = np.vstack((input_data, input_data)) 
# for i in range(5):
#        batch = np.vstack((batch, input_data))

batch = []
for i in range(data_batch_size):
    batch.append(input_data)

model  = pipeline(
    task=task,
    model=model_name,
    batch_size=model_batch_size)

res = model(batch)
print(res)