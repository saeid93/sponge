import os
import numpy as np
import torch
from transformers import pipeline
import time
import logging

try:
    ITERATIONS = int(os.environ['ITERATIONS'])
    logging.warning(f'ITERATIONS set to: {ITERATIONS}')
except KeyError as e:
    ITERATIONS = 60
    logging.warning(
        f"ITERATIONS env variable not set, using default value: {ITERATIONS}")

# torch.set_num_interop_threads(1)
# torch.set_num_threads(1)


batch = "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel"

start = time.time()
model = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english")
logging.warning(f"model loading time: {time.time() - start}")

logging.warning('starting the experiments')
model_times = []
softmax_times = []
for i in range(ITERATIONS):
    start = time.time()
    out = model(batch)
    iter_time = time.time() - start
    model_times.append(iter_time)
    logging.warning(f'iteration {i} time: {iter_time}')
    start = time.time()

logging.warning('total times average:')
logging.warning(np.average(model_times))

logging.warning('total times p99:')
logging.warning(np.percentile(model_times, 99))
