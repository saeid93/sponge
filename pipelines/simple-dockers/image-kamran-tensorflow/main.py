import os
import numpy as np
import tensorflow as tf
# import torch
# from torchvision import models
# from torchvision import transforms
from keras_vggface.utils import preprocess_input, decode_predictions
from PIL import Image
# import timm
import time
import datetime
import logging

try:
    ITERATIONS = int(os.environ['ITERATIONS'])
    logging.warning(f'ITERATIONS set to: {ITERATIONS}')
except KeyError as e:
    ITERATIONS = 60
    logging.warning(
        f"ITERATIONS env variable not set, using default value: {ITERATIONS}")
try:
    NUM_THREADS = int(os.environ['NUM_THREADS'])
    logging.warning(f'NUM_THREADS set to: {NUM_THREADS}')
except KeyError as e:
    NUM_THREADS = 1
    logging.warning(
        f"NUM_THREADS env variable not set, using default value: {NUM_THREADS}")


dir = os.path.dirname(__file__)
image_name = 'input-sample.JPEG'
path = os.path.join(dir, image_name)

start = time.time()
image = Image.open(path)
image = image.resize((224, 224))
image = np.asarray(image)
batch = image.astype('float32')
batch = np.expand_dims(batch, axis=0)
postprocessing_time = time.time() - start
logging.warning(f"preprocessing time: {postprocessing_time}")

start = time.time()
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)
model = tf.keras.models.load_model(
    os.path.join(dir, 'face-resnet50.h5'), compile=False)
logging.warning(f"model loading time: {time.time() - start}")

logging.warning('starting the experiments')

model_times = []

for i in range(ITERATIONS):
    start = time.time()
    # logging.warning(f'inference time: {end-start}')
    preprocess_start = time.time()
    preprocessed = preprocess_input(batch, version=2)
    logging.warning(f'preprocess time: {time.time()-preprocess_start}')
    prediction_start = time.time()
    predicted = model.predict(preprocessed)
    logging.warning(f'prediction time: {time.time()-prediction_start}')
    decoding_start = time.time()
    predictions = decode_predictions(predicted)
    logging.warning(f'decoding time: {time.time()-decoding_start}') 
    end = time.time()
    model_times.append(end-start)
    logging.warning(f'totatl inference time: {end-start}')

logging.warning('model times:')
logging.warning(model_times)

total_times = postprocessing_time + np.array(model_times)
logging.warning('total times:')
logging.warning(total_times)

logging.warning('total times average:')
logging.warning(np.average(total_times))

logging.warning('total times p99:')
logging.warning(np.percentile(total_times, 99))


time.sleep(100)
