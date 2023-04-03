import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(
    project_dir, '..')))

from barazmoon.twitter import twitter_workload_generator

from experiments.utils.constants import PROJECT_PATH


model_path = os.path.join(PROJECT_PATH, "lstm-module", "lstm_saved_model")
fig_path = os.path.join(PROJECT_PATH, 'lstm-module', 'lstm_prediction.png')

model = load_model(model_path)
workload = twitter_workload_generator('1-26')
workload = list(filter(lambda x: x != 0, workload)) # for removing missing hours
hour = 60 * 60
day = hour * 24
test_idx = 18 * day
test_data = workload[test_idx:test_idx + 2 * hour]

def get_x_y(data):
    """
    For each 60 seconds it taeks the max of last 60 seconds
    and returns an output with length of len(data)/60 that
    each entry is the maximum rps in each aggregated 60 seconds
    x: series of max of every 1 minute
    y: target of the 10 minutes
    """
    x = []
    y = []
    history_seconds = 600
    for i in range(0, len(data) - history_seconds, 60):
        t = data[i: i + history_seconds]
        for j in range(0, len(t), 60):
            x.append(max(t[j : j + 60]))
        y.append(max(data[i + history_seconds : i + history_seconds + 60]))
    return x, y

test_x, test_y = get_x_y(test_data)

test_x = tf.convert_to_tensor(
    np.array(test_x).reshape((-1, 10, 1)), dtype=tf.float32)
prediction = model.predict(test_x)
plt.plot(list(range(len(test_y))), list(test_y), label="real values")
plt.plot(list(range(len(test_y))), list(prediction), label="predictions")
plt.xlabel("time (minute)")
plt.ylabel("load (RPS)")
plt.legend()
plt.savefig(fig_path)