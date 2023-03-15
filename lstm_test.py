import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

model = load_model(f"./lstm_saved_model")
with open(f"./workload.txt", "r") as f:
    workload = f.readlines()
workload = workload[0].split()
workload = list(map(int, workload))
workload = list(filter(lambda x:x!=0, workload)) # for removing missing hours
hour = 60 * 60
day = hour * 24
test_idx = 18 * day
test_data = workload[test_idx:test_idx + 2 * hour]

def get_x_y(data):
    x = []
    y = []
    history_seconds = 600
    for i in range(0, len(data) - history_seconds, 60):
        t = data[i:i+history_seconds]
        for j in range(0, len(t), 60):
            x.append(max(t[j:j+60]))
        y.append(max(data[i+history_seconds:i+history_seconds+60]))
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
plt.savefig('lstm_prediction.png')