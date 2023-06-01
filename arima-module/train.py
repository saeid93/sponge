import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from barazmoon.twitter import twitter_workload_generator
import time
import os


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
    history_seconds = 120
    step = 10
    for i in range(0, len(data) - history_seconds, step):
        t = data[i : i + history_seconds]
        for j in range(0, len(t), step):
            x.append(max(t[j : j + step]))
        y.append(max(data[i + history_seconds : i + history_seconds + 2 * step]))
    return x, y


last_day = 21 * 24 * 60 * 60
# load the per second RPS of the Twitter dataset
workload = twitter_workload_generator(f"{0}-{last_day}", damping_factor=5)
workload = list(filter(lambda x: x != 0, workload))
train_to_idx = 14 * 24 * 60 * 60
hour = 60 * 60
day = hour * 24
test_idx = 18 * day
workload = workload[test_idx: test_idx + 2 * hour]
data_x, data_y = get_x_y(workload)
data_x = np.array(data_x).reshape((-1, 12))

preds = []
actual = []

for i in range(len(data_x)):
    model = ARIMA(list(data_x[i]), order=(1,0,0))
    model_fit = model.fit()
    pred = int(model_fit.forecast(steps=2)[1])
    preds.append(pred)
    actual.append(data_y[i])
    if i % 100 == 0:
        print(pred, data_y[i], i, len(data_x))


plt.plot(list(range(len(preds))), actual, label="actual")
plt.plot(list(range(len(preds))), preds, label="pred")
plt.savefig(f"{os.path.dirname(__file__)}/arima.png")
