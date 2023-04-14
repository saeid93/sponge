import numpy as np
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import regularizers

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(project_dir, "..")))

from barazmoon.twitter import twitter_workload_generator

from experiments.utils.constants import LSTM_PATH, LSTM_INPUT_SIZE


def create_model():
    """
    LSTM model
    """
    model = Sequential()
    model.add(Input(shape=(LSTM_INPUT_SIZE, 1)))
    model.add(LSTM(25, activation="relu", kernel_regularizer=regularizers.L1(0.00001)))
    model.add(Dense(1))
    return model


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
        t = data[i : i + history_seconds]
        for j in range(0, len(t), 60):
            x.append(max(t[j : j + 60]))
        y.append(max(data[i + history_seconds : i + history_seconds + 60]))
    return x, y


def get_data():
    """
    read the dataset
    """
    # load the per second RPS of the Twitter dataset
    workload = twitter_workload_generator("1-26")
    workload = list(filter(lambda x: x != 0, workload))

    # Twitter dataset is for 21 days
    # we take the data of the first 14 days for training
    # we take the data of the next 7 days for testing
    train_to_idx = 14 * 24 * 60 * 60
    workload_train = workload[:train_to_idx]
    workload_test = workload[train_to_idx:]

    # TODO
    train_x, train_y = get_x_y(workload_train)
    test_x, test_y = get_x_y(workload_test)

    return (
        tf.convert_to_tensor(
            np.array(train_x).reshape((-1, LSTM_INPUT_SIZE, 1)), dtype=tf.int32
        ),
        tf.convert_to_tensor(np.array(train_y), dtype=tf.int32),
        tf.convert_to_tensor(
            np.array(test_x).reshape((-1, LSTM_INPUT_SIZE, 1)), dtype=tf.int32
        ),
        tf.convert_to_tensor(np.array(test_y), dtype=tf.int32),
    )


if __name__ == "__main__":
    tf.random.set_seed(7)
    train_x, train_y, test_x, test_y = get_data()
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    model = create_model()
    print(model.summary())
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        train_x, train_y, epochs=30, batch_size=64, validation_data=(test_x, test_y)
    )
    predictions = model.predict(test_x)
    plt.plot(list(range(len(test_y))), list(test_y), label="real values")
    plt.plot(list(range(len(test_y))), list(predictions), label="predictions")
    plt.legend()
    plt.show()
    model.save(LSTM_PATH)
