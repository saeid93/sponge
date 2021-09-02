import numpy as np

from smart_kube.util import Histogram

# ------------- histogram test --------------
#  to percentiles on the following histogram: { 1: 1, 2: 2, 3: 3, 4: 4 }.
#  with size of 10
#  based-on:
# https://github.com/kubernetes/autoscaler/blob/master/
# vertical-pod-autoscaler/pkg/recommender/
# util/histogram_test.go

timesteps = 10
time_interval = 60

first_bucket_size = 1
max_value = 10
ratio = 1
time = np.arange(timesteps) * time_interval
# to make this histogram { 1: 1, 2: 2, 3: 3, 4: 4}

histogram = Histogram(
    max_value=max_value,
    first_bucket_size=first_bucket_size,
    ratio=1,
    time_decay=False
)
for i in range(1, 5):
    histogram.add_sample(value=i, weight=i)


assert histogram.percentile(percentile=0.0) == 2
assert histogram.percentile(percentile=0.1) == 2
assert histogram.percentile(percentile=0.2) == 3
assert histogram.percentile(percentile=0.3) == 3
assert histogram.percentile(percentile=0.4) == 4
assert histogram.percentile(percentile=0.5) == 4
assert histogram.percentile(percentile=0.6) == 4
assert histogram.percentile(percentile=0.7) == 5
assert histogram.percentile(percentile=0.8) == 5
assert histogram.percentile(percentile=0.9) == 5
assert histogram.percentile(percentile=1.0) == 5
