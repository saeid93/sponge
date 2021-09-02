from smart_kube.util import Histogram

# ------------- test histogram option --------------
#  based-on:
# https://github.com/kubernetes/autoscaler/blob/
# master/vertical-pod-autoscaler/pkg/recommender/
# util/histogram_test.go

first_bucket_size = 40.0
max_value = 500
ratio = 1.5

histogram = Histogram(
    max_value=max_value,
    first_bucket_size=first_bucket_size,
    ratio=ratio,
    time_decay=False
)

assert histogram.num_buckets == 6
assert histogram.get_bucket_start(0) == 0.0
assert histogram.get_bucket_start(1) == 40.0
assert histogram.get_bucket_start(2) == 100.0
assert histogram.get_bucket_start(3) == 190.0
assert histogram.get_bucket_start(4) == 325.0
assert histogram.get_bucket_start(5) == 527.5


assert histogram.find_bucket(-1.0) == 0
assert histogram.find_bucket(39.99) == 0
assert histogram.find_bucket(40.0) == 1
assert histogram.find_bucket(100.0) == 2
assert histogram.find_bucket(900.0) == 5
