from smart_kube.util import Histogram

# ------------- test histogram option --------------
#  based-on:
# https://github.com/kubernetes/autoscaler/blob/
# master/vertical-pod-autoscaler/pkg/recommender/
# util/decaying_histogram_test.go

first_bucket_size = 1.0
max_value = 10.0
ratio = 1.0
epsilon = 1e-15
half_life = 3600
reference_timestamp = 1234569600
start_timestamp = 1234567890

histogram = Histogram(
    max_value=max_value,
    first_bucket_size=first_bucket_size,
    epsilon=epsilon,
    ratio=ratio,
    half_life=half_life,
    reference_timestamp=reference_timestamp
)

histogram.add_sample(value=2, weight=1000, timestamp=start_timestamp)
histogram.add_sample(value=1, weight=1, timestamp=start_timestamp+3600*20)
assert 2 == histogram.percentile(0.999)
assert 3 == histogram.percentile(1.0)
