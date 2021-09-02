import numpy as np
from math import log


# look TestPercentileEstimator in the estimator_test.go for every option
class Histogram:
    def __init__(self,
                 max_value,
                 first_bucket_size,
                 ratio=1.05,
                 epsilon=0.0001,
                 half_life=24*3600,
                 time_interval=60,
                 time_decay=True,
                 reference_timestamp=0) -> None:
        """A Python implementation of the vertical pod autoscaler histogram

        From:
            This is the Python implementation of these parts of the vpa code:
            https://github.com/kubernetes/autoscaler/blob/master/
            vertical-pod-autoscaler/pkg/recommender/util/histogram_options.go
            https://github.com/kubernetes/autoscaler/blob/master/
            vertical-pod-autoscaler/pkg/recommender/util/histogram.go
            https://github.com/kubernetes/autoscaler/blob/master/
            vertical-pod-autoscaler/pkg/recommender/util/decaying_histogram.go

            For testing the go codes and sampeles see:
            https://github.com/kubernetes/autoscaler/blob/master/
            vertical-pod-autoscaler/pkg/recommender/util/histogram_options_test.go
            https://github.com/kubernetes/autoscaler/blob/master/
            vertical-pod-autoscaler/pkg/recommender/util/histogram_test.go
            https://github.com/kubernetes/autoscaler/blob/master/
            vertical-pod-autoscaler/pkg/recommender/util/decaying_histogram_test.go
            https://github.com/kubernetes/autoscaler/blob/master/
            vertical-pod-autoscaler/pkg/recommender/logic/estimator_test.go

        Args:
            ratio (float, optional):
            the bucket bounderay growth. Defaults to 0.05.

            half_life ([type], optional):
            half life according to the autopilot paper. Defaults to 24*3600.

            time_intervals (int, optional): timestep time interval.
            Defaults to 60.

            ratio (float, optional): Make each bucket 5% larger than the
                previous one.

            episolon (int, optional): epsilon is the minimal weight kept in
                histograms,
            it should be small enough that old samples
            (just inside MemoryAggregationWindowLength)
            added with minSampleWeight are still kept

            histogram with exponentially growing bucket boundaries. The first
            bucket covers the range [0..firstBucketSize).
            Bucket with index n has size equal to
            firstBucketSize * ratio^n.
            It follows that the bucket with index n >= 1 starts at:
                firstBucketSize * (1 + ratio + ratio^2 + ... + ratio^(n-1)) =
                firstBucketSize * (ratio^n - 1) / (ratio - 1).
            The last bucket start is larger or equal to maxValue.
            Requires maxValue > 0, firstBucketSize > 0, ratio > 1, epsilon > 0.
        """
        self.max_value = max_value
        self.first_bucket_size = first_bucket_size
        self.ratio = ratio
        self.half_life = half_life
        self.time_interval = time_interval
        self.epsilon = epsilon
        self.time_decay = time_decay
        self.total_timesteps = 0
        self.bin_boundaries = self.gen_bin_boundaries()
        self.bucket_weight = np.zeros(self.num_buckets)
        self.min_bucket = self.num_buckets
        self.max_bucket = 0
        self.reference_time = reference_timestamp
        self.total_sample_count = 0

    def add_sample(self, value: float, weight: float, timestamp: float = 1.0):
        """add a new sample to the histogram

        From:
            histogram.go
            decaying_histogram.go

        Args:
            value (float): the value of the resource usage
            weight (float): the weight of the resource usage
            (might be decayed based on timestamp)
            timestamp (float, optional): timestamp for the decaying
            histograams. Defaults to 1.0.

        Raises:
            ValueError: weights should not be negative
        """
        if weight < 0:
            raise ValueError("sample weight must be non-negative")
        if self.time_decay:
            weight *= self.decay_factor(timestamp)
        bucket = self.find_bucket(value)
        self.bucket_weight[bucket] += weight
        if bucket < self.min_bucket\
           and self.bucket_weight[bucket] >= self.epsilon:
            self.min_bucket = bucket
        if bucket > self.max_bucket\
           and self.bucket_weight[bucket] >= self.epsilon:
            self.max_bucket = bucket
        self.total_sample_count += 1

    def gen_bin_boundaries(self) -> list:
        """make growing bins according to the vpa algroithm

        From:
            histrogram_options.go

        Returns:
            list: bins boundries
        """
        bins = [0, self.first_bucket_size]
        next_bucket = self.first_bucket_size
        while next_bucket <= self.max_value:
            bucket_length = bins[-1] - bins[-2]
            bucket_length *= self.ratio
            next_bucket += bucket_length
            bins.append(next_bucket)
        return bins

    def percentile(self, percentile: float) -> float:
        """compute the percentile of the usage histogram

        From:
            histogram.go

        Args:
            percentile (float): the requested percentile

        Returns:
            float: requetsted percentile bin value
        """
        partial_sum = 0
        threshold = percentile * self.total_weight
        bucket = self.min_bucket
        for bucket in range(self.min_bucket, self.max_bucket):
            partial_sum += self.bucket_weight[bucket]
            if partial_sum >= threshold:
                break
        else:
            bucket += 1
        if bucket < self.num_buckets-1:
            return self.get_bucket_start(bucket+1)
        return self.get_bucket_start(bucket)

    def decay_factor(self, timestamp: float) -> float:
        """ USed in A histogram that gives newer samples a higher weight than
        the old samples, gradually decaying ("forgetting") the past samples.
        The weight of each sample is multiplied by the factor of
        2^((sampleTime - referenceTimestamp) / halfLife).
        This means that the sample loses half of its weight ("importance") with
        each halfLife period.

        From:
            decaying_histogram.go

        Args:
            timestamp : timestamp of the adding sample

        Returns:
            float: the multiplier decaying facotr
        """
        # TODO add other stuff in the decay_factor code
        time_elapsed = float(timestamp - self.reference_time)
        decay_factor = np.exp2(time_elapsed / self.half_life)
        return decay_factor

    def find_bucket(self, value: float) -> int:
        """Returns the index of the bucket for given value.
        This is the inverse function to
        GetBucketStart(), which yields the following formula for
        the bucket index:
        bucket(value) = \
            floor(log(value/firstBucketSize*(ratio-1)+1) / log(ratio))

        From:
            histogram_options.go

        Args:
            value (int): the bucket that value belongs
        """
        # linear histogram
        if self.ratio == 1:
            bucket = int(value/self.first_bucket_size)
            if bucket < 0:
                return 0
            if bucket >= self.num_buckets:
                return self.num_buckets - 1
            return bucket
        # exponential histogram
        else:
            if value < self.first_bucket_size:
                return 0
            bucket = int(log(
                value*(self.ratio-1)/self.first_bucket_size+1, self.ratio))
            if bucket >= self.num_buckets:
                return self.num_buckets - 1
            return bucket

    @property
    def num_buckets(self):
        return len(self.bin_boundaries)

    @property
    def total_weight(self):
        return sum(self.bucket_weight)

    def get_bucket_start(self, bucket):
        """
        From:
            histogram_options.go
        """
        if bucket < 0 or bucket >= self.num_buckets:
            raise ValueError("index {} out of range [0..{}]".format(
                bucket, self.num_buckets-1))
        return self.bin_boundaries[bucket]

    def get_bucket_end(self, bucket):
        return self.bin_boundaries[bucket+1]

# TODO remaining fucntionalities for later (if needed)
# -------- from decaying_histogram_options.go --------
    def shift_reference_timestamp(self, other):
        """
        From:
            decaying_histogram_options.go
        """
        raise NotImplementedError

# -------- from histogram.go --------
    def scale(self, factor: float):
        """
        From:
            histogram.go
        """
        raise NotImplementedError

    def update_min_and_max_bucket(self):
        """
        From:
            histogram.go
        """
        raise NotImplementedError

# -------- both histogram.go and decaying_histogram_optiones.go --------

    def equals(self, other):
        """
        From:
            histogram.go and decaying_histogram_options.go
        """
        raise NotImplementedError

    def merge(self, other):
        """
        From:
            histogram.go and decaying_histogram_options.go
        """
        raise NotImplementedError

    def is_empty(self, other):
        """
        From:
            histogram.go and decaying_histogram_options.go
        """
        raise NotImplementedError

    def save_to_checkpoint(self, other):
        """
        From:
            histogram.go and decaying_histogram_options.go
            produce a compact representation of the
            histogram with dictionaries
        """
        raise NotImplementedError

    def load_from_checkpoint(self, other):
        """
        From:
            histogram.go and decaying_histogram_options.go
            returns back the histogram from the compact
            representation
        """
        raise NotImplementedError
