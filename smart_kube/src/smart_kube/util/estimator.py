import numpy as np
from smart_kube.util import Histogram
from typing import Tuple


class Estimator:
    """
    From:
        https://github.com/kubernetes/autoscaler/blob/master/
        vertical-pod-autoscaler/pkg/recommender/logic/estimator.go
    """
    def percentile_estimatior(self, h_mem: Histogram, h_cpu: Histogram,
                              p_mem: float,
                              p_cpu: float) -> Tuple[float, float]:
        """
        From:
            estimator.go
        """
        return h_mem.percentile(p_mem), h_cpu.percentile(p_cpu)

    def confidence_multiplier_estimator(
            self, resource_value: float, first_sample_start_time: int,
            last_sample_start_time: int, total_sample_count: int,
            multiplier: float, exponent: float) -> float:
        """
        From:
            estimator.go
        """
        if first_sample_start_time == 0 and last_sample_start_time == 0:
            return resource_value
        confidence = get_confidence(
            first_sample_start_time,
            last_sample_start_time,
            total_sample_count)
        scaled_resource_value = resource_value * np.power(
            1+multiplier/confidence, exponent)
        return scaled_resource_value

    def margin_estimator(self, resource_value: float,
                         margin_fraction: float = 0.15) -> float:
        """
        From:
            estimator.go
        """
        margin = resource_value * margin_fraction
        new_resource_value = resource_value + margin
        return new_resource_value

    def min_resources_estimator(self, resource_value: float,
                                min_resource: float) -> float:
        """
        From:
            estimator.go
        """
        if resource_value < min_resource:
            return min_resource
        return resource_value


def get_confidence(first_sample_start_time: int, last_sample_start_time: int,
                   total_sample_count: int):
    """
    Returns a non-negative real number that heuristically measures how much
    confidence the history aggregated in the AggregateContainerState provides.
    For a workload producing a steady stream of samples over N days at the rate
    of 1 sample per minute, this metric is equal to N.
    This implementation is a very simple heuristic which looks at the total
    count of samples and the time between the first and the last sample.
    From:
        estimator.go
    """
    # Distance between the first and the last observed sample time, measured
    # in days.
    day_length = 3600 * 24
    life_span_in_days = (
        last_sample_start_time - first_sample_start_time) / day_length
    # Total count of samples normalized such that it equals the number of days
    # for frequency of 1 sample/minute.
    # TODO find out the problem
    # doubt between (60 * 24) or (3600 * 24)
    sample_amount = total_sample_count / (3600*24)
    return np.min([life_span_in_days, sample_amount])
