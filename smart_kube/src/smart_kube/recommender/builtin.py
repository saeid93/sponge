from smart_kube.util.types import bytes_to_megabytes
from .nonml_interface import NonMLInterface
from typing import Dict, Any
import numpy as np

from smart_kube.util import (
    Histogram,
    Estimator,
    cores_to_millicores,
    millicores_to_cores,
    bytes_to_int_bytes,
    megabytes_to_bytes
)


class Builtin(NonMLInterface):
    def __init__(self, config: Dict[str, Any]):
        config_histogram = config['histogram']
        # cpu values in histogram in cores
        self.cpu_first_bucket_size = config_histogram['cpu'][
            'first_bucket_size']
        self.cpu_max_value = config_histogram['cpu']['max_value']
        # memory values in histogram in bytes
        self.memory_first_bucket_size = config_histogram['memory'][
            'first_bucket_size']
        self.memory_max_value = config_histogram['memory']['max_value']
        self.cpu_histogram = Histogram(
            max_value=self.cpu_max_value,
            first_bucket_size=self.cpu_first_bucket_size
        )
        self.memory_histogram = Histogram(
            max_value=self.memory_max_value,
            first_bucket_size=self.memory_first_bucket_size
        )
        self.timestamps = []
        self.total_sample_count = 0
        self.action_space = config['action_space']
        self.margin = config['margin']
        self.confidence = config['confidence']
        self.min_resource = config['min_resource']
        self.estimator = Estimator()

    def update(self, observation: np.array, timestamp: float):
        """update resource usage with the new observatin from
        the siulator

        Args:
            observation (np.array): the observation from the simulator
            timestamp (float): timestamp of the current observation
        """
        # units in observation -> memory: Megabytes, cpu: Milicores
        # units in histgrams -> memory: bytes, cpu: cores
        self.memory_histogram.add_sample(
            value=megabytes_to_bytes(observation[0]),
            # TODO check definitive guide: based on the current
            # Container’s CPU request value.
            # TODO check autopilot paper
            weight=1.0,
            timestamp=timestamp)
        self.cpu_histogram.add_sample(
            value=millicores_to_cores(observation[1]),
            weight=1.0,
            timestamp=timestamp)
        self.timestamps.append(timestamp)
        self.total_sample_count += 1

    def reset(self):
        self.cpu_histogram = Histogram(
            max_value=self.cpu_max_value,
            first_bucket_size=self.cpu_first_bucket_size
        )
        self.memory_histogram = Histogram(
            max_value=self.memory_max_value,
            first_bucket_size=self.memory_first_bucket_size
        )
        self.timestamps = []
        self.total_sample_count = 0

    def recommender(self):
        """checks if the config of the worklaod is in
        in the correct format
        From:
            CreatePodResourceRecommender in
            https://github.com/kubernetes/autoscaler/blob/
            master/vertical-pod-autoscaler/pkg/recommender/
            logic/recommender.go

        recommendation format
                 ram_lower_bound   cpu_lower_bound
                [                |                |

                 ram_target   cpu_target
                |           |            |

                 ram_higher_bound   cpu_higher_bound
                |                 |                 ]
        """
        if self.memory_histogram.total_sample_count == 0 and\
           self.cpu_histogram.total_sample_count == 0:
            return np.concatenate((
               self.action_space.low[0:4],
               self.action_space.high[0:2]
               ))

        # TODO start here
        # percentiles estimations
        # units in histgrams -> memory: bytes (float), cpu: cores (float)
        # units of returned values -> memory: bytes (int), cpu: cores (float)
        target_bound_cpu_percentile = 0.9
        lower_bound_cpu_percentile = 0.5
        upper_bound_cpu_percentile = 0.95
        target_bound_memory_percentile = 0.9
        lower_bound_memory_percentile = 0.5
        upper_bound_memory_percentile = 0.95
        target_bound_cpu = self.cpu_histogram.percentile(
            target_bound_cpu_percentile)
        lower_bound_cpu = self.cpu_histogram.percentile(
            lower_bound_cpu_percentile)
        upper_bound_cpu = self.cpu_histogram.percentile(
            upper_bound_cpu_percentile)
        target_bound_memory = bytes_to_int_bytes(
            self.memory_histogram.percentile(
                target_bound_memory_percentile))
        lower_bound_memory = bytes_to_int_bytes(
            self.memory_histogram.percentile(
                lower_bound_memory_percentile))
        upper_bound_memory = bytes_to_int_bytes(
            self.memory_histogram.percentile(
                upper_bound_memory_percentile))

        if self.margin:
            # units of inputs -> memory: bytes (int), cpu: cores (float)
            # units of returned values -> memory: bytes (int),
            #                             cpu: milicores (float)
            # adding margin estimations
            margin_fraction = 0.15
            target_bound_cpu = self.estimator.margin_estimator(
                resource_value=cores_to_millicores(target_bound_cpu),
                margin_fraction=margin_fraction)
            lower_bound_cpu = self.estimator.margin_estimator(
                resource_value=cores_to_millicores(lower_bound_cpu),
                margin_fraction=margin_fraction)
            upper_bound_cpu = self.estimator.margin_estimator(
                resource_value=cores_to_millicores(upper_bound_cpu),
                margin_fraction=margin_fraction)
            target_bound_memory = bytes_to_int_bytes(
                self.estimator.margin_estimator(
                    resource_value=target_bound_memory,
                    margin_fraction=margin_fraction))
            lower_bound_memory = bytes_to_int_bytes(
                self.estimator.margin_estimator(
                    resource_value=lower_bound_memory,
                    margin_fraction=margin_fraction))
            upper_bound_memory = bytes_to_int_bytes(
                self.estimator.margin_estimator(
                    resource_value=upper_bound_memory,
                    margin_fraction=margin_fraction))

        if self.confidence:
            # with upper and lower bound confidence
            first_sample_start_time = self.timestamps[0]
            last_sample_start_time = self.timestamps[-1]
            lower_bound_cpu = self.estimator.confidence_multiplier_estimator(
                resource_value=lower_bound_cpu,
                first_sample_start_time=first_sample_start_time,
                last_sample_start_time=last_sample_start_time,
                total_sample_count=self.total_sample_count,
                multiplier=0.001, exponent=-2.0)
            upper_bound_cpu = self.estimator.confidence_multiplier_estimator(
                resource_value=upper_bound_cpu,
                first_sample_start_time=first_sample_start_time,
                last_sample_start_time=last_sample_start_time,
                total_sample_count=self.total_sample_count,
                multiplier=1.0, exponent=2.0)
            lower_bound_memory =\
                self.estimator.confidence_multiplier_estimator(
                    resource_value=lower_bound_cpu,
                    first_sample_start_time=first_sample_start_time,
                    last_sample_start_time=last_sample_start_time,
                    total_sample_count=self.total_sample_count,
                    multiplier=0.001, exponent=-2.0)
            upper_bound_memory =\
                self.estimator.confidence_multiplier_estimator(
                    resource_value=upper_bound_cpu,
                    first_sample_start_time=first_sample_start_time,
                    last_sample_start_time=last_sample_start_time,
                    total_sample_count=self.total_sample_count,
                    multiplier=1.0, exponent=2.0)

            # handle inf in upper bound
            if upper_bound_memory == np.inf:
                upper_bound_memory = bytes_to_int_bytes(
                    megabytes_to_bytes(
                        self.action_space.high[0]))
            if upper_bound_cpu == np.inf:
                upper_bound_cpu = self.action_space.high[1]

        if self.min_resource:
            # with min resource check
            pod_min_cpu_millicore = 25
            pod_min_memory_bytes = 250 * 10e6
            target_bound_cpu = self.estimator.min_resources_estimator(
                resource_value=target_bound_cpu,
                min_resource=pod_min_cpu_millicore)
            lower_bound_cpu = self.estimator.min_resources_estimator(
                resource_value=lower_bound_cpu,
                min_resource=pod_min_cpu_millicore)
            upper_bound_cpu = self.estimator.min_resources_estimator(
                resource_value=upper_bound_cpu,
                min_resource=pod_min_cpu_millicore)
            target_bound_memory = self.estimator.min_resources_estimator(
                resource_value=target_bound_memory,
                min_resource=pod_min_memory_bytes)
            lower_bound_memory = self.estimator.min_resources_estimator(
                resource_value=lower_bound_memory,
                min_resource=pod_min_memory_bytes)
            upper_bound_memory = self.estimator.min_resources_estimator(
                resource_value=upper_bound_memory,
                min_resource=pod_min_memory_bytes)

        # concatenate all recomms in the simulator format
        # units of inputs -> memory: bytes (int),
        #                    cpu: milicores (float)
        # units of returned values -> memory: Megabytes (float),
        #                             cpu: milicores (float)
        recommendation = np.array([
            bytes_to_megabytes(lower_bound_memory), lower_bound_cpu,
            bytes_to_megabytes(target_bound_memory), target_bound_cpu,
            bytes_to_megabytes(upper_bound_memory), upper_bound_cpu
            ])

        # capping
        recommendation = np.clip(
            recommendation,
            a_min=self.action_space.low,
            a_max=self.action_space.high)

        # make it granular as millicores for cpu
        # and megabytes for memory
        recommendation = recommendation.astype(int)

        return recommendation

    def _check_config(self):
        """check the config structure according to
        the recommender method
        """
        pass
