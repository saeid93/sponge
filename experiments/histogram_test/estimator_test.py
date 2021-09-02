from smart_kube.util import (
    Histogram,
    Estimator,
    cores_to_millicores,
    millicores_to_cores,
    bytes_to_int_bytes
)


cpu_first_bucket_size = 0.01
cpu_max_value = 1000

memory_first_bucket_size = 1e7
memory_max_value = 1e12

estimator = Estimator()

# ------------- percentile estimator test -------------
memory_histogram = Histogram(
    max_value=memory_max_value,
    first_bucket_size=memory_first_bucket_size,
    time_decay=False
)

cpu_histogram = Histogram(
    max_value=cpu_max_value,
    first_bucket_size=cpu_first_bucket_size,
    time_decay=False
)

cpu_histogram.add_sample(value=1, weight=1.0)
cpu_histogram.add_sample(value=2, weight=1.0)
cpu_histogram.add_sample(value=3, weight=1.0)

memory_histogram.add_sample(value=1e9, weight=1.0)
memory_histogram.add_sample(value=2e9, weight=1.0)
memory_histogram.add_sample(value=3e9, weight=1.0)

mem_e, cpu_e = estimator.percentile_estimatior(
    h_cpu=cpu_histogram,
    h_mem=memory_histogram,
    p_cpu=0.2,
    p_mem=0.5)

assert bytes_to_int_bytes(mem_e) == 2093479957
assert cores_to_millicores(cpu_e) == 1016

# ------------- margin test -------------

cpu = estimator.margin_estimator(
    resource_value=cores_to_millicores(3.14),
    margin_fraction=0.1)

mem = estimator.margin_estimator(
    resource_value=3.14e9,
    margin_fraction=0.1)

assert millicores_to_cores(cpu) == 3.454
assert mem == 3454000000

# ------------- confidence multiplier test -------------
cpu = estimator.confidence_multiplier_estimator(
    resource_value=3.14,
    first_sample_start_time=0,
    last_sample_start_time=960,
    total_sample_count=9,
    multiplier=0.1,
    exponent=2.0)

mem = estimator.confidence_multiplier_estimator(
    resource_value=3.14e9,
    first_sample_start_time=0,
    last_sample_start_time=960,
    total_sample_count=9,
    multiplier=0.1,
    exponent=2.0)

assert cpu == 907.46
assert mem == 907460000000.0

# ------------- confidence min resources test -------------

cpu = estimator.min_resources_estimator(
    resource_value=cores_to_millicores(3.14),
    min_resource=cores_to_millicores(0.2))

mem = estimator.min_resources_estimator(
    resource_value=2e7,
    min_resource=4e8)

assert millicores_to_cores(cpu) == 3.14
assert mem == 4e8
