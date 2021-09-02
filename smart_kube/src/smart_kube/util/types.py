# https://github.com/kubernetes/autoscaler/blob/master/
# vertical-pod-autoscaler/pkg/recommender/model/types.go
def cores_to_millicores(value):
    """
    ResourceAmount represents quantity of a certain resource
    within a container.
    Note this keeps CPU in millicores
    From:
        types.go
    """
    return int(value * 1000)


def millicores_to_cores(value: float):
    """
    CoresFromCPUAmount converts ResourceAmount to number of
    cores expressed as float64.
    From:
        types.go
    """
    return value / 1000.0


def bytes_to_int_bytes(value: float) -> int:
    """
    BytesFromMemoryAmount converts ResourceAmount to number
    of bytes expressed as float64.
    Note this keeps memory in bytes
    From:
        types.go
    """
    return int(value)


def megabytes_to_bytes(value: float) -> float:
    """
    """
    return value * 10e6


def bytes_to_megabytes(value: float) -> float:
    """
    """
    return value / 10e6
