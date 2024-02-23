from barazmoon.slas import slas_workload_generator


def make_slas(image_size: int, sla: int, length: int):
    available_slas = slas_workload_generator(image_size=image_size, sla=sla)
    if length <= len(available_slas):
        return available_slas[:length]
    else:
        num = length // len(available_slas)
        output = available_slas * (num + 1)
        return output[:length]