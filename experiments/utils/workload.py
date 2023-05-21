from barazmoon.twitter import twitter_workload_generator

def make_workload(config: dict):
    workload_type = config["workload_type"]
    workload_config = config["workload_config"]

    if workload_type == "static":
        loads_to_test = workload_config["loads_to_test"]
        load_duration = workload_config["load_duration"]
        workload = [loads_to_test] * load_duration
    elif workload_type == "twitter":
        loads_to_test = []
        for w_config in workload_config:
            damping_factor = w_config["damping_factor"]
            start = w_config["start"]
            end = w_config["end"]
            load_to_test = start + "-" + end
            loads_to_test.append(load_to_test)
        workload = twitter_workload_generator(
            loads_to_test[0], damping_factor=damping_factor
        )
        load_duration = len(workload)
    return load_duration, workload