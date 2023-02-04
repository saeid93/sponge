from simulator import (
    Model,
    ResourceAllocation,
    Profile,
    Task,
    Pipeline,
    Optimizer)

# ---------- first task ----------
task_a_model_1 = Model(
    name='yolo5n',
    resource_allocation=ResourceAllocation(cpu=1),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5
)

task_a_model_2 = Model(
    name='yolo5n',
    resource_allocation=ResourceAllocation(cpu=2),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5
)

task_a_model_3 = Model(
    name='yolo5s',
    resource_allocation=ResourceAllocation(cpu=1),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8
)

task_a_model_4 = Model(
    name='yolo5s',
    resource_allocation=ResourceAllocation(cpu=2),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8
)

task_a = Task(
    name='crop',
    available_model_profiles = [
        task_a_model_1,
        task_a_model_2,
        task_a_model_3,
        task_a_model_4
    ],
    active_variant = 'yolo5s',
    active_allocation=ResourceAllocation(cpu=2),
    replica=2,
    batch=1,
    gpu_mode=False
)

# ---------- second task ----------
task_b_model_1 = Model(
    name='resnet18',
    resource_allocation=ResourceAllocation(cpu=1),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5
)

task_b_model_2 = Model(
    name='resnet18',
    resource_allocation=ResourceAllocation(cpu=2),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5
)

task_b_model_3 = Model(
    name='resnet34',
    resource_allocation=ResourceAllocation(cpu=1),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8
)

task_b_model_4 = Model(
    name='resnet34',
    resource_allocation=ResourceAllocation(cpu=2),
    measured_profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        # Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8
)

task_b = Task(
    name='classification',
    available_model_profiles = [
        task_b_model_1,
        task_b_model_2,
        task_b_model_3,
        task_b_model_4
    ],
    active_variant = 'resnet34',
    active_allocation=ResourceAllocation(cpu=1),
    replica=1,
    batch=1,
    gpu_mode=False
)

inference_graph = [
    task_a,
    task_b
]

pipeline = Pipeline(
    inference_graph=inference_graph,
    gpu_mode=False
)

optimizer = Optimizer(
    pipeline=pipeline
)


print(f"{pipeline.stage_wise_throughput = }")
print(f"{pipeline.stage_wise_latencies = }")
print(f"{pipeline.stage_wise_replicas = }")
print(f"{pipeline.stage_wise_cpu = }")
print(f"{pipeline.stage_wise_gpu = }")
print(f"{pipeline.cpu_usage = }")
print(f"{pipeline.gpu_usage = }")
print(f"{pipeline.pipeline_latency = }")

print(f"{optimizer.can_sustain_load(arrival_rate=4) = }")
print(f"{optimizer.find_load_bottlenecks(arrival_rate=30) = }")
print(f"{optimizer.objective(alpha=0.5, beta=0.5) = }")

# scaling, sla and arrival rate metrics
scaling_cap = 2
sla = 5
arrival_rate = 10

# all states
states = optimizer.all_states(scaling_cap=scaling_cap)
print(f"{states = }")
states.to_markdown('all-states.csv')

# all feasibla states
all_states = optimizer.all_states(
    check_constraints=True, scaling_cap=scaling_cap,
    arrival_rate=arrival_rate, sla=sla)
print(f"{all_states = }")
all_states.to_markdown(
    f'feasible_scaling_cap_{scaling_cap}_sla_{sla}_load_{arrival_rate}.csv')

# optimal states
optimal = optimizer.optimize(
    scaling_cap=scaling_cap, sla=sla, arrival_rate=arrival_rate)
optimal.to_markdown(
    f'optimal_scaling_cap_{scaling_cap}_sla_{sla}_load_{arrival_rate}.csv')
