from simulator import (
    Model,
    ResourceRequirement,
    Profile,
    Task,
    Pipeline,
    Optimizer)

# ---------- first task ----------
task_a_model_1 = Model(
    name='yolo5n',
    resources=ResourceRequirement(cpu=1),
    profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5
)

task_a_model_2 = Model(
    name='yolo5s',
    resources=ResourceRequirement(cpu=2),
    profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8
)

task_a = Task(
    name='crop',
    available_variants = {
        task_a_model_1.name: task_a_model_1,
        task_a_model_2.name: task_a_model_2
    },
    active_variant = task_a_model_1.name,
    replica=2,
    batch=1,
    gpu_mode=False
)

# ---------- second task ----------
task_b_model_1 = Model(
    name='resnet18',
    resources=ResourceRequirement(cpu=1),
    profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        Profile(batch=8, latency=0.8),
    ],
    accuracy=0.5
)

task_b_model_2 = Model(
    name='resnet34',
    resources=ResourceRequirement(cpu=1),
    profiles=[
        Profile(batch=1, latency=0.1),
        Profile(batch=2, latency=0.2),
        Profile(batch=4, latency=0.4),
        Profile(batch=8, latency=0.8),
    ],
    accuracy=0.8
)

task_b = Task(
    name='classification',
    available_variants = {
        task_b_model_1.name: task_b_model_1,
        task_b_model_2.name: task_b_model_2
    },
    active_variant = task_b_model_1.name,
    replica=1,
    batch=1,
    gpu_mode=False
)

inference_graph = [
    task_a,
    task_b
]

pipeline = Pipeline(
    sla=7,
    inference_graph=inference_graph,
    gpu_mode=False
)

optimizer = Optimizer(
    pipeline=pipeline
)

print(f"{pipeline.stage_wise_throughput = }")
print(f"{pipeline.stage_wise_latencies = }")
print(f"{pipeline.stage_wise_replicas = }")
print(f"{pipeline.stage_wise_cpu_usage = }")
print(f"{pipeline.stage_wise_gpu_usage = }")
print(f"{pipeline.cpu_usage = }")
print(f"{pipeline.gpu_usage = }")
print(f"{pipeline.pipeline_latency = }")
print(f"{optimizer.can_sustain_load(arrival_rate=4) = }")
print(f"{optimizer.find_load_bottlenecks(arrival_rate=6) = }")
print(f"{optimizer.objective(alpha=0.5, beta=0.5) = }")