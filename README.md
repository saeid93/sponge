# Structure

This repository contains the code for our upcoming ML inference pipeline autoconfiguring projects.

**Project 1 Summary** [Joint Optimization of Qualitative and Performance Metrics on ML inference DAGs](https://www.overleaf.com/read/pfnwptxyktff):

Efficiently optimizing multi-model inference pipelines for fast, accurate, and cost-effective inference is a crucial challenge in ML production systems, given their tight end-to-end latency requirements. To simplify the exploration of the vast and intricate trade-off space of accuracy and cost in inference pipelines, providers frequently opt to consider one of them. However, the challenge lies in reconciling accuracy and cost trade-offs.
To address this challenge and propose a solution to efficiently manage model variants in inference pipelines. Model variants are different versions of pre-trained models for the same Deep Learning task with variations in resource requirements, latency, and accuracy. We present IPA, an online deep-learning Inference Pipeline Adaptation system that efficiently leverages model variants for each deep learning task. IPA dynamically configures batch size, replication, and model variants to optimize accuracy, minimize costs, and meet user-defined latency SLAs using Integer Programming. It supports multi-objective settings for achieving different trade-offs between accuracy and cost objectives while remaining adaptable to varying workloads and dynamic traffic patterns. Extensive experiments on a Kubernetes implementation with five real-world inference pipelines demonstrate that IPA improves normalized accuracy by up to 35% with a minimal cost increase of less than 5%.

# Related project Repository

1. **Twitter Trace preprocessing** code is available at [web-service-datasets](https://github.com/reconfigurable-ml-pipeline/web-service-datasets) repository.
2. **Load tester** code is available at [load_tester](https://github.com/reconfigurable-ml-pipeline/load_tester) repository.
3. **Forked MLServer** [MLServer](https://github.com/SeldonIO/MLServer) is a production ready ML Serving platform, due to some modification we needed to make in this project we use a [forked version](https://github.com/saeid93/MLServer) of it

# Project Setup Steps
1. Install the forked MLServer, in the root folder of it do:
```
make install-dev
```
2. Install the barAzmoon library for load testing, in the root folder of the `saeed` branch:
```
pip install -e .
```
3. Install depandancies of the project:
```
pip install -r requirements.txt
```
4. Go to the parent folder `generate_dirs.py` file and run it to populate necessary data folders:
```
python generate_dirs.py
```
5. Go to the lstm-module and train the lstm NN:
```
python lstm_train.py
```
# Adding a new pipeline step
1. Profiling
2. Finding the best optimization variables with setting the three variables `threshold`, `sla_factor` to approperiate values.
