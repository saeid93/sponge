# Structure

This repository contains the code for our upcoming ML inference pipeline autoconfiguring projects.

**Project 1 Summary** [Joint Optimization of Qualitative and Performance Metrics on ML inference DAGs](https://www.overleaf.com/read/pfnwptxyktff):

Dataflow and computational pipelines have a longstanding history in the field of computing. ML inference is becoming one of the important factors in ML production systems life cycle. Many scenarios in production machine learning inference systems consist of more than one step for multiple steps of a machine learning inference scenario e.g. prepossessing, feature engineering, inference, and post-processing. ML pipelines may also represent a DAG like structure e.g several inference on multiple trained models could be aggregated into an ensemble stage or data sources of input data might be from multiple sources. A good ML inference pipelining system must be able to find the optimal resource and configuration assignment to the nodes of each node of the dataflow pipeline in a way that met both performance metrics like tail latency SLO and qualitative metrics like end to end accuracy of the inference pipeline subject to the users limited budget. Also incoming workloads from the users in production are variable and the initial configuration/resource assignment should be able to adapt to the fluctuating workloads in the cloud. In this work, we have proposed a solution for optimizing both qualitative and performance metrics jointly.

For setting infrastructure refer to accompanying repo [infrastructure](https://github.com/reconfigurable-ml-pipeline/infrastructure)

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
