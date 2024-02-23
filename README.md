# Abstract
This paper presents DynaInf, a novel deep learning inference serving system designed to guarantee dynamic Service Level Objectives (SLO) in a dynamic network environment. DynaInf uses in-place vertical scaling, dynamic batching, and request reordering to optimize resource utilization and user satisfaction in mobile and IoT applications. Moreover, we introduce an Integer Programming formulation to encapsulate the problem of resource allocation in a dynamically changing network bandwidths, providing a mathematical model of the relationship between latency, batch size, and CPU cores in inference serving systems. Preliminary evaluations indicate that DynaInf reduces latency SLO violation to less than 1% while minimizing CPU resource allocation, demonstrating its potential for effective inference serving in dynamically changing network bandwidths.

## 1 Project Setup Steps
1. Go to the [infrastructure](/infrastructure/README.md) for the guide to set up the K8S cluster and related depandancies, the complete installtion takes ~30 minutes.

2. Dyaninf uses config yaml files for running experiments, the config files used in the paper are stored in the `data/configs/final` folder. And then do the following:
```bash
cd experiments/runner
```
and run the experiments for the appropriate config file:

```bash
python runner_script.py --config-name <config-name>
```
