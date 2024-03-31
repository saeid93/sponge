# Abstract
Mobile and IoT applications increasingly adopt deep learning inference to provide intelligence. Inference requests are typically sent to a cloud infrastructure over a wireless network that is highly variable, leading to the challenge of dynamic Service Level Objectives (SLOs) at the request level. 
This paper presents Sponge, a novel deep learning inference serving system that maximizes resource efficiency while guaranteeing dynamic SLOs. Sponge achieves its goal by applying in-place vertical scaling, dynamic batching, and request reordering. Specifically, we introduce an Integer Programming formulation to capture the resource allocation problem, providing a mathematical model of the relationship between latency, batch size, and resources. We demonstrate the potential of Sponge through a prototype implementation and preliminary experiments and discuss future works.

## 1 Project Setup Steps
1. Go to the [infrastructure](/infrastructure/README.md) for the guide to set up the K8S cluster and related depandancies, the complete installtion takes ~30 minutes.

2. Get the minikube ip using `minikube ip` and add it to your configs.
3. Dyaninf uses config yaml files for running experiments, the config files used in the paper are stored in the `data/configs/final` folder. And then do the following:
```bash
cd experiments/runner
```
and run the experiments for the appropriate config file:

```bash
python runner_script.py --config-name <config-name>
```

## Citation
Please use the following citation if you use this framework:

```
@TODO: add citation
```
