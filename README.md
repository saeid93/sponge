# Structure
This repository contains the code of a smart (ML-based) Kuberntest vertical pod autoscaler. For background on the kubernetes vpa see the kubernetes [vpa design proposal](https://github.com/kubernetes/community/blob/master/contributors/design-proposals/autoscaling/vertical-pod-autoscaler.md) and this [blog post](https://povilasv.me/vertical-pod-autoscaling-the-definitive-guide/).


## General Project Structure
The project consists of three parts:
1. [smart_kube](smart_kube): The library for implementation of the simulator, emulator and backend Kubernetes wrappers and the vpa recommender, simulator, emulator and updater.
2. [experiments](experiments): The directory consisting all the client codes e.g. triggering learning algorithms in the vpa library, saving and loading models and scripts for checking simulator and a learned agent etc.
3. [stress-dockerfiles](stress-dockerfiles) stress testing docker images.
4. [crd](crd): crd version of the implementations TODO.

# Project Setup

## Prerequisite
1. Install vertical pod autoscaler with on of the following methods
   1. [Kubernetes open source autoscaler installation](https://cloud.google.com/kubernetes-engine/docs/how-to/vertical-pod-autoscaling)
   2. [Google cloud installation](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler)

2. Go to the [/smart_kube](./smart_kube) and install the framework in editable mode
   ```
   pip install -e .
   ```

4. For building the stress tests docker images and uploading them to your intended Docker regiestry go to the [stress-dockerfiles](stress-dockerfiles) and use the <code>./build.sh</code> in each of the images folders (change the address to you docker registery). 
5. Install prometheus+grafana monitoring from [prometues installation guide](docs/prometeus-monitoring.md).

6. Go to [/experiments/utils/constants.py](/experiments/utils/constants.py) and set the path to your data and project folders in the file. For example:
   ```
   DATA_PATH = "/Users/saeid/Codes/arabesque/smart-kube/data"
   ```


# Complete System

For easier modeling purposes we have used [OpenAI Gym](https://gym.openai.com/) interface (even for non-RL) methods for both the simulated and emulated environments.

States:

         cpu_usage  ram_usage  cpu_request  ram_request
        [          |          |            |            ]

Actions:

         cpu_lower_bound  cpu_target  cpu_higher_bound   cpu_uncapped_target
        [               |          |                |                   |  ...

         ram_lower_bound ram_target ram_higher_bound ram_uncapped_target
        |               |          |                |                  ]

At each timestep the simulation/Emulations
1. The environment gets an action from the vpa agent which could be any type of vpa algorithm
2. The action/recommendation from the vpa is in the format of the (lowerBound, target, upperBound) per resource
3. The vpa updater decides whether to change the <code>request</code> and <code>limits</code>:\
   3.1. checks if <code>lowerBound<resource-usage<upperBound</code>\
   3.2. If yes, then it updates the pod and restarts it with the new <code>request</code> and <code>limit</code>.\
   3.3. If no, nothing happens.
4. The environment moves to the next step with the new <code>request</code> and <code>limit</code>.

| ![our vpa model](/docs/figures/our-vpa-structure.svg) |
| :--: |
| *vpa model* |

## Environment ([smart_kube](smart_kube))
### Simulations
The simulator uses the OpenAI gym interface as explained above. however, the resource usages are represented in a numpy array rather than actully having a container running in the background.

### Emulations

The same code of the simulation side has been resued. The only differences are:
1. Load creation is done using stress testing tool explained above.
2. A recontainer is deployed instead a pod in a Kubernetes deployment object.
3. Resource usages are fetched from the kubernetes' metric server.

### Custrom resource & Operator Implementation

To make the mentioend method into a real-world VPA we have used Kubernetes [Custom Resource Definition (CRD)](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) and [Operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/).

TODO write it after implementation

## Implemented vertical pod autoscalers ([vpa](vpa))
All the algorithms are inherited following the [vpa interface](smart_kube/src/smart_kube/vpa) implemented in the smart_kube library

1. built-in predictor: Using weigthed histogram of recent resource usages algorithm explained [here](https://povilasv.me/vertical-pod-autoscaling-the-definitive-guide/) and [here](https://research.google/pubs/pub49065/). This is a Python implementation of the [recommender code](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler/pkg/recommender) in the [Kubernetes built-in vertical pod autoscaler](https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler). TODO

2. timeseries methods TODO

3. RL details TODO

4. LSTMs TODO

5. TODO 


## Difference between training and testing time

In trainig time we use historical gcloud and prometheus resource usage data, however during the tests we only use metric server and sometimes the prometheus for live monitoring of data usage.

# Stress Testing Functionalities
For implemeneting a workload generated by the worklaod generatinon scripts introduced in the previous section on the containers we have used unix [stress-ng tool](https://wiki.ubuntu.com/Kernel/Reference/stress-ng). There are two [dockerised app](stress-dockerfiles) designed implemented for that. A [utilisation server](stress-dockerfiles/stress-utilization-server) sends the current intended resource usage (ram in Megabytes and cpu in milicores) to a [stress pod](stress-dockerfiles/stress-pod) app. To build the docker images and upload it to your intended docker registry. For a test of them use [experiments/kubernetes_operations/stress_test.py](experiments/kubernetes_operations/stress_test.py). The utilisation server and the stress pod have the following lifecycle.

1. uploading the workload to utilisation server

![stress-step-1](/docs/figures/stress-step-1.svg)

2. registring the pod and its asociated service to the utilisation server

![stress-step-2](/docs/figures/stress-step-2.svg)

3. eastablish the connection and sends its initial resource usage

![stress-step-3](/docs/figures/stress-step-3.svg)

4. sends resource usage priodically within a specified time interval to the pod and the pod use stress to update its resource usage

![stress-step-4](/docs/figures/stress-step-4.svg)

# Code Guide

## [Examples of Kubernetes Python Interface](experiments/kubernetes_operations)

1. Implemented cluster level operations \
[experiments/kubernetes_operations/clusters_operations.py](experiments/kubernetes_operations/clusters_operations.py)

2. Pods operations \
[experiments/kubernetes_operations/pods_operations.py](experiments/kubernetes_operations/pods_operations.py)

3. Deployments operations \
[experiments/kubernetes_operations/deployments_operations.py](experiments/kubernetes_operations/deployments_operations.py)

4. Accessing the builtin vertical pod autoscaler \
[experiments/kubernetes_operations/builtin_vpa.py](experiments/kubernetes_operations/builtin_vpa.py)

5. Stress test the pods example, see [Stress Testing Functionalities](##stress-testing-functionalities) section for more details\
[experiments/kubernetes_operations/stress_test.py](experiments/kubernetes_operations/stress_test.py)

## [Generating the Workloads](experiments/workload/generate_workload.py)

Go to the your dataset generation config [data/configs/workloads](data/configs/workloads) and choose the apporoperiate config e.g. for step workloads you can change the variables in [data/configs/workloads/step.json](data/configs/workloads/step.json)
```
python generate_workload.py --help
Usage: generate_workload.py [OPTIONS]

Options:
  --workload-type TEXT
  --help                Show this message and exit.
```
Currently available workload generation methods:

1. **step**: Generates workloads with variable steps.
2. **sinusoidal**: Generates workloads based-on a sinusoidal wave.
3. **low_high**: A workload with only two values a high value and a low value.
4. **arabesque**: Generates workloads the arabasque resource suage dataset.
5. **azure**: Generates workloads based-on Azure dataset.
6. **alibaba**: Generates workloads based-on a sinusoidal wave.
7. **borg**: Generates workloads based-on a borg dataset.

The generated workload will be saved in <code>data/workloads/[workload-id]</code>.

## Links to the Video Demos

1. [**video demo of the kubernetes client**](https://drive.google.com/file/d/1JbbFb5BPxA6iKbxrKZ_KRSgHelasMkKx/view?usp=sharing)
2. [**video demo of the gym and the recommender simulation**](https://drive.google.com/file/d/1LbuJ0B6vbBHOc5OVmhdtEFljmtohu1vT/view?usp=sharing)
3. [**video demo of the load generation module**](https://drive.google.com/file/d/1iRhhxnc_9aQ63fuBMitNVuPUOQdclKWU/view?usp=sharing)

## Check scripts
TODO



