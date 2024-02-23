## GPU
 

We are aware of leveraging GPUs to speed up the intensive calculations required for tasks like Deep Learning. Using GPUs with Kubernetes allows you to extend the scalability of K8s to ML applications.

However, Kubernetes does not inherently have the ability to schedule GPU resources, so this approach requires the use of third-party device plugins. Additionally, there is no native way to determine utilization, per-device request statistics, or other metrics—this information is an important input to analyzing GPU efficiency and cost, which can be a significant expenditure.

This article will explore the use of GPUs in Kubernetes, outline the key metrics you should be tracking, and detail the process of setting up the tools required to schedule and monitor your GPU resources.

Requesting GPUsPermalink
Although the syntax for requests and limits is similar to that of CPUs, Kubernetes does not inherently have the ability to schedule GPU resources. To handle the nvidia.com/gpuresource, the nvidia-device-plugin DaemonSet must be running. It is possible this DaemonSet is installed by default—you can check by running kubectl get ds -A. On a node with a GPU, run kubectl describe node <gpu-node-name>and check if the nvidia.com/gpu resource is allocable.

Theoretically, seeing GPU utilization is as simple as using the nvidia-smi (System Management Interface) in any container with the proper runtime. To see this in action, first create a pod using the NVIDIA dcgmproftester to generate a test GPU load.


## Installation
This should be as easy with a helm install. However, if you go this route, not all the GPU metrics are enabled by default. One I was missing in particular was GPU utilization metric.

```
git clone https://github.com/NVIDIA/gpu-monitoring-tools.git
```
Now edit default-counters.csv and dcp-metrics-include.csv files under gpu-monitoring-tools/etc/dcgm-exporter folder, to add metrics we want.

Look for DCGM_FI_DEV_GPU_UTIL tag under Utilization section in the csv file. Uncomment this line. While you are at it, you can comment any other metric else you don’t need in this file. Some of the metrics are computationally intensive to collect, so we should decrease the load on the system by commenting the unwanted metrics.

Now under the folder gpu-monitoring-tools/deployment , execute the following command to install dcgm-exporter.

```
helm install — generate-name dcgm-exporter/
```

You can look at the pods in the default name space for dcgm-exporter. Note that I have one GPU nodes in my cluster.

```kubectl get po```

## Prometheus dashboard

Now let’s verify the GPU metrics exposed in Prometheus dashboard, by opening http://<machine-ip-address>:30090 . You can type DCGM_GPU_UTIL in the box (or any other metric you are interested in) and see the results.

There are a few metrics which are relevant to usage.

- DCGM_FI_DEV_GPU_UTIL is what we will be focusing on. It represents a simple GPU utilization percentile consistent with the above GPU-Util field in the SMI. However, there are more specific metrics available.

- DCGM_FI_PROF_GR_ENGINE_ACTIVE represents the average portion of time any graphics/compute engine was active.

- DCGM_FI_PROF_SM_OCCUPANCY is the fraction of warps on a multiprocessor relative to the total concurrent warps supported.


