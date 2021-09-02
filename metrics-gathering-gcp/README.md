## Example
An example for fetching metrics is given below
```python
from smart_kube.metrics.gcp_monitoring_query import MonitoringQuery
# init by specifying each project
mq = MonitoringQuery(project_id='sandbox-ba851a37')
# define which metrics to fetch
metrics=[
    "container:cores",
    "container:cpu_usage",
    "container:mem_limit",
    "container:mem_limit_usage",
    "container:mem_request",
    "container:mem_request_usage",
    "container:mem_used_bytes",
    "container:uptime",
]
# define the labels for resources
resource_labels = ["resource_type", "pod_name","container_name"]
# fetch
resource = mq.get_metrics(metrics,resource_labels=resource_labels)
```
Output is as follows
```python
resource
                                                                               container:cores  container:cpu_usage  ...  container:mem_used_bytes  container:uptime
name                                               date                                                              ...                                            
k8s_container_my-rec-deployment-5578cbdb64-4crh... 2021-06-10 10:15:43.507162              0.0                  NaN  ...                       NaN               NaN
                                                   2021-06-10 10:16:43.507162              0.0                  NaN  ...                       NaN               NaN
                                                   2021-06-10 10:17:43.507162              0.0                  NaN  ...                       NaN               NaN
k8s_container_my-rec-deployment-5578cbdb64-82jr... 2021-06-10 10:15:43.507162              0.0                  NaN  ...                       NaN               NaN
                                                   2021-06-10 10:16:43.507162              0.0                  NaN  ...                       NaN               NaN
...                                                                                        ...                  ...  ...                       ...               ...
k8s_container_nginx-6799fc88d8-v4l6d_nginx         2021-06-11 08:00:43.507162              NaN                  NaN  ...                       NaN      78117.550399
                                                   2021-06-11 08:01:43.507162              NaN                  NaN  ...                       NaN      78177.564884
                                                   2021-06-11 08:02:43.507162              NaN                  NaN  ...                       NaN      78237.587522
                                                   2021-06-11 08:03:43.507162              NaN                  NaN  ...                       NaN      78297.571336
                                                   2021-06-11 08:04:43.507162              NaN                  NaN  ...                       NaN      78357.561885

[10531 rows x 8 columns]
```

## GCP Metrics
The following are metrics which are available for querying from Google Cloud Operations via its python client which are collection from [this source](https://cloud.google.com/monitoring/api/metrics_kubernetes) (please refer to the aforementioned for more detail).

- "cores": "kubernetes.io/container/cpu/request_cores": Number of CPU cores requested by the container. Sampled every 60 seconds.
- "cpu_usage": "kubernetes.io/container/cpu/request_utilization": The fraction of the requested CPU that is currently in use on the instance. This value can be greater than 1 as usage can exceed the request. Sampled every 60 seconds. After sampling, data is not visible for up to 240 seconds.
- "mem_limit": "kubernetes.io/container/memory/limit_bytes": Local ephemeral storage limit in bytes. Sampled every 60 seconds. 
- "mem_limit_usage": "kubernetes.io/container/memory/limit_utilization": The fraction of the memory limit that is currently in use on the instance. This value cannot exceed 1 as usage cannot exceed the limit. Sampled every 60 seconds. After sampling, data is not visible for up to 120 seconds. memory_type: Either `evictable` or `non-evictable`. Evictable memory is memory that can be easily reclaimed by the kernel, while non-evictable memory cannot.
- "mem_request": "kubernetes.io/container/memory/request_bytes": Memory request of the container in bytes. Sampled every 60 seconds.
- "mem_request_usage": "kubernetes.io/container/memory/request_utilization": The fraction of the requested memory that is currently in use on the instance. This value can be greater than 1 as usage can exceed the request. Sampled every 60 seconds. After sampling, data is not visible for up to 120 seconds. memory_type: Either `evictable` or `non-evictable`. Evictable memory is memory that can be easily reclaimed by the kernel, while non-evictable memory cannot
- "mem_used_bytes": "kubernetes.io/container/memory/used_bytes": Memory usage in bytes. Sampled every 60 seconds.memory_type: Either `evictable` or `non-evictable`. Evictable memory is memory that can be easily reclaimed by the kernel, while non-evictable memory cannot.
- "restart": "kubernetes.io/container/restart_count": Number of times the container has restarted. Sampled every 60 seconds.
- "uptime": "kubernetes.io/container/uptime": Time in seconds that the container has been running. Sampled every 60 seconds.


- "alloc_util": "kubernetes.io/node/cpu/allocatable_utilization": The fraction of the allocatable CPU that is currently in use on the instance. Sampled every 60 seconds. After sampling, data is not visible for up to 240 seconds.
- "core_usagetime": "kubernetes.io/node/cpu/core_usage_time": Cumulative CPU usage on all cores used on the node in seconds. Sampled every 60 seconds. 
- "total_cores": "kubernetes.io/node/cpu/total_cores": Total number of CPU cores on the node. Sampled every 60 seconds.
- "mem_allocatable": "kubernetes.io/node/memory/allocatable_bytes": Cumulative memory bytes used by the node. Sampled every 60 seconds. 
- "mem_alloc_util": "kubernetes.io/node/memory/allocatable_utilization": The fraction of the allocatable memory that is currently in use on the instance. This value cannot exceed 1 as usage cannot exceed allocatable memory bytes. Sampled every 60 seconds. After sampling, data is not visible for up to 120 seconds. memory_type: Either `evictable` or `non-evictable`. Evictable memory is memory that can be easily reclaimed by the kernel, while non-evictable memory cannot. component: Name of the respective system daemon.
- "mem_total_bytes": "kubernetes.io/node/memory/total_bytes": Number of bytes of memory allocatable on the node. Sampled every 60 seconds. 
- "mem_used_bytes": "kubernetes.io/node/memory/used_bytes": Cumulative memory bytes used by the node. Sampled every 60 seconds.
memory_type: Either `evictable` or `non-evictable`. Evictable memory is memory that can be easily reclaimed by the kernel, while non-evictable memory cannot.
- "network_received": "kubernetes.io/node/network/received_bytes_count": Cumulative number of bytes received by the node over the network. Sampled every 60 seconds. 
- "network_sent": "kubernetes.io/node/network/sent_bytes_count": Cumulative number of bytes transmitted by the node over the network. Sampled every 60 seconds.
