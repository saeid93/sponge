from prometheus_api_client import PrometheusConnect
from prometheus_api_client.utils import parse_datetime, parse_timedelta
from matplotlib import pyplot as plt
import pandas as pd
from plot import PlotBuilder
import requests
PROMETHEUS = "http://localhost:30090"
prom = PrometheusConnect(url ="http://localhost:30090", disable_ssl=True)
pod_name = "sklearn-prof-daqiq-default-0-classifier-f68b57bd7-zm9wh"
# Get the list of all the metrics that the Prometheus host scrapes
pc = prom
response =requests.get(PROMETHEUS + '/api/v1/query', params={'query': 'container_cpu_user_seconds_total'}) 

for data in response.json()['data']['result']:
    if 'pod' in data['metric'] and 'daqiq' in data['metric']['pod']:
        print(data['value'])


response = requests.get(PROMETHEUS + '/api/v1/query', params={'query':'container_memory_usage_bytes{pod="sklearn-prof-daqiq-default-0-classifier-f68b57bd7-zm9wh"}'})
print("###############")
for data in response.json()['data']['result']:
    print(data['value'])


# rate(seldon_api_executor_client_requests_seconds_count{pod=~"sklearn.*"}[1m])
print("######")
response_request_count = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : 'rate(seldon_api_executor_client_requests_seconds_count{pod=~"sklearn-prof-daqiq-default.*"}[2s])[40m:15s]'})

response_request_istion = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : 'rate(istio_response_bytes_bucket{pod=~"sklearn.*"}[1m])[10m:10s]'})

response_cpu_usage = requests.get(PROMETHEUS + '/api/v1/query', params={'query' : 'rate(container_cpu_usage_seconds_total{pod=~"sklearn-prof-daqiq-default.*"}[1m])[40m:15s]'})

# rate(istio_response_bytes_bucket{pod=~"sklearn.*"}[1m])

# rate(container_cpu_usage_seconds_total{pod=~"sklearn-prof-daqiq-default.*"}[1m])
# data = pd.read_csv("users.csv")
# pb = PlotBuilder(data)
# pb.user_rate('s')
# data = pb.user_rate_second_interval(15)

plot_values = []
for value in response_request_count.json()['data']['result'][0]['values']:
    print(value[1])
    if int(float(value[1]))>0:
        plot_values.append(int(float(value[1])))

fig, ax = plt.subplots(figsize=(10, 6))

axb = ax.twinx()
axb.set_ylabel('rate')
axb.plot([i for i in range(1,len(plot_values)+ 1)], plot_values, color='black', label='pressure')
plt.savefig('usertool.png')


# plot_values = []
# for value in response_cpu_usage.json()['data']['result'][0]['values']:
#     plot_values.append((float(value[1]))*100)

# fig, ax = plt.subplots(figsize=(10, 6))

# axb = ax.twinx()
# axb.set_ylabel('latency')
# print(plot_values)
# axb.plot([i for i in range(1,len(plot_values)+ 1)], plot_values, color='blue', label='pressure')
# plt.savefig('cpu.png')