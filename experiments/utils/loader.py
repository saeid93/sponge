import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import time
import json
import yaml

class Loader:
    def __init__(self, series_path,
                 config_key_mapper, second_node=False) -> None:
        self.series_path = series_path
        self.config_path = os.path.join(series_path, config_key_mapper)
        self.second_node = second_node

    def load_configs(self) -> Dict[str, Dict[str, Any]]:
        config_files = {}
        for file in os.listdir(self.series_path):
            # check only text files
            if file.endswith('.yaml'):
                config_path = os.path.join(self.series_path, file)
                with open(config_path, 'r') as cf:
                    config = yaml.safe_load(cf)
                config_files[file] = config
        return config_files

    def key_config_mapper(self):
        key_config_mapper = pd.read_csv(self.config_path)
        return key_config_mapper

    def get_experiment_detail(self, experiment_id: int):
        key_config_mapper = self.key_config_mapper()
        return key_config_mapper[
            key_config_mapper["experiment_id"]==experiment_id]

    def get_result_file_names(self):
        files = []
        key_config_mapper = self.key_config_mapper()
        experiments_keys = list(key_config_mapper["experiment_id"])
        for path in os.listdir(self.series_path):
            if os.path.isfile(os.path.join(self.series_path, path)):
                file_name = path.split(".")[0]
                try:
                    if int(file_name) in experiments_keys:
                        files.append(path)
                except:
                    pass
        return files

    def read_results(self):
        files = self.get_result_file_names()
        results = {}
        for file in files:
            name = file.split(".")[0].split("/")[-1]
            full_path = os.path.join(
                self.series_path, file
            )
            json_file = open(full_path)
            results[name] = json.load(json_file)
        return results

    def flatten_results(self, per_second_latencies):
        """
        change the results format from
        [[second_1], [second_2], ...]
        to:
        [req_1, req_2, ...]
        """
        flattend_results = []
        for second_results in per_second_latencies:
            for request_result in second_results:
                flattend_results.append(request_result)
        return flattend_results

    def latency_calculator(self, results: Dict[Dict, Any]):
        client_to_server_latencies = []
        model_latencies = []
        model_to_server_latencies = []
        latencies = {
            'client_to_server_latencies': [],
            'model_latencies': [],
            'model_to_server_latencies': [] 
        }
        timeout_count = 0
        for result in results:
            try:
                # outer times
                outter_times = result[
                    'timing'] if 'timing' in result.keys() else result['time']
                sending_time = outter_times["sending_time"]
                arrival_time = outter_times["arrival_time"]
                # inner times
                data = result['outputs'][0]['data']
                data = json.loads(data[0])
                inner_times = data["time"]
                model_name = result['model_name']

                # TEMP to be fixed with a consistent time format
                if self.second_node:
                    inner_times = inner_times[model_name + '_times'][0]
                try:
                    arrival_key = "arrival_" + model_name
                    serve_key   = "serving_" + model_name
                    model_arrival_time = inner_times[arrival_key]
                    model_serving_time = inner_times[serve_key]
                except KeyError:
                    arrival_key = "arrival_" + model_name.replace('-', '_')
                    serve_key   = "serving_" + model_name.replace('-', '_')
                    model_arrival_time = inner_times[arrival_key]
                    model_serving_time = inner_times[serve_key]
                # all three latencies
                client_to_server_latency =\
                    model_arrival_time - sending_time
                model_latency =\
                    model_serving_time - model_arrival_time
                model_to_server_latency =\
                    arrival_time - model_serving_time
                client_to_server_latencies.append(client_to_server_latency)
                model_latencies.append(model_latency)
                model_to_server_latencies.append(model_to_server_latency)
                latencies = {
                    'client_to_server_latencies': client_to_server_latencies,
                    'model_latencies': model_latencies,
                    'model_to_server_latencies': model_to_server_latencies
                }
            except KeyError:
                timeout_count += 1
        return latencies, timeout_count

    def metric_summary(self, metric, values):
        summary = {}
        if values != [] and values != None:
            try:
                summary[f'{metric}_avg'] = np.average(values)
            except TypeError:
                print('excepted!')
            summary[f'{metric}_p99'] = np.percentile(values, 99)
            summary[f'{metric}_p50'] = np.percentile(values, 50)
            summary[f'{metric}_var'] = np.var(values)
            summary[f'{metric}_max'] = max(values)
            summary[f'{metric}_min'] = min(values)
        else:
            summary[f'{metric}_avg'] = None
            summary[f'{metric}_p99'] = None
            summary[f'{metric}_p50'] = None
            summary[f'{metric}_var'] = None
            summary[f'{metric}_max'] = None
            summary[f'{metric}_min'] = None
        return summary

    def latency_summary(self, latencies):
        summary = {}
        for metric_name, values in latencies.items():
            summary.update(
                self.metric_summary(
                    metric=metric_name, values=values))
        return summary

    def result_processing(self):
        results = self.read_results()
        final_dataframe = []
        for experiment_id, result in results.items():
            processed_exp = {'experiment_id': int(experiment_id)}
            latencies, timeout_count = self.latency_calculator(
                self.flatten_results(
                    results[str(experiment_id)]['responses']))
            latencies = self.latency_summary(latencies)
            processed_exp.update(latencies)
            processed_exp['start_time'] = time.ctime(
                result['start_time_experiment'])
            processed_exp['end_time'] = time.ctime(
                result['end_time_experiment'])
            processed_exp['duration'] = round(
                result['end_time_experiment'] - result[
                    'start_time_experiment'])
            processed_exp['timeout_count'] = timeout_count
            skipped_metrics = [
                'time_cpu_usage_count',
                'time_cpu_usage_rate',
                'time_cpu_throttled_count',
                'time_cpu_throttled_rate',
                'time_memory_usage',
                'time_throughput',
                'responses',
                'start_time_experiment',
                'end_time_experiment'   
            ]
            for metric, values in result.items():
                if metric in skipped_metrics:
                    continue
                processed_exp.update(self.metric_summary(
                    metric=metric, values=values))
            final_dataframe.append(processed_exp)
        return pd.DataFrame(final_dataframe)

    def table_maker(
        self,
        experiment_ids: List[int],
        metadata_columns: List[str], results_columns: List[str]):
        # extract full data
        results = self.result_processing()
        metadata = self.key_config_mapper()
        # retrieve rows
        selected_results = results[results[
            'experiment_id'].isin(experiment_ids)]
        selected_metadata = metadata[metadata[
            'experiment_id'].isin(experiment_ids)]
        merged_results = selected_metadata.merge(selected_results)
        columns = metadata_columns + results_columns
        output = merged_results[columns]
        return output
