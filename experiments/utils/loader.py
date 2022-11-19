import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import time
import json
import yaml
from json import JSONDecodeError

class Loader:
    def __init__(self, series_path,
                 config_key_mapper, second_node=False, type_of='node') -> None:
        self.series_path = series_path
        self.config_path = os.path.join(series_path, config_key_mapper)
        self.second_node = second_node
        self.type_of = type_of
        legal_types = ['node', 'pipeline', 'node_with_log']
        if type_of not in legal_types:
            raise ValueError(f'Invalid type: {type_of}')

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

    def _get_experiments_with_logs(self):
        key_config_mapper = self.key_config_mapper()
        experiments_with_logs = key_config_mapper[
            key_config_mapper['no_engine']==False]['experiment_id'].tolist()
        return experiments_with_logs

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

    def read_results(self, selected=None):
        files = self.get_result_file_names()
        results = {}
        for file in files:
            # try:
            name = file.split(".")[0].split("/")[-1]
            if selected is not None:
                if int(name) in selected:
                    full_path = os.path.join(
                        self.series_path, file
                    )
                    json_file = open(full_path)
                    try:
                        results[name] = json.load(json_file)
                    except JSONDecodeError:
                        pass
                        # print('excepted-1!')
            else:
                full_path = os.path.join(
                    self.series_path, file
                )
                json_file = open(full_path)
                try:
                    results[name] = json.load(json_file)
                except JSONDecodeError:
                    pass
                    # print('excepted-1!')                
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

    def _node_latency_calculator(self, results: Dict[Dict, Any]):
        client_to_model_latencies = []
        model_latencies = []
        model_to_client_latencies = []
        latencies = {
            'client_to_model_latencies': [],
            'model_latencies': [],
            'model_to_client_latencies': [] 
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
                client_to_model_latency =\
                    model_arrival_time - sending_time
                model_latency =\
                    model_serving_time - model_arrival_time
                model_to_client_latency =\
                    arrival_time - model_serving_time
                client_to_model_latencies.append(client_to_model_latency)
                model_latencies.append(model_latency)
                model_to_client_latencies.append(model_to_client_latency)
                latencies = {
                    'client_to_model_latencies': client_to_model_latencies,
                    'model_latencies': model_latencies,
                    'model_to_client_latencies': model_to_client_latencies
                }
            except KeyError:
                timeout_count += 1
        return latencies, timeout_count

    def _node_latency_calculator_with_log(
        self, results: Dict[Dict, Any], log: Dict[Dict, Any]):
        client_to_svc_latencies =  []
        svc_latencies = []
        svc_to_model_latencies = []
        model_latencies =  []
        model_to_client_latencies = []
        latencies = {
            'client_to_svc_latencies': [],
            'svc_latencies': [],
            'svc_to_model_latencies': [],
            'model_latencies': [],
            'model_to_client_latencies': [] 
        }
        timeout_count = 0
        to_svc_logs = log['to_svc_logs']
        to_model_logs = log['to_model_logs']
        for result, to_svc_log, to_model_log in zip(
            results, to_svc_logs, to_model_logs):
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
                client_to_svc_latency = to_svc_log - sending_time
                svc_latency = to_model_log - to_svc_log
                svc_to_model_latency = model_arrival_time - model_serving_time
                model_latency =\
                    model_serving_time - model_arrival_time
                model_to_client_latency =\
                    arrival_time - model_serving_time
                client_to_svc_latencies.append(client_to_svc_latency)
                svc_latencies.append(svc_latency)
                svc_to_model_latencies.append(svc_to_model_latency)
                model_latencies.append(model_latency)
                model_to_client_latencies.append(model_to_client_latency)
                latencies = {
                    'client_to_svc_latencies': client_to_svc_latencies,
                    'svc_latencies': svc_latencies, 
                    'svc_to_model_latencies': svc_to_model_latencies,
                    'model_latencies': model_latencies,
                    'model_to_client_latencies': model_to_client_latencies 
                }
            except KeyError:
                timeout_count += 1
        return latencies, timeout_count

    def _pipeline_latency_calculator(
        self, results: Dict[Dict, Any]):
        sample_time_entry = json.loads(
            results[0]['outputs'][0]['data'][0])['time']
        model_name_raws = list(
            sample_time_entry.keys())
        model_time_types = list(
            map(lambda l: type(l), list(
                json.loads(results[0][
                    'outputs'][0]['data'][0])[
                        'time'].values())))
        time_variables = []
        for model_name_raw, model_time_type in zip(
            model_name_raws, model_time_types):
            if model_time_type == list:
                time_variables.append(
                    list(sample_time_entry[
                        model_name_raw][0].keys()))
            else:
                time_variables.append(model_name_raw)
        flattened_time_variables = []
        for time_variable in time_variables:
            if type(time_variable) == str:
                flattened_time_variables.append(time_variable)
            elif type(time_variable) == list:
                for sub_time_var in time_variable:
                    flattened_time_variables.append(sub_time_var)
        
        raw_latencies = {
            time_var: [] for time_var in flattened_time_variables}
        raw_latencies.update({
            'sending_time': [],
            'arrival_time': []
        })
        # for time_variable in time_variables:
        timeout_count = 0
        for result in results:
            try:
                # outer times
                outter_times = result[
                    'timing'] if 'timing' in result.keys() else result['time']
                raw_latencies['sending_time'].append(
                    outter_times["sending_time"])
                raw_latencies['arrival_time'].append(
                    outter_times["arrival_time"])
                times = json.loads(
                    result['outputs'][0]['data'][0])['time']
                for time_var, value in times.items():
                    if type(value) == list:
                        reversed = {key: [] for key in list(value[0].keys())}
                        for item in value:
                            for entry, val in item.items():
                                reversed[entry].append(val)
                        reversed_to_add = {
                            k: max(v) for k, v in reversed.items()}
                        for latency_var, value in reversed_to_add.items():
                            raw_latencies[latency_var].append(value)
                    else:
                        raw_latencies[time_var].append(value)
            except KeyError:
                timeout_count += 1
        latencies = raw_latencies
        nodes_order = self._find_node_orders()
        latencies = {
            'client_to_model': []
        }
        for node_index in range(len(nodes_order)):
            latencies.update({
                f'{nodes_order[node_index]}': [],
            })
            if node_index + 1 != len(nodes_order):
                latencies.update({
                    f'{nodes_order[node_index]}_to_{nodes_order[node_index+1]}': [],
                })
            else: break
        latencies.update({
            f'server_to_client': []
        })
        for key, value in latencies.items():
            if 'client_to_model' == key:
                length = min(
                    len(raw_latencies['sending_time']),
                    len(raw_latencies[
                        f"arrival_{nodes_order[0]}"]))
                # TODO might be buggy
                latencies[key] = np.array(
                    raw_latencies[
                        f"arrival_{nodes_order[0]}"][:length]) - np.array(
                    raw_latencies['sending_time'][:length])
            elif 'server_to_client' == key:
                latencies[key] = np.array(
                    raw_latencies[
                        'arrival_time'][:length]) - np.array(
                    raw_latencies[
                        f"serving_{nodes_order[len(nodes_order)-1]}"][:length])
                break
            elif key in nodes_order:
                latencies[key] = np.array(
                    raw_latencies[
                        f"serving_{key}"] - np.array(
                            raw_latencies[f"arrival_{key}"]))
            elif '_to_' in key and 'client' not in key:
                nodes = key.split('_to_')
                latencies[key] = np.array(
                    raw_latencies[f'arrival_{nodes[1]}']) - np.array(
                        raw_latencies[f"serving_{nodes[0]}"]
                    )
        latencies = {k: v.tolist() for k, v in latencies.items()}
        return latencies, timeout_count

    def latency_calculator(self, results: Dict[Dict, Any], log=None):
        """symmetric input meaning:
            per each input at the first node of the pipeline
            we only have one output going to the second node of the
            pipeline
        """
        if self.type_of == 'node':
            return self._node_latency_calculator(results)
        if self.type_of == 'node_with_log':
            return self._node_latency_calculator_with_log(results, log)
        elif self.type_of == 'pipeline':
            return self._pipeline_latency_calculator(results)

    def metric_summary(self, metric, values):
        summary = {}
        if values != [] and values != None:
            try:
                summary[f'{metric}_avg'] = np.average(values)
            except TypeError:
                pass
                # print('excepted-2!')
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
        log = None
        selected = None
        if self.type_of == 'node_with_log':
            selected = self._get_experiments_with_logs()
            log = self._read_logs()
        results = self.read_results(selected)
        final_dataframe = []
        for experiment_id, result in results.items():
            processed_exp = {'experiment_id': int(experiment_id)}
            flattened_results = self.flatten_results(
                results[str(experiment_id)]['responses'])
            if log is not None:
                latencies, timeout_count = self.latency_calculator(
                    flattened_results, log[experiment_id])
            else:
                latencies, timeout_count = self.latency_calculator(
                    flattened_results, log)                
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
            if self.type_of=='node' or self.type_of=='node_with_log':
                for metric, values in result.items():
                    if metric in skipped_metrics:
                        continue
                    processed_exp.update(self.metric_summary(
                        metric=metric, values=values))
                final_dataframe.append(processed_exp)
            elif self.type_of=='pipeline':
                nodes_order = self._find_node_orders()
                for model in nodes_order:
                    for pod_name, pod_values in result[model].items():
                        pod_index = 1
                        for metric, values in pod_values.items():
                            if metric in skipped_metrics:
                                continue
                            processed_exp.update(self.metric_summary(
                                metric=f'{model}_pod{pod_index}_{metric}',
                                values=values))
                        pod_index += 1
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

    def _read_logs(self):
        files = self.get_result_file_names()
        results = {}
        to_svc_logs = []
        to_model_logs = []
        for file in files:
            if 'txt' in file:
                name = file.split(".")[0].split("/")[-1]
                full_path = os.path.join(
                    self.series_path, file
                )
                with open(full_path) as f:
                    lines = [line for line in f]
                for line in lines:
                    line = json.loads(line)
                    if line['msg'] == "Predictions called":
                        to_svc_logs.append(line)
                    elif line['msg'] == "Calling HTTP":
                        to_model_logs.append(line)
                to_svc_logs_pd = pd.DataFrame(to_svc_logs)
                to_model_logs_pd = pd.DataFrame(to_model_logs)
                to_svc_logs_ts = to_svc_logs_pd['ts'].tolist()
                to_model_logs_ts = to_model_logs_pd['ts'].tolist()
                to_svc_logs_ts.sort()
                to_model_logs_ts.sort()
                results[name] = {
                    'to_svc_logs': to_svc_logs_ts,
                    'to_model_logs': to_model_logs_ts}
        return results

    def _find_node_orders(self):
        config = self.load_configs()
        sample_config_key = list(config.keys())[0]
        node_order = list(
            map(lambda l: l['node_name'],
            config[sample_config_key]['nodes']))
        return node_order
