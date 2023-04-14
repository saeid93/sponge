import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import time
import json
import yaml
from json import JSONDecodeError


class Parser:
    def __init__(
        self,
        series_path,
        config_key_mapper,
        model_name,
        second_node=False,
        type_of="node",
    ) -> None:
        self.series_path = series_path
        self.config_path = os.path.join(series_path, config_key_mapper)
        self.model_name = model_name
        self.second_node = second_node
        self.type_of = type_of
        legal_types = ["node", "pipeline", "node_with_log"]
        if type_of not in legal_types:
            raise ValueError(f"Invalid type: {type_of}")
        if type_of == "pipeline":
            self.node_orders = list(
                map(
                    lambda l: l["node_name"],
                    list(self.load_configs().values())[0]["nodes"],
                )
            )

    def load_configs(self) -> Dict[str, Dict[str, Any]]:
        config_files = {}
        for file in os.listdir(self.series_path):
            # check only text files
            if file.endswith(".yaml"):
                config_path = os.path.join(self.series_path, file)
                with open(config_path, "r") as cf:
                    config = yaml.safe_load(cf)
                config_files[file] = config
        return config_files

    def key_config_mapper(self):
        key_config_mapper = pd.read_csv(self.config_path)
        return key_config_mapper

    def get_experiment_detail(self, experiment_id: int):
        key_config_mapper = self.key_config_mapper()
        return key_config_mapper[key_config_mapper["experiment_id"] == experiment_id]

    def _get_experiments_with_logs(self):
        key_config_mapper = self.key_config_mapper()
        experiments_with_logs = key_config_mapper[
            key_config_mapper["no_engine"] == False
        ]["experiment_id"].tolist()
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
                    full_path = os.path.join(self.series_path, file)
                    json_file = open(full_path)
                    try:
                        results[name] = json.load(json_file)
                    except JSONDecodeError:
                        pass
                        # print('excepted-1!')
            else:
                full_path = os.path.join(self.series_path, file)
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
            "client_to_model_latencies": [],
            "model_latencies": [],
            "model_to_client_latencies": [],
        }
        timeout_count = 0
        for result in results:
            try:
                times = result["times"]
                request_times = times["request"]
                model_times = times["models"][self.model_name]
                client_to_model_latency = (
                    model_times["arrival"] - request_times["sending"]
                )
                model_latency = model_times["serving"] - model_times["arrival"]
                model_to_client_latency = (
                    request_times["arrival"] - model_times["serving"]
                )
                client_to_model_latencies.append(client_to_model_latency)
                model_latencies.append(model_latency)
                model_to_client_latencies.append(model_to_client_latency)
                latencies = {
                    "client_to_model_latencies": client_to_model_latencies,
                    "model_latencies": model_latencies,
                    "model_to_client_latencies": model_to_client_latencies,
                }
            except KeyError:
                timeout_count += 1
        return latencies, timeout_count

    def _pipeline_latency_calculator(self, results: Dict[Dict, Any]):
        latencies = {
            "client_to_pipeline_latencies": [],
            "pipeline_to_client_latencies": [],
        }
        # client_to_pipeline_latencies = []
        # model_to_pipeline_latencies = []
        for index, model in enumerate(self.node_orders):
            latencies[f"task_{index}_model_latencies"] = []
            if index < len(self.node_orders) - 1:
                latencies[f"task_{index}_to_task_{index+1}_latencies"] = []
        timeout_count = 0
        for result in results:
            try:
                times = result["times"]
                request_times = times["request"]
                # model_times = times['models'][self.model_name]
                model_times = times["models"]
                for index, model in enumerate(self.node_orders):
                    if index == 0:
                        latencies["client_to_pipeline_latencies"].append(
                            model_times[model]["arrival"] - request_times["sending"]
                        )
                    if index == len(self.node_orders) - 1:
                        latencies["pipeline_to_client_latencies"].append(
                            request_times["arrival"] - model_times[model]["serving"]
                        )
                    latencies[f"task_{index}_model_latencies"].append(
                        model_times[model]["serving"] - model_times[model]["arrival"]
                    )
                    if index < len(self.node_orders) - 1:
                        latencies[f"task_{index}_to_task_{index+1}_latencies"].append(
                            model_times[self.node_orders[index + 1]]["arrival"]
                            - model_times[model]["serving"]
                        )
            except KeyError:
                timeout_count += 1
        return latencies, timeout_count

    def latency_calculator(self, results: Dict[Dict, Any], log=None):
        """symmetric input meaning:
        per each input at the first node of the pipeline
        we only have one output going to the second node of the
        pipeline
        """
        if self.type_of == "node":
            return self._node_latency_calculator(results)
        # if self.type_of == 'node_with_log':
        #     return self._node_latency_calculator_with_log(results, log)
        elif self.type_of == "pipeline":
            return self._pipeline_latency_calculator(results)

    def metric_summary(self, metric, values):
        summary = {}
        if values != [] and values != None:
            try:
                summary[f"{metric}_avg"] = np.average(values)
            except TypeError:
                pass
                # print('excepted-2!')
            summary[f"{metric}_p99"] = np.percentile(values, 99)
            summary[f"{metric}_p95"] = np.percentile(values, 95)
            summary[f"{metric}_p50"] = np.percentile(values, 50)
            summary[f"{metric}_var"] = np.var(values)
            summary[f"{metric}_max"] = max(values)
            summary[f"{metric}_min"] = min(values)
        else:
            summary[f"{metric}_avg"] = None
            summary[f"{metric}_p99"] = None
            summary[f"{metric}_p95"] = None
            summary[f"{metric}_p50"] = None
            summary[f"{metric}_var"] = None
            summary[f"{metric}_max"] = None
            summary[f"{metric}_min"] = None
        return summary

    def latency_summary(self, latencies):
        summary = {}
        for metric_name, values in latencies.items():
            summary.update(self.metric_summary(metric=metric_name, values=values))
        return summary

    def result_processing(self):
        log = None
        selected = None
        # if self.type_of == 'node_with_log':
        #     selected = self._get_experiments_with_logs()
        #     log = self._read_logs()
        results = self.read_results(selected)
        final_dataframe = []
        for experiment_id, result in results.items():
            processed_exp = {"experiment_id": int(experiment_id)}
            flattened_results = self.flatten_results(
                results[str(experiment_id)]["responses"]
            )
            if log is not None:
                latencies, timeout_count = self.latency_calculator(
                    flattened_results, log[experiment_id]
                )
            else:
                latencies, timeout_count = self.latency_calculator(
                    flattened_results, log
                )
            latencies = self.latency_summary(latencies)
            processed_exp.update(latencies)
            processed_exp["start_time"] = time.ctime(result["start_time_experiment"])
            processed_exp["end_time"] = time.ctime(result["end_time_experiment"])
            processed_exp["duration"] = round(
                result["end_time_experiment"] - result["start_time_experiment"]
            )
            processed_exp["timeout_count"] = timeout_count
            skipped_metrics = [
                "time_cpu_usage_count",
                "time_cpu_usage_rate",
                "time_cpu_throttled_count",
                "time_cpu_throttled_rate",
                "time_memory_usage",
                "time_throughput",
                "responses",
                "start_time_experiment",
                "end_time_experiment",
            ]
            if self.type_of == "node" or self.type_of == "node_with_log":
                for metric, values in result.items():
                    if metric in skipped_metrics:
                        continue
                    processed_exp.update(
                        self.metric_summary(metric=metric, values=values)
                    )
                final_dataframe.append(processed_exp)
            elif self.type_of == "pipeline":
                # nodes_order = self._find_node_orders()
                # for model in self.node_orders:
                #     for pod_name, pod_values in result[model].items():
                #         pod_index = 1
                #         for metric, values in pod_values.items():
                #             if metric in skipped_metrics:
                #                 continue
                #             processed_exp.update(self.metric_summary(
                #                 metric=f'{model}_pod{pod_index}_{metric}',
                #                 values=values))
                #         pod_index += 1
                # final_dataframe.append(processed_exp)
                for index, model in enumerate(self.node_orders):
                    for pod_name, pod_values in result[model].items():
                        pod_index = 1
                        for metric, values in pod_values.items():
                            if metric in skipped_metrics:
                                continue
                            processed_exp.update(
                                self.metric_summary(
                                    metric=f"task_{index}_{metric}", values=values
                                )
                            )
                        pod_index += 1
                final_dataframe.append(processed_exp)
        return pd.DataFrame(final_dataframe)

    def table_maker(
        self,
        experiment_ids: List[int],
        metadata_columns: List[str],
        results_columns: List[str],
    ):
        # extract full data
        results = self.result_processing()
        metadata = self.key_config_mapper()
        # retrieve rows
        selected_results = results[results["experiment_id"].isin(experiment_ids)]
        selected_metadata = metadata[metadata["experiment_id"].isin(experiment_ids)]
        merged_results = selected_metadata.merge(selected_results)
        columns = metadata_columns + results_columns
        output = merged_results[columns]
        return output

    # def _read_logs(self):
    #     files = self.get_result_file_names()
    #     results = {}
    #     to_svc_logs = []
    #     to_model_logs = []
    #     for file in files:
    #         if 'txt' in file:
    #             name = file.split(".")[0].split("/")[-1]
    #             full_path = os.path.join(
    #                 self.series_path, file
    #             )
    #             with open(full_path) as f:
    #                 lines = [line for line in f]
    #             for line in lines:
    #                 line = json.loads(line)
    #                 if line['msg'] == "Predictions called":
    #                     to_svc_logs.append(line)
    #                 elif line['msg'] == "Calling HTTP":
    #                     to_model_logs.append(line)
    #             to_svc_logs_pd = pd.DataFrame(to_svc_logs)
    #             to_model_logs_pd = pd.DataFrame(to_model_logs)
    #             to_svc_logs_ts = to_svc_logs_pd['ts'].tolist()
    #             to_model_logs_ts = to_model_logs_pd['ts'].tolist()
    #             to_svc_logs_ts.sort()
    #             to_model_logs_ts.sort()
    #             results[name] = {
    #                 'to_svc_logs': to_svc_logs_ts,
    #                 'to_model_logs': to_model_logs_ts}
    #     return results

    # def _find_node_orders(self):
    #     config = self.load_configs()
    #     sample_config_key = list(config.keys())[0]
    #     node_order = list(
    #         map(lambda l: l['node_name'],
    #         config[sample_config_key]['nodes']))
    #     return node_order
