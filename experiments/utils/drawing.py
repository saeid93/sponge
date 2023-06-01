import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Union


def draw_temporal(
    dict_to_draw: Dict[str, Dict[str, List[int]]],
    adaptation_interval=None,
    ylabel="Value",
    multiple_experiments=False,
):
    if not multiple_experiments:
        num_keys = len(dict_to_draw.keys())
        x_values = range(len(list(dict_to_draw.values())[0]))
        if num_keys > 1:
            fig, axs = plt.subplots(nrows=num_keys, ncols=1, figsize=(10, num_keys * 2))
            if adaptation_interval is not None:
                x_values = [item * adaptation_interval for item in list(x_values)]
            for i, key in enumerate(dict_to_draw.keys()):
                axs[i].plot(x_values, dict_to_draw[key], label=key)
                axs[i].set_title(key)
                axs[i].set_ylabel(ylabel=ylabel)
                axs[i].legend()
        else:
            fig, axs = plt.subplots(figsize=(10, num_keys * 2))
            key = list(dict_to_draw.keys())[0]
            axs.plot(x_values, dict_to_draw[key], label=key)
            axs.set_title(key)
            axs.set_ylabel(ylabel=ylabel)
            axs.legend()

    else:
        # extract number of keys
        sample_dict_item_key = list(dict_to_draw.keys())[0]
        sample_dict_item = dict_to_draw[sample_dict_item_key]
        keys_to_draw = sample_dict_item.keys()
        num_keys = len(keys_to_draw)

        # draw this set of keys
        if num_keys > 1:
            fig, axs = plt.subplots(nrows=num_keys, ncols=1, figsize=(10, num_keys * 2))
            for i, key in enumerate(sample_dict_item.keys()):
                for experiment_id, dict_to_draw_exp in dict_to_draw.items():
                    x_values = range(len(list(dict_to_draw_exp.values())[0]))
                    if adaptation_interval is not None:
                        x_values = [
                            item * adaptation_interval[experiment_id]
                            for item in list(x_values)
                        ]
                    axs[i].plot(x_values, dict_to_draw_exp[key], label=experiment_id)
                    axs[i].set_title(key)
                    axs[i].set_ylabel(ylabel=ylabel)
                    axs[i].legend()
        else:
            fig, axs = plt.subplots(figsize=(10, num_keys * 2))
            for i, key in enumerate(sample_dict_item.keys()):
                for experiment_id, dict_to_draw_exp in dict_to_draw.items():
                    x_values = range(len(list(dict_to_draw_exp.values())[0]))
                    if adaptation_interval is not None:
                        x_values = [
                            item * adaptation_interval[experiment_id]
                            for item in list(x_values)
                        ]
                    axs.plot(x_values, dict_to_draw_exp[key], label=experiment_id)
                    axs.set_title(key)
                    axs.set_ylabel(ylabel=ylabel)
                    axs.legend()

    plt.tight_layout()
    plt.show()


def draw_temporal_final(
    dicts_to_draw: Dict[str, Dict[str, List[int]]],
    series_names: List[str],
    selected_experiments: Dict[str, Dict[str, Union[str, List[str]]]],
    adaptation_interval=None,
    fig_size: int = 10,
):
    num_keys = sum(map(lambda l: len(l["selection"]), selected_experiments.values()))
    _, axs = plt.subplots(
        nrows=num_keys + 1, ncols=1, figsize=(fig_size, num_keys * 2 + 1)
    )

    axs[0].plot(
        dicts_to_draw["load"]["recieved_load_x"],
        dicts_to_draw["load"]["recieved_load"],
        label="recieved_load",
    )
    axs[0].plot(
        dicts_to_draw["load"]["sent_load_x"],
        dicts_to_draw["load"]["sent_load"],
        label="sent_load",
    )
    axs[0].plot(
        dicts_to_draw["load"]["predicted_load_x"],
        dicts_to_draw["load"]["predicted_load"],
        label="predicted_load",
    )
    axs[0].set_ylabel(ylabel="load")

    figure_index = 1
    for metric, metric_to_draw in dicts_to_draw.items():
        if metric not in selected_experiments.keys():
            continue
        sample_experiment = list(metric_to_draw.keys())[0]
        metrics = metric_to_draw[sample_experiment]
        for key in metrics.keys():
            if key not in selected_experiments[metric]["selection"]:
                continue
            for (
                experiment_id,
                dict_to_draw_exp,
            ) in metric_to_draw.items():  # draw different experiments
                x_values = range(len(list(dict_to_draw_exp.values())[0]))
                if adaptation_interval is not None and metric not in [
                    "measured_latencies"
                ]:
                    x_values = [
                        item * adaptation_interval[experiment_id]
                        for item in list(x_values)
                    ]
                axs[figure_index].plot(
                    x_values, dict_to_draw_exp[key], label=series_names[experiment_id]
                )
                axs[figure_index].set_title(selected_experiments[metric]['title'])
                axs[figure_index].set_ylabel(ylabel=selected_experiments[metric]['ylabel'])
                axs[figure_index].legend()
            figure_index += 1

    plt.tight_layout()
    plt.show()


def draw_cumulative(
    dict_to_draw: Dict[str, Dict[str, List[int]]],
    ylabel="Value",
    xlabel = "Stage",
    multiple_experiments=False,
    series_names=None,
):
    if not multiple_experiments:
        dict_to_draw_cul = {key: sum(value) for key, value in dict_to_draw.items()}
        fig, axs = plt.subplots(figsize=(4, 3))
        x_values = list(dict_to_draw_cul.keys())
        y_values = list(dict_to_draw_cul.values())

        axs.bar(x_values, y_values)
        axs.set_xlabel("Stage")
        axs.set_ylabel(ylabel=ylabel)
        axs.set_xticklabels(x_values)
    else:
        dict_to_draw_cul = {}
        for series, series_dict in dict_to_draw.items():
            dict_to_draw_cul[series] = {
                key: sum(list(filter(lambda x: x is not None, value)))
                for key, value in series_dict.items()
            }
        fig, axs = plt.subplots(figsize=(4, 3))
        experiments = list(dict_to_draw_cul.keys())
        model_names = list(dict_to_draw_cul[experiments[0]].keys())
        num_experiments = len(experiments)

        bar_width = 1 / (num_experiments + 1)
        bar_positions = np.arange(len(model_names))

        for i, experiment in enumerate(experiments):
            y_values = list(dict_to_draw_cul[experiment].values())
            x_positions = bar_positions + i * bar_width
            label = str(experiment) if series_names is None else series_names[experiment]
            axs.bar(x_positions, y_values, width=bar_width, label=label)

        axs.set_xlabel(xlabel=xlabel)
        axs.set_ylabel(ylabel=ylabel)
        axs.set_title("Comparison of Experiments")
        axs.set_xticks(bar_positions + bar_width * (num_experiments - 1) / 2)
        axs.set_xticklabels(model_names)
        axs.legend(title="Experiments")

    plt.tight_layout()
    plt.show()
