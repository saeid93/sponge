import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Union


font = {"size": 12}
plt.rc("font", size=12)
plt.rc("axes", titlesize=12)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
color_list = ["#ffff99", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]

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
                axs[figure_index].set_title(selected_experiments[metric]["title"])
                axs[figure_index].set_ylabel(
                    ylabel=selected_experiments[metric]["ylabel"]
                )
                axs[figure_index].legend()
            figure_index += 1

    plt.tight_layout()
    plt.show()


def draw_temporal_final2(
    dicts_to_draw: Dict[str, Dict[str, List[int]]],
    series_names: List[str],
    selected_experiments: Dict[str, Dict[str, Union[str, List[str]]]],
    filename: str,
    adaptation_interval=None,
    draw_load: bool = True,
    fig_size: int = 10,
    bbox_to_anchor=(0.8, 6.1),
    save=False,
):
    num_keys = sum(map(lambda l: len(l["selection"]), selected_experiments.values()))

    if draw_load:
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
        axs[0].set_ylabel(ylabel="Load (RPS)")
        axs[0].set_xticklabels([])
        axs[-1].set_xlabel("Time (s)")
        figure_index = 1
    else:
        _, axs = plt.subplots(nrows=num_keys, ncols=1, figsize=(fig_size, num_keys * 2))
        figure_index = 0

    for metric, metric_to_draw in dicts_to_draw.items():
        if metric not in selected_experiments.keys():
            continue
        sample_experiment = list(metric_to_draw.keys())[0]
        metrics = metric_to_draw[sample_experiment]

        for key in metrics.keys():
            axs[figure_index].grid(axis="y", linestyle="dashed", color="gray")
            if key not in selected_experiments[metric]["selection"]:
                continue
            color_idx = 0
            for experiment_id, dict_to_draw_exp in metric_to_draw.items():
                color = color_list[color_idx]
                x_values = range(len(list(dict_to_draw_exp.values())[0]))
                if adaptation_interval is not None and metric not in [
                    "measured_latencies"
                ]:
                    x_values = [
                        item * adaptation_interval[experiment_id]
                        for item in list(x_values)
                    ]
                axs[figure_index].plot(
                    x_values,
                    dict_to_draw_exp[key],
                    label=series_names[experiment_id],
                    color=color,
                )
                axs[figure_index].set_ylabel(
                    ylabel=selected_experiments[metric]["ylabel"]
                )
                color_idx += 1

            if figure_index < len(list(selected_experiments.keys())) - 1:
                axs[figure_index].set_xticklabels([])
            else:
                axs[figure_index].set_xlabel("Time (s)")
            figure_index += 1

    if draw_load:
        plt.legend(
            fontsize=13,
            fancybox=False,
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.8, 6.1),
            handlelength=1,
            columnspacing=0.8,
        )
    else:
        # pass
        plt.legend(
            fontsize=13,
            fancybox=False,
            ncol=3,
            frameon=False,
            bbox_to_anchor=bbox_to_anchor,
            handlelength=1,
            columnspacing=0.8,
        )
    if save:
        plt.savefig(filename)
    else:
        plt.show()


# def draw_temporal_final3(
#     final_by_load_type: Dict[str, Dict[str, List[int]]],
#     selected_experiments: Dict[str, Dict[str, Union[str, List[str]]]],
#     filename: str,
#     serie_color: dict,
#     adaptation_interval=None,
#     bbox_to_anchor=(0.8, 6.1),
#     save=False,
# ):
#     num_keys = sum(map(lambda l: len(l["selection"]), selected_experiments.values()))

#     fig = plt.figure(figsize=(10, 10), dpi=600)

#     subfigs = fig.subfigures(2, 2)

#     subfig_idx = 0
#     for load_type in final_by_load_type.keys():
#         subfig = subfigs.flat[subfig_idx]
#         subfig.suptitle(load_type, fontsize=13)
#         subfig_idx += 1
#         # subfig.suptitle(f'Subfig {outerind}')
#         axs = subfig.subplots(len(final_by_load_type[load_type].keys()), 1)
#         axs_idx = 0

#         for metric in final_by_load_type[load_type]:
#             ax = axs.flat[axs_idx]
#             ax.grid(axis="y", linestyle="dashed", color="gray")
#             if subfig_idx % 2 == 1:
#                 ax.set_ylabel(selected_experiments[metric]["ylabel"])
#             axs_idx += 1
#             if axs_idx < len(list(selected_experiments.keys())):
#                 ax.set_xticklabels([])
#             else:
#                 if subfig_idx > len(final_by_load_type.keys()) // 2:
#                     ax.set_xlabel("Time (s)")
#             for serie_name in final_by_load_type[load_type][metric]:
               
#                 for key, values in final_by_load_type[load_type][metric][
#                     serie_name
#                 ].items():
#                     if key in selected_experiments[metric]["selection"]:
#                         ax.plot(values, label=serie_name, color=serie_color[serie_name])

#     plt.legend(
#         fontsize=13,
#         fancybox=False,
#         ncol=3,
#         frameon=False,
#         bbox_to_anchor=bbox_to_anchor,
#         handlelength=1,
#         columnspacing=0.8,
#     )
#     if save:
#         plt.savefig(filename)
#     else:
#         plt.show()

def draw_temporal_final3(
    final_by_load_type: Dict[str, Dict[str, List[int]]],
    selected_experiments: Dict[str, Dict[str, Union[str, List[str]]]],
    filename: str,
    adaptation_interval=None,
    bbox_to_anchor=(0.8, 6.1),
    save=False,
):
    num_keys = sum(map(lambda l: len(l["selection"]), selected_experiments.values()))

    fig = plt.figure(figsize=(8, 11), dpi=600)
    fig.tight_layout()
    gs = fig.add_gridspec(ncols=2, nrows=2)
    subfig_x = 0
    subfig_y = 0
    num_works = 0
    for load_type in final_by_load_type.keys():
        sgs = gs[subfig_x, subfig_y].subgridspec(
            len(final_by_load_type[load_type].keys()), 1
        )

        axs_idx = 0

        for metric in final_by_load_type[load_type]:
            ax = fig.add_subplot(sgs[axs_idx])
            if axs_idx == 0:
                ax.set_title(load_type, fontsize=13)
            ax.grid(axis="y", linestyle="dashed", color="gray")
            if subfig_y == 0:
                ax.set_ylabel(selected_experiments[metric]["ylabel"])

            axs_idx += 1
            color_idx = 0
            if axs_idx < len(list(selected_experiments.keys())):
                ax.set_xticklabels([])
            else:
                if subfig_x == 1:
                    ax.set_xlabel("Time (s)")
            num_works = 0
            for serie_name in final_by_load_type[load_type][metric]:
                num_works += 1
                color = color_list[color_idx]
                color_idx += 1
                for key, values in final_by_load_type[load_type][metric][
                    serie_name
                ].items():
                    if key in selected_experiments[metric]["selection"]:
                        ax.plot(values, label=serie_name, color=color)
        subfig_y += 1
        if subfig_y == 2:
            subfig_y = 0
            subfig_x += 1

    print(f"{num_works = }")
    plt.legend(
        fontsize=13,
        fancybox=False,
        ncol=num_works,
        frameon=False,
        bbox_to_anchor=bbox_to_anchor,
        handlelength=1,
        columnspacing=0.8,
    )
    if save:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()


def draw_temporal_final4(
    final_by_load_type: Dict[str, Dict[str, List[int]]],
    selected_experiments: Dict[str, Dict[str, Union[str, List[str]]]],
    filename: str,
    serie_color: dict,
    hl_for_metric = None,
    adaptation_interval=None,
    bbox_to_anchor=(0.8, 6.1),
    save=False,
):
    num_keys = sum(map(lambda l: len(l["selection"]), selected_experiments.values()))

    metrics_len = len(final_by_load_type[list(final_by_load_type.keys())[0]].keys())
    fig = plt.figure(figsize=(8, 1 + metrics_len * 2.5), dpi=600)
    fig.tight_layout()
    gs = fig.add_gridspec(ncols=2, nrows=2)
    subfig_x = 0
    subfig_y = 0
    num_works = 0
    for load_type in final_by_load_type.keys():
        sgs = gs[subfig_x, subfig_y].subgridspec(
            len(final_by_load_type[load_type].keys()), 1
        )

        axs_idx = 0

        for metric in final_by_load_type[load_type]:
            ax = fig.add_subplot(sgs[axs_idx])
            if axs_idx == 0:
                ax.set_title(load_type, fontsize=13)
            ax.grid(axis="y", linestyle="dashed", color="gray")
            if subfig_y == 0:
                ax.set_ylabel(selected_experiments[metric]["ylabel"])

            axs_idx += 1
            if axs_idx < metrics_len:
                ax.set_xticklabels([])
            else:
                if subfig_x == 1:
                    ax.set_xlabel("Time (s)")
            
            if hl_for_metric.get(metric):
                ax.axhline(
                    y=hl_for_metric[metric]["value"],
                    # label=hl_for_metric[metric]["label"],
                    color=hl_for_metric[metric]["color"],
                    linestyle="dashed"
                )
            num_works = 0
            for serie_name in final_by_load_type[load_type][metric]:
                num_works += 1
                for key, values in final_by_load_type[load_type][metric][
                    serie_name
                ].items():
                    if key in selected_experiments[metric]["selection"]:
                        ax.plot(values, label=serie_name, color=serie_color[serie_name])
        subfig_y += 1
        if subfig_y == 2:
            subfig_y = 0
            subfig_x += 1

    print(f"{num_works = }")
    plt.legend(
        fontsize=13,
        fancybox=False,
        ncol=num_works,
        frameon=False,
        bbox_to_anchor=bbox_to_anchor,
        handlelength=1,
        columnspacing=0.8,
    )
    if save:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()


def draw_cumulative(
    dict_to_draw: Dict[str, Dict[str, List[int]]],
    ylabel="Value",
    xlabel="Stage",
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
            label = (
                str(experiment) if series_names is None else series_names[experiment]
            )
            axs.bar(x_positions, y_values, width=bar_width, label=label)

        axs.set_xlabel(xlabel=xlabel)
        axs.set_ylabel(ylabel=ylabel)
        axs.set_title("Comparison of Experiments")
        axs.set_xticks(bar_positions + bar_width * (num_experiments - 1) / 2)
        axs.set_xticklabels(model_names)
        axs.legend(title="Experiments")

    plt.tight_layout()
    plt.show()


def draw_cumulative_with_grouping(
    dict_to_draw: Dict[str, Dict[str, List[int]]],
    series_meta: Dict[str, Dict[str, int]],
    ylabel="Value",
    xlabel="Stage",
):
    dict_to_draw_cul = {}
    for series, series_dict in dict_to_draw.items():
        dict_to_draw_cul[series] = sum(series_dict)

    categories = list(series_meta.keys())
    group_names = list(next(iter(series_meta.values())).keys())

    values = {}
    for category in categories:
        values[category] = {}
        for group_name in group_names:
            values[category][group_name] = dict_to_draw_cul[series_meta[category][group_name]]

    bar_width = 0.2
    x = np.arange(len(group_names))

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the bars for each category and group
    for i, category in enumerate(categories):
        category_values = [values[category][group] for group in group_names]
        ax.bar(x + i * bar_width, category_values, bar_width, label=category)

    # Set the x-axis labels and title
    ax.set_xticks(x + (len(categories) - 1) * bar_width / 2)
    ax.set_xticklabels(group_names)

    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_title("Comparison of Experiments")
    # axs.set_xticks(bar_positions + bar_width * (num_experiments - 1) / 2)
    # axs.set_xticklabels(model_names)
    ax.legend(title="Experiments")

    plt.tight_layout()
    plt.show()


def draw_cumulative_final(
    results: Dict[str, Dict[int, float]],
    series_metadata: Dict[int, dict],
    metrics_metadata: Dict[str, dict],
    filename,
):
    fig, axs = plt.subplots(1, len(results), figsize=(8, 2))
    ax_idx = 0
    for metric in results.keys():
        ax = axs[ax_idx]
        ax.set_title(metrics_metadata[metric]["ylabel"])
        x = 0
        width = 1
        for serie, metric_result in results[metric].items():
            ax.bar(
                x,
                metric_result,
                color=series_metadata[serie]["color"],
                label=series_metadata[serie]["label"],
                width=width,
            )
            # ax.set_xticks(
            #     range(len(series_metadata.keys())),
            #     [series_metadata[serie]["label"] for serie in series_metadata.keys()]
            # )
            ax.set_xticks([])
            x += width

        ax_idx += 1

    plt.legend(
        fontsize=13,
        fancybox=False,
        ncol=len(series_metadata.keys()),
        frameon=False,
        bbox_to_anchor=(0.85, 1.5),
        handlelength=1,
        columnspacing=0.8,
    )
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)
    plt.savefig(
        f"{filename}.pdf",
        dpi=600,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0,
    )




def draw_cdf(data_dict: dict, x: float):
    import seaborn as sns
    
    rep = 3
    fig, axs = plt.subplots(1, rep, figsize=(8, 2.2))
    
    for idx in range(rep):
        color_idx = 0
        for label, data in data_dict.items():
            sns.kdeplot(
                data, label=label, cumulative=True, linestyle="dashed", color=color_list[color_idx], ax=axs[idx]
            )
            color_idx += 1
        if idx > 0:
            axs[idx].set_yticklabels([])
            axs[idx].set_ylabel(None)
        else:
            axs[idx].set_ylabel("CDF")
        
        axs[idx].set_xticks([0, x])
        axs[idx].set_xlim(0)
        axs[idx].vlines(x, ymin=0, ymax=1, colors="black", ls="--", label="x")
    plt.legend(
        fontsize=13,
        fancybox=False,
        ncol=len(data_dict.keys()) + 1,
        frameon=False,
        bbox_to_anchor=(0.2, 1.25),
        handlelength=1,
        columnspacing=0.8,
    )
    plt.savefig("cdf.pdf")
        

if __name__ == "__main__":
    data = np.random.random(100)
    data2 = np.random.random(100) * 0.9
    data3 = np.random.random(100) * 0.85
    data4 = np.random.random(100) * 0.75
    data5 = np.random.random(100) * 0.6

    data_dict = {"1": data, "2": data2, "3": data3, "4": data4, "5": data5}
    draw_cdf(data_dict, 0.9)
