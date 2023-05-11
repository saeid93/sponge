import matplotlib.pyplot as plt
from typing import Dict, List

def draw_temporal(dict_to_draw: Dict[str, List[int]], adaptation_interval=None):
    num_keys = len(dict_to_draw.keys())
    fig, axs = plt.subplots(nrows=num_keys, ncols=1, figsize=(10, num_keys * 2))
    x_values = range(len(list(dict_to_draw.values())[0]))
    if adaptation_interval is not None:
        x_values = [item * adaptation_interval for item in list(x_values)]
    for i, key in enumerate(dict_to_draw.keys()):
        axs[i].plot(x_values, dict_to_draw[key], label=key)
        axs[i].set_title(key)
        axs[i].legend() 
    # if num_keys > 1:
    #     for i, key in enumerate(dict_to_draw.keys()):
    #         axs[i].plot(x_values, dict_to_draw[key], label=key)
    #         axs[i].set_title(key)
    #         axs[i].legend()
    # else:
    #     key = list(dict_to_draw.keys())[0]
    #     axs.plot(x_values, dict_to_draw[key], label=key)
    #     axs.set_title(key)
    #     axs.legend()

    plt.tight_layout()
    plt.show()

def draw_cumulative(dict_to_draw: Dict[str, List[int]]):
    num_keys = len(dict_to_draw.keys())
    dict_to_draw_cul = {key: sum(value) for key, value in dict_to_draw.items()}
    fig, axs = plt.subplots(figsize=(6, 4))
    x_values = list(dict_to_draw_cul.keys())
    y_values = list(dict_to_draw_cul.values())

    axs.bar(x_values, y_values)
    axs.set_xlabel('Models')

    axs.set_xticklabels(x_values)

    plt.tight_layout()
    plt.show()