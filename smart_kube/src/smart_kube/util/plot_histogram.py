import numpy as np
import matplotlib.pyplot as plt
from .histogram import Histogram


def plot_histogram(
    # memory_histogram: Histogram,
    histogram: Histogram,
    # last_bucket_memory: int,,
    title: str,
    x_label: str,
        ):

    """
    plot histgraom from the Histogram class
    """
    # plot cpu
    fig, axs = plt.subplots(1, 1)
    axs.bar(
        histogram.bin_boundaries,
        histogram.bucket_weight,
        edgecolor='black',
        align='edge', width=5)

    axs.title.set_text(title)
    axs.set_xlabel(x_label)
    axs.set_ylabel('number of containers')
    axs.grid(True)

    # plot memory
    # axs[1].bar(memory_histogram.bin_boundaries[0: last_bucket_memory],
    #            memory_histogram.bucket_weight[0: last_bucket_memory],
    #            edgecolor='black', width=width_memory)

    # axs[1].title.set_text('Memory')
    # axs[1].set_xlabel('bins')
    # axs[1].set_ylabel('bucket_weight')
    # axs[1].grid(True)

    fig.tight_layout()


    # TODO complete for cpu and memrory later
    # # plot cpu
    # fig, axs = plt.subplots(2, 1)
    # axs[0].bar(cpu_histogram.bin_boundaries[0: last_bucket_cpu],
    #            cpu_histogram.bucket_weight[0: last_bucket_cpu],
    #            edgecolor='black', width=width_cpu)

    # axs[0].title.set_text('CPU')
    # axs[0].set_xlabel('bins')
    # axs[0].set_ylabel('bucket_weight')
    # axs[0].grid(True)

    # # plot memory
    # axs[1].bar(memory_histogram.bin_boundaries[0: last_bucket_memory],
    #            memory_histogram.bucket_weight[0: last_bucket_memory],
    #            edgecolor='black', width=width_memory)

    # axs[1].title.set_text('Memory')
    # axs[1].set_xlabel('bins')
    # axs[1].set_ylabel('bucket_weight')
    # axs[1].grid(True)

    # fig.tight_layout()

    return fig
