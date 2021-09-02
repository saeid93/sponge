import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter


def plot_slack(
    timestamps: np.array,
    workload: np.array,
    request_cpu: float,
    request_memory: float,
        ):

    """
    plot the workloads per each resource
    resources:
        1. memory
        2. cpu
    return:
        fig object
        ready to be plotted
    """

    memory = workload[0, :]
    cpu = workload[1, :]

    # timesteps = np.arange(0, timesteps)
    fig, axs = plt.subplots(2, 1)

    requests_memory = np.ones(memory.shape) * request_memory
    requests_cpu = np.ones(cpu.shape) * request_cpu

    # plot memory
    axs[0].plot(timestamps, memory)
    axs[0].plot(timestamps, requests_memory)
    # axs[0].axhline(request_mem, color='g', linestyle='-')
    # axs[0].axhline(limit_mem, color='r', linestyle='-')
    # axs[0].plot(timesteps, savgol_filter(memory, plot_smoothing, 3),
    # color='red')
    axs[0].title.set_text('Memory')
    axs[0].set_xlabel('timesteps in second')
    axs[0].set_ylabel('usage in Megabyte')
    axs[0].grid(True)
    axs[0].fill_between(
        timestamps,
        memory,
        requests_memory,
        color='red',
        alpha=0.2)
    axs[0].legend(["usage", "request"],
                  loc ="lower right")

    # plot cpu
    axs[1].plot(timestamps, cpu)
    axs[1].plot(timestamps, requests_cpu)
    # axs[1].axhline(request_cpu, color='g', linestyle='-')
    # axs[1].axhline(limit_cpu, color='r', linestyle='-')
    # axs[1].plot(timesteps, savgol_filter(cpu, plot_smoothing, 3),
    # color='red')
    axs[1].title.set_text('CPU')
    axs[1].set_xlabel('timesteps in second')
    axs[1].set_ylabel('usage in Milicores')
    axs[1].grid(True)
    axs[1].fill_between(
        timestamps,
        cpu,
        requests_cpu,
        color='red',
        alpha=0.2)
    axs[1].legend(["usage", "request"],
                  loc ="lower right")

    fig.tight_layout()

    return fig
