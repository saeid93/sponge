import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter


def plot_workload(
        timestamps: np.array,
        workload: np.array,
        request_cpu: int,
        limit_cpu: int,
        request_memory: int,
        limit_memory: int):  # plot_smoothing):
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

    # plot memory
    axs[0].plot(timestamps, memory)
    axs[0].axhline(request_memory, color='g', linestyle='-')
    axs[0].axhline(limit_memory, color='r', linestyle='-')
    # axs[0].plot(timesteps, savgol_filter(memory, plot_smoothing, 3),
    # color='red')
    axs[0].title.set_text('Memory')
    axs[0].set_xlabel('timesteps in second')
    axs[0].set_ylabel('usage in Megabyte')
    axs[0].grid(True)
    axs[0].legend(["workload", "request", "limit"],
                  loc ="lower right")

    # plot cpu
    axs[1].plot(timestamps, cpu)
    axs[1].axhline(request_cpu, color='g', linestyle='-')
    axs[1].axhline(limit_cpu, color='r', linestyle='-')
    # axs[1].plot(timesteps, savgol_filter(cpu, plot_smoothing, 3),
    # color='red')
    axs[1].title.set_text('CPU')
    axs[1].set_xlabel('timesteps in second')
    axs[1].set_ylabel('usage in Milicores')
    axs[1].grid(True)

    axs[1].legend(["workload", "request", "limit"],
                  loc ="lower right")

    fig.tight_layout()

    return fig
