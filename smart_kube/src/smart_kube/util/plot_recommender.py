import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter


def plot_recommender(
    timestamps: np.array,
    workload: np.array,
    recommendations_memory: np.array,
    recommendations_cpu: np.array,
    # request_cpu: int,
    # limit_cpu: int,
    # request_mem: int,
    # limit_mem: int
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

    # plot memory
    axs[0].plot(timestamps, memory)
    axs[0].plot(timestamps, recommendations_memory[0, :])
    axs[0].plot(timestamps, recommendations_memory[1, :])
    axs[0].plot(timestamps, recommendations_memory[2, :])
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
        recommendations_memory[0, :],
        recommendations_memory[2, :],
        alpha=0.2)
    axs[0].legend(["usage", "lower_bound", "target", "upper_bound"],
                  loc ="lower right")

    # plot cpu
    axs[1].plot(timestamps, cpu)
    axs[1].plot(timestamps, recommendations_cpu[0, :])
    axs[1].plot(timestamps, recommendations_cpu[1, :])
    axs[1].plot(timestamps, recommendations_cpu[2, :])
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
        recommendations_cpu[0, :],
        recommendations_cpu[2, :],
        alpha=0.2)
    axs[1].legend(["usage", "lower_bound", "target", "upper_bound"],
                  loc ="lower right")

    fig.tight_layout()

    return fig
