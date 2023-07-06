import math, os, subprocess

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_losses(
    losses_dicts: dict[str, list[float]],
    num_epoches: int | None = None,
    single_plot: bool = False,
) -> None:
    """Plot multiple training/test/validation losses.

    Args:
        losses_dicts (dict[str, list[float]]): Dict where the keys are the \
            plot title, value is the losses.
        num_epoches (int | None, optional): Fix the number of epoches on the \
            plots' X-axis, else it'll plot however many epoches are given. \
            Defaults to None.
        single_plot (bool, optional): Whether to plot everything in 1 plot. \
            Defaults to False.
    """
    if single_plot:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.set()

        for title, losses in losses_dicts.items():
            if num_epoches:
                sns.lineplot(x=range(1, num_epoches+1), y=losses[:num_epoches], label=f'{title}', ax=ax)
            else:
                sns.lineplot(x=range(1, len(losses)+1), y=losses, label=f'{title}', ax=ax)

        ax.set_title("Combined Losses")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return


    num_graphs = len(losses_dicts)
    cols = 1 if num_graphs == 1 \
        else 2 if num_graphs <= 4 \
        else 3
    rows = math.ceil(num_graphs / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axs = axs.flatten()

    for i, (title, losses) in enumerate(losses_dicts.items()):
        ax = axs[i]
        sns.set()

        if num_epoches:
            sns.lineplot(x=range(1, num_epoches+1), y=losses[:num_epoches], label=f'{title}', ax=ax)
        else:
            sns.lineplot(x=range(1, len(losses)+1), y=losses, label=f'{title}', ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()

    # remove unused graphs
    for i in range(num_graphs, cols * rows):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def get_least_utilized_and_allocated_gpu() -> torch.device:
    """
    Returns the `torch.device` of the GPU with the lowest utilization and memory
    allocation.

    Returns:
        torch.device: The `torch.device` of the least utilized and memory allocated GPU.
    """
    result = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,utilization.gpu,memory.used',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')

    # GPU stats are returned in separate lines
    gpu_stats = result.strip().split('\n')
    assert len(gpu_stats) > 0, "No visible GPU."
    assert len(gpu_stats) > 1, "Only 1 GPU (expected to run on a machine with multiple GPUs)."

    parsed_stats: list[tuple[int, int, int]] = []
    for gpu_stat in gpu_stats:
        stats = gpu_stat.split(', ')
        gpu_id = int(stats[0])
        utilization = int(stats[1])
        memory = int(stats[2])
        parsed_stats.append((gpu_id, utilization, memory))

        # Printing GPU stats for debugging.
        print(f'GPU-{gpu_id}: Util = {utilization:>3.0f}%, MemAlloc = {(memory / 1024):>4.1f} GiB')

    # Sort GPUs by utilization first, then memory allocation.
    sorted_gpus = sorted(parsed_stats, key=lambda x: (x[1], x[2]))

    least_used_gpu_id = sorted_gpus[0][0]
    print(f'Using GPU-{least_used_gpu_id}...\n')
    return torch.device(f'cuda:{least_used_gpu_id}')


def _is_in_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython
        return bool(get_ipython())
    except ImportError:
        return False

def abs_path(relative_path: str) -> str:
    if _is_in_notebook():
        from IPython.core.getipython import get_ipython
        current_dir = get_ipython().starting_dir # type: ignore
    else:
        current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, relative_path)
