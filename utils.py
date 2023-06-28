import math, subprocess
from typing import TypedDict

import matplotlib.pyplot as plt
import seaborn as sns
import torch


class LossPlotData(TypedDict):
    train_loss: list[float] | None
    test_loss: list[float] | None
    val_loss: list[float] | None

def plot_losses(losses_dicts: dict[str, LossPlotData]) -> None:
    """Plot multiple training/test/validation losses.

    Args:
        losses_dicts (dict[str, LossPlotData]): Dict where the keys are the plot title, value is the losses.
    """
    num_graphs = len(losses_dicts)
    cols = 1 if num_graphs == 1 \
        else 2 if num_graphs <= 4 \
        else 3
    rows = math.ceil(num_graphs / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axs = axs.flatten()

    for i, (title, losses_dict) in enumerate(losses_dicts.items()):
        train_loss, test_loss, val_loss = \
            losses_dict['train_loss'], losses_dict["test_loss"], losses_dict["val_loss"]

        ax = axs[i]
        sns.set()
        if train_loss:
            sns.lineplot(x=range(1, len(train_loss)+1), y=train_loss, label="train", ax=ax)
        if test_loss:
            sns.lineplot(x=range(1, len(test_loss)+1), y=test_loss, label="test", ax=ax)
        if val_loss:
            sns.lineplot(x=range(1, len(val_loss)+1), y=val_loss, label="val", ax=ax)
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
        print(f'GPU-{gpu_id}: Util = {utilization:>3.0f}%, MemAlloc = {(memory / 1024):>4.1f}')

    # Sort GPUs by utilization first, then memory allocation.
    sorted_gpus = sorted(parsed_stats, key=lambda x: (x[1], x[2]))

    least_used_gpu_id = sorted_gpus[0][0]
    print(f'Using GPU-{least_used_gpu_id}...\n')
    return torch.device(f'cuda:{least_used_gpu_id}')
