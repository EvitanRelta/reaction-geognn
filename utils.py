import math, os, shutil, subprocess
from typing import TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml


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

LIGHTNING_LOG_DIR = abs_path("lightning_logs")

HPARAM: TypeAlias = dict
METRIC_DF: TypeAlias = pd.DataFrame
def load_version_log(version_num: int) -> tuple[HPARAM, METRIC_DF]:
    """Loads the `hparams.yaml` and `metrics.csv` files from a version dir in
    the lightning-logs.

    Args:
        version_num (int): Version number.

    Returns:
        tuple[HPARAM, METRIC_DF]: Tuple containing the verison's hparam `dict` \
            and metrics `DataFrame`.
    """
    version_dir = f'version_{version_num}'
    version_path = os.path.join(LIGHTNING_LOG_DIR, version_dir)
    assert os.path.isdir(version_path), f'"{version_path}" dir not found.'

    hparams_path = os.path.join(version_path, "hparams.yaml")
    metrics_path = os.path.join(version_path, "metrics.csv")
    assert os.path.isfile(hparams_path), f'"{hparams_path}" file not found.'
    assert os.path.isfile(metrics_path), f'"{metrics_path}" file not found.'

    with open(hparams_path, "r") as hparams_file:
        hparams = yaml.safe_load(hparams_file)

    metrics_df = pd.read_csv(metrics_path)
    return hparams, metrics_df


def concat_version(
    version_nums: list[int],
    delete_old_versions: bool = False,
    suppress_hparam_check: bool = False,
) -> None:
    """Concatenate multiple version directories, where the 1st provided version
    is the 1st PyTorch Lightning training version, the 2nd is the training
    resumed from the 1st version, 3rd is resumed from the 2nd, and so on.

    The concatenated files are then saved in the last version's directory,
    and the other versions are deleted.

    Args:
        version_nums (list[int]): List of versions to concatenate, starting from \
            the first training version, followed by the versions resumed from the \
            first, etc.
        delete_old_versions (bool, optional): Whether to delete the old versions' \
            directories, else suffix the dirs/metric-file with `_old`. \
            Defaults to False.
        suppress_hparam_check (bool, optional): Whether to suppress checking for \
            matching `hparams.yaml` values. Defaults to False.
    """
    assert len(version_nums) > 0, "no version numbers given."
    save_to_dir = os.path.join(LIGHTNING_LOG_DIR, f'version_{version_nums[-1]}')

    hparams, all_metrics_df = load_version_log(version_nums[0])
    for i, (curr_hparams, metrics_df) in enumerate(map(load_version_log, version_nums[1:])):
        if not suppress_hparam_check:
            assert len(curr_hparams) == len(hparams), "hparams doesn't match."
            for key in hparams.keys():
                assert curr_hparams[key] == hparams[key], "hparams doesn't match."

        assert len(metrics_df.keys()) == len(all_metrics_df.keys()), 'metrics have different columns.'
        assert metrics_df["epoch"].min() == all_metrics_df["epoch"].max() + 1, "epoch doesn't match"

        new_all_metrics_df = pd.concat([all_metrics_df, metrics_df], axis=0, ignore_index=True)
        assert len(new_all_metrics_df) == len(all_metrics_df) + len(metrics_df), 'error concatenating metrics dataframes.'
        all_metrics_df = new_all_metrics_df

    # Saving and deleting/renaming part
    metrics_path = os.path.join(save_to_dir, "metrics.csv")
    if delete_old_versions:
        all_metrics_df.to_csv(metrics_path, index=False)
    else:
        os.rename(metrics_path, os.path.join(save_to_dir, "metrics_old.csv"))
        all_metrics_df.to_csv(metrics_path, index=False)

    for version_num in version_nums[:-1]:
        old_version_dir = os.path.join(LIGHTNING_LOG_DIR, f'version_{version_num}')
        if delete_old_versions:
            shutil.rmtree(old_version_dir)
        else:
            os.rename(old_version_dir, old_version_dir + "_old")


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
    Returns the `torch.device` of the GPU with the lowest weighed badness value
    calculated from utilization % and allocated memory %.

    Returns:
        torch.device: The `torch.device` of the least utilized and memory allocated GPU.
    """
    result = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,utilization.gpu,memory.used,memory.total',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')

    # GPU stats are returned in separate lines
    gpu_stats = result.strip().split('\n')
    assert len(gpu_stats) > 0, "No visible GPU."
    assert len(gpu_stats) > 1, "Only 1 GPU (expected to run on a machine with multiple GPUs)."

    parsed_stats: list[tuple[int, float]] = []
    for gpu_stat in gpu_stats:
        stats = gpu_stat.split(', ')
        gpu_id = int(stats[0])
        utilization = int(stats[1]) / 100
        memory_used = int(stats[2])
        memory_total = int(stats[3])
        memory_ratio = memory_used / memory_total

        badness_value = 0.5 * utilization + 0.5 * memory_ratio
        parsed_stats.append((gpu_id, badness_value))

        # Printing GPU stats for debugging.
        print(f'GPU-{gpu_id}: Util = {utilization * 100:>3.0f}%, MemAlloc = {memory_ratio * 100:>4.1f}%, Badness = {badness_value:>4.2f}')

    # Sort GPUs by badness value
    sorted_gpus = sorted(parsed_stats, key=lambda x: x[1])

    least_used_gpu_id = sorted_gpus[0][0]
    print(f'Using GPU-{least_used_gpu_id}...\n')
    return torch.device(f'cuda:{least_used_gpu_id}')
