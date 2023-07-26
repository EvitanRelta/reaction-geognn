import math, os, shutil, subprocess, time
from typing import Callable, TypeAlias

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml


def abs_path(relative_path: str, caller_file_path: str) -> str:
    current_dir = os.path.dirname(os.path.realpath(caller_file_path))
    return os.path.join(current_dir, relative_path)

LIGHTNING_LOGS_DIR = abs_path("lightning_logs", __file__)

HPARAMS_DICT: TypeAlias = dict
METRIC_DF: TypeAlias = pd.DataFrame
def load_version_log(version_num: int) -> tuple[HPARAMS_DICT, METRIC_DF]:
    """Loads the `hparams.yaml` and `metrics.csv` files from a version dir in
    the lightning-logs.

    Args:
        version_num (int): Version number.

    Returns:
        tuple[HPARAM, METRIC_DF]: Tuple containing the verison's hparam `dict` \
            and metrics `DataFrame`.
    """
    version_dir = f'version_{version_num}'
    version_path = os.path.join(LIGHTNING_LOGS_DIR, version_dir)
    assert os.path.isdir(version_path), f'"{version_path}" dir not found.'

    hparams_path = os.path.join(version_path, "hparams.yaml")
    metrics_path = os.path.join(version_path, "metrics.csv")
    assert os.path.isfile(hparams_path), f'"{hparams_path}" file not found.'
    assert os.path.isfile(metrics_path), f'"{metrics_path}" file not found.'

    with open(hparams_path, "r") as hparams_file:
        hparams = yaml.load(hparams_file, Loader=yaml.Loader)

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
    save_to_dir = os.path.join(LIGHTNING_LOGS_DIR, f'version_{version_nums[-1]}')

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
        old_version_dir = os.path.join(LIGHTNING_LOGS_DIR, f'version_{version_num}')
        if delete_old_versions:
            shutil.rmtree(old_version_dir)
        else:
            os.rename(old_version_dir, old_version_dir + "_old")

Color = tuple[int, int, int] | str
def plot_losses(
    version_nums: list[int],
    plot_title: str = "Losses",
    x_y_labels: tuple[str, str] = ("Epochs", "Loss"),
    titles: list[str] | str | Callable[[int, dict], str] = lambda v, hparams: f"(v{v}) {hparams['_logged_hparams']['notes']}",
    epoch_range: tuple[int, int] = (0, 9999),
    metric_col_names: list[str] = ["train_loss", "val_loss"],
    metric_map: list[Callable[[list[float]], list[float]] | None] | None = None,
    linestyles: list[str] = ["solid", "dashed", "dotted", "dashdot"],
    colors: list[Color] = sns.color_palette("deep"), # type: ignore
    figsize: tuple[int, int] = (10, 5),
    x_lim: tuple[float | None, float | None] = (None, None),
    y_lim: tuple[float | None, float | None] = (None, None),
    points: list[tuple[float, float]] = [],
    point_annotations: list[str] = [],
    point_colors: list[Color] = sns.color_palette("deep"), # type: ignore
) -> None:
    """Plot multiple PyTorch-Lightning version-logs on the same graph, with the
    option to plot

    Args:
        version_nums (list[int]): PyTorch-Lightning version-log numbers.
        plot_title (str, optional): Title of the plot. Defaults to "Losses".
        x_y_labels (tuple[str, str], optional): Labels for X and Y-axes respectively. \
            Defaults to ("Epochs", "Loss"),
        titles (list[str] | str | Callable[[int, dict], str], optional): Title(s) \
            for each version defined in `version_nums`. The string-interpolation-values \
            available are: `version_num` and the values in the `hparams.yaml` file \
            (eg. `"v{version_num}, loss={train_loss}"`). Or you can give a \
            `Callable[[int, dict], str]` where the 1st arg is `version_num`, 2nd arg \
            is the `hparam` dict. Defaults to `lambda v, hparams: f"(v{v}) {hparams['_logged_hparams']['notes']}"`.
        epoch_range (tuple[int, int], optional): Min/max epoch to plot. Defaults to (0, 9999).
        metric_col_names (list[str], optional): The column names in the `metrics.csv` \
            to plot. Defaults to ["train_loss", "val_loss"].
        metric_map (list[Callable[[list[float]], list[float]] | None] | None, optional): \
            Function to map over the values of each metric defined in `metric_col_names`. \
            Defaults to None.
        linestyles (list[str], optional): Linestyles for each metric line defined in \
            `metric_col_names`. Defaults to ["solid", "dashed", "dotted", "dashdot"].
        colors (list[tuple[float, float, float]], optional): Color palette to use
            for each line. Defaults to sns.color_palette("deep").
        figsize (tuple[int, int], optional): Plot size (ie. the `figsize` arg \
            passed to `plt.subplots`). Defaults to (10, 5).
        x_lim (tuple[float | None, float | None], optional): Limits of the plot's X-axis. \
            Defaults to (None, None).
        y_lim (tuple[float | None, float | None], optional): Limits of the plot's Y-axis. \
            Defaults to (None, None).
        points (list[tuple[float, float]]): X-Y coords of points to plot. Defaults to [].
        point_annotations (list[str]): Annotations to write at each point defined \
            in `points`. Defaults to [].
        point_colors (list[Color]): Color palette to use for each point. \
            Defaults to sns.color_palette("deep").
    """
    fig, ax = plt.subplots(figsize=figsize)
    for i, version_num in enumerate(version_nums):
        color = colors[i]
        hparams, metrics_df = load_version_log(version_num)

        # Handle getting of title.
        if isinstance(titles, str):
            title = titles
            title = title.format(version_num=version_num, **hparams)
        elif isinstance(titles, list):
            title = titles[i]
            title = title.format(version_num=version_num, **hparams)
        else:
            title = titles(version_num, hparams)

        # Plot lines.
        for j, metric_name in enumerate(metric_col_names):
            loss = metrics_df[metric_name].dropna().tolist()[epoch_range[0]:epoch_range[1]]
            if metric_map != None and metric_map[j] != None:
                loss = metric_map[j](loss) # type: ignore
            if j == 0:
                sns.lineplot(x=range(len(loss)), y=loss, ax=ax, color=color, linestyle=linestyles[j], label=title)
            else:
                sns.lineplot(x=range(len(loss)), y=loss, ax=ax, color=color, linestyle=linestyles[j])

    # Draw points.
    if len(points) > 0:
        X, Y = zip(*points)
        plt.scatter(x=X, y=Y, c=point_colors[:len(points)], zorder=2)

    # Draw points' annotations.
    for i, (x, y) in enumerate(points):
        if len(point_annotations) > i:
            ax.annotate(point_annotations[i], (x, y), xytext=(10,10), textcoords="offset pixels")

    ax.set_title(plot_title)
    ax.set_xlabel(x_y_labels[0])
    ax.set_ylabel(x_y_labels[1])
    ax.legend()
    if y_lim:
        plt.ylim(y_lim)
    if x_lim:
        plt.xlim(x_lim)
    plt.tight_layout()
    plt.show()

def get_least_utilized_and_allocated_gpu(monitor_duration: float = 3) -> torch.device:
    """
    Returns the `torch.device` of the GPU with the lowest weighed badness value
    calculated from utilization % and allocated memory %.

    Args:
        monitor_duration (float, optional): Seconds to monitor GPUs for \
            (polling every 0.5s), after which the highest utilization values \
            while monitoring are used. Defaults to 3.

    Returns:
        torch.device: The `torch.device` of the least utilized and memory allocated GPU.
    """
    POLLING_INTERVAL = 0.5

    gpu_stats = _get_gpu_stats()
    num_polls = int(monitor_duration / POLLING_INTERVAL)
    if num_polls > 1:
        print('Monitoring GPUs...')
    for _ in range(num_polls - 1): # -1 cuz we've already polled once above.
        time.sleep(POLLING_INTERVAL)
        for gpu_id, (utilization, memory_ratio) in enumerate(_get_gpu_stats()):
            gpu_stats[gpu_id] = (
                max(gpu_stats[gpu_id][0], utilization),
                max(gpu_stats[gpu_id][1], memory_ratio),
            )
    assert len(gpu_stats) > 0, "No visible GPU."
    assert len(gpu_stats) > 1, "Only 1 GPU (expected to run on a machine with multiple GPUs)."

    badnesses: list[tuple[int, float]] = []
    for gpu_id, (utilization, memory_ratio) in enumerate(gpu_stats):
        # `f` is mostly linear, then exponentially increases when close to 1.
        # This is to rank almost full/fully-utilized GPUs worse.
        # Visualization: https://www.desmos.com/calculator/bunpkkrg57
        f = lambda x : 10 ** (5 * (x - 1)) + x

        badness_value = 0.6 * f(utilization) + 0.4 * f(memory_ratio) # weigh utilization more, cuz my models don't use much memory.
        badnesses.append((gpu_id, badness_value))

        # Printing GPU stats for debugging.
        print(f'GPU-{gpu_id}: Util = {utilization * 100:>3.0f} %, MemAlloc = {memory_ratio * 100:>3.0f} %, Badness = {badness_value:>4.2f}')

    # Sort GPUs by badness value
    sorted_gpus = sorted(badnesses, key=lambda x: x[1])

    least_used_gpu_id = sorted_gpus[0][0]
    print(f'Using GPU-{least_used_gpu_id}...\n')
    return torch.device(f'cuda:{least_used_gpu_id}')

def _get_gpu_stats() -> list[tuple[float, float]]:
    """Get each GPU's current utilization and memory usage percentages.

    Returns:
        list[tuple[float, float]]: `(utilization, memory_usage)` for each GPU. \
            The values are in the range [0, 1], `1` being 100% usage.
    """
    result = subprocess.check_output([
        'nvidia-smi',
        '--query-gpu=index,utilization.gpu,memory.used,memory.total',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')

    # GPU stats are returned in separate lines
    gpu_stats = result.strip().split('\n')
    parsed_stats: list[tuple[float, float]] = []
    for gpu_stat in gpu_stats:
        stats = gpu_stat.split(', ')
        utilization = int(stats[1]) / 100
        memory_used = int(stats[2])
        memory_total = int(stats[3])
        memory_ratio = memory_used / memory_total
        parsed_stats.append((utilization, memory_ratio))
    return parsed_stats
