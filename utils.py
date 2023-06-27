import math
from typing import TypedDict

import matplotlib.pyplot as plt
import seaborn as sns


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
