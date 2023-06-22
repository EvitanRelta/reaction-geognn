import os
from typing import Literal, overload

from .Wb97SplitDataset import Wb97SplitDataset


@overload
def get_wb97_fold_dataset(
    fold_num: Literal[0, 1, 2, 3, 4],
    include_pretrain: Literal[True],
) -> tuple[
    tuple[Wb97SplitDataset, Wb97SplitDataset, Wb97SplitDataset],
    tuple[Wb97SplitDataset, Wb97SplitDataset, Wb97SplitDataset],
]: ...
@overload
def get_wb97_fold_dataset(
    fold_num: Literal[0, 1, 2, 3, 4],
    include_pretrain: Literal[False] = False,
) -> tuple[Wb97SplitDataset, Wb97SplitDataset, Wb97SplitDataset]: ...
def get_wb97_fold_dataset(
    fold_num: Literal[0, 1, 2, 3, 4],
    include_pretrain: bool = False,
):
    split_dir = './wb97xd3/splits'

    train_path = _abs_path(f'{split_dir}/fold_{fold_num}/aam_train.csv')
    test_path = _abs_path(f'{split_dir}/fold_{fold_num}/aam_test.csv')
    val_path = _abs_path(f'{split_dir}/fold_{fold_num}/aam_val.csv')

    pretrain_train_path = _abs_path(f'{split_dir}/fold_{fold_num}/aam_pretrain_train.csv')
    pretrain_test_path = _abs_path(f'{split_dir}/fold_{fold_num}/aam_pretrain_test.csv')
    pretrain_val_path = _abs_path(f'{split_dir}/fold_{fold_num}/aam_pretrain_val.csv')

    datasets = (
        Wb97SplitDataset(train_path),
        Wb97SplitDataset(test_path),
        Wb97SplitDataset(val_path),
    )

    if not include_pretrain:
        return datasets

    return (
        datasets,
        (
            Wb97SplitDataset(pretrain_train_path),
            Wb97SplitDataset(pretrain_test_path),
            Wb97SplitDataset(pretrain_val_path),
        ),
    )


def _abs_path(relative_path: str) -> str:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current_dir, relative_path)
