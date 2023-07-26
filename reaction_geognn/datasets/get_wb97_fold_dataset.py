from typing import Literal, TypeAlias, overload

from utils import abs_path

from .Wb97SplitDataset import Wb97SplitDataset

Wb97FoldSplitTuple: TypeAlias = tuple[Wb97SplitDataset, Wb97SplitDataset, Wb97SplitDataset]
B97FoldSplitTuple: TypeAlias = Wb97FoldSplitTuple

@overload
def get_wb97_fold_dataset(
    fold_num: Literal[0, 1, 2, 3, 4],
    include_pretrain: Literal[True],
) -> tuple[Wb97FoldSplitTuple, B97FoldSplitTuple]: ...
@overload
def get_wb97_fold_dataset(
    fold_num: Literal[0, 1, 2, 3, 4],
    include_pretrain: Literal[False] = False,
) -> Wb97FoldSplitTuple: ...
def get_wb97_fold_dataset(
    fold_num: Literal[0, 1, 2, 3, 4],
    include_pretrain: bool = False,
):
    """Gets the wB97X-D3 (and optionally the B97-D3) fold-splits as defined by
    the paper - `"Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction"`

    Args:
        fold_num (Literal[0, 1, 2, 3, 4]): wB97X-D3 fold-split number as defined \
            by the paper - `"Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction"`
        include_pretrain (bool, optional): Whether to include the respective \
            fold-split from pretraining-dataset (ie. B97-D3). Defaults to False.

    Returns:
        Wb97FoldSplitTuple | tuple[Wb97FoldSplitTuple, B97FoldSplitTuple]: \
            If `include_pretrain=False`, returns only the fold-split for wB97X-D3 \
            dataset as `(train_split, test_split, val_split)`. \
            If `include_pretrain=True`, returns both fold-splits for wB97X-D3 and \
            B97-D3 datasets as `(wb97xd3_split_tuple, b97d3_split_tuple)` where \
            each split tuple is in the form `(train_split, test_split, val_split)`.
    """
    SPLIT_DIR = './wb97xd3/splits'

    train_path = abs_path(f'{SPLIT_DIR}/fold_{fold_num}/aam_train.csv', __file__)
    test_path = abs_path(f'{SPLIT_DIR}/fold_{fold_num}/aam_test.csv', __file__)
    val_path = abs_path(f'{SPLIT_DIR}/fold_{fold_num}/aam_val.csv', __file__)

    pretrain_train_path = abs_path(f'{SPLIT_DIR}/fold_{fold_num}/aam_pretrain_train.csv', __file__)
    pretrain_test_path = abs_path(f'{SPLIT_DIR}/fold_{fold_num}/aam_pretrain_test.csv', __file__)
    pretrain_val_path = abs_path(f'{SPLIT_DIR}/fold_{fold_num}/aam_pretrain_val.csv', __file__)

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
