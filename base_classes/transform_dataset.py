from collections.abc import Sized
from typing import Callable, TypeVar, cast

from torch.utils.data import Dataset

INPUT_TYPE = TypeVar('INPUT_TYPE')
OUTPUT_TYPE = TypeVar('OUTPUT_TYPE', covariant=True)

class TransformDataset(Dataset[OUTPUT_TYPE]):
    """Transforms a dataset of type `Dataset[INPUT_TYPE]` to type `Dataset[OUTPUT_TYPE]`
    using the function `transfrom: Callable[[INPUT_TYPE], OUTPUT_TYPE]` defined
    in the constructor.

    ### Warning
    Setting `transform_on_get = True` in the constructor while data-loading with
    `num_workers > 0` might throw an error.
    """
    def __init__(
        self,
        dataset: Dataset[INPUT_TYPE],
        transform: Callable[[INPUT_TYPE], OUTPUT_TYPE],
        transform_on_get: bool = False,
    ):
        """
        Args:
            dataset (Dataset[INPUT_TYPE]): Dataset to transform.
            transform (Callable[[INPUT_TYPE], OUTPUT_TYPE]): Function to transform \
                each element in `dataset`.
            transform_on_get (bool, optional): If `True`, only transform a data \
                element when it's being getted. If `False`, transforms ALL \
                elements in the constructor. (WARNING: Data-loading with \
                `num_workers > 0` might throw an error when this is set to `True`) \
                Defaults to False.
        """
        self.raw_dataset = dataset
        self.transform = transform
        self.data_list: list[OUTPUT_TYPE]

        if transform_on_get:
            assert isinstance(self.raw_dataset, Sized)
            self.data_list = cast(list[OUTPUT_TYPE], [None for _ in range(len(self.raw_dataset))])
        else:
            self.data_list = [transform(data) for data in self.raw_dataset]

    def __getitem__(self, idx: int) -> OUTPUT_TYPE:
        if self.data_list[idx] == None:
            self.data_list[idx] = self.transform(self.raw_dataset[idx])
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)
