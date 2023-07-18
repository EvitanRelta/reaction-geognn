from typing import Callable, TypeVar

from torch.utils.data import Dataset

INPUT_TYPE = TypeVar('INPUT_TYPE')
OUTPUT_TYPE = TypeVar('OUTPUT_TYPE', covariant=True)

class TransformDataset(Dataset[OUTPUT_TYPE]):
    """Transforms a dataset of type `Dataset[INPUT_TYPE]` to type `Dataset[OUTPUT_TYPE]`
    using the function `transfrom: Callable[[INPUT_TYPE], OUTPUT_TYPE]` defined
    in the constructor.
    """
    def __init__(
        self,
        dataset: Dataset[INPUT_TYPE],
        transform: Callable[[INPUT_TYPE], OUTPUT_TYPE],
    ):
        """
        Args:
            dataset (Dataset[INPUT_TYPE]): Dataset to transform.
            transform (Callable[[INPUT_TYPE], OUTPUT_TYPE]): Function to transform \
                each element in `dataset`.
        """
        self.data_list: list[OUTPUT_TYPE] = [transform(data) for data in dataset]

    def __getitem__(self, idx: int) -> OUTPUT_TYPE:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)
