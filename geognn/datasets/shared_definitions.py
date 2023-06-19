from typing import Sequence, TypedDict, cast

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class GeoGNNDataElement(TypedDict):
    """A data entry for GeoGNN."""

    smiles: str
    """SMILES string of the data's molecule."""

    data: Tensor
    """Ground truth data. Size `(num_of_feats, num_of_entries)`"""


class GeoGNNDataset(Dataset[GeoGNNDataElement]):
    """Base class for a dataset used by GeoGNN."""

    def __init__(
        self,
        smiles_column_name: str,
        data_columns_to_use: list[str],
        csv_path: str,
    ) -> None:
        raw_df = pd.read_csv(csv_path, sep=',')
        smiles_list = raw_df[smiles_column_name].values
        smiles_list = cast(Sequence[str], smiles_list)

        filtered_data = torch.tensor(raw_df[data_columns_to_use].values, dtype=torch.float32)

        self.data_list: list[GeoGNNDataElement] = []
        for i in range(len(filtered_data)):
            self.data_list.append({
                'smiles': smiles_list[i],
                'data': filtered_data[i]
            })

    def __getitem__(self, index: int) -> GeoGNNDataElement:
        return self.data_list[index]

    def __len__(self) -> int:
        return len(self.data_list)
