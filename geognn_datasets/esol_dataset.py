"""
Processing of esol dataset.

ESOL (delaney) is a standard regression data set,which is also called delaney dataset. In the dataset, you can find  the structure and water solubility data of 1128 compounds.  It's a good choice to validate machine learning models and to estimate solubility directly based on molecular structure which was encoded in SMILES string.

You can download the dataset from
https://moleculenet.org/datasets-1 and load it into pahelix reader creators.

This is a PyTorch equivalent of GeoGNN's `esol_dataset.py`:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/datasets/esol_dataset.py
"""

from typing import cast
import pandas as pd
import torch
from torch.utils.data import Dataset

from .shared_definitions import GeoGNNDataElement


class ESOLDataset(Dataset[GeoGNNDataElement]):
    """
    Water solubility data (log-solubility in mols/L) for common small organic
    molecules.

    The `data` Tensor in each element is the ground truth water log-solublity
    in mols/L, size `(1, )`.

    The dataset can be downloaded from:
    https://moleculenet.org/datasets-1
    """

    def __init__(
        self,
        csv_path: str = './geognn_datasets/chemrl_downstream_datasets/esol/raw/delaney-processed.csv'
    ) -> None:
        """
        Args:
            csv_path (str, optional): Path to the dataset's `.csv` file. \
                Defaults to './geognn_datasets/chemrl_downstream_datasets/esol/raw/delaney-processed.csv'.
        """
        columns_to_use = ['measured log solubility in mols per litre']

        raw_df = pd.read_csv(csv_path, sep=',')
        smiles_list = raw_df['smiles'].values
        filtered_data = torch.tensor(raw_df[columns_to_use].values, dtype=torch.float32)

        self.data_list: list[GeoGNNDataElement] = []
        for i in range(len(filtered_data)):
            self.data_list.append({
                'smiles': cast(str, smiles_list[i]),
                'data': filtered_data[i]
            })

    def __getitem__(self, index: int) -> GeoGNNDataElement:
        return self.data_list[index]

    def __len__(self) -> int:
        return len(self.data_list)
