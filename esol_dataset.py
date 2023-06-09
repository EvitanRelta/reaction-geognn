#!/usr/bin/python
#-*-coding:utf-8-*-
#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processing of esol dataset.

ESOL (delaney) is a standard regression data set,which is also called delaney dataset. In the dataset, you can find  the structure and water solubility data of 1128 compounds.  It's a good choice to validate machine learning models and to estimate solubility directly based on molecular structure which was encoded in SMILES string.

You can download the dataset from
https://moleculenet.org/datasets-1 and load it into pahelix reader creators.

This is a PyTorch equivalent of GeoGNN's `esol_dataset.py`:
https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/datasets/esol_dataset.py
"""

from typing import TypedDict
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class ESOLDataElement(TypedDict):
    """A data entry in the ESOL dataset."""

    smiles: str
    """SMILES string of the data's molecule."""

    label: Tensor
    """Ground truth log-solublity in mols/L."""

class ESOLDataset(Dataset):
    def __init__(self) -> None:
        csv_path = './geognn_datasets/chemrl_downstream_datasets/esol/raw/delaney-processed.csv'
        self.dataset, self.mean, self.std = load_esol_dataset(csv_path)

    def __getitem__(self, index: int) -> ESOLDataElement:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

def load_esol_dataset(csv_path: str) -> tuple[list[ESOLDataElement], Tensor, Tensor]:
    """
    Loads the ESOL dataset, and the dataset's mean and standard deviation.
    
    Example of a data element: {
        'smiles': 'OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O',
        'label': tensor([-0.7700], dtype=torch.float64)
    }

    Args:
        csv_path (str, optional): Path to the dataset's `.csv` file. \
            Defaults to './geognn_datasets/chemrl_downstream_datasets/esol/raw/delaney-processed.csv'.

    Returns:
        tuple[DatasetData, Tensor, Tensor]: Returns the dataset data, their mean \
            and their standard deviation in that order.
    """
    task_names = ['measured log solubility in mols per litre']
    input_df = pd.read_csv(csv_path, sep=',')
    smiles_list = input_df['smiles']
    labels = input_df[task_names]

    data_list: list[ESOLDataElement] = []
    for i in range(len(labels)):
        data: ESOLDataElement = {
            'smiles': smiles_list[i],
            'label': torch.tensor(labels.values[i], dtype=torch.float32),
        }
        data_list.append(data)
    return data_list, *get_esol_stat(labels)


def get_esol_stat(labels: pd.DataFrame) -> tuple[Tensor, Tensor]:
    """Return mean and std of labels"""
    label_values = torch.tensor(labels.values)
    return torch.mean(label_values, 0).float(), torch.std(label_values, 0).float()
