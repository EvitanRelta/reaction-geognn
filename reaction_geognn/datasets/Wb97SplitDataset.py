import os
from typing import Sized, cast

import pandas as pd
import torch
from geognn.datasets.shared_definitions import GeoGNNDataElement, GeoGNNDataset
from torch.utils.data import Dataset
from typing_extensions import override


class Wb97SplitDataset(GeoGNNDataset):
    """
    Base dataset class for loading a wb97xd3 split CSV file.

    Split CSV files are downloadable at: \\
    https://github.com/hesther/reactiondatabase/tree/main/data_splits

    Which were split and used by: \\
    "Machine Learning of Reaction Properties via Learned Representations of
    the Condensed Graph of Reaction"
    https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00975
    """

    def __init__(self, csv_path: str) -> None:
        """
        Args:
            csv_path (str): Path to the wb97xd3 split dataset's `.csv` file.
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        super().__init__(
            smiles_column_name = 'AAM',
            data_columns_to_use = ['ea'],
            csv_path = os.path.join(current_dir, csv_path), # Set path relative to this file.
        )
