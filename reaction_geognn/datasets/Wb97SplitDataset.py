from geognn.datasets.shared_definitions import load_smiles_csv
from geognn_base_classes import GeoGNNDataset
from utils import abs_path


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
        data_list = load_smiles_csv(
            csv_path = abs_path(csv_path, __file__), # Set path relative to this file.
            smiles_column_name = 'AAM',
            data_columns_to_load = ['ea'],
        )
        super().__init__(data_list)
