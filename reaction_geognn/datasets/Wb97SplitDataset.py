from base_classes import GeoGNNDataset
from geognn.datasets.shared_definitions import load_smiles_csv
from utils import abs_path


class Wb97SplitDataset(GeoGNNDataset):
    """Dataset class for loading a SINGLE wB97X-D3 fold-split CSV file.

    Computational activation energies (but lacking the enthalpy values found in
    the original wB97X-D3 dataset) of forward and reversed reaction at the
    ωB97X-D3/def2-TZVP level of theory.

    Data/Labels are `tensor([activation_energy])`.

    The fold-split is as defined in the paper -
    `"Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction"`.

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
