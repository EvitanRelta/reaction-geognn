from typing import Literal

import torch
from base_classes import GeoGNNCacheDataModule, GeoGNNDataset, GeoGNNGraphs
from typing_extensions import override

from .datasets import get_wb97_fold_dataset
from .graph_utils import superimpose_reactant_products_graphs
from .preprocessing import reaction_smart_to_graph


class Wb97DataModule(GeoGNNCacheDataModule):
    """`LightningDataModule` class for wB97X-D3 fold-split.

    The fold-split is as defined in the paper -
    `"Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction"`.

    Contains computational activation energies (but lacking the enthalpy values
    found in the original wB97X-D3 dataset) of forward AND reversed reaction at the
    ωB97X-D3/def2-TZVP level of theory.

    Data/Labels are `tensor([activation_energy])`.

    Split CSV files are downloadable at: \\
    https://github.com/hesther/reactiondatabase/tree/main/data_splits

    Which were split and used by: \\
    "Machine Learning of Reaction Properties via Learned Representations of
    the Condensed Graph of Reaction"
    https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00975
    """

    @override
    def get_dataset_splits(self) -> tuple[GeoGNNDataset, GeoGNNDataset, GeoGNNDataset]:
        return get_wb97_fold_dataset(self.fold_num)

    @override
    @classmethod
    def compute_graphs(cls, smiles: str) -> GeoGNNGraphs:
        atom_bond_graph, bond_angle_graph = reaction_smart_to_graph(smiles, torch.device('cpu'))
        return (
            atom_bond_graph,
            bond_angle_graph,
            superimpose_reactant_products_graphs(atom_bond_graph),
        )

    def __init__(
        self,
        fold_num: Literal[0, 1, 2, 3, 4],
        batch_size: int,
        shuffle: bool = False,
        cache_path: str | None = None,
    ):
        self.fold_num: Literal[0, 1, 2, 3, 4] = fold_num
        super().__init__(batch_size, shuffle, cache_path)


class B97DataModule(Wb97DataModule):
    """`LightningDataModule` class for B97-D3 fold-split.

    The fold-split is as defined in the paper -
    `"Machine Learning of Reaction Properties via Learned Representations of the Condensed Graph of Reaction"`.

    Contains computational activation energies (but lacking the enthalpy values 
    found in the original wB97X-D3 dataset) of forward AND reversed reaction at the
    B97-D3/def2-mSVP level of theory.

    Labels are `tensor([activation_energy])`.

    Split CSV files are downloadable at: \\
    https://github.com/hesther/reactiondatabase/tree/main/data_splits

    Which were split and used by: \\
    "Machine Learning of Reaction Properties via Learned Representations of
    the Condensed Graph of Reaction"
    https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.1c00975
    """

    @override
    def get_dataset_splits(self) -> tuple[GeoGNNDataset, GeoGNNDataset, GeoGNNDataset]:
        wb97, b97 = get_wb97_fold_dataset(self.fold_num, include_pretrain=True)
        return b97
