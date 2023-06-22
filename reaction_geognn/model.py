import os
from typing import Callable, cast

import dgl
import pandas as pd
import torch
from dgl import DGLGraph
from geognn import GeoGNNModel
from geognn.datasets import GeoGNNDataElement, GeoGNNDataLoader, GeoGNNDataset
from geognn.layers import DropoutMLP
from torch import Tensor, nn
from torch.utils.data import Dataset

from .preprocessing import reaction_smart_to_graph


class ProtoModel(nn.Module):
    def __init__(self, out_size: int, dropout_rate: float = 0.2) -> None:
        super().__init__()
        self.embed_dim = 32
        self.geognn = GeoGNNModel(dropout_rate=dropout_rate)
        self.norm = nn.LayerNorm(self.geognn.embed_dim)
        self.mlp = DropoutMLP(
            num_of_layers = 2,
            in_size = self.geognn.embed_dim,
            hidden_size = 128,
            out_size = out_size,
            activation = nn.LeakyReLU(),
            dropout_rate = dropout_rate,
        )

    def forward(self, atom_bond_graph: DGLGraph, bond_angle_graph: DGLGraph) -> Tensor:
        """
        Args:
            atom_bond_graph (DGLGraph): Graph of a molecule, with atoms as \
                nodes, bonds as edges.
            bond_angle_graph (DGLGraph): Graph of a molecule, with bonds as \
                nodes, bond-angles as edges.

        Returns:
            Tensor: Predicted values with size `(self.out_size, )`.
        """
        batched_node_repr, batched_edge_repr, batched_graph_repr \
            = self.geognn.forward(atom_bond_graph, bond_angle_graph)


        pred_list: list[Tensor] = []
        for node_repr, graph in ProtoModel._split_batched_data(batched_node_repr, atom_bond_graph):
            reactant_node_repr, product_node_repr \
                = ProtoModel._split_reactant_product_nodes(node_repr, graph)

            diff_node_repr = product_node_repr - reactant_node_repr

            # Sum over the node dimension
            diff_node_repr = diff_node_repr.sum(dim=0)  # shape is now (embed_dim, )

            diff_node_repr = self.norm.forward(diff_node_repr)
            pred = self.mlp.forward(diff_node_repr)
            pred_list.append(pred)

        return torch.stack(pred_list)

    @staticmethod
    def _split_batched_data(
        batched_node_repr: Tensor,
        batched_atom_bond_graph: DGLGraph,
    ) -> list[tuple[Tensor, DGLGraph]]:
        output: list[tuple[Tensor, DGLGraph]] = []
        start_index = 0
        for graph in dgl.unbatch(batched_atom_bond_graph):
            num_nodes = graph.number_of_nodes()
            node_repr = batched_node_repr[start_index : start_index + num_nodes]
            output.append((node_repr, graph))
        return output

    @staticmethod
    def _split_reactant_product_nodes(
        node_repr: Tensor,
        atom_bond_graph: DGLGraph,
    ) -> tuple[Tensor, Tensor]:
        assert '_is_reactant' in atom_bond_graph.ndata, \
            'Atom-bond graphs needs to have .ndata["_is_reactant"] of dtype=bool.'
        assert len(node_repr) % 2 == 0, 'Odd number of nodes in node_repr.'

        mask = atom_bond_graph.ndata['_is_reactant']
        assert isinstance(mask, Tensor) and mask.dtype == torch.bool
        reactant_node_repr = node_repr[mask]
        product_node_repr = node_repr[~mask]

        assert len(reactant_node_repr) == len(node_repr) // 2
        assert len(product_node_repr) == len(node_repr) // 2
        return reactant_node_repr, product_node_repr


class ProtoDataLoader(GeoGNNDataLoader):
    def __init__(
        self,
        dataset: Dataset[GeoGNNDataElement],
        fit_mean: Tensor,
        fit_std: Tensor,
        batch_size: int,
        shuffle: bool = True,
        device: torch.device = torch.device('cpu'),
        cached_graphs: dict[str, tuple[DGLGraph, DGLGraph]] = {},
        worker_init_fn: Callable[[int], None] | None = None,
        generator: torch.Generator | None = None
    ) -> None:
        super().__init__(
            dataset,
            fit_mean,
            fit_std,
            batch_size,
            shuffle,
            device,
            cached_graphs,
            self._collate_fn,
            worker_init_fn,
            generator,
        )

    def _collate_fn(self, batch: list[GeoGNNDataElement]) -> tuple[DGLGraph, DGLGraph, Tensor]:
        atom_bond_graphs: list[DGLGraph] = []
        bond_angle_graphs: list[DGLGraph] = []
        data_list: list[Tensor] = []
        for elem in batch:
            smiles, data = elem['smiles'], elem['data']

            if smiles in self._cached_graphs:
                atom_bond_graph, bond_angle_graph = self._cached_graphs[smiles]
            else:
                graphs = reaction_smart_to_graph(smiles, self.device)
                atom_bond_graph, bond_angle_graph = graphs
                self._cached_graphs[smiles] = graphs

            atom_bond_graphs.append(atom_bond_graph)
            bond_angle_graphs.append(bond_angle_graph)
            data_list.append(data)

        data = torch.stack(data_list).to(self.device)
        return (
            dgl.batch(atom_bond_graphs),
            dgl.batch(bond_angle_graphs),
            GeoGNNDataLoader._standardize_data(data, self.fit_mean, self.fit_std),
        )
