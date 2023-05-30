import dgl, torch, numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem
from dgl import DGLGraph
from dgl.transforms.functional import add_reverse_edges, to_simple
from typing import TypeAlias, Any, Callable, Literal
from dataclasses import dataclass


Atom: TypeAlias = rdchem.Atom
Bond: TypeAlias = rdchem.Bond
Mol: TypeAlias = rdchem.Mol
Conformer: TypeAlias = rdchem.Conformer

FeatureCategory: TypeAlias = Literal['atom_feats', 'bond_feats']
FeatureName: TypeAlias = str

@dataclass
class Feature:
    get_value: Callable[[Atom | Bond], Any]
    """Gets the feature value from an `rdchem.Atom` / `rdchem.Bond` instance."""
    possible_values: list[Any]
    """All possible values this feature can take on."""

    def get_encoded_feat_value(self, x: Atom | Bond) -> int:
        """
        Gets the label-encoded value of the feature from `x`.

        For the feature value of `x` obtained by `self.get_value`, get the index
        of said value relative to the `self.possible_values` list, and +1 to it.

        For example:
        ```
        possible_values = ['a', 'b', 'c']
        value = get_value(x)  # if value == 'b', ie. index 1
        get_index(x)  # then `get_index` returns 1+1 = 2
        ```

        ## Note:

        The +1 is modelled after the +1 to the atom/bond ID in GeoGNN's
        preprocessing:
        https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/utils/compound_tools.py#L609

        But idk what the +1 is for. It seems to be an index reserved for
        Out-Of-Vocabulary (OOV) values based on the GeoGNN comment `# 0: OOV`.
        But since `safe_index` returns the last index (which is the `"misc"`
        label of the features), unknown/unseen feature values are already mapped
        to the last index. So the "OOV" 0-th index will completely unused?

        Args:
            x (Atom | Bond): The `rdchem.Atom` / `rdchem.Bond` instance.

        Returns:
            int: Index of feature value, relative to the `self.possible_values` list.
        """
        value = self.get_value(x)
        return self.possible_values.index(value) + 1


def to_bidirected_copy(g: DGLGraph) -> DGLGraph:
    """Exactly the same as `dgl.to_bidirected`, but copies both node and edge
    features.

    Args:
        g (DGLGraph): The input directed graph.

    Returns:
        DGLGraph: Graph `g` but bidirected and with copied node/edge features.
    """
    g = add_reverse_edges(g, copy_ndata=True, copy_edata=True)
    g = to_simple(g, return_counts=None, copy_ndata=True, copy_edata=True)
    return g

class Utils:
    RdChemEnum: TypeAlias = Any     # RdChem's enums have no proper typings.
    @staticmethod
    def _rdchem_enum_to_list(rdchem_enum: RdChemEnum) -> list[RdChemEnum]:
        """Converts an enum from `rdkit.Chem.rdchem` (eg. `rdchem.ChiralType`)
        to a list of all the possible enum valuess.

        Args:
            rdchem_enum (RdChemEnum): An enum defined in `rdkit.Chem`.

        Returns:
            list[RdChemEnum]: All possible enum values in a list.
        """
        return [rdchem_enum.values[i] for i in range(len(rdchem_enum.values))]

    FEATURES: dict[FeatureCategory, dict[FeatureName, Feature]] = {
        'atom_feats': {
            'atomic_num': Feature(
                lambda atom: atom.GetAtomicNum(),
                list(range(1, 119)) + ['misc'],
            ),
            'chiral_tag': Feature(
                lambda atom: atom.GetChiralTag(),
                _rdchem_enum_to_list(rdchem.ChiralType),
            ),
            'degree': Feature(
                lambda atom: atom.GetDegree(),
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
            ),
            'formal_charge': Feature(
                lambda atom: atom.GetFormalCharge(),
                [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
            ),
            'hybridization': Feature(
                lambda atom: atom.GetHybridization(),
                _rdchem_enum_to_list(rdchem.HybridizationType),
            ),
            'is_aromatic': Feature(
                lambda atom: int(atom.GetIsAromatic()),
                [0, 1],
            ),
            'total_numHs': Feature(
                lambda atom: atom.GetTotalNumHs(),
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
            ),
        },
        'bond_feats': {
            'bond_dir': Feature(
                lambda bond: bond.GetBondDir(),
                _rdchem_enum_to_list(rdchem.BondDir),
            ),
            'bond_type': Feature(
                lambda bond: bond.GetBondType(),
                _rdchem_enum_to_list(rdchem.BondType),
            ),
            'is_in_ring': Feature(
                lambda bond: int(bond.IsInRing()),
                [0, 1],
            ),
        }
    }
    """
    Features as defined in GeoGNN's utility functions/classes:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/pahelix/utils/compound_tools.py

    Not all features in the above `compound_tools.py` are included, as not all
    was actually used by the GeoGNN model. Only those specified in the GeoGNN's
    config are included:
    https://github.com/PaddlePaddle/PaddleHelix/blob/e93c3e9/apps/pretrained_compound/ChemRL/GEM/model_configs/geognn_l8.json
    """

    @staticmethod
    def smiles_to_graph(smiles: str) -> DGLGraph:
        """Convert a molecule's SMILES string into a DGL graph.

        Args:
            smiles (str): A molecule's SMILES string.

        Returns:
            DGLGraph: The molecule in graph form.
        """
        mol = AllChem.MolFromSmiles(smiles)
        # mol, conf = Utils._generate_conformer(mol)

        # Create an undirected DGL graph with all the molecule's nodes and edges.
        num_bonds = mol.GetNumBonds()
        edges = torch.zeros(num_bonds, dtype=torch.int32), torch.zeros(num_bonds, dtype=torch.int32)
        for i, bond in enumerate(mol.GetBonds()):
            edges[0][i] = bond.GetBeginAtomIdx()
            edges[1][i] = bond.GetEndAtomIdx()
        graph = dgl.graph(edges, idtype=torch.int32)

        # Add node features.
        for feat_name, feat in Utils.FEATURES['atom_feats'].items():
            graph.ndata[feat_name] = torch.tensor([feat.get_encoded_feat_value(atom) for atom in mol.GetAtoms()])

        # Add edge features.
        for feat_name, feat in Utils.FEATURES['bond_feats'].items():
            graph.edata[feat_name] = torch.tensor([feat.get_encoded_feat_value(bond) for bond in mol.GetBonds()])

        graph = to_bidirected_copy(graph)   # Convert to undirected graph.
        graph = graph.to('cuda:0')  # Copies graph to GPU. (https://docs.dgl.ai/guide/graph-gpu.html)
        return graph

    @staticmethod
    def _generate_conformer(mol: Mol, numConfs: int = 10) -> tuple[Mol, Conformer]:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        index = np.argmin([x[1] for x in res])
        conf = new_mol.GetConformer(id=int(index))
        return new_mol, conf