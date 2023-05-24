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
            graph.ndata[feat_name] = torch.tensor([feat.get_value(atom) for atom in mol.GetAtoms()])

        # Add edge features.
        for feat_name, feat in Utils.FEATURES['bond_feats'].items():
            graph.edata[feat_name] = torch.tensor([feat.get_value(bond) for bond in mol.GetBonds()])

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