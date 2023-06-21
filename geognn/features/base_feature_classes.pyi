from numbers import Number
from typing import Any

from dgl import DGLGraph
from torch import FloatTensor, IntTensor, Tensor

from .rdkit_types import Atom, Bond, Conformer, Mol

class Feature:
    """Base class for a feature."""

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): Name of this feature, which will be the feature's key \
                in `DGLGraph.ndata` / `.edata`.
        """

    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor:
        """
        Gets the feature values of all atoms/bonds from a `rdkit.Chem.rdchem.Mol`
        and `Conformer` instance.

        Args:
            mol (Mol): Target molecule, type `rdkit.Chem.rdchem.Mol`.
            conf (Conformer): Target molecular conformer, type `rdkit.Chem.rdchem.Conformer`.
            atom_bond_graph (DGLGraph | None, optional): Graph of the molecule, \
                with atoms as nodes, bonds as edges. Defaults to `None`.

        Returns:
            Tensor: Feature values, size `(num_of_atoms/bonds, )`.
        """



class FloatFeature(Feature):
    """Base class for a feature with values of datatype `float`."""

    centers: list[float]
    gamma: float
    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> FloatTensor: ...



class LabelEncodedFeature(Feature):
    possible_values: list[Any]
    """All the possible unencoded values this feature can take on."""

    def _get_unencoded_value(self, x: Atom | Bond, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Any:
        """
        Gets the feature's unencoded value from a `rdkit.Chem.rdchem.Atom`
        / `Bond` instance.

        Args:
            x (Atom | Bond): Target `rdkit.Chem.rdchem.Atom` / `Bond` instance.
            mol (Mol): Molecule of the atom, type `rdkit.Chem.rdchem.Mol`.
            conf (Conformer): Molecular conformer of the atom, type `rdkit.Chem.rdchem.Conformer`.
            atom_bond_graph (DGLGraph | None, optional): Graph of the molecule, \
                with atoms as nodes, bonds as edges. Defaults to `None`.

        Returns:
            Any: The unencoded value of the feature.
        """

    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> IntTensor: ...

    def _get_value(self, x: Atom | Bond, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> int:
        """
        Gets the label-encoded (one-indexed, relative to `self.possible_values`)
        value of the feature from a `rdkit.Chem.rdchem.Atom` / `Bond` instance.

        For the feature value of `x` obtained by `self._get_unencoded_value`, get the index
        of said value relative to the `self.possible_values` list, and +1 to it.

        For example:
        ```
        possible_values = ['a', 'b', 'c']
        value = _get_unencoded_value(x)  # if value == 'b', ie. index 1
        _get_value(x)  # then `_get_value` returns 1 + 1 = 2
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
            x (Atom | Bond): Target `rdkit.Chem.rdchem.Atom` / `Bond` instance.
            mol (Mol): Molecule of the atom, type `rdkit.Chem.rdchem.Mol`.
            conf (Conformer): Molecular conformer of the atom, type `rdkit.Chem.rdchem.Conformer`.
            atom_bond_graph (DGLGraph | None, optional): Graph of the molecule, \
                with atoms as nodes, bonds as edges. Defaults to `None`.

        Returns:
            int: Label-encoded feature value.
        """



class AtomFeature(Feature):
    """Base class for atom features."""

    def _get_value(self, x: Atom, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Number:
        """Gets the feature's value from a `rdkit.Chem.rdchem.Atom` instance."""

    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor: ...




class BondFeature(Feature):
    """Base class for bond features."""

    def _get_value(self, x: Bond, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Number:
        """Gets the feature's value from a `rdkit.Chem.rdchem.Bond` instance."""

    def get_feat_values(self, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> Tensor: ...
