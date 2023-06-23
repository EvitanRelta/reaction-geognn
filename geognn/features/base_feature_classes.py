from typing import Any, Callable, Literal, cast, overload

import torch
from dgl import DGLGraph
from torch import FloatTensor, IntTensor, Tensor

from mygnn.geognn.features.rdkit_types import Conformer, Mol

from .rdkit_types import Atom, Bond, Conformer, Mol


class Feature:
    @overload
    def __init__(self, *, name: str, get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], Tensor]) -> None: ...
    @overload
    def __init__(self, *, name: str, feat_type: Literal['atom'], get_value: Callable[[Atom, Mol, Conformer, DGLGraph | None], int | float], dtype: torch.dtype) -> None: ...
    @overload
    def __init__(self, *, name: str, feat_type: Literal['bond'], get_value: Callable[[Bond, Mol, Conformer, DGLGraph | None], int | float], dtype: torch.dtype) -> None: ...
    def __init__(
        self,
        *,
        name: str,
        dtype: torch.dtype | None = None,
        get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], Tensor] | None = None,
        feat_type: Literal['atom', 'bond'] | None = None,
        get_value: Callable[[Atom | Bond, Mol, Conformer, DGLGraph | None], int | float] | None = None,
    ) -> None:
        self.name: str = name
        self.dtype: torch.dtype | None = dtype

        if get_feat_values != None:
            self.get_feat_values = get_feat_values
            return
        assert feat_type != None and get_value != None and dtype != None

        if feat_type == 'atom':
            self.get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], Tensor] \
                = lambda mol, conf, atom_bond_graph : torch.tensor([
                    get_value(atom, mol, conf, atom_bond_graph) for atom in mol.GetAtoms() # type: ignore
                ], dtype=self.dtype)
        elif feat_type == 'bond':
            self.get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], Tensor] \
                = lambda mol, conf, atom_bond_graph : torch.tensor([
                    get_value(bond, mol, conf, atom_bond_graph) for bond in mol.GetBonds() # type: ignore
                ], dtype=self.dtype)
        else:
            raise ValueError(f'Expected `feat_type` to be `Literal["atom", "bond"]`, but got "{feat_type}".')



class FloatFeature(Feature):
    @overload
    def __init__(self, *, name: str, centers: Tensor, gamma: float, get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], FloatTensor]) -> None: ...
    @overload
    def __init__(self, *, name: str, centers: Tensor, gamma: float, feat_type: Literal['atom'], get_value: Callable[[Atom, Mol, Conformer, DGLGraph | None], float], dtype: torch.dtype=torch.float32) -> None: ...
    @overload
    def __init__(self, *, name: str, centers: Tensor, gamma: float, feat_type: Literal['bond'], get_value: Callable[[Bond, Mol, Conformer, DGLGraph | None], float], dtype: torch.dtype=torch.float32) -> None: ...
    def __init__(
        self,
        *,
        name: str,
        centers: Tensor,
        gamma: float,
        dtype: torch.dtype | None = torch.float32,
        get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], FloatTensor] | None = None,
        feat_type: Literal['atom', 'bond'] | None = None,
        get_value: Callable[[Atom | Bond, Mol, Conformer, DGLGraph | None], float] | None = None,
    ) -> None:
        super().__init__(name=name, dtype=dtype, get_feat_values=get_feat_values, feat_type=feat_type, get_value=get_value) # type: ignore
        self.centers = centers
        self.gamma = gamma


class LabelEncodedFeature(Feature):
    @overload
    def __init__(self, *, name: str, feat_type: Literal['atom'], possible_values: list[Any], get_raw_value: Callable[[Atom, Mol, Conformer, DGLGraph | None], Any], dtype: torch.dtype = torch.uint8) -> None: ...
    @overload
    def __init__(self, *, name: str, feat_type: Literal['bond'], possible_values: list[Any], get_raw_value: Callable[[Bond, Mol, Conformer, DGLGraph | None], Any], dtype: torch.dtype = torch.uint8) -> None: ...
    def __init__(
        self,
        *,
        name: str,
        feat_type: Literal['atom', 'bond'],
        possible_values: list[Any],
        get_raw_value: Callable[[Atom | Bond, Mol, Conformer, DGLGraph | None], Any],
        dtype: torch.dtype = torch.uint8,
    ) -> None:
        super().__init__(name=name, feat_type=feat_type, get_value=self._get_encoded_value, dtype=dtype) # type: ignore
        self.possible_values = possible_values
        self.get_raw_value = get_raw_value

        # Update the return type of `self.get_feat_values` to be `IntTensor`.
        self.get_feat_values: Callable[[Mol, Conformer, DGLGraph | None], IntTensor] \
            = cast(Callable[[Mol, Conformer, DGLGraph | None], IntTensor], self.get_feat_values)

    def _get_encoded_value(self, x: Atom | Bond, mol: Mol, conf: Conformer, atom_bond_graph: DGLGraph | None = None) -> int:
        value = self.get_raw_value(x, mol, conf, atom_bond_graph)
        try:
            return self.possible_values.index(value) + 1
        except ValueError:
            # If `value` doesn't exist in `self.possible_values`,
            # return the last element's index (assumes 'misc' is the last index).
            return len(self.possible_values) - 1
