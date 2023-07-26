"""
Aliases for RDKit's types, as RDKit doesn't have proper type hintings.
"""

from typing import Any, NewType, TypeAlias

from rdkit.Chem import rdchem  # type: ignore

Atom: TypeAlias = rdchem.Atom
"""A molecular atom (alias for `rdkit.Chem.rdchem.Atom`)."""

Bond: TypeAlias = rdchem.Bond
"""A molecular bond (alias for `rdkit.Chem.rdchem.Bond`)."""

Mol: TypeAlias = rdchem.Mol
"""A molecule (alias for `rdkit.Chem.rdchem.Mol`)."""

Conformer: TypeAlias = rdchem.Conformer
"""A molecular conformer (alias for `rdkit.Chem.rdchem.Conformer`)."""

RDKitEnum = NewType('RDKitEnum', Any)
"""An enum from `rdkit.Chem.rdchem` (eg. `rdchem.ChiralType`)."""

RDKitEnumValue = NewType('RDKitEnumValue', Any)
"""A value for an enum in `rdkit.Chem.rdchem` (eg. `rdchem.ChiralType.CHI_UNSPECIFIED`)."""
