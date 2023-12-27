from enum import IntEnum, auto
import numpy as np


class TokenType(IntEnum):
    SPECIAL = 0
    ATOM = 1
    BOND = 2
    FEAT = 3
    TGT = 4


def pack_atom_properties(
    *, prop_atom_in_ring, prop_atom_charge, prop_atom_hybridization, prop_atom_chirality
):
    return np.stack(
        [
            prop_atom_in_ring,
            prop_atom_charge,
            prop_atom_hybridization,
            prop_atom_chirality,
        ],
        axis=0,
    )


def unpack_atom_properties(atom_properties):
    if len(atom_properties.shape) == 2:
        return {
            "prop_atom_in_ring": atom_properties[0],
            "prop_atom_charge": atom_properties[1],
            "prop_atom_hybridization": atom_properties[2],
            "prop_atom_chirality": atom_properties[3],
        }
    else:
        assert len(atom_properties.shape) == 3
        return {
            "prop_atom_in_ring": atom_properties[:, 0],
            "prop_atom_charge": atom_properties[:, 1],
            "prop_atom_hybridization": atom_properties[:, 2],
            "prop_atom_chirality": atom_properties[:, 3],
        }


def pack_bond_properties(*, prop_bond_aromatic, prop_bond_conjugated, prop_bond_stereo):
    return np.stack(
        [prop_bond_aromatic, prop_bond_conjugated, prop_bond_stereo], axis=0
    )


def unpack_bond_properties(bond_properties):
    if len(bond_properties.shape) == 2:
        return {
            "prop_bond_aromatic": bond_properties[0],
            "prop_bond_conjugated": bond_properties[1],
            "prop_bond_stereo": bond_properties[2],
        }
    else:
        assert len(bond_properties.shape) == 3
        return {
            "prop_bond_aromatic": bond_properties[:, 0],
            "prop_bond_conjugated": bond_properties[:, 1],
            "prop_bond_stereo": bond_properties[:, 2],
        }
