import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem.rdchem import StereoType

from ..utils import TokenType, unpack_bond_properties
from .base import ModellingHead


class BondPropModellingHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_aromatic_type = 3
        self.num_conjugated_type = 3
        self.num_stereo_types = len(StereoType.values) + 1

        self.aromatic_head = ModellingHead(self.num_aromatic_type, config)
        self.conjugated_head = ModellingHead(self.num_conjugated_type, config)
        self.stereo_head = ModellingHead(self.num_stereo_types, config)

    @staticmethod
    def adjust_for_input(bond_props, mm_mask, token_type_ids):
        bond_mask = token_type_ids == TokenType.BOND
        final_mask = bond_mask & mm_mask
        # For now, just setting all masked tokens to padding idx
        return bond_props.masked_fill(final_mask.unsqueeze(1), 0.0)

    @staticmethod
    def adjust_for_loss(bond_props, mm_mask, token_type_ids):
        bond_mask = token_type_ids == TokenType.BOND
        final_mask = bond_mask & mm_mask
        # any prop=-100 will be ignored by cross entropy loss
        # set props of any tokens not selected in final_mask to -100
        return bond_props.masked_fill(~final_mask.unsqueeze(1), -100)

    def forward(self, features, bond_props, mm_mask, token_type_ids):
        bond_is_aromatic = self.aromatic_head(features)
        bond_is_conjugated = self.conjugated_head(features)
        bond_stereo = self.stereo_head(features)

        loss = None
        if bond_props is not None:
            # any prop=-100 is ignored by cross_entropy
            bond_props = self.adjust_for_loss(bond_props, mm_mask, token_type_ids)

            bond_props = unpack_bond_properties(bond_props)
            prop_bond_aromatic = bond_props["prop_bond_aromatic"]
            prop_bond_conjugated = bond_props["prop_bond_conjugated"]
            prop_bond_stereo = bond_props["prop_bond_stereo"]

            bond_aromatic_loss = F.cross_entropy(
                bond_is_aromatic.view(-1, self.num_aromatic_type),
                prop_bond_aromatic.reshape(-1).long(),
            )

            bond_conjugated_loss = F.cross_entropy(
                bond_is_conjugated.view(-1, self.num_conjugated_type),
                prop_bond_conjugated.reshape(-1).long(),
            )

            bond_stero_loss = F.cross_entropy(
                bond_stereo.view(-1, self.num_stereo_types),
                prop_bond_stereo.reshape(-1).long(),
            )

            loss = bond_aromatic_loss + bond_stero_loss + bond_conjugated_loss

        prediction_dict = {
            "bond_is_aromatic": bond_is_aromatic,
            "bond_is_conjugated": bond_is_conjugated,
            "bond_stereo": bond_stereo,
        }

        return loss, prediction_dict
