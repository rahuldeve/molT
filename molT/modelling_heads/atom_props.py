import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem.rdchem import ChiralType, HybridizationType

from ..utils import TokenType, unpack_atom_properties
from .base import ModellingHead


class AtomPropModellingHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_in_ring_type = 3
        self.num_hybridization_type = len(HybridizationType.values) + 1
        self.num_chirality_type = len(ChiralType.values) + 1
        self.num_charge_types = 4

        self.in_ring_head = ModellingHead(self.num_in_ring_type, config)
        self.hybridization_head = ModellingHead(self.num_hybridization_type, config)
        self.chirality_head = ModellingHead(self.num_chirality_type, config)
        self.charge_head = ModellingHead(self.num_charge_types, config)

    @staticmethod
    @torch.no_grad()
    def adjust_for_input(atom_props, mm_mask, token_type_ids):
        atom_mask = token_type_ids == TokenType.ATOM        
        final_mask = atom_mask & mm_mask
        # final mask has elements that needs to be masked
        # we can invert this mask and multiply it with atom_props
        # to simulate masked_fill with 0.0 on atom_props
        return atom_props * ~(final_mask.unsqueeze(-1))

    @staticmethod
    @torch.no_grad()
    def adjust_for_loss(atom_props, mm_mask, token_type_ids):
        atom_mask = token_type_ids == TokenType.ATOM
        final_mask = atom_mask & mm_mask
        # any prop=-100 will be ignored by cross entropy loss
        return atom_props.to_dense().masked_fill(~final_mask.unsqueeze(-1), -100)

    def forward(self, features, atom_props, mm_mask, token_type_ids):
        atom_in_ring = self.in_ring_head(features)
        atom_hybridization = self.hybridization_head(features)
        atom_chirality = self.chirality_head(features)
        atom_charge = self.charge_head(features)

        loss = None
        if atom_props is not None:
            # any prop=-100 is ignored by cross_entropy
            atom_props = self.adjust_for_loss(atom_props, mm_mask, token_type_ids)

            atom_props = unpack_atom_properties(atom_props)
            prop_atom_charge = atom_props["prop_atom_charge"]
            prop_atom_chirality = atom_props["prop_atom_chirality"]
            prop_atom_hybridization = atom_props["prop_atom_hybridization"]
            prop_atom_in_ring = atom_props["prop_atom_in_ring"]

            atom_in_ring_loss = F.cross_entropy(
                atom_in_ring.view(-1, self.num_in_ring_type),
                prop_atom_in_ring.reshape(-1).long(),
            )

            atom_hybridization_loss = F.cross_entropy(
                atom_hybridization.view(-1, self.num_hybridization_type),
                prop_atom_hybridization.reshape(-1).long(),
            )

            atom_chirality_loss = F.cross_entropy(
                atom_chirality.view(-1, self.num_chirality_type),
                prop_atom_chirality.reshape(-1).long(),
            )

            atom_charge_loss = F.cross_entropy(
                atom_charge.view(-1, self.num_charge_types),
                prop_atom_charge.reshape(-1).long(),
            )

            loss = (
                atom_in_ring_loss
                + atom_hybridization_loss
                + atom_chirality_loss
                + atom_charge_loss
            )

        prediction_dict = {
            "atom_hybridization": atom_hybridization,
            "atom_chirality": atom_chirality,
            "atom_charge": atom_charge,
        }

        return loss, prediction_dict
