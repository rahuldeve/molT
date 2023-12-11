import torch
import torch.nn as nn

from ..utils import TokenType
from .base import ModellingHead


class ExpDive(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp):
        return torch.exp(inp) - torch.exp(-inp)


class MolDescriptorModellingHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.projection = ModellingHead(1, config)
        self.exp_dive = ExpDive()

    @staticmethod
    def adjust_for_input(mol_descriptors, mm_mask, token_type_ids):
        mol_descriptor_token_mask = token_type_ids == TokenType.DESC
        final_mask = mm_mask & mol_descriptor_token_mask
        return torch.where(final_mask, 1.0, mol_descriptors)

    def forward(self, features, mol_descriptors, mm_mask, token_type_ids):
        preds = self.projection(features)
        preds = self.exp_dive(preds).squeeze()

        # calculate loss only for tokens that are mol descriptors and have been masked
        # we do this by zeroing out rmse error based on final_mask
        mol_desc_token_mask = token_type_ids == TokenType.DESC
        final_mask = mol_desc_token_mask & mm_mask
        rmse_error = torch.norm(preds - mol_descriptors, 2)
        # zero out any token not in final_mask
        rmse_error = rmse_error.masked_fill(~final_mask, 0.0)
        loss = rmse_error.sum(dim=-1).mean()
        return loss, preds
