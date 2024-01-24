import torch
import torch.nn as nn

from ..utils import TokenType
from .base import ModellingHead


class ExpDive(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp):
        return torch.exp(inp) - torch.exp(-inp)


class MolFeatureModellingHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1)
        )
        # self.exp_dive = ExpDive()

    @staticmethod
    @torch.no_grad()
    def adjust_for_input(mol_features, mm_mask, token_type_ids):
        mol_feature_token_mask = token_type_ids == TokenType.FEAT
        final_mask = mm_mask & mol_feature_token_mask
        return torch.where(final_mask, 1.0, mol_features)

    def forward(self, features, mol_features, mm_mask, token_type_ids):
        preds = self.projection(features).squeeze()
        # preds = self.exp_dive(preds).squeeze()

        # calculate loss only for tokens that are mol descriptors and have been masked
        # we do this by zeroing out rmse error based on final_mask
        mol_feature_token_mask = token_type_ids == TokenType.FEAT
        final_mask = mol_feature_token_mask & mm_mask
        mse_error = torch.pow(preds - mol_features, 2)
        # zero out any token not in final_mask
        mse_error = mse_error.masked_fill(~final_mask, 0.0)
        loss = mse_error.mean(dim=0).sum()
        return loss, preds.masked_fill(~mol_feature_token_mask, 0.0)
