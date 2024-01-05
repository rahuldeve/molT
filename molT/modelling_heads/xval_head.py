import torch
import torch.nn as nn

from ..utils import TokenType
from .base import ExpDive, ModellingHead


class XValTargetRegressionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.projection = ModellingHead(1, config)
        self.exp_dive = ExpDive()

    @staticmethod
    def adjust_for_input(target_values, mm_mask, token_type_ids, in_training=True):
        target_token_mask = token_type_ids == TokenType.TGT
        if in_training:
            final_mask = mm_mask & target_token_mask
        else:
            final_mask = target_token_mask

        return torch.where(final_mask, 1.0, target_values)

    def forward(self, features, target_values, mm_mask, token_type_ids):
        preds = self.projection(features)
        preds = self.exp_dive(preds).squeeze()

        # calculate loss only for tokens that are mol descriptors and have been masked
        # we do this by zeroing out rmse error based on final_mask
        target_token_mask = token_type_ids == TokenType.TGT
        final_mask = target_token_mask & mm_mask
        mse_error = torch.pow(preds - target_values, 2)
        # zero out any token not in final_mask
        mse_error = mse_error.masked_fill(~final_mask, 0.0)
        loss = mse_error.mean(dim=0).sum()
        return loss, preds[target_token_mask], target_values[target_token_mask]
