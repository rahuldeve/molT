import torch.nn as nn

from ..utils import TokenType
from .base import ExpDive, ModellingHead


class CLSTargetRegressionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, 1, bias=False)
        self.exp_dive = ExpDive()

    def forward(self, features, target_values, mm_mask, token_type_ids):
        features = self.linear(features[:, 0, :])
        preds = self.exp_dive(features).squeeze()

        # calculate loss only for tokens that are mol descriptors and have been masked
        # we do this by zeroing out rmse error based on final_mask
        target_token_mask = token_type_ids == TokenType.TGT
        target_values = target_values[target_token_mask]
        loss = nn.functional.mse_loss(preds, target_values)
        return loss, preds, target_values
