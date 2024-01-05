from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.utils import ModelOutput

from ..utils import TokenType
from .base import ModellingHead, ExpDive


# @dataclass
# class CLSRegressionOutput(ModelOutput):
#     loss: Optional[torch.FloatTensor] = None
#     target_loss: Optional[torch.FloatTensor] = None
#     pred_target_values: Optional[torch.FloatTensor] = None
#     true_target_values: Optional[torch.FloatTensor] = None


class CLSTargetRegressionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.projection = ModellingHead(1, config)
        self.exp_dive = ExpDive()

    def forward(self, features, target_values, mm_mask, token_type_ids):
        preds = self.projection(features[:, 0, :])
        preds = self.exp_dive(preds).squeeze()

        # calculate loss only for tokens that are mol descriptors and have been masked
        # we do this by zeroing out rmse error based on final_mask
        target_token_mask = token_type_ids == TokenType.TGT
        target_values = target_values[target_token_mask]
        loss = nn.functional.mse_loss(preds, target_values)
        return loss, preds, target_values
