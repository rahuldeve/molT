import torch.nn as nn

from ..utils import TokenType


class SumRegressionHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, features, target_values, mm_mask, token_type_ids):
        relevant_mask = (token_type_ids != TokenType.SPECIAL)
        features = relevant_mask.unsqueeze(-1) * features
        preds = self.linear(features.sum(dim=-2)).squeeze()

        # calculate loss only for tokens that are mol descriptors and have been masked
        # we do this by zeroing out rmse error based on final_mask
        loss = nn.functional.mse_loss(preds, target_values)
        return loss, preds, target_values
