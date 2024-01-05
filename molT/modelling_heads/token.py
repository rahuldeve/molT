import torch.nn as nn
import torch.nn.functional as F

from ..utils import TokenType
from .base import ModellingHead


class TokenModellingHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.token_head = ModellingHead(config.vocab_size, config)

    def forward(self, features, labels, mm_mask, token_type_ids):
        prediction_scores = self.token_head(features)
        loss = None
        if labels is not None:
            atom_mask = token_type_ids == TokenType.ATOM
            bond_mask = token_type_ids == TokenType.BOND
            final_mask = mm_mask & (atom_mask | bond_mask)
            labels = labels.masked_fill(~final_mask, -100)
            loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        return loss, prediction_scores
