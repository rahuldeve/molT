import torch
import torch.nn as nn


class ModellingHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, out_size, config):
        super().__init__()
        self.out_size = out_size
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, out_size, bias=False)

    def forward(self, features):
        # x = self.dense(features)
        x = self.layer_norm(features)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


class ExpDive(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp):
        return torch.exp(inp) - torch.exp(-inp)
