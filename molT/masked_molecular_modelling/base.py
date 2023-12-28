from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.activations import gelu
from transformers.utils import ModelOutput


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


@dataclass
class MoleculeModellingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    molecule_modelling_loss: Optional[torch.FloatTensor] = None
    atom_prop_loss: Optional[torch.FloatTensor] = None
    bond_prop_loss: Optional[torch.FloatTensor] = None
    mol_feature_loss: Optional[torch.FloatTensor] = None
    target_loss: Optional[torch.FloatTensor] = None

    target_mask: Optional[torch.FloatTensor] = None
    pred_target_values: Optional[torch.FloatTensor] = None
    true_target_values: Optional[torch.FloatTensor] = None
