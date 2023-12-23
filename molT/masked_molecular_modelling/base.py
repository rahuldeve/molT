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
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, out_size)
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.decoder.bias = self.bias

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias

    def forward(self, features):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x
    

@dataclass
class MoleculeModellingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    molecule_modelling_loss: Optional[torch.FloatTensor] = None
    atom_prop_loss: Optional[torch.FloatTensor] = None
    bond_prop_loss: Optional[torch.FloatTensor] = None
    mol_desc_loss: Optional[torch.FloatTensor] = None
    target_loss: Optional[torch.FloatTensor] = None

    target_mask: Optional[torch.FloatTensor] = None
    pred_target_values: Optional[torch.FloatTensor] = None
    true_target_values: Optional[torch.FloatTensor] = None
    
    # logits: Optional[torch.FloatTensor] = None
    # hidden_states: Optional[torch.FloatTensor] = None
    # attentions: Optional[torch.FloatTensor] = None
