from typing import Optional, Tuple, Union

import torch
from transformers.utils import logging

from ..tranformer import MolTModel, MolTPreTrainedModel
from .atom_props import AtomPropModellingHead
from .base import MoleculeModellingOutput
from .bond_props import BondPropModellingHead
from .mlm import TokenModellingHead
from .mol_descriptors import MolDescriptorModellingHead

logger = logging.get_logger(__name__)


class MolTForMaskedMM(MolTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.graph_transformer = MolTModel(config, add_pooling_layer=False)
        self.token_head = TokenModellingHead(config)
        self.atom_prop_head = AtomPropModellingHead(config)
        self.bond_prop_head = BondPropModellingHead(config)
        self.mol_desc_head = MolDescriptorModellingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pos_embeds: Optional[torch.Tensor] = None,
        pos_embeds_shape: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        token_idxs: Optional[torch.Tensor] = None,
        mm_mask: Optional[torch.Tensor] = None,
        atom_props: Optional[torch.Tensor] = None,
        bond_props: Optional[torch.Tensor] = None,
        mol_desc: Optional[torch.Tensor] = None,

        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple[torch.Tensor], MoleculeModellingOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.graph_transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pos_embeds=pos_embeds,
            pos_embeds_shape=pos_embeds_shape,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,

            atom_props=AtomPropModellingHead.adjust_for_input(
                atom_props, mm_mask, token_type_ids
            ),
            bond_props=BondPropModellingHead.adjust_for_input(
                bond_props, mm_mask, token_type_ids
            ),
            mol_desc=MolDescriptorModellingHead.adjust_for_input(
                mol_desc, mm_mask, token_type_ids
            ),
        )

        sequence_output = outputs[0]
        molecule_modelling_loss, _ = self.token_head(
            sequence_output, labels, mm_mask, token_type_ids
        )

        atom_prop_loss, _ = self.atom_prop_head(
            sequence_output, atom_props, mm_mask, token_type_ids
        )

        bond_prop_loss, _ = self.bond_prop_head(
            sequence_output, bond_props, mm_mask, token_type_ids
        )

        mol_desc_loss, _ = self.mol_desc_head(
            sequence_output, mol_desc, mm_mask, token_type_ids
        )

        loss = None
        if (
            molecule_modelling_loss is not None
            and atom_prop_loss is not None
            and bond_prop_loss is not None
            and mol_desc_loss is not None
        ):
            loss = (
                molecule_modelling_loss
                + atom_prop_loss
                + bond_prop_loss
                + 0.01 * mol_desc_loss
            )

        return MoleculeModellingOutput(
            loss=loss,
            molecule_modelling_loss=molecule_modelling_loss,
            atom_prop_loss=atom_prop_loss,
            bond_prop_loss=bond_prop_loss,
            mol_desc_loss=mol_desc_loss,
        )
