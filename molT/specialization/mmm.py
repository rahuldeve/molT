from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from transformers.utils import ModelOutput, logging

from ..modelling_heads import (
    AtomPropModellingHead,
    BondPropModellingHead,
    MolFeatureModellingHead,
    TokenModellingHead,
)
from ..tranformer import MolTModel, MolTPreTrainedModel

logger = logging.get_logger(__name__)


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
        self.mol_feat_head = MolFeatureModellingHead(config)
        self.target_head = TokenModellingHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pos_embed_idxs: Optional[torch.Tensor] = None,
        lp_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        mm_mask: Optional[torch.Tensor] = None,
        atom_props: Optional[torch.Tensor] = None,
        bond_props: Optional[torch.Tensor] = None,
        mol_features: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
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
            pos_embed_idxs=pos_embed_idxs,
            lp_embeds=lp_embeds,
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
            mol_features=MolFeatureModellingHead.adjust_for_input(
                mol_features, mm_mask, token_type_ids
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

        mol_feature_loss, _ = self.mol_feat_head(
            sequence_output, mol_features, mm_mask, token_type_ids
        )

        loss = None
        if (
            molecule_modelling_loss is not None
            and atom_prop_loss is not None
            and bond_prop_loss is not None
            and mol_feature_loss is not None
        ):
            loss = (
                molecule_modelling_loss
                + atom_prop_loss
                + bond_prop_loss
                + mol_feature_loss
            )

        return MoleculeModellingOutput(
            loss=loss,
            molecule_modelling_loss=molecule_modelling_loss,
            atom_prop_loss=atom_prop_loss,
            bond_prop_loss=bond_prop_loss,
            mol_feature_loss=mol_feature_loss,
        )

    @staticmethod
    def report_metrics(eval_results):
        (
            mm_loss,
            atom_prop_loss,
            bond_prop_loss,
            mol_feature_loss,
        ) = eval_results.predictions

        return {
            "mm_loss": mm_loss.mean(),
            "atom_prop_loss": atom_prop_loss.mean(),
            "bond_prop_loss": bond_prop_loss.mean(),
            "mol_feature_loss": mol_feature_loss.mean(),
        }
