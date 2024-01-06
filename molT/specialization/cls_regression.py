from typing import Optional, Tuple, Union
from dataclasses import dataclass
import torch
from transformers.utils import logging, ModelOutput
from sklearn import metrics

from ..tranformer import MolTModel, MolTPreTrainedModel
from ..modelling_heads import CLSTargetRegressionHead


logger = logging.get_logger(__name__)

@dataclass
class CLSRegressionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    target_loss: Optional[torch.FloatTensor] = None
    pred_target_values: Optional[torch.FloatTensor] = None
    true_target_values: Optional[torch.FloatTensor] = None



class CLSRegression(MolTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.graph_transformer = MolTModel(config, add_pooling_layer=False)
        self.target_head = CLSTargetRegressionHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pos_embed_ids: Optional[torch.Tensor] = None,
        lp_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        mm_mask: Optional[torch.Tensor] = None,
        atom_props: Optional[torch.Tensor] = None,
        bond_props: Optional[torch.Tensor] = None,
        mol_features: Optional[torch.Tensor] = None,
        target_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # the following args are not needed but huggingface tokenizer
        # will not send them to the collator if its not specified here
        token_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], CLSRegressionOutput]:
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
            pos_embed_ids=pos_embed_ids,
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
            atom_props=atom_props,
            bond_props=bond_props,
            mol_features=mol_features,
        )

        sequence_output = outputs[0]
        target_loss, pred_target_values, target_values = self.target_head(
            sequence_output, target_values, mm_mask, token_type_ids
        )

        return CLSRegressionOutput(
            loss=target_loss,
            target_loss=target_loss,
            pred_target_values=pred_target_values,
            true_target_values=target_values,  # type: ignore
        )
    
    @staticmethod
    def report_metrics(eval_results):
        (
            target_loss,
            pred_target_values,
            true_target_values,
        ) = eval_results.predictions

        y_true = pred_target_values
        y_pred = true_target_values

        r2_score = metrics.r2_score(y_true, y_pred)
        mae = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)

        return {
            "target_loss": target_loss.mean(),
            "target_r2": r2_score,
            "target_mae": mae,
            "target_mse": mse,
        }
